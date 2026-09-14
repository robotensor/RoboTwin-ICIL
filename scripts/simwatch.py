"""Run a simulator command under a watchdog that reruns it when it hangs or loses the GPU.

SAPIEN's camera read can hang for good on a Blackwell GPU that another process is loading, and the
renderer can lose the GPU outright (see "Rendering" in docs/install.md). Each attempt ends one of
three ways:

  stalled     The attempt's process tree used under --min-cpu-rate cores on average over the last
              --stall seconds, or the attempt ran longer than --max-wall seconds. A hung camera
              read sleeps or polls the GPU with a trickle of CPU, so the test is a rate over a
              sliding window, not whether CPU time moved at all. Every process of the attempt is
              killed (SIGTERM, then SIGKILL) and, once all are gone, the command rerun.
  transient   The attempt exited non-zero and its own output, everything it appended to --log,
              matches a --rerun-on pattern (by default Vulkan's device-lost error in either
              spelling, ErrorDeviceLost or VK_ERROR_DEVICE_LOST). It is rerun.
  final       Any other exit. The watch ends with the attempt's exit code (128+N for signal N).

An attempt ends when its command exits. Anything it left running is killed first, so it can neither
write a marker into the next attempt's output nor share the GPU with it.

At most --retries reruns follow the first attempt. When the last allowed attempt also stalls or
exits transiently, the watch gives up with exit code 124. `robotwin-icil eval` and `survey` resume
from what they already wrote, so a rerun costs only the episode or seed that failed.

    python scripts/simwatch.py --stall 120 --max-wall 900 --retries 8 --log run.log -- \\
        /root/miniforge3/envs/robotwin/bin/python -m robotwin_icil.cli eval ...

Standard library only; runs under any Python 3.10+.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
from collections import deque
from typing import NamedTuple

GAVE_UP = 124
CHUNK = 1024 * 1024
OVERLAP = 64 * 1024
# What robotwin_icil.robotwin.gpu_lost() takes for a lost GPU: SAPIEN's C++ spelling and Vulkan's C one.
DEFAULT_RERUN_ON = ("ErrorDeviceLost", "VK_ERROR_DEVICE_LOST")
DEFAULT_MIN_CPU_RATE = 0.25
KILL_WAIT = 10.0


class Proc(NamedTuple):
    ppid: int
    pgrp: int
    session: int
    state: str
    start: int  # clock ticks from boot to its start; with the pid, it tells a reused pid apart
    cpu: float  # user + system seconds of every thread, plus children it has reaped


def read_procs() -> dict[int, Proc]:
    """Every process on the host, from /proc/<pid>/stat.

    That file already sums all of a process's threads, including threads that have exited, so a
    trickle of CPU in a worker thread counts; /proc/<pid>/task/*/stat would lose the exited ones.
    """
    ticks = os.sysconf("SC_CLK_TCK")
    procs = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        cpu = sum(int(tick) for tick in fields[11:15]) / ticks
        pid, ppid, pgrp, session = int(entry), int(fields[1]), int(fields[2]), int(fields[3])
        procs[pid] = Proc(ppid, pgrp, session, fields[0], int(fields[19]), cpu)
    return procs


def process_start(pid: int) -> int | None:
    """When `pid` started, in clock ticks after boot, or None once it is gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def attempt_procs(leader: int, start: int | None, procs: dict[int, Proc]) -> dict[int, Proc]:
    """The attempt's processes in `procs`: its leader, the leader's descendants and its session.

    The session still holds a descendant that init adopted when its parent exited; a worker that
    starts a session of its own is found through its parent while that parent lives. The leader is
    walked only while its pid names the process simwatch started (`start`). The session needs no
    such check: the kernel does not hand out a session's id while any process is in it.
    """
    children: dict[int, list[int]] = {}
    for pid, proc in procs.items():
        children.setdefault(proc.ppid, []).append(pid)
    stack = [pid for pid, proc in procs.items() if proc.session == leader]
    if leader in procs and procs[leader].start == start:
        stack.append(leader)
    members: dict[int, Proc] = {}
    while stack:
        pid = stack.pop()
        if pid not in members:
            members[pid] = procs[pid]
            stack.extend(children.get(pid, []))
    return members


def signal_process(pid: int, start: int, sig: int) -> None:
    """Send `sig` to `pid` only while that pid is still the process that started at `start`."""
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    except OSError:  # a kernel without pidfds: check, then signal by number
        try:
            if process_start(pid) == start:
                os.kill(pid, sig)
        except ProcessLookupError:
            pass
        return
    try:
        # The descriptor holds the process that had the pid when it was opened. If that one started
        # at `start`, a process that reuses the number later cannot receive the signal.
        if process_start(pid) == start:
            signal.pidfd_send_signal(fd, sig)
    except ProcessLookupError:
        pass
    finally:
        os.close(fd)


def kill_attempt(proc: subprocess.Popen, start: int | None, grace: float) -> int:
    """Stop every process of the attempt, and say how many were still running.

    Each gets SIGTERM, then SIGKILL once `grace` seconds have passed. This returns only when none
    is left running, so a rerun never shares the GPU with what it replaces, or KILL_WAIT seconds
    after the SIGKILL when a process is stuck in the kernel.
    """
    signalled: dict[tuple[int, int], int] = {}  # (pid, start) -> the last signal sent to it
    deadline = time.monotonic() + grace
    while True:
        procs = read_procs()
        keys = {(pid, p.start) for pid, p in attempt_procs(proc.pid, start, procs).items()}
        # One signalled already can have left the tree and session: its parent died before it did.
        keys |= {key for key in signalled if key[0] in procs and procs[key[0]].start == key[1]}
        running = [key for key in keys if procs[key[0]].state not in "ZX"]
        now = time.monotonic()
        if not running or now >= deadline + KILL_WAIT:
            break
        sig = signal.SIGKILL if now >= deadline else signal.SIGTERM
        for key in running:
            if signalled.get(key) != sig:
                signal_process(*key, sig)
                signalled[key] = sig
        time.sleep(0.1)
    proc.wait()
    return len(signalled)


def exit_status(returncode: int) -> int:
    """A Popen return code as a shell reports it: a death by signal N is 128+N."""
    return 128 - returncode if returncode < 0 else returncode


def run_once(
    cmd: list[str],
    log,
    *,
    stall: float,
    min_cpu_rate: float,
    max_wall: float,
    poll: float,
    grace: float = 20.0,
) -> tuple[int | None, str]:
    """Run one attempt: its exit status, or None when the watchdog killed it, and why it ended.

    CPU is sampled every `poll` seconds. The attempt is stalled once a window of at least `stall`
    seconds of samples shows under `min_cpu_rate` cores on average, or after `max_wall` seconds.
    """
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    start = process_start(proc.pid)  # not reaped before poll(), so readable even if it exited
    started = time.monotonic()
    used, last_total = 0.0, 0.0
    window = deque([(started, used)])
    while True:
        code = proc.poll()
        if code is not None:
            # What it left running could still write to the shared log, a marker landing in the
            # next attempt's part of it, or hold the GPU; it goes before the output is searched.
            left = kill_attempt(proc, start, grace)
            why = f"exited {exit_status(code)}"
            if left:
                why += f" (killed {left} process{'es' if left > 1 else ''} it left running)"
            return exit_status(code), why
        # A child reaped inside the tree moves into its parent's cutime, so the sum keeps it.
        total = sum(p.cpu for p in attempt_procs(proc.pid, start, read_procs()).values())
        now = time.monotonic()
        # A member reaped outside the tree takes its CPU with it; that is not negative work.
        used += max(0.0, total - last_total)
        last_total = total
        window.append((now, used))
        while len(window) > 1 and window[1][0] <= now - stall:
            window.popleft()
        since, used_then = window[0]
        why = None
        if now - since >= stall and (used - used_then) / (now - since) < min_cpu_rate:
            rate = (used - used_then) / (now - since)
            why = (
                f"stalled: {rate:.3f} cores over the last {now - since:.0f}s, "
                f"under --min-cpu-rate {min_cpu_rate:g}"
            )
        elif now - started > max_wall:
            why = f"stalled: still running after {max_wall:.0f}s wall"
        if why:
            print(
                f"[simwatch] {why} ({used:.1f}s CPU in all); killing the attempt's processes",
                file=log,
                flush=True,
            )
            kill_attempt(proc, start, grace)
            return None, why
        time.sleep(poll)


def find_marker(path: str, offset: int, patterns: list[re.Pattern[str]]) -> str | None:
    """The first of `patterns` found in what was appended to `path` after `offset`, or None.

    All of it is searched: pytest prints a failed test's captured output after the error, so the
    marker can be megabytes from the end. It is read CHUNK bytes at a time, each searched together
    with the last OVERLAP bytes of the one before, so a match up to OVERLAP long is never split.
    """
    with open(path, "rb") as f:
        f.seek(offset)
        carry = b""
        while chunk := f.read(CHUNK):
            buffer = carry + chunk
            text = buffer.decode(errors="replace")
            for pattern in patterns:
                if pattern.search(text):
                    return pattern.pattern
            carry = buffer[-OVERLAP:]
    return None


def watch(
    cmd: list[str],
    log_path: str,
    *,
    retries: int,
    rerun_on: list[re.Pattern[str]],
    **limits,
) -> int:
    """Run `cmd` until an attempt ends for good or `retries` reruns are spent; the exit status."""
    attempts = retries + 1
    with open(log_path, "a", buffering=1) as log:

        def say(line: str) -> None:
            print(f"[simwatch] {line}", file=log, flush=True)

        for attempt in range(1, attempts + 1):
            say(f"attempt {attempt}/{attempts}: {shlex.join(cmd)}")
            # Attempts share the log, so only what this attempt appended is searched for a marker.
            offset = os.fstat(log.fileno()).st_size
            code, why = run_once(cmd, log, **limits)
            end = f"attempt {attempt}/{attempts} {why}"
            if code == 0:
                say(f"{end}; done")
                return 0
            if code is not None:
                marker = find_marker(log_path, offset, rerun_on)
                if marker is None:
                    say(f"{end}, no --rerun-on match; exiting {code}")
                    return code
                end = f"{end}, output matches {marker!r}"
            say(f"{end}; {'rerunning' if attempt < attempts else 'no retries left'}")
        say(f"gave up after {attempts} attempts; exiting {GAVE_UP}")
        return GAVE_UP


def _regex(text: str) -> re.Pattern[str]:
    try:
        return re.compile(text)
    except re.error as e:
        raise argparse.ArgumentTypeError(f"bad regular expression {text!r}: {e}") from e


def _positive(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, not {text}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="simwatch.py",
        usage="%(prog)s [options] --log FILE -- CMD [ARG ...]",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stall",
        type=_positive,
        default=180.0,
        metavar="SECONDS",
        help="length of the CPU-rate window (default: %(default)g)",
    )
    parser.add_argument(
        "--min-cpu-rate",
        type=float,
        default=DEFAULT_MIN_CPU_RATE,
        metavar="CORES",
        help="an attempt whose process tree averaged fewer cores than this over the last --stall "
        "seconds is stalled; 1.0 is one core busy (default: %(default)g)",
    )
    parser.add_argument(
        "--max-wall",
        type=_positive,
        default=600.0,
        metavar="SECONDS",
        help="an attempt still running after this long is stalled (default: %(default)g)",
    )
    parser.add_argument(
        "--rerun-on",
        type=_regex,
        action="append",
        metavar="REGEX",
        help="rerun an attempt that exits non-zero when this pattern matches anything it wrote to "
        "--log; repeatable, and any use replaces the default (default: "
        + " ".join(DEFAULT_RERUN_ON)
        + ")",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        metavar="N",
        help="reruns allowed after the first attempt; exit 124 once they are spent "
        "(default: %(default)d)",
    )
    parser.add_argument(
        "--poll",
        type=_positive,
        default=5.0,
        metavar="SECONDS",
        help="how often the process tree's CPU is sampled (default: %(default)g)",
    )
    parser.add_argument(
        "--log",
        required=True,
        metavar="FILE",
        help="regular file the command's stdout and stderr, and simwatch's own lines, are appended "
        "to; it is reread for --rerun-on, so not a pipe or device",
    )
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not args.cmd:
        parser.error("no command given after --")
    if args.retries < 0 or args.min_cpu_rate < 0:
        parser.error("--retries and --min-cpu-rate must not be negative")
    try:
        mode = os.stat(args.log).st_mode
    except FileNotFoundError:
        pass  # appending creates a regular file
    else:
        # A pipe cannot seek and a device gives nothing back, so no attempt's output could be searched.
        if not stat.S_ISREG(mode):
            parser.error(
                f"--log {args.log} is not a regular file; simwatch rereads it for --rerun-on"
            )
    args.rerun_on = args.rerun_on or [re.compile(p) for p in DEFAULT_RERUN_ON]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return watch(
        args.cmd,
        args.log,
        retries=args.retries,
        rerun_on=args.rerun_on,
        stall=args.stall,
        min_cpu_rate=args.min_cpu_rate,
        max_wall=args.max_wall,
        poll=args.poll,
    )


if __name__ == "__main__":
    sys.exit(main())
