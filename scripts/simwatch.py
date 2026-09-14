"""Run a simulator command under a watchdog that reruns it when it hangs or loses the GPU.

SAPIEN's camera read can hang for good on a Blackwell GPU that another process is loading, and the
renderer can lose the GPU outright (see "Rendering" in docs/install.md). Each attempt ends one of
three ways:

  stalled     The attempt's process tree used under --min-cpu-rate cores on average over the last
              --stall seconds, or the attempt ran longer than --max-wall seconds. A hung camera
              read sleeps or polls the GPU with a trickle of CPU, so the test is a rate over a
              sliding window, not whether CPU time moved at all. The process group is killed
              (SIGTERM, then SIGKILL) and the command rerun.
  transient   The attempt exited non-zero and its own output, everything it appended to --log,
              matches a --rerun-on pattern (by default Vulkan's device-lost error in either
              spelling, ErrorDeviceLost or VK_ERROR_DEVICE_LOST). It is rerun.
  final       Any other exit. The watch ends with the attempt's exit code (128+N for signal N).

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


class Proc(NamedTuple):
    ppid: int
    pgrp: int
    session: int
    state: str
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
        procs[int(entry)] = Proc(int(fields[1]), int(fields[2]), int(fields[3]), fields[0], cpu)
    return procs


def tree_cpu_seconds(root: int) -> float:
    """CPU seconds used so far by `root`, its descendants and anything else in its session.

    A child that exits and is reaped inside the tree moves into its parent's cutime and cstime, so
    the sum does not drop when it goes; the session catches descendants reparented away.
    """
    procs = read_procs()
    children: dict[int, list[int]] = {}
    for pid, proc in procs.items():
        children.setdefault(proc.ppid, []).append(pid)
    members: set[int] = set()
    stack = [root, *(pid for pid, proc in procs.items() if proc.session == root)]
    while stack:
        pid = stack.pop()
        if pid in members or pid not in procs:
            continue
        members.add(pid)
        stack.extend(children.get(pid, []))
    return sum(procs[pid].cpu for pid in members)


def group_alive(pgid: int) -> bool:
    """Whether any process in the group is still running; zombies left to a lazy reaper do not count."""
    return any(p.pgrp == pgid and p.state not in "ZX" for p in read_procs().values())


def kill_group(proc: subprocess.Popen, grace: float) -> None:
    """SIGTERM the attempt's process group, then SIGKILL whatever is left after `grace` seconds."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    while True:
        proc.poll()
        if not group_alive(proc.pid):
            break
        if time.monotonic() >= deadline:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
        time.sleep(0.2)
    proc.wait()


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
    started = time.monotonic()
    used, last_total = 0.0, 0.0
    window = deque([(started, used)])
    while True:
        code = proc.poll()
        if code is not None:
            return exit_status(code), f"exited {exit_status(code)}"
        total = tree_cpu_seconds(proc.pid)
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
                f"[simwatch] {why} ({used:.1f}s CPU in all); killing process group {proc.pid}",
                file=log,
                flush=True,
            )
            kill_group(proc, grace)
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
