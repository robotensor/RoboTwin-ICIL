"""Run a simulator command under a watchdog that reruns it when it hangs or loses the GPU.

SAPIEN's camera read can hang for good on a Blackwell GPU that another process is loading, and the
renderer can lose the GPU outright (see "Rendering" in docs/install.md). Each attempt ends one of
three ways:

  stalled     The attempt's process tree has not gained CPU time for --stall seconds, or the
              attempt ran longer than --max-wall seconds. The process group is killed and the
              command rerun.
  transient   The attempt exited non-zero and its own output, the last 64 KB it appended to --log,
              matches a --rerun-on pattern (by default SAPIEN's ErrorDeviceLost). It is rerun.
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
import subprocess
import sys
import time

GAVE_UP = 124
TAIL_BYTES = 64 * 1024
DEFAULT_RERUN_ON = ("ErrorDeviceLost",)


def tree_cpu_seconds(root: int) -> float:
    """User + system CPU seconds of `root` and every descendant, read from /proc."""
    ticks = os.sysconf("SC_CLK_TCK")
    children: dict[int, list[int]] = {}
    stats: dict[int, float] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        pid, ppid = int(entry), int(fields[1])
        children.setdefault(ppid, []).append(pid)
        stats[pid] = (int(fields[11]) + int(fields[12])) / ticks
    total, stack = 0.0, [root]
    while stack:
        pid = stack.pop()
        total += stats.get(pid, 0.0)
        stack.extend(children.get(pid, []))
    return total


def exit_status(returncode: int) -> int:
    """A Popen return code as a shell reports it: a death by signal N is 128+N."""
    return 128 - returncode if returncode < 0 else returncode


def run_once(
    cmd: list[str], log, *, stall: float, max_wall: float, poll: float
) -> tuple[int | None, str]:
    """Run one attempt: its exit status, or None when the watchdog killed it, and why it ended.

    A hung SAPIEN camera read still burns a trickle of CPU polling the GPU, so besides the
    CPU-progress test there is a wall-clock cap per attempt: a run that has not finished by then
    is treated as hung too.
    """
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    started = time.monotonic()
    last_cpu, last_change = -1.0, started
    while True:
        code = proc.poll()
        if code is not None:
            return exit_status(code), f"exited {exit_status(code)}"
        cpu = tree_cpu_seconds(proc.pid)
        now = time.monotonic()
        if cpu > last_cpu + 0.5:
            last_cpu, last_change = cpu, now
        if now - last_change > stall or now - started > max_wall:
            why = (
                f"stalled: no CPU progress for {stall:.0f}s"
                if now - last_change > stall
                else f"stalled: still running after {max_wall:.0f}s wall"
            )
            print(f"[simwatch] {why} at {cpu:.1f}s CPU; killing", file=log, flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return None, why
        time.sleep(poll)


def log_tail(path: str, offset: int, limit: int = TAIL_BYTES) -> str:
    """The last `limit` bytes appended to `path` after `offset`, decoded leniently."""
    with open(path, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        f.seek(max(offset, size - limit))
        return f.read().decode(errors="replace")


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
                tail = log_tail(log_path, offset)
                marker = next((p.pattern for p in rerun_on if p.search(tail)), None)
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
        help="an attempt whose CPU time has not grown for this long is stalled "
        "(default: %(default)g)",
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
        help="rerun an attempt that exits non-zero when this pattern matches the last 64 KB of "
        "its output; repeatable, and any use replaces the default (default: "
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
        help="file the command's stdout and stderr, and simwatch's own lines, are appended to",
    )
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not args.cmd:
        parser.error("no command given after --")
    if args.retries < 0:
        parser.error("--retries must not be negative")
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
        max_wall=args.max_wall,
        poll=args.poll,
    )


if __name__ == "__main__":
    sys.exit(main())
