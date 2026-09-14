"""Run a simulator command under a hang watchdog, rerunning it when it stops making progress.

SAPIEN's camera read can hang for good on a Blackwell GPU that another process is loading (see
"Rendering" in docs/install.md). A hung process sleeps, or polls the GPU with a trickle of CPU,
so two signals decide that an attempt is dead: its process tree's CPU time has not advanced for
--stall seconds, or the attempt has run longer than --max-wall seconds. The tree is then killed
and the command rerun, up to --retries times. `robotwin-icil eval` and `survey` resume from what
they already wrote, so a rerun costs only the episode or seed that hung.

    python scripts/simwatch.py --stall 120 --max-wall 300 --retries 8 --log run.log -- \
        /root/miniforge3/envs/robotwin/bin/python -m robotwin_icil.cli eval ...

Standard library only; runs under any Python 3.10+.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


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


def run_once(cmd: list[str], log, stall: float, poll: float, max_wall: float) -> int | None:
    """Exit code of the command, or None when the watchdog killed it for stalling.

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
            return code
        cpu = tree_cpu_seconds(proc.pid)
        now = time.monotonic()
        if cpu > last_cpu + 0.5:
            last_cpu, last_change = cpu, now
        if now - last_change > stall or now - started > max_wall:
            why = (
                f"no CPU progress for {stall:.0f}s"
                if now - last_change > stall
                else f"over {max_wall:.0f}s wall"
            )
            print(f"[simwatch] {why} at {cpu:.1f}s CPU; killing", file=log, flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return None
        time.sleep(poll)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stall", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument("--max-wall", type=float, default=600.0, help="seconds per attempt")
    parser.add_argument("--log", required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    with open(args.log, "a", buffering=1) as log:
        for attempt in range(1, args.retries + 2):
            print(f"[simwatch] attempt {attempt}: {' '.join(cmd)}", file=log, flush=True)
            code = run_once(cmd, log, args.stall, args.poll, args.max_wall)
            if code is not None:
                print(f"[simwatch] exited {code}", file=log, flush=True)
                return code
        print("[simwatch] gave up: every attempt stalled", file=log, flush=True)
        return 124


if __name__ == "__main__":
    sys.exit(main())
