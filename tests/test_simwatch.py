"""scripts/simwatch.py against small child processes: which exits are rerun, which hangs are killed."""

import importlib.util
import os
import re
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "simwatch.py"
spec = importlib.util.spec_from_file_location("simwatch", SCRIPT)
simwatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simwatch)

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")

# A child that burns `rate` cores in `threads` worker threads, never finishing on its own. The
# main thread only waits, so the CPU is visible only when every thread of the process is counted.
TRICKLE = """
import os, sys, threading, time
rate, threads = float(sys.argv[1]), int(sys.argv[2])
print(f"pid={os.getpid()}", flush=True)
def burn():
    while True:
        end = time.thread_time() + 0.02 * rate / threads
        while time.thread_time() < end:
            pass
        time.sleep(0.02 * (1 - rate / threads))
for _ in range(threads):
    threading.Thread(target=burn, daemon=True).start()
time.sleep(3600)
"""


def watch(tmp_path, *args, log_name="run.log"):
    log = tmp_path / log_name
    started = time.monotonic()
    code = simwatch.main(["--log", str(log), *args])
    return code, log.read_text(), time.monotonic() - started


def stall_rate(log):
    return float(re.search(r"stalled: ([0-9.]+) cores over the last", log).group(1))


def python(source, *argv):
    return ["--", sys.executable, "-c", source, *map(str, argv)]


def live_group(pgid):
    return [pid for pid, p in simwatch.read_procs().items() if p.pgrp == pgid and p.state != "Z"]


def alive(pid):
    """Whether `pid` still runs; a zombie waiting for its reaper does not."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] not in "ZX"
    except OSError:
        return False


def pids(log, name):
    return [int(pid) for pid in re.findall(rf"\b{name}=(\d+)", log)]


@pytest.mark.parametrize(
    "error",
    [
        "RuntimeError: vk::Queue::submit: ErrorDeviceLost",
        # robotwin_icil's gpu_lost() takes Vulkan's C spelling as a lost GPU too.
        "error: the renderer lost the GPU while the expert ran seed 3: VK_ERROR_DEVICE_LOST",
    ],
)
def test_a_device_lost_crash_is_rerun_until_the_retries_run_out(tmp_path, error):
    crash = f"print({error!r}); raise SystemExit(1)"
    code, log, _ = watch(tmp_path, "--retries", "2", "--poll", "0.1", *python(crash))
    marker = next(m for m in ("ErrorDeviceLost", "VK_ERROR_DEVICE_LOST") if m in error)

    assert code == simwatch.GAVE_UP == 124
    assert [n for n in (1, 2, 3, 4) if f"[simwatch] attempt {n}/3: " in log] == [1, 2, 3]
    assert log.count(f"output matches {marker!r}; rerunning") == 2
    assert f"attempt 3/3 exited 1, output matches {marker!r}; no retries left" in log
    assert log.rstrip().endswith("gave up after 3 attempts; exiting 124")


def test_a_rerun_that_succeeds_ends_the_watch_with_zero(tmp_path):
    # The first attempt loses the device and leaves a marker file; the second finds it and passes.
    flag = tmp_path / "crashed-once"
    source = (
        "import pathlib, sys; flag = pathlib.Path(sys.argv[1])\n"
        "if not flag.exists(): flag.touch(); print('ErrorDeviceLost'); sys.exit(1)\n"
        "print('scored')"
    )
    code, log, _ = watch(tmp_path, "--poll", "0.1", *python(source, flag))

    assert code == 0
    assert "attempt 1/6 exited 1, output matches 'ErrorDeviceLost'; rerunning" in log
    assert "attempt 2/6 exited 0; done" in log


def test_any_other_failure_ends_the_watch_with_its_exit_code(tmp_path):
    code, log, _ = watch(tmp_path, "--poll", "0.1", *python("print('boom'); raise SystemExit(3)"))

    assert code == 3
    assert (
        log.count("attempt ") == 2 and "attempt 1/6 exited 3, no --rerun-on match; exiting 3" in log
    )


def test_only_this_attempts_output_is_searched_for_the_marker(tmp_path):
    # An earlier run left ErrorDeviceLost in the shared log; a plain failure now must not match it.
    (tmp_path / "run.log").write_text("old run\nvk::ErrorDeviceLost\n")
    code, log, _ = watch(tmp_path, "--poll", "0.1", *python("raise SystemExit(2)"))

    assert code == 2 and "no --rerun-on match" in log


def test_a_marker_followed_by_a_lot_of_output_is_still_found(tmp_path):
    # pytest prints a failed test's captured output after the error and cuts the summary line short,
    # so the marker can sit far from the end: here 3 MiB, across several of the chunks read.
    source = (
        "import sys\n"
        "print('RuntimeError: vk::Device::waitForFences: ErrorDeviceLost')\n"
        "for i in range(3 * 1024):\n"
        "    print(f'captured {i:06d} ' + 'x' * 1008)\n"
        "sys.exit(1)"
    )
    code, log, _ = watch(tmp_path, "--retries", "1", "--poll", "0.1", *python(source))

    assert len(log) > 6 * 1024 * 1024
    assert code == 124 and "attempt 2/2 exited 1, output matches 'ErrorDeviceLost'" in log


def test_a_marker_split_between_two_reads_is_found(tmp_path):
    # The patterns see each chunk together with the tail of the one before it.
    patterns = [re.compile("ErrorDeviceLost")]
    path = tmp_path / "run.log"
    for split in range(1, len("ErrorDeviceLost")):
        # The attempt starts after "earlier"; its first read ends `split` bytes into the marker.
        path.write_bytes(b"earlier" + b"y" * (simwatch.CHUNK - split) + b"ErrorDeviceLost" + b"z")
        assert simwatch.find_marker(str(path), len("earlier"), patterns) == "ErrorDeviceLost"
        assert simwatch.find_marker(str(path), path.stat().st_size - 1, patterns) is None


def test_rerun_on_patterns_replace_the_default(tmp_path):
    source = "print('CUDA error: an illegal memory access'); raise SystemExit(1)"
    args = ["--retries", "1", "--poll", "0.1", "--rerun-on", "illegal memory", "--rerun-on", "xid"]
    code, log, _ = watch(tmp_path, *args, *python(source))
    assert code == 124 and "output matches 'illegal memory'" in log

    code, log, _ = watch(
        tmp_path, "--poll", "0.1", "--rerun-on", "xid", *python("print('ErrorDeviceLost'); exit(1)")
    )
    assert code == 1


def test_a_death_by_signal_is_reported_as_128_plus_the_signal(tmp_path):
    segv = "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"
    code, _, _ = watch(tmp_path, "--poll", "0.1", *python(segv))
    assert code == 128 + 11


def test_a_trickle_of_cpu_is_a_stall_and_its_process_group_is_killed(tmp_path):
    # The shape of the hung camera read: about 0.1 core, no end. Killed after --stall, not the cap.
    limits = ["--stall", "3", "--poll", "0.25", "--max-wall", "30", "--retries", "0"]
    code, log, took = watch(tmp_path, *limits, *python(TRICKLE, 0.1, 1))

    assert code == 124
    assert 0.05 < stall_rate(log) < 0.2 and "under --min-cpu-rate 0.25" in log
    assert "still running after" not in log
    assert 3 <= took < 8
    assert live_group(int(re.search(r"^pid=(\d+)$", log, re.MULTILINE).group(1))) == []


def test_cpu_trickling_in_worker_threads_is_counted(tmp_path):
    # Four threads at 0.15 core each is 0.6 cores: over the 0.25 threshold only if all are summed.
    limits = ["--stall", "2", "--poll", "0.25", "--max-wall", "5", "--retries", "0"]
    code, log, _ = watch(tmp_path, *limits, *python(TRICKLE, 0.6, 4), log_name="busy.log")

    assert code == 124
    assert "still running after 5s wall" in log and "cores over the last" not in log

    # The same threads at a sixth of the rate are a stall, measured at their true rate.
    code, log, _ = watch(tmp_path, *limits, *python(TRICKLE, 0.1, 4), log_name="idle.log")
    assert code == 124 and 0.05 < stall_rate(log) < 0.2


def test_a_child_working_at_full_speed_outlasts_the_stall_window(tmp_path):
    busy = (
        "import time\nend = time.monotonic() + 5\nwhile time.monotonic() < end: pass\nprint('done')"
    )
    code, log, took = watch(tmp_path, "--stall", "2", "--poll", "0.25", *python(busy))

    assert code == 0 and "stalled" not in log
    assert took >= 5


def test_a_grandchild_that_ignores_sigterm_is_killed_too(tmp_path):
    # The whole group goes: a worker that shrugs off SIGTERM gets SIGKILL after the grace period.
    source = (
        "import os, signal, subprocess, sys, time\n"
        "worker = 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(3600)'\n"
        "child = subprocess.Popen([sys.executable, '-c', worker])\n"
        "print(f'pid={os.getpid()} worker={child.pid}', flush=True)\n"
        "time.sleep(3600)\n"
    )
    log_path = tmp_path / "run.log"
    started = time.monotonic()
    with open(log_path, "a", buffering=1) as log:
        code, why = simwatch.run_once(
            [sys.executable, "-c", source],
            log,
            stall=1.5,
            min_cpu_rate=0.25,
            max_wall=30,
            poll=0.25,
            grace=1.0,
        )
    text = log_path.read_text()
    pgid, worker = (int(word.split("=")[1]) for word in text.split("\n")[0].split())

    assert code is None and why.startswith("stalled")
    assert time.monotonic() - started < 8
    # Gone when run_once returns, not merely signalled: SIGKILL takes a moment on a busy host.
    assert live_group(pgid) == []
    assert not alive(worker)


def test_a_worker_in_its_own_session_is_killed_with_the_attempt(tmp_path):
    # Its CPU counts toward the attempt, so the stall kill must reach it too, not only the group.
    source = (
        "import os, subprocess, sys, time\n"
        "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3600)'],"
        " start_new_session=True)\n"
        "print(f'worker={worker.pid}', flush=True)\n"
        "time.sleep(3600)\n"
    )
    limits = ["--stall", "1.5", "--poll", "0.25", "--max-wall", "30", "--retries", "1"]
    code, log, _ = watch(tmp_path, *limits, *python(source))
    workers = pids(log, "worker")

    assert code == 124 and log.count("stalled: ") == 4 and len(workers) == 2
    assert [pid for pid in workers if alive(pid)] == []


def test_a_log_that_cannot_be_reread_is_a_usage_error(capsys):
    # simwatch rereads the log for --rerun-on; a pipe cannot seek, and /dev/null gives nothing back.
    read_end, write_end = os.pipe()
    try:
        for log in (f"/proc/self/fd/{write_end}", "/dev/null"):
            with pytest.raises(SystemExit) as exit:
                simwatch.main(["--poll", "0.1", "--log", log, *python("raise SystemExit(3)")])
            assert exit.value.code == 2 and "is not a regular file" in capsys.readouterr().err
    finally:
        os.close(read_end)
        os.close(write_end)


def test_no_command_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit:
        simwatch.main(["--log", str(tmp_path / "run.log"), "--"])
    assert exit.value.code == 2 and "no command given" in capsys.readouterr().err
