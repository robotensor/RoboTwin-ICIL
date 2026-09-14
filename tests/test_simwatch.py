"""scripts/simwatch.py against small child processes: which exits are rerun, which hangs are killed."""

import importlib.util
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "simwatch.py"
spec = importlib.util.spec_from_file_location("simwatch", SCRIPT)
simwatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simwatch)

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")


def watch(tmp_path, *args, log_name="run.log"):
    log = tmp_path / log_name
    started = time.monotonic()
    code = simwatch.main(["--log", str(log), *args])
    return code, log.read_text(), time.monotonic() - started


def python(source, *argv):
    return ["--", sys.executable, "-c", source, *map(str, argv)]


def test_a_device_lost_crash_is_rerun_until_the_retries_run_out(tmp_path):
    crash = "print('RuntimeError: vk::Queue::submit: ErrorDeviceLost'); raise SystemExit(1)"
    code, log, _ = watch(tmp_path, "--retries", "2", "--poll", "0.1", *python(crash))

    assert code == simwatch.GAVE_UP == 124
    assert [n for n in (1, 2, 3, 4) if f"[simwatch] attempt {n}/3: " in log] == [1, 2, 3]
    assert log.count("output matches 'ErrorDeviceLost'; rerunning") == 2
    assert "attempt 3/3 exited 1, output matches 'ErrorDeviceLost'; no retries left" in log
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


def test_no_command_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit:
        simwatch.main(["--log", str(tmp_path / "run.log"), "--"])
    assert exit.value.code == 2 and "no command given" in capsys.readouterr().err
