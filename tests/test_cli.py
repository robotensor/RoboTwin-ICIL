import json

import numpy as np
import pytest

from robotwin_icil import camera_profiles, cli, png, robotwin, runner, survey
from robotwin_icil.records import RunDir
from robotwin_icil.robotwin import RoboTwinError
from test_records import manifest, record


def test_tasks_lists_categories_and_suite_membership(capsys):
    assert cli.main(["tasks"]) == 0
    out = capsys.readouterr().out
    assert "Pick and Place (pick_and_place)" in out
    assert "click_bell  [v1]" in out


def test_report_reads_a_run_directory_without_a_simulator(tmp_path, capsys):
    run = RunDir(tmp_path)
    run.start(manifest())
    run.append(record(0))
    run.append(record(1, success=False))

    assert cli.main(["report", str(tmp_path)]) == 0
    assert "50.0%" in capsys.readouterr().out
    assert cli.main(["report", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["overall"]["success_rate"] == 0.5


def test_report_on_a_non_run_directory_fails_cleanly(tmp_path, capsys):
    assert cli.main(["report", str(tmp_path)]) == 1
    assert "no manifest.json" in capsys.readouterr().err


def test_eval_rejects_bad_arguments_before_touching_the_simulator(tmp_path, capsys):
    base = ["eval", "--policy", "replay", "--run-dir", str(tmp_path)]
    assert cli.main([*base, "--suite", "v1", "--episodes", "0"]) == 2
    assert cli.main([*base, "--suite", "no_such_suite", "--episodes", "1"]) == 1
    assert "unknown suite" in capsys.readouterr().err


EVAL = ["eval", "--policy", "replay", "--task", "click_bell", "--episodes", "1"]


@pytest.fixture
def stop_at_run(monkeypatch):
    """Capture what `eval` would run, then stop before the simulator."""
    seen = {}

    def run(spec, policy, config):
        seen.update(spec=spec, config=config)
        raise RoboTwinError("stopped before the simulator")

    monkeypatch.setattr(runner, "run", run)
    return seen


def test_eval_runs_the_stock_cameras_by_default(tmp_path, stop_at_run):
    assert cli.main([*EVAL, "--run-dir", str(tmp_path)]) == 1
    assert stop_at_run["config"].camera_profile == "stock"
    assert stop_at_run["spec"].video_camera == "head_camera"


def test_eval_takes_a_camera_profile(tmp_path, stop_at_run):
    assert cli.main([*EVAL, "--run-dir", str(tmp_path), "--camera-profile", "far_side"]) == 1
    assert stop_at_run["config"].camera_profile == "far_side"
    assert stop_at_run["spec"].video_camera == camera_profiles.get("far_side").video_camera


def test_an_unknown_camera_profile_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([*EVAL, "--run-dir", str(tmp_path), "--camera-profile", "nope"])
    assert exc.value.code == 2
    assert "invalid choice: 'nope'" in capsys.readouterr().err


def test_survey_takes_a_camera_profile(monkeypatch, capsys):
    seen = []

    def survey_task(task_env, task, seeds, config):
        seen.append(config)
        return survey.TaskSurvey(task=task)

    monkeypatch.setattr(robotwin, "load_task", lambda name: object())
    monkeypatch.setattr(survey, "survey_task", survey_task)
    assert cli.main(["survey", "--task", "click_bell", "--seeds", "1"]) == 0
    assert cli.main(["survey", "--task", "click_bell", "--camera-profile", "far_side"]) == 0
    assert [config.camera_profile for config in seen] == ["stock", "far_side"]


def test_cameras_writes_one_png_per_camera(tmp_path, monkeypatch, capsys):
    far_side = np.full((180, 320, 3), 7, dtype=np.uint8)
    seen = {}

    def snapshot(task, seed, config):
        seen.update(task=task, seed=seed, config=config)
        return {"head_camera": np.zeros((240, 320, 3), dtype=np.uint8), "far_side_camera": far_side}

    monkeypatch.setattr(robotwin, "snapshot", snapshot)
    out = tmp_path / "cameras"
    argv = ["cameras", "--profile", "far_side", "--task", "click_bell", "--seed", "3"]
    assert cli.main([*argv, "--out", str(out)]) == 0
    assert (seen["task"], seen["seed"], seen["config"].camera_profile) == (
        "click_bell",
        3,
        "far_side",
    )
    assert sorted(path.name for path in out.iterdir()) == ["far_side_camera.png", "head_camera.png"]
    assert (out / "far_side_camera.png").read_bytes() == png.encode(far_side)
    assert "far_side_camera: 320x180" in capsys.readouterr().out


def test_cameras_refuses_a_task_outside_the_table(tmp_path, capsys):
    assert cli.main(["cameras", "--task", "nope", "--out", str(tmp_path)]) == 1
    assert "unknown task 'nope'" in capsys.readouterr().err
