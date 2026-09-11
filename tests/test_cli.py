import json

import numpy as np
import pytest

from robotwin_icil import camera_profiles, cli, png, robotwin, runner, survey
from robotwin_icil.policy import ReplayPolicy
from robotwin_icil.records import RunDir
from robotwin_icil.robotwin import RoboTwinError
from test_records import manifest, record


def test_tasks_lists_categories_and_suite_membership(capsys):
    assert cli.main(["tasks"]) == 0
    out = capsys.readouterr().out
    assert "Pick and Place (pick_and_place)" in out
    assert "click_bell  [v1]" in out


def test_a_broken_cameras_yml_is_reported_not_raised(monkeypatch, capsys):
    # The parser lists the profiles, so every subcommand reads cameras.yml.
    def broken():
        raise camera_profiles.ProfileError("cameras.yml must hold a 'profiles' mapping")

    monkeypatch.setattr(camera_profiles, "names", broken)
    assert cli.main(["tasks"]) == 1
    assert "must hold a 'profiles' mapping" in capsys.readouterr().err


def test_report_reads_a_run_directory_without_a_simulator(tmp_path, capsys):
    run = RunDir(tmp_path)
    run.start(manifest())
    run.append(record(0))
    run.append(record(1, success=False))

    assert cli.main(["report", str(tmp_path)]) == 0
    assert "50.0%" in capsys.readouterr().out
    assert cli.main(["report", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["overall"]["success_rate"] == 0.5


def test_report_refuses_a_run_that_failed_the_frozen_policy_audit(tmp_path, capsys):
    run = RunDir(tmp_path)
    run.start(manifest())
    run.append(record(0))
    run.fail_audit({"policy": "learning", "parameter_checksum": {"start": "a", "end": "b"}})

    for flags in ([], ["--json"]):
        assert cli.main(["report", str(tmp_path), *flags]) == 1
        captured = capsys.readouterr()
        assert "failed the frozen-policy audit" in captured.err
        assert captured.out == ""


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
        seen.update(spec=spec, config=config, policy=policy)
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
    far_side = camera_profiles.get("far_side").sha256[:12]
    assert f"camera profile far_side (sha256 {far_side})" in capsys.readouterr().out


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


@pytest.mark.parametrize(
    "item, expected",
    [
        ("steps=3", ("steps", 3)),
        ("temperature=0.5", ("temperature", 0.5)),
        ("lr=1.0e-4", ("lr", 0.0001)),
        ("lr=1e-4", ("lr", "1e-4")),  # YAML 1.1: no dot, no float
        ("lr=1.0e4", ("lr", "1.0e4")),  # nor without a signed exponent
        ("deterministic=true", ("deterministic", True)),
        ("deterministic=False", ("deterministic", False)),
        ("checkpoint=null", ("checkpoint", None)),
        ("checkpoint=", ("checkpoint", None)),
        ("config=configs/bpp.yml", ("config", "configs/bpp.yml")),
        ("arm=left", ("arm", "left")),
        ("revision='0123'", ("revision", "0123")),  # quoted: the string, not octal 83
        ("date=2024-01-01", ("date", "2024-01-01")),
        ("tags=[a, b]", ("tags", "[a, b]")),
        ("tag=#1", ("tag", "#1")),
        ("expr=a=b", ("expr", "a=b")),
    ],
)
def test_a_policy_arg_is_read_as_yaml_and_anything_else_stays_a_string(item, expected):
    assert cli.parse_policy_arg(item) == expected


class Configured(ReplayPolicy):
    name = "configured"

    def __init__(self, config=None, temperature=1.0, deterministic=False):
        super().__init__()
        self.config, self.temperature, self.deterministic = config, temperature, deterministic


def test_eval_passes_policy_args_to_the_policy_and_the_run(tmp_path, stop_at_run):
    argv = [*EVAL, "--run-dir", str(tmp_path), "--policy", "test_cli:Configured"]
    args = ["config=configs/bpp.yml", "temperature=0.5", "deterministic=true"]
    assert cli.main([*argv, *(x for arg in args for x in ("--policy-arg", arg))]) == 1
    policy = stop_at_run["policy"]
    assert (policy.config, policy.temperature, policy.deterministic) == (
        "configs/bpp.yml",
        0.5,
        True,
    )
    assert stop_at_run["spec"].policy_config == {
        "config": "configs/bpp.yml",
        "temperature": 0.5,
        "deterministic": True,
    }


def test_eval_without_policy_args_builds_the_policy_with_none(tmp_path, stop_at_run):
    assert cli.main([*EVAL, "--run-dir", str(tmp_path)]) == 1
    assert stop_at_run["spec"].policy_config == {}


@pytest.mark.parametrize(
    "items, message",
    [
        (["temperature"], "'temperature' is not KEY=VALUE"),
        (["=3"], "'' is not a Python identifier"),
        (["top-k=3"], "'top-k' is not a Python identifier"),
        (["1st=3"], "'1st' is not a Python identifier"),
        (["scale=.inf"], "a number must be finite"),
        (["steps=3", "steps=4"], "'steps' is given twice"),
    ],
)
def test_a_bad_policy_arg_is_a_usage_error_before_the_simulator(
    tmp_path, stop_at_run, capsys, items, message
):
    argv = [
        *EVAL,
        "--run-dir",
        str(tmp_path),
        *(x for item in items for x in ("--policy-arg", item)),
    ]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert stop_at_run == {}


def test_a_policy_arg_the_policy_does_not_take_fails_cleanly(tmp_path, stop_at_run, capsys):
    argv = [*EVAL, "--run-dir", str(tmp_path), "--policy-arg", "checkpoint=x.pt"]
    assert cli.main(argv) == 1
    assert "policy 'replay' does not take these arguments" in capsys.readouterr().err
    assert stop_at_run == {}
