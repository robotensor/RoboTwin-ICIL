import json

import pytest

from robotwin_icil import cli, tasks
from robotwin_icil.records import RunDir
from test_records import manifest, record

EVAL = ["eval", "--policy", "replay", "--task", "click_bell", "--episodes", "1", "--run-dir", "r"]
SURVEY = ["survey", "--task", "click_bell"]


def listed(out: str) -> list[str]:
    return [line.split()[0] for line in out.splitlines() if line.startswith("  ")]


def test_tasks_lists_categories_arms_and_suite_membership(capsys):
    assert cli.main(["tasks"]) == 0
    out = capsys.readouterr().out
    assert "Pick and Place (pick_and_place)" in out
    assert "click_bell  (one arm)  [v1]" in out
    assert "stack_bowls_two  (switching arms)  [v1]" in out
    assert "lift_pot  (two arms)" in out
    assert listed(out) == list(tasks.table().tasks)


def test_tasks_with_arms_1_lists_exactly_the_one_arm_tasks(capsys):
    assert cli.main(["tasks", "--arms", "1"]) == 0
    out = capsys.readouterr().out
    one_arm = [task.name for task in tasks.table().tasks.values() if task.arms == "1"]
    assert listed(out) == one_arm
    assert len(one_arm) == 26
    assert "Bimanual Manipulation" not in out  # no one-arm task there, so no empty heading
    assert "(two arms)" not in out and "(switching arms)" not in out


def test_arms_1_refuses_a_task_that_needs_both_arms_before_touching_the_simulator(tmp_path, capsys):
    base = ["eval", "--policy", "replay", "--episodes", "1", "--run-dir", str(tmp_path)]
    assert cli.main([*base, "--task", "lift_pot", "--arms", "1"]) == 1
    err = capsys.readouterr().err
    assert (
        "task 'lift_pot' needs two arms; --arms 1 runs only tasks whose expert uses one arm" in err
    )
    assert cli.main(["survey", "--task", "handover_block", "--arms", "1"]) == 1
    assert "task 'handover_block' needs two arms" in capsys.readouterr().err


def test_arms_accepts_only_1_or_2():
    with pytest.raises(SystemExit):
        cli.main(["tasks", "--arms", "switching"])


def test_report_reads_a_run_directory_without_a_simulator(tmp_path, capsys):
    run = RunDir(tmp_path)
    run.start(manifest())
    run.append(record(0))
    run.append(record(1, success=False))

    assert cli.main(["report", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "50.0%" in out and "Embodiment:                  aloha-agilex" in out
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


@pytest.mark.parametrize("command", [EVAL, SURVEY])
def test_a_run_chooses_its_robot_or_keeps_the_task_configs(command, capsys):
    parser = cli.build_parser()
    # Without the flag the task config's own robot runs (None reaches `SceneConfig.embodiment`),
    # so `--task-config` still decides it and nothing is silently replaced.
    assert parser.parse_args(command).embodiment is None
    assert parser.parse_args([*command, "--embodiment", "franka-panda"]).embodiment == (
        "franka-panda"
    )
    assert parser.parse_args([*command, "--embodiment", "aloha-agilex"]).embodiment == (
        "aloha-agilex"
    )
    # A robot the benchmark cannot form is refused by the parser, before any simulator import.
    with pytest.raises(SystemExit) as exc:
        cli.main([*command, "--embodiment", "ur5-wsg"])
    assert exc.value.code == 2 and "aloha-agilex" in capsys.readouterr().err


@pytest.mark.parametrize("command", [EVAL, SURVEY])
def test_a_run_takes_its_robot_and_its_arms_together(command):
    # Independent choices: --embodiment picks the robot, --arms which tasks it is given.
    argv = [*command, "--embodiment", "franka-panda", "--arms", "1"]
    args = cli.build_parser().parse_args(argv)
    assert (args.embodiment, args.arms) == ("franka-panda", "1")


def test_a_one_arm_survey_of_every_task_selects_the_26_one_arm_tasks():
    argv = ["survey", "--suite", "all", "--embodiment", "franka-panda", "--arms", "1"]
    args = cli.build_parser().parse_args(argv)
    selected = tasks.table().select(suite=args.suite, task=args.task, arms=args.arms)
    assert len(selected) == 26 and {task.arms for task in selected} == {"1"}


MATERIALIZE = ["materialize", "--task", "click_bell", "--scene-seed", "11"]


@pytest.fixture
def fake_sim(monkeypatch):
    """Route materialize and run-unit to fake envs; `envs` is what a test wants load_task to give."""
    from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
    from robotwin_icil import robotwin

    envs = {"next": lambda name: FakeTaskEnv()}
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "SceneConfig", FakeConfig)
    monkeypatch.setattr(robotwin, "load_task", lambda name: envs["next"](name))
    return envs


def test_materialize_then_run_unit_through_the_cli(tmp_path, fake_sim, capsys):
    pytest.importorskip("imageio_ffmpeg")
    out, run = tmp_path / "prompt", tmp_path / "run"
    assert cli.main([*MATERIALIZE, "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True and printed["scene_seed"] == 11
    assert {p.name for p in out.iterdir()} == {"prompt.npz", "demonstration.mp4", "result.json"}

    assert (
        cli.main(
            [
                "run-unit",
                "--prompt",
                str(out / "prompt.npz"),
                "--policy",
                "replay",
                "--out",
                str(run),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["success"] is True and printed["void"] is False
    assert json.loads((run / "result.json").read_text()) == printed
    assert (run / "evaluation.mp4").is_file()


def test_a_rejected_seed_exits_3_with_its_result(tmp_path, fake_sim, capsys):
    from fake_robotwin import FakeTaskEnv

    fake_sim["next"] = lambda name: FakeTaskEnv(unstable_seeds={11})
    assert cli.main([*MATERIALIZE, "--out", str(tmp_path)]) == cli.EXIT_REJECTED == 3
    assert json.loads(capsys.readouterr().out)["rejection"] == "unstable"
    assert json.loads((tmp_path / "result.json").read_text())["ok"] is False
    assert not (tmp_path / "prompt.npz").exists()


def test_a_task_the_benchmark_does_not_score_is_refused_before_the_simulator(
    tmp_path, fake_sim, capsys
):
    assert (
        cli.main(["materialize", "--task", "nope", "--scene-seed", "1", "--out", str(tmp_path)])
        == 1
    )
    assert "unknown task 'nope'" in capsys.readouterr().err


def test_run_unit_exits_0_on_a_failed_policy_and_1_on_a_harness_error(tmp_path, fake_sim, capsys):
    from robotwin_icil import robotwin

    assert cli.main([*MATERIALIZE, "--out", str(tmp_path / "p")]) == 0
    prompt = str(tmp_path / "p" / "prompt.npz")
    assert (
        cli.main(
            ["run-unit", "--prompt", prompt, "--policy", "dummy", "--out", str(tmp_path / "r")]
        )
        == 0
    )
    capsys.readouterr()
    assert json.loads((tmp_path / "r" / "result.json").read_text())["success"] is False

    def broken(name):
        raise robotwin.RoboTwinError("no RoboTwin checkout")

    fake_sim["next"] = broken
    assert (
        cli.main(
            ["run-unit", "--prompt", prompt, "--policy", "dummy", "--out", str(tmp_path / "r2")]
        )
        == 1
    )
    assert "no RoboTwin checkout" in capsys.readouterr().err


class KwargPolicy(cli.make_policy("replay").__class__):
    """A replay policy that records how it was constructed."""

    name = "kwarg"
    created: list = []

    def __init__(self, checkpoint, extra="none"):
        super().__init__()
        self.created.append({"checkpoint": checkpoint, "extra": extra})


def test_policy_args_reach_the_policys_constructor(tmp_path, fake_sim, capsys):
    assert cli.main([*MATERIALIZE, "--out", str(tmp_path / "p")]) == 0
    prompt = str(tmp_path / "p" / "prompt.npz")
    base = [
        "run-unit",
        "--prompt",
        prompt,
        "--policy",
        "test_cli:KwargPolicy",
        "--out",
        str(tmp_path / "r"),
    ]
    KwargPolicy.created.clear()
    assert cli.main([*base, "--policy-arg", "checkpoint=ckpt.pt", "--policy-arg", "extra=a=b"]) == 0
    assert KwargPolicy.created == [{"checkpoint": "ckpt.pt", "extra": "a=b"}]
    assert cli.main([*base, "--policy-arg", "novalue"]) == 1
    assert "key=value" in capsys.readouterr().err


def test_materialize_chooses_its_robot_and_records_it(tmp_path, fake_sim, capsys):
    from robotwin_icil import prompt

    parser = cli.build_parser()
    # As eval's: without the flag the task config's own robot runs, and the prompt records it.
    assert parser.parse_args([*MATERIALIZE, "--out", "d"]).embodiment is None
    assert cli.main([*MATERIALIZE, "--out", str(tmp_path), "--embodiment", "franka-panda"]) == 0
    _, meta = prompt.read_raw(tmp_path / "prompt.npz")
    assert meta["embodiment"]["name"] == "franka-panda"
    assert json.loads(capsys.readouterr().out)["embodiment"] == "franka-panda"
