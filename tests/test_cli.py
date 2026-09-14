import json
import os
from pathlib import Path

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
    assert "click_bell  (one arm)  [franka_1arm, v1]" in out
    assert "stack_bowls_two  (switching arms)  [franka_1arm, v1]" in out
    assert "place_a2b_left  (one arm)  [v1]" in out
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


def test_a_rejected_seed_exits_0_with_its_result(tmp_path, fake_sim, capsys):
    # The orchestrator reads result.json whatever the exit; a non-zero exit for an outcome it
    # expects would have it void the unit on the log tail and lose the rejection's reason.
    from fake_robotwin import FakeTaskEnv

    fake_sim["next"] = lambda name: FakeTaskEnv(unstable_seeds={11})
    assert cli.main([*MATERIALIZE, "--out", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["rejection"] == "unstable"
    assert json.loads((tmp_path / "result.json").read_text())["ok"] is False
    assert not (tmp_path / "prompt.npz").exists()


def test_materialize_takes_every_candidate_seed_in_order(tmp_path, fake_sim, capsys):
    from fake_robotwin import FakeTaskEnv

    fake_sim["next"] = lambda name: FakeTaskEnv(unstable_seeds={11, 12})
    argv = [*MATERIALIZE, "--scene-seed", "12", "--scene-seed", "13", "--out", str(tmp_path)]
    assert cli.build_parser().parse_args(argv).scene_seeds == [11, 12, 13]
    assert cli.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True and printed["scene_seed"] == 13
    assert [a["seed"] for a in printed["attempts"]] == [11, 12, 13]

    # Every candidate rejected is still a result, and still exit 0.
    fake_sim["next"] = lambda name: FakeTaskEnv(unstable_seeds={11, 12, 13})
    assert cli.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["void"] is True and printed["void_cause"] == "harness"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.json"]

    # A seed given twice is a caller's mistake: exit 1, and nothing written.
    assert cli.main([*MATERIALIZE, "--scene-seed", "11", "--out", str(tmp_path)]) == 1
    assert "given more than once" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_a_command_built_for_another_benchmark_source_refuses_to_run(tmp_path, fake_sim, capsys):
    # The plugin builds its argv against the robotwin_icil it imports; an interpreter running
    # another one may read the same flags differently (one --scene-seed kept of four, say).
    import robotwin_icil

    other = "0" * 64
    out = tmp_path / "p"
    assert cli.main([*MATERIALIZE, "--out", str(out), "--expect-source-sha256", other]) == 1
    err = capsys.readouterr().err
    assert f"digests to {robotwin_icil.source_sha256()}, not the {other}" in err
    assert not out.exists()

    argv = ["run-unit", "--prompt", str(out / "prompt.npz"), "--policy", "replay"]
    assert cli.main([*argv, "--out", str(tmp_path / "r"), "--expect-source-sha256", other]) == 1
    assert "another robotwin_icil" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()

    # Its own digest runs, and the result says which source wrote it.
    mine = ["--expect-source-sha256", robotwin_icil.source_sha256()]
    assert cli.main([*MATERIALIZE, "--out", str(out), *mine]) == 0
    assert json.loads(capsys.readouterr().out)["source_sha256"] == robotwin_icil.source_sha256()
    assert cli.main([*argv, "--out", str(tmp_path / "r"), *mine]) == 0
    assert json.loads(capsys.readouterr().out)["source_sha256"] == robotwin_icil.source_sha256()


def test_the_denoiser_is_a_flag_that_reaches_the_simulator_seam(tmp_path, fake_sim, monkeypatch):
    # The orchestrator hands a benchmark subprocess only allow-listed variables, so the override
    # for a denoiser that hangs camera reads travels as an argument.
    from fake_robotwin import FakeTaskEnv

    monkeypatch.setenv("ROBOTWIN_ICIL_DENOISER", "")
    seen = []

    def load_task(name):
        seen.append(os.environ.get("ROBOTWIN_ICIL_DENOISER"))
        return FakeTaskEnv()

    fake_sim["next"] = load_task
    assert cli.main([*MATERIALIZE, "--out", str(tmp_path / "p"), "--denoiser", "none"]) == 0
    argv = ["run-unit", "--prompt", str(tmp_path / "p" / "prompt.npz"), "--policy", "replay"]
    assert cli.main([*argv, "--out", str(tmp_path / "r"), "--denoiser", "oidn"]) == 0
    assert seen == ["none", "oidn"]
    with pytest.raises(SystemExit) as refused:
        cli.main([*MATERIALIZE, "--out", str(tmp_path / "q"), "--denoiser", "fast"])
    assert refused.value.code == 2


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


def test_a_harness_fault_mid_unit_is_a_void_result_and_a_logged_traceback(
    tmp_path, fake_sim, monkeypatch, capsys
):
    # The orchestrator reads result.json; a command that died would leave it a log tail to read.
    from robotwin_icil import robotwin

    assert cli.main([*MATERIALIZE, "--out", str(tmp_path / "p")]) == 0
    capsys.readouterr()

    def broken(env):
        raise KeyError("head_camera")

    monkeypatch.setattr(robotwin, "fingerprint", broken)
    argv = ["run-unit", "--prompt", str(tmp_path / "p" / "prompt.npz"), "--policy", "replay"]
    assert cli.main([*argv, "--out", str(tmp_path / "r")]) == 0
    printed, err = capsys.readouterr()
    result = json.loads(printed)
    assert result["void"] is True and "KeyError: 'head_camera'" in result["error"]
    assert json.loads((tmp_path / "r" / "result.json").read_text()) == result
    assert "Traceback" in err and "KeyError: 'head_camera'" in err


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


class WherePolicy(cli.make_policy("replay").__class__):
    """A replay policy that notes the working directory it was built and reset in."""

    name = "where"
    seen: dict = {}

    def __init__(self, checkpoint):
        super().__init__()
        self.seen.update(built=Path.cwd(), checkpoint=Path(checkpoint).resolve())

    def _reset(self):
        super()._reset()
        self.seen["reset"] = Path.cwd()


def test_a_policy_is_built_before_the_simulator_moves_the_working_directory(
    tmp_path, fake_sim, monkeypatch, capsys
):
    # docs/policies.md tells adapters to resolve a relative checkpoint in __init__: that holds
    # only while the constructor runs before RoboTwin chdirs into vendor/RoboTwin, as here.
    from fake_robotwin import FakeTaskEnv

    caller, simulator = tmp_path / "caller", tmp_path / "vendor"
    caller.mkdir()
    simulator.mkdir()
    monkeypatch.chdir(caller)
    assert cli.main([*MATERIALIZE, "--out", str(tmp_path / "p")]) == 0
    capsys.readouterr()

    def load_task(name):
        os.chdir(simulator)
        return FakeTaskEnv()

    fake_sim["next"] = load_task
    WherePolicy.seen.clear()
    argv = ["run-unit", "--prompt", str(tmp_path / "p" / "prompt.npz")]
    argv += ["--policy", "test_cli:WherePolicy", "--policy-arg", "checkpoint=ckpt/model.pt"]
    assert cli.main([*argv, "--out", str(tmp_path / "r")]) == 0
    assert json.loads(capsys.readouterr().out)["success"] is True
    assert WherePolicy.seen["built"] == caller and WherePolicy.seen["reset"] == simulator
    assert WherePolicy.seen["checkpoint"] == caller / "ckpt" / "model.pt"


def test_materialize_chooses_its_robot_and_records_it(tmp_path, fake_sim, capsys):
    from robotwin_icil import prompt

    parser = cli.build_parser()
    # As eval's: without the flag the task config's own robot runs, and the prompt records it.
    assert parser.parse_args([*MATERIALIZE, "--out", "d"]).embodiment is None
    assert cli.main([*MATERIALIZE, "--out", str(tmp_path), "--embodiment", "franka-panda"]) == 0
    _, meta = prompt.read_raw(tmp_path / "prompt.npz")
    assert meta["embodiment"]["name"] == meta["embodiment"]["choice"] == "franka-panda"
    assert json.loads(capsys.readouterr().out)["embodiment"] == "franka-panda"


def _stale(directory, *names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text('{"success": true, "void": false, "ok": true}')


@pytest.mark.parametrize(
    ("argv", "error"),
    [
        (["--policy", "nope"], "unknown policy 'nope'"),
        (["--policy", "replay", "--policy-arg", "novalue"], "key=value"),
        (["--policy", "replay", "--policy-arg", "checkpoint=x"], "cannot construct policy"),
    ],
)
def test_run_unit_failing_before_it_runs_leaves_no_earlier_result(
    tmp_path, fake_sim, capsys, argv, error
):
    # A reused --out must never hand a reader an earlier command's verdict as this one's.
    out = tmp_path / "run"
    _stale(out, "result.json", "evaluation.mp4")
    base = ["run-unit", "--prompt", str(tmp_path / "p" / "prompt.npz"), "--out", str(out)]
    assert cli.main([*base, *argv]) == 1
    assert error in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == []


def test_materialize_failing_before_it_runs_leaves_no_earlier_prompt(tmp_path, fake_sim, capsys):
    _stale(tmp_path, "prompt.npz", "demonstration.mp4", "result.json")
    argv = ["materialize", "--task", "nope", "--scene-seed", "1", "--out", str(tmp_path)]
    assert cli.main(argv) == 1
    assert "unknown task 'nope'" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_run_unit_refuses_to_write_over_the_prompts_own_result(tmp_path, fake_sim, capsys):
    out = tmp_path / "p"
    assert cli.main([*MATERIALIZE, "--out", str(out)]) == 0
    capsys.readouterr()
    before = (out / "result.json").read_text()
    argv = ["run-unit", "--prompt", str(out / "prompt.npz"), "--policy", "replay"]
    assert cli.main([*argv, "--out", str(out)]) == 1
    assert "holds the prompt" in capsys.readouterr().err
    assert (out / "result.json").read_text() == before
    assert sorted(p.name for p in out.iterdir()) == [
        "demonstration.mp4",
        "prompt.npz",
        "result.json",
    ]


RUN_UNIT = ["run-unit", "--prompt", "p/prompt.npz"]


def test_run_unit_takes_a_policy_or_an_address_never_both(tmp_path, capsys):
    base = [*RUN_UNIT, "--out", str(tmp_path / "r")]
    address = ["--policy-address", "/tmp/policy.sock", "--authkey-env", "POLICY_KEY"]
    with pytest.raises(SystemExit) as refused:
        cli.main([*base, "--policy", "replay", *address])
    assert refused.value.code == 2 and "not allowed with argument" in capsys.readouterr().err
    with pytest.raises(SystemExit) as refused:
        cli.main(base)
    assert refused.value.code == 2 and "--policy --policy-address" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize(
    ("flags", "reason"),
    [
        (["--policy-address", "/tmp/policy.sock"], "--policy-address needs --authkey-env"),
        (
            ["--policy-address", "/tmp/p.sock", "--authkey-env", "K", "--policy-arg", "a=b"],
            "--policy-arg configures a policy run in this process",
        ),
        (["--policy", "replay", "--authkey-env", "K"], "--authkey-env go with --policy-address"),
        (
            ["--policy", "replay", "--act-timeout-s", "5", "--policy-log", "x.log"],
            "--act-timeout-s",
        ),
        (["--policy", "replay", "--policy-budget-s", "5"], "--policy-budget-s go with"),
        (["--policy", "replay", "--unit-timeout-s", "600"], "--unit-timeout-s go with"),
    ],
)
def test_run_unit_refuses_flags_that_belong_to_the_other_kind_of_policy(
    tmp_path, capsys, flags, reason
):
    assert cli.main([*RUN_UNIT, "--out", str(tmp_path / "r"), *flags]) == 2
    assert reason in capsys.readouterr().err
    assert not (tmp_path / "r").exists()  # refused before anything was cleared or written


@pytest.mark.parametrize("flag", ["--act-timeout-s", "--policy-budget-s", "--unit-timeout-s"])
@pytest.mark.parametrize("seconds", ["0", "-1", "inf", "soon"])
def test_a_served_policys_limits_are_positive_numbers_of_seconds(flag, seconds, capsys):
    flags = ["--policy-address", "/tmp/p.sock", "--authkey-env", "K", flag, seconds]
    with pytest.raises(SystemExit) as refused:
        cli.main([*RUN_UNIT, "--out", "r", *flags])
    assert refused.value.code == 2 and "positive number of seconds" in capsys.readouterr().err


def test_a_served_policy_without_its_key_is_refused_before_any_result(
    tmp_path, fake_sim, monkeypatch, capsys
):
    # A key that is not there is the caller's configuration, not the unit's outcome: exit 1, and
    # no result.json the orchestrator could take for one.
    monkeypatch.delenv("NO_SUCH_POLICY_KEY", raising=False)
    assert cli.main([*MATERIALIZE, "--out", str(tmp_path / "p")]) == 0
    capsys.readouterr()
    argv = [
        "run-unit",
        "--prompt",
        str(tmp_path / "p" / "prompt.npz"),
        "--out",
        str(tmp_path / "r"),
    ]
    argv += ["--policy-address", "/nonexistent/policy.sock", "--authkey-env", "NO_SUCH_POLICY_KEY"]
    assert cli.main(argv) == 1
    assert "NO_SUCH_POLICY_KEY: the environment variable holds no key" in capsys.readouterr().err
    assert sorted(p.name for p in (tmp_path / "r").iterdir()) == []
