"""The command builders' argv, parsed by the benchmark's real command-line parser."""

import os

import pytest

import robotwin_icil
from icil_benchmark_robotwin import BENCHMARK, commands
from robotwin_icil import cli

UNIT = BENCHMARK.derive_units(
    seed_material="duel|franka_press_push", count=1, suite="franka_1arm", category="press_push"
)[0]
#: The unit as a duel hands it over: the orchestrator's reserved keys beside the benchmark's own.
DUEL_UNIT = {
    **UNIT,
    "unit_id": "fu-000",
    "skill": "franka_press_push",
    "benchmark": "robotwin",
    "index": 0,
    "seed": 1234,
    "instance": 0,
    "demo": "fu-000",
}
KEY = "ab" * 32


def parse(argv):
    """The argv's arguments as `robotwin-icil` parses them, after the interpreter and module."""
    assert all(isinstance(arg, str) for arg in argv)
    assert argv[1:3] == ["-m", "robotwin_icil.cli"]
    return cli.build_parser().parse_args(argv[3:])


def test_the_commands_run_under_the_simulator_environments_interpreter(monkeypatch):
    monkeypatch.delenv(commands.PYTHON_ENV, raising=False)
    argv = BENCHMARK.materialize_command(unit=UNIT, out_dir="/runs/prompt")
    assert argv[0] == commands.DEFAULT_PYTHON == "/root/miniforge3/envs/robotwin/bin/python"
    monkeypatch.setenv("ROBOTWIN_ICIL_PYTHON", "/opt/sim/bin/python")
    assert BENCHMARK.run_command(
        unit=UNIT, prompt="/p.npz", out_dir="/o", policy_address="/s", authkey_env="K"
    )[0] == ("/opt/sim/bin/python")


def test_materialize_passes_every_candidate_in_order_with_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parse(BENCHMARK.materialize_command(unit=DUEL_UNIT, out_dir="duel/fu-000/prompt"))
    assert args.command == "materialize" and args.task == UNIT["task"]
    assert args.scene_seeds == UNIT["instance_params"]["scene_seeds"]
    assert len(args.scene_seeds) == 4
    assert args.embodiment == "franka-panda"
    assert (args.task_config, args.save_freq) == ("demo_clean", 15)
    assert args.out == str(tmp_path / "duel" / "fu-000" / "prompt")
    # The command refuses to run under another robotwin_icil than the one that built it.
    assert args.expect_source_sha256 == robotwin_icil.source_sha256()


def test_run_unit_drives_the_served_policy_and_never_names_its_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ICIL_POLICY_AUTHKEY", KEY)
    argv = BENCHMARK.run_command(
        unit=DUEL_UNIT,
        prompt="duel/fu-000/prompt/prompt.npz",
        out_dir="duel/fu-000/challenger",
        policy_address="duel/policy.sock",
        authkey_env="ICIL_POLICY_AUTHKEY",
        act_timeout_s=30,
        policy_budget_s=300,
        unit_timeout_s=587.25,
        policy_log="duel/fu-000/challenger/policy.log",
        added_by_a_later_duel=True,
    )
    args = parse(argv)
    assert args.command == "run-unit" and args.policy is None
    assert args.prompt == str(tmp_path / "duel" / "fu-000" / "prompt" / "prompt.npz")
    assert args.out == str(tmp_path / "duel" / "fu-000" / "challenger")
    assert args.policy_address == str(tmp_path / "duel" / "policy.sock")
    assert args.authkey_env == "ICIL_POLICY_AUTHKEY" and args.act_timeout_s == 30.0
    assert args.policy_budget_s == 300.0 and args.unit_timeout_s == 587.25
    assert args.policy_log == str(tmp_path / "duel" / "fu-000" / "challenger" / "policy.log")
    assert cli._run_unit_usage(args) is None  # the flags go together, as main() requires
    assert args.expect_source_sha256 == robotwin_icil.source_sha256()
    assert not any(KEY in arg for arg in argv)


def test_a_tcp_address_is_kept_and_optional_flags_are_left_out():
    args = parse(
        BENCHMARK.run_command(
            unit=UNIT,
            prompt="/p/prompt.npz",
            out_dir="/o",
            policy_address="policy-host:5555",
            authkey_env="K",
        )
    )
    assert args.policy_address == "policy-host:5555"
    assert args.act_timeout_s is None and args.policy_log is None
    assert args.policy_budget_s is None and args.unit_timeout_s is None


@pytest.mark.parametrize("key", ["act_timeout_s", "policy_budget_s", "unit_timeout_s"])
@pytest.mark.parametrize("value", [0, -1.0, float("inf"), float("nan"), True, "30"])
def test_a_time_limit_run_unit_would_refuse_is_refused_when_the_argv_is_built(key, value):
    # Passed on, run-unit would exit 2 once the unit ran, and the unit would be void.
    with pytest.raises(ValueError, match=f"{key} must be a positive, finite number"):
        BENCHMARK.run_command(
            unit=UNIT,
            prompt="/p.npz",
            out_dir="/o",
            policy_address="/s",
            authkey_env="K",
            **{key: value},
        )


def test_the_layout_benchmarks_check_uses_parses():
    root = "/nonexistent/icil-orchestrator-check/fu-000"
    materialize = parse(BENCHMARK.materialize_command(unit=DUEL_UNIT, out_dir=f"{root}/prompt"))
    run = parse(
        BENCHMARK.run_command(
            unit=DUEL_UNIT,
            prompt=f"{root}/prompt/prompt.npz",
            out_dir=f"{root}/challenger",
            policy_address="/nonexistent/icil-orchestrator-check/policy.sock",
            authkey_env="ICIL_POLICY_AUTHKEY",
        )
    )
    assert materialize.out == f"{root}/prompt" and run.out == f"{root}/challenger"


def _with(**params):
    return {**UNIT, "instance_params": {**UNIT["instance_params"], **params}}


@pytest.mark.parametrize(
    ("unit", "reason"),
    [
        ({**UNIT, "task": "nope"}, "task 'nope' is not a task"),
        ({**UNIT, "instance_params": None}, "embodiment None"),
        (_with(embodiment="ur5-wsg"), "embodiment 'ur5-wsg'"),
        (_with(scene_seeds=[]), "scene_seeds []"),
        (_with(scene_seeds=[3, 3]), "not distinct"),
        (_with(scene_seeds=[-1]), "not distinct integers in [0, 2**32)"),
        (_with(scene_seeds="11"), "scene_seeds '11'"),
    ],
)
def test_a_unit_that_cannot_run_is_refused_by_both_builders(unit, reason):
    with pytest.raises(ValueError) as refused:
        BENCHMARK.materialize_command(unit=unit, out_dir="/o")
    assert reason in str(refused.value)
    with pytest.raises(ValueError):
        BENCHMARK.run_command(
            unit=unit, prompt="/p.npz", out_dir="/o", policy_address="/s", authkey_env="K"
        )


@pytest.mark.parametrize("authkey_env", ["", "KEY=abcd", None])
def test_the_key_variable_must_be_a_name(authkey_env):
    with pytest.raises(ValueError, match="authkey_env"):
        BENCHMARK.run_command(
            unit=UNIT, prompt="/p.npz", out_dir="/o", policy_address="/s", authkey_env=authkey_env
        )


def test_every_argv_is_plain_strings_with_absolute_paths():
    argv = BENCHMARK.materialize_command(unit=UNIT, out_dir="relative/dir")
    assert os.path.isabs(argv[-1]) and all(isinstance(a, str) for a in argv)
