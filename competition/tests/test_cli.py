"""The argv the plugin builds is the argv the CLI parses.

This is the seam that rots silently: the orchestrator never imports the simulator half, it only
runs it, so a renamed flag would be found by a failing duel rather than by a failing test.
"""

from __future__ import annotations

import json

import pytest
from icil_benchmark_robotwin import BENCHMARK
from icil_benchmark_robotwin.cli import build_parser, main
from icil_benchmark_robotwin.plugin import DEFAULT_VIEW, VIEWS
from icil_benchmark_robotwin.units import derive_units

UNIT = derive_units(seed_material="duel-1", count=1, suite="v1")[0].as_dict()


def _run_argv(**extra):
    return BENCHMARK.run_command(
        unit=UNIT,
        prompt="/prompt/u0/prompt.npz",
        out_dir="/work/u0",
        policy_address="/work/p.sock",
        **extra,
    )


def _parse(argv):
    # The plugin names the console script; the parser is what that script runs.
    assert argv[0] == "robotwin-icil-competition"
    return build_parser().parse_args([str(a) for a in argv[1:]])


def test_the_materialize_argv_parses():
    args = _parse(BENCHMARK.materialize_command(unit=UNIT, out_dir="/work/u0"))
    assert args.cmd == "materialize"
    assert args.task == UNIT["task"]
    assert args.scene_seed == UNIT["scene_seed"]
    assert args.out == "/work/u0"
    assert args.max_expert_attempts == UNIT["max_expert_attempts"]


def test_the_run_argv_parses():
    args = _parse(_run_argv())
    assert args.cmd == "run-unit"
    assert args.prompt == "/prompt/u0/prompt.npz"
    assert args.policy_address == "/work/p.sock"
    assert args.view == DEFAULT_VIEW


@pytest.mark.parametrize("view", VIEWS)
def test_the_run_argv_carries_either_view_the_competition_asks_for(view):
    """Both fields run through this one argv, so both spellings have to survive it."""
    args = _parse(_run_argv(view=view))
    assert args.view == view
    assert view in BENCHMARK.info()["views"]


def test_a_view_this_benchmark_does_not_serve_is_still_refused():
    """Serving the sensorimotor view did not turn `--view` into a free-text field."""
    with pytest.raises(SystemExit):
        _parse(_run_argv(view="telepathy"))


def test_an_option_the_orchestrator_adds_travels_without_a_change_here():
    """`**extra` is the seam that lets the competition pass something new without editing this."""
    argv = BENCHMARK.materialize_command(unit=UNIT, out_dir="/w", task_config="demo_randomized")
    args = _parse(argv)
    assert args.task_config == "demo_randomized"
    # A false or absent option is not passed at all, rather than passed as the string "False".
    assert "--no-video" not in BENCHMARK.materialize_command(
        unit=UNIT, out_dir="/w", no_video=False
    )
    assert "--no-video" in BENCHMARK.materialize_command(unit=UNIT, out_dir="/w", no_video=True)


def test_the_pure_subcommands_run_with_no_simulator(capsys):
    assert main(["info"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "robotwin"
    assert main(["catalogue"]) == 0
    assert "v1" in json.loads(capsys.readouterr().out)["suites"]
    assert main(["units", "--seed-material", "d", "--count", "3"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 3


def test_units_can_be_written_where_the_orchestrator_asked(tmp_path, capsys):
    out = tmp_path / "units.json"
    assert main(["units", "--seed-material", "d", "--count", "2", "--out", str(out)]) == 0
    assert len(json.loads(out.read_text())) == 2


def test_verify_exits_non_zero_on_a_prompt_that_is_not_the_one_asked_for(tmp_path, capsys):
    assert main(["verify", "--prompt", str(tmp_path / "absent.npz")]) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]


def test_every_subcommand_is_reachable():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    for cmd in ("info", "catalogue", "units", "materialize", "run-unit", "verify"):
        assert cmd in parser.format_help()
