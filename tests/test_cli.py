import json

import pytest

from robotwin_icil import cli
from robotwin_icil.records import RunDir
from test_records import manifest, record

EVAL = ["eval", "--policy", "replay", "--task", "click_bell", "--episodes", "1", "--run-dir", "r"]
SURVEY = ["survey", "--task", "click_bell"]


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
def test_a_run_chooses_its_robot_and_defaults_to_aloha(command, capsys):
    parser = cli.build_parser()
    assert parser.parse_args(command).embodiment == "aloha-agilex"
    assert parser.parse_args([*command, "--embodiment", "franka-panda"]).embodiment == (
        "franka-panda"
    )
    # A robot the benchmark cannot form is refused by the parser, before any simulator import.
    with pytest.raises(SystemExit) as exc:
        cli.main([*command, "--embodiment", "ur5-wsg"])
    assert exc.value.code == 2 and "aloha-agilex" in capsys.readouterr().err
