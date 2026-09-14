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
