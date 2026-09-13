from pathlib import Path

import pytest

from robotwin_icil import arms, tasks

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = REPO_ROOT / "vendor" / "RoboTwin"

needs_robotwin = pytest.mark.skipif(
    not (ROBOTWIN_ROOT / "envs").is_dir(), reason="RoboTwin submodule not initialised"
)


def test_every_task_has_a_known_category():
    table = tasks.table()
    assert table.tasks
    for task in table.tasks.values():
        assert task.category in table.categories
        assert task.category_label == table.categories[task.category]


def test_every_task_declares_its_arms():
    table = tasks.table()
    assert all(task.arms in arms.ARMS for task in table.tasks.values())
    # All three kinds of expert exist upstream; an empty kind would mean the table lost one.
    assert {task.arms for task in table.tasks.values()} == set(arms.ARMS)


def test_categories_are_all_populated():
    # An empty category would report a 0-episode row in every run.
    assert all(members for members in tasks.table().by_category().values())


def test_v1_suite_spans_at_least_three_categories():
    suite = tasks.table().suite("v1")
    assert len({task.category for task in suite}) >= 3


def test_all_suite_is_the_whole_table():
    table = tasks.table()
    assert set(table.suites["all"]) == set(table.tasks)


def test_unknown_names_are_rejected():
    table = tasks.table()
    with pytest.raises(tasks.TaskTableError):
        table["no_such_task"]
    with pytest.raises(tasks.TaskTableError):
        table.suite("no_such_suite")


def test_select_narrows_to_one_arm_tasks_only_when_asked():
    table = tasks.table()
    v1 = table.suite("v1")
    assert table.select(suite="v1") == v1
    assert table.select(suite="v1", arms=arms.TWO) == v1
    one_arm = table.select(suite="v1", arms=arms.ONE)
    assert one_arm == tuple(task for task in v1 if task.arms == arms.ONE)
    assert 0 < len(one_arm) < len(v1)  # v1 has both one-arm tasks and switching ones
    assert table.select(task="click_bell", arms=arms.ONE) == (table["click_bell"],)
    assert table.select(task="lift_pot") == (table["lift_pot"],)


def test_select_refuses_a_task_or_suite_with_no_one_arm_expert():
    table = tasks.table()
    with pytest.raises(tasks.TaskTableError, match="task 'lift_pot' needs two arms; --arms 1"):
        table.select(task="lift_pot", arms=arms.ONE)
    with pytest.raises(tasks.TaskTableError, match="'stack_bowls_two' needs switching arms"):
        table.select(task="stack_bowls_two", arms=arms.ONE)
    two_arm_only = tasks.TaskTable(
        tasks=table.tasks, categories=table.categories, suites={"lifts": ("lift_pot",)}
    )
    with pytest.raises(tasks.TaskTableError, match="suite 'lifts' has no one-arm task"):
        two_arm_only.select(suite="lifts", arms=arms.ONE)
    with pytest.raises(tasks.TaskTableError, match="arms 1 or 2"):
        table.select(suite="v1", arms="switching")
    with pytest.raises(tasks.TaskTableError, match="either a suite or a task"):
        table.select()


def write_table(tmp_path, entries: str) -> Path:
    path = tmp_path / "tasks.yml"
    path.write_text(
        "categories:\n  press_push: Press / Push\ntasks:\n" + entries + "suites:\n  all: '*'\n"
    )
    return path


def test_a_task_without_arms_fails_to_load_naming_it(tmp_path):
    path = write_table(tmp_path, "  click_bell: {category: press_push}\n")
    with pytest.raises(tasks.TaskTableError, match="'click_bell' needs exactly"):
        tasks.load_table(path)
    path = write_table(tmp_path, "  click_bell: press_push\n")
    with pytest.raises(tasks.TaskTableError, match="'click_bell' needs exactly"):
        tasks.load_table(path)


def test_an_unknown_arms_value_fails_to_load_naming_it(tmp_path):
    path = write_table(tmp_path, "  click_bell: {category: press_push, arms: 3}\n")
    with pytest.raises(tasks.TaskTableError, match="'click_bell' has arms 3"):
        tasks.load_table(path)


def test_arms_load_as_the_table_vocabulary(tmp_path):
    path = write_table(
        tmp_path,
        "  click_bell: {category: press_push, arms: 1}\n"
        "  stack: {category: press_push, arms: switching}\n"
        "  lift: {category: press_push, arms: 2}\n",
    )
    table = tasks.load_table(path)
    assert [task.arms for task in table.tasks.values()] == [arms.ONE, arms.SWITCHING, arms.TWO]


@needs_robotwin
def test_table_matches_the_pinned_robotwin_checkout():
    # The point of the table: an upstream rename must fail here rather than silently shrink a suite.
    tasks.check_against_robotwin(ROBOTWIN_ROOT)


@needs_robotwin
def test_arms_match_a_static_read_of_every_expert():
    tasks.check_arms_against_robotwin(ROBOTWIN_ROOT)


@needs_robotwin
def test_a_wrong_arms_value_fails_naming_the_task():
    table = tasks.table()
    wrong = dict(table.tasks)
    click_bell = wrong["click_bell"]
    wrong["click_bell"] = tasks.Task(
        click_bell.name, click_bell.category, click_bell.category_label, arms.TWO
    )
    altered = tasks.TaskTable(tasks=wrong, categories=table.categories, suites=table.suites)
    with pytest.raises(
        tasks.TaskTableError, match="click_bell: tasks.yml says 2, play_once says 1"
    ):
        tasks.check_arms_against_robotwin(ROBOTWIN_ROOT, altered)


@needs_robotwin
def test_one_arm_tasks_are_the_twenty_six_the_experts_show():
    # The set a --arms 1 run draws from; each member's play_once drives exactly one arm.
    one_arm = sorted(task.name for task in tasks.table().tasks.values() if task.arms == arms.ONE)
    assert len(one_arm) == 26
    assert "put_bottles_dustbin" not in one_arm  # hands right-side bottles to the left arm
    assert "shake_bottle_horizontally" in one_arm  # one arm, like shake_bottle
