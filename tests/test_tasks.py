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


def test_franka_1arm_is_the_surveyed_suite_with_an_arm_switching_stand_in():
    # docs/survey.md, "Franka one-arm survey": at least 2 of 3 seeds solved on two Frankas, and
    # every successful demonstration moved one arm. Stacking has no one-arm task, so
    # stack_bowls_two stands in for it.
    table = tasks.table()
    suite = table.suite("franka_1arm")
    names = [task.name for task in suite]
    assert names == ["place_empty_cup", "stack_bowls_two", "click_bell", "press_stapler"]
    assert names == [name for name in table.tasks if name in names]  # table order
    assert [task.category for task in suite] == [
        "pick_and_place",
        "stacking",
        "press_push",
        "press_push",
    ]
    assert {task.name: task.arms for task in suite if task.arms != arms.ONE} == {
        "stack_bowls_two": arms.SWITCHING
    }
    assert not [t for t in table.tasks.values() if t.category == "stacking" and t.arms == arms.ONE]
    # The suite keeps its stand-in; only --arms 1, which keeps one-arm tasks alone, drops it.
    assert table.select(suite="franka_1arm") == suite
    assert [task.name for task in table.select(suite="franka_1arm", arms=arms.ONE)] == [
        "place_empty_cup",
        "click_bell",
        "press_stapler",
    ]


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


def test_a_suite_may_hold_tasks_of_any_arms(tmp_path):
    # A suite is a list of tasks and says nothing of arms: franka_1arm holds an arm-switching
    # stand-in for a category with no one-arm task. What a run keeps is --arms's to narrow.
    path = tmp_path / "tasks.yml"
    path.write_text(
        "categories:\n  press_push: Press / Push\n  stacking: Stacking\ntasks:\n"
        "  click_bell: {category: press_push, arms: 1}\n"
        "  stack: {category: stacking, arms: switching}\n"
        "suites:\n  track: [click_bell, stack]\n"
    )
    table = tasks.load_table(path)
    assert [task.arms for task in table.suite("track")] == [arms.ONE, arms.SWITCHING]
    assert table.select(suite="track", arms=arms.ONE) == (table["click_bell"],)


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


# The set a --arms 1 run draws from: issue #82's list, corrected from the experts' source —
# put_bottles_dustbin hands right-side bottles to the left arm (2), shake_bottle_horizontally
# drives one arm like shake_bottle (1). Spelled out so a swap that keeps the count (one task's
# 1 becoming 2 while another's 2 becomes 1) cannot pass on the count alone; that every member's
# play_once drives one arm is test_arms_match_a_static_read_of_every_expert's job.
ONE_ARM_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "click_alarmclock",
    "click_bell",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "place_a2b_left",
    "place_a2b_right",
    "place_container_plate",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "rotate_qrcode",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stamp_seal",
    "turn_switch",
)


def test_one_arm_tasks_are_the_twenty_six_the_experts_show():
    one_arm = sorted(task.name for task in tasks.table().tasks.values() if task.arms == arms.ONE)
    assert one_arm == sorted(ONE_ARM_TASKS)
    assert len(ONE_ARM_TASKS) == 26
