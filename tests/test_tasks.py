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


def test_unknown_names_are_rejected():
    table = tasks.table()
    with pytest.raises(tasks.TaskTableError):
        table["no_such_task"]


def test_select_narrows_to_one_arm_tasks_only_when_asked():
    table = tasks.table()
    assert table.select() == tuple(table.tasks.values())
    assert table.select(arms=arms.TWO) == tuple(table.tasks.values())
    assert table.select(arms=arms.ONE) == tuple(
        task for task in table.tasks.values() if task.arms == arms.ONE
    )
    assert 0 < len(table.select(arms=arms.ONE)) < len(table.tasks)
    assert table.select(task="click_bell", arms=arms.ONE) == (table["click_bell"],)
    assert table.select(task="lift_pot") == (table["lift_pot"],)


def test_select_refuses_a_task_with_no_one_arm_expert():
    table = tasks.table()
    with pytest.raises(tasks.TaskTableError, match="task 'lift_pot' needs two arms; --arms 1"):
        table.select(task="lift_pot", arms=arms.ONE)
    with pytest.raises(tasks.TaskTableError, match="'stack_bowls_two' needs switching arms"):
        table.select(task="stack_bowls_two", arms=arms.ONE)
    two_arm_only = tasks.TaskTable(
        tasks={"lift_pot": table["lift_pot"]}, categories=table.categories
    )
    with pytest.raises(tasks.TaskTableError, match="the task catalog has no one-arm task"):
        two_arm_only.select(arms=arms.ONE)
    with pytest.raises(tasks.TaskTableError, match="arms 1 or 2"):
        table.select(arms="switching")


def write_table(tmp_path, entries: str) -> Path:
    path = tmp_path / "tasks.yml"
    path.write_text("categories:\n  press_push: Press / Push\ntasks:\n" + entries)
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
    # The point of the table: an upstream rename must fail here rather than silently shrink a run.
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
    altered = tasks.TaskTable(tasks=wrong, categories=table.categories)
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
