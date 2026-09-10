from pathlib import Path

import pytest

from robotwin_icil import tasks

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = REPO_ROOT / "vendor" / "RoboTwin"


def test_every_task_has_a_known_category():
    table = tasks.table()
    assert table.tasks
    for task in table.tasks.values():
        assert task.category in table.categories
        assert task.category_label == table.categories[task.category]


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


@pytest.mark.skipif(
    not (ROBOTWIN_ROOT / "envs").is_dir(), reason="RoboTwin submodule not initialised"
)
def test_table_matches_the_pinned_robotwin_checkout():
    # The point of the table: an upstream rename must fail here rather than silently shrink a suite.
    tasks.check_against_robotwin(ROBOTWIN_ROOT)
