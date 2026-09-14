"""What the plugin draws units from: tasks, their categories and arms, and the suites.

Tasks, their categories, the arms each task's expert uses (`Task.arms`, read off its `play_once`)
and the benchmark's own suites are `robotwin_icil.tasks`'s. One thing is not in that table yet,
and is PROVISIONAL here: the `franka_1arm` suite the orchestrator's one-arm Franka track draws
from. The Franka survey (robotensor/ICIL-robotwin-benchmark#83) names it and has not finished;
until the table names it, `provisional_franka_1arm` derives it from each task's `arms`.
"""

from __future__ import annotations

from typing import Any

from robotwin_icil import tasks

#: How many arms a task's expert uses, as `Task.arms` spells it.
ONE, SWITCHING, TWO = "1", "switching", "2"

#: The suite the orchestrator's one-arm Franka track names in `spec.json`.
FRANKA_1ARM = "franka_1arm"
#: The categories that track scores, as its skills name them.
FRANKA_1ARM_CATEGORIES = ("pick_and_place", "stacking", "press_push")
#: What a category of the track with no one-arm task serves in their place. At the pinned RoboTwin
#: that category is stacking, whose every expert picks the arm nearest the next object: its
#: arm-switching tasks keep the track's `franka_stacking` skill drawing units while it is
#: provisional. A category with neither kind of task has no `franka_1arm` task, and `derive_units`
#: refuses to draw units for it.
STAND_IN = SWITCHING
STAND_IN_RULE = (
    "a category with no one-arm task serves its arm-switching tasks in their place; one with "
    "neither has no franka_1arm task and derives no units"
)

#: The robot a suite's units run on: the Franka suite on two Franka arms, every other suite on
#: the benchmark's default robot.
FRANKA = "franka-panda"
DEFAULT_EMBODIMENT = "aloha-agilex"


def _in_category(table: tasks.TaskTable, category: str, arms: str) -> list[str]:
    return [t.name for t in table.tasks.values() if t.category == category and t.arms == arms]


def categories_without_one_arm_task(table: tasks.TaskTable) -> tuple[str, ...]:
    """The track's categories in which no task's expert uses one arm: stacking, at the pinned
    RoboTwin."""
    return tuple(c for c in FRANKA_1ARM_CATEGORIES if not _in_category(table, c, ONE))


def provisional_franka_1arm(table: tasks.TaskTable) -> tuple[str, ...]:
    """PROVISIONAL `franka_1arm`, in table order: per category of the track, its one-arm tasks.

    A category with no one-arm task (`categories_without_one_arm_task`) gives its `STAND_IN`
    arm-switching tasks instead, so that every skill of the track can draw a unit; one with
    neither gives nothing. Two-arm tasks are never in it.
    """
    members: list[str] = []
    for category in FRANKA_1ARM_CATEGORIES:
        members += _in_category(table, category, ONE) or _in_category(table, category, STAND_IN)
    return tuple(members)


def suites(table: tasks.TaskTable) -> dict[str, tuple[str, ...]]:
    """Every suite units can be drawn from: the table's, and `franka_1arm` until it has one."""
    out = dict(table.suites)
    out.setdefault(FRANKA_1ARM, provisional_franka_1arm(table))
    return out


def franka_1arm_basis(table: tasks.TaskTable) -> dict[str, Any]:
    """What `franka_1arm` is made of, for `info()`: whether it is provisional, each category's
    tasks and the arms they use, the categories with no one-arm task, and what stands in."""
    members = set(suites(table)[FRANKA_1ARM])
    categories = {}
    for category in FRANKA_1ARM_CATEGORIES:
        names = [n for n, t in table.tasks.items() if t.category == category and n in members]
        categories[category] = {"tasks": names, "arms": sorted({table[n].arms for n in names})}
    return {
        "provisional": FRANKA_1ARM not in table.suites,
        "categories": categories,
        "without_one_arm_task": list(categories_without_one_arm_task(table)),
        "stand_in": STAND_IN_RULE,
    }


def _listed(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def provisional_franka_1arm_note(table: tasks.TaskTable) -> str:
    """Said in `info()` for as long as `franka_1arm` is served in place of a surveyed one."""
    without = categories_without_one_arm_task(table)
    one_arm = [c for c in FRANKA_1ARM_CATEGORIES if c not in without]
    parts = [f"the one-arm tasks of {_listed(one_arm)}"] if one_arm else []
    for category in without:
        if _in_category(table, category, STAND_IN):
            parts.append(f"{category} has no one-arm task, so its arm-switching tasks stand in")
        else:
            parts.append(f"{category} has no one-arm or arm-switching task, so it has no units")
    return (
        "franka_1arm is PROVISIONAL until the Franka survey (robotensor/ICIL-robotwin-benchmark#83) "
        "names it: " + "; ".join(parts)
    )


def provisional(table: tasks.TaskTable) -> list[str]:
    """What the catalogue serves that no survey has settled yet."""
    return [provisional_franka_1arm_note(table)] if FRANKA_1ARM not in table.suites else []


def embodiment_of(suite: str) -> str:
    """The robot a suite's units run on, as `robotwin-icil --embodiment` names it."""
    return FRANKA if suite == FRANKA_1ARM else DEFAULT_EMBODIMENT


def task_label(name: str) -> str:
    """A task's name for people: `click_bell` -> `Click bell`."""
    return name.replace("_", " ").capitalize()


def catalogue(table: tasks.TaskTable | None = None) -> dict[str, Any]:
    """`{"suites": {suite: [task]}, "categories": {id: label}, "tasks": {task: {"category",
    "arms", "label"}}, "embodiments": {suite: robot}, "provisional": [note]}`."""
    table = table or tasks.table()
    return {
        "suites": {name: list(members) for name, members in suites(table).items()},
        "categories": dict(table.categories),
        "tasks": {
            name: {"category": task.category, "arms": task.arms, "label": task_label(name)}
            for name, task in table.tasks.items()
        },
        "embodiments": {name: embodiment_of(name) for name in suites(table)},
        "provisional": provisional(table),
    }
