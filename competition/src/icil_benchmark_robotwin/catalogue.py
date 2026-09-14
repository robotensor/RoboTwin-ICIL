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

#: The robot a suite's units run on: the Franka suite on two Franka arms, every other suite on
#: the benchmark's default robot.
FRANKA = "franka-panda"
DEFAULT_EMBODIMENT = "aloha-agilex"

#: Said in `info()` for as long as the suite below is served in place of a surveyed one.
PROVISIONAL_FRANKA_1ARM = (
    "franka_1arm is PROVISIONAL until the Franka survey (robotensor/ICIL-robotwin-benchmark#83) "
    "names it: the one-arm tasks of pick_and_place and press_push, and the arm-switching tasks of "
    "stacking, which has no one-arm task"
)


def provisional_franka_1arm(table: tasks.TaskTable) -> tuple[str, ...]:
    """PROVISIONAL `franka_1arm`, in table order: per category of the track, its one-arm tasks.

    A category with no one-arm task gives its arm-switching tasks instead, so that every skill of
    the track can draw a unit: that is stacking, where each expert picks the arm nearest the next
    object. Two-arm tasks are never in it.
    """
    members: list[str] = []
    for category in FRANKA_1ARM_CATEGORIES:
        in_category = [t for t in table.tasks.values() if t.category == category]
        one_arm = [t.name for t in in_category if t.arms == ONE]
        members += one_arm or [t.name for t in in_category if t.arms == SWITCHING]
    return tuple(members)


def suites(table: tasks.TaskTable) -> dict[str, tuple[str, ...]]:
    """Every suite units can be drawn from: the table's, and `franka_1arm` until it has one."""
    out = dict(table.suites)
    out.setdefault(FRANKA_1ARM, provisional_franka_1arm(table))
    return out


def provisional(table: tasks.TaskTable) -> list[str]:
    """What the catalogue serves that no survey has settled yet."""
    return [PROVISIONAL_FRANKA_1ARM] if FRANKA_1ARM not in table.suites else []


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
