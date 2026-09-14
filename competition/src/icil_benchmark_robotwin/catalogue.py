"""What the plugin draws units from: tasks, their categories and arms, and the suites.

All of it is `robotwin_icil.tasks`'s: the tasks, their categories, the arms each task's expert uses
(`Task.arms`, read off its `play_once`) and the suites. One of them is `franka_1arm`, the suite the
orchestrator's one-arm Franka track draws from, which the Franka survey chose
(robotensor/ICIL-robotwin-benchmark#83, written up at `SURVEY`).
"""

from __future__ import annotations

from typing import Any

from robotwin_icil import tasks

#: The suite the orchestrator's one-arm Franka track names in `spec.json`.
FRANKA_1ARM = "franka_1arm"
#: The categories that track scores, as its skills name them.
FRANKA_1ARM_CATEGORIES = ("pick_and_place", "stacking", "press_push")
#: Where the survey that chose `franka_1arm` gives its table, its rule and its decision.
SURVEY = "docs/survey.md#franka-one-arm-survey"

#: The robot a suite's units run on: the Franka suite on two Franka arms, every other suite on
#: the benchmark's default robot.
FRANKA = "franka-panda"
DEFAULT_EMBODIMENT = "aloha-agilex"


def suites(table: tasks.TaskTable) -> dict[str, tuple[str, ...]]:
    """Every suite units can be drawn from: the task table's, each in its own order."""
    return dict(table.suites)


def franka_1arm_basis(table: tasks.TaskTable) -> dict[str, Any]:
    """What `franka_1arm` is made of, for `info()`: the survey that chose it, and each category of
    the track with its tasks and the arms their experts use."""
    members = table.suites.get(FRANKA_1ARM, ())
    categories = {}
    for category in FRANKA_1ARM_CATEGORIES:
        names = [name for name in members if table[name].category == category]
        categories[category] = {"tasks": names, "arms": sorted({table[n].arms for n in names})}
    return {"survey": SURVEY, "categories": categories}


def embodiment_of(suite: str) -> str:
    """The robot a suite's units run on, as `robotwin-icil --embodiment` names it."""
    return FRANKA if suite == FRANKA_1ARM else DEFAULT_EMBODIMENT


def task_label(name: str) -> str:
    """A task's name for people: `click_bell` -> `Click bell`."""
    return name.replace("_", " ").capitalize()


def catalogue(table: tasks.TaskTable | None = None) -> dict[str, Any]:
    """`{"suites": {suite: [task]}, "categories": {id: label}, "tasks": {task: {"category",
    "arms", "label"}}, "embodiments": {suite: robot}}`."""
    table = table or tasks.table()
    return {
        "suites": {name: list(members) for name, members in suites(table).items()},
        "categories": dict(table.categories),
        "tasks": {
            name: {"category": task.category, "arms": task.arms, "label": task_label(name)}
            for name, task in table.tasks.items()
        },
        "embodiments": {name: embodiment_of(name) for name in suites(table)},
    }
