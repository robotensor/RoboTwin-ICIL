"""What the plugin draws units from: tasks, their categories and arms, and the suites.

The tasks, their categories and the arms each task's expert uses (`Task.arms`, read off its
`play_once`) are `robotwin_icil.tasks`'s. The suites are the competition's own: `franka_1arm`, the
suite the orchestrator's one-arm Franka track draws from, chosen from the Franka survey
(robotensor/RoboTwin-ICIL#83, measurements at `SURVEY`, rule and decision in this package's README).
"""

from __future__ import annotations

from typing import Any

from robotwin_icil import tasks
from robotwin_icil.arms import ONE, SWITCHING, TWO

#: The suite the orchestrator's one-arm Franka track names in `spec.json`.
FRANKA_1ARM = "franka_1arm"
#: The categories that track scores, as its skills name them.
FRANKA_1ARM_CATEGORIES = ("pick_and_place", "stacking", "press_push")
#: The track's tasks, in task-table order: the tasks whose expert, on two Franka arms, solved at
#: least 2 of 3 surveyed seeds and moved one arm in every demonstration it produced. Stacking has
#: no one-arm task, so stack_bowls_two, an arm-switching task, stands in for it.
SUITES: dict[str, tuple[str, ...]] = {
    FRANKA_1ARM: ("place_empty_cup", "stack_bowls_two", "click_bell", "press_stapler"),
}
#: Where the survey that chose `franka_1arm` gives its measurements.
SURVEY = "docs/survey.md#franka-panda-survey"
#: What `info()` calls a `franka_1arm` task whose expert is not a one-arm one.
STAND_IN_KINDS = {SWITCHING: "an arm-switching", TWO: "a two-arm"}

#: The robot a suite's units run on: the Franka suite on two Franka arms, any other suite on the
#: benchmark's default robot.
FRANKA = "franka-panda"
DEFAULT_EMBODIMENT = "aloha-agilex"


def suites(table: tasks.TaskTable) -> dict[str, tuple[str, ...]]:
    """Every suite units can be drawn from, each in its own order. A member the task table does
    not hold is refused, naming it."""
    for members in SUITES.values():
        for name in members:
            table[name]
    return dict(SUITES)


def stand_ins(table: tasks.TaskTable) -> dict[str, str]:
    """The `franka_1arm` tasks whose expert is not a one-arm one, in suite order, each with what it
    stands in for and the limit that comes with it: stack_bowls_two, for stacking, which has no
    one-arm task at the pinned RoboTwin. Read off the table, so a note cannot outlive it."""
    notes = {}
    for name in suites(table)[FRANKA_1ARM]:
        task = table[name]
        if task.arms == ONE:
            continue
        without = not any(
            other.category == task.category and other.arms == ONE for other in table.tasks.values()
        )
        where = f"{task.category}, which has no one-arm task" if without else task.category
        notes[name] = (
            f"{name} is {STAND_IN_KINDS[task.arms]} stand-in for {where}: every demonstration the "
            "Franka survey kept moved one arm, but a scene the survey did not see can still make "
            "its expert move both arms, and neither materialize nor verify_prompt refuses such a "
            "demonstration"
        )
    return notes


def franka_1arm_basis(table: tasks.TaskTable) -> dict[str, Any]:
    """What `franka_1arm` is made of, for `info()`: the survey that chose it, each category of the
    track with its tasks and the arms their experts use, and `stand_ins`."""
    members = suites(table)[FRANKA_1ARM]
    categories = {}
    for category in FRANKA_1ARM_CATEGORIES:
        names = [name for name in members if table[name].category == category]
        categories[category] = {"tasks": names, "arms": sorted({table[n].arms for n in names})}
    return {"survey": SURVEY, "categories": categories, "stand_ins": stand_ins(table)}


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
