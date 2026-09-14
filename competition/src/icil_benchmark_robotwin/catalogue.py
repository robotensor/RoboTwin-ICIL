"""What the plugin draws units from: tasks, their categories and arms, and the suites.

Tasks, categories and the benchmark's own suites are `robotwin_icil.tasks`'s. Two things are not
in that table on this branch yet, and are PROVISIONAL here:

- **The arms each task's expert uses.** They are read off each task's `play_once` on branch
  `arms-and-embodiment`, whose table records them as `Task.arms`; this branch is not stacked on
  it. `PROVISIONAL_ARMS` is a copy of that table, used only while `Task` has no `arms`, and a test
  holds the two equal whenever both exist.
- **The `franka_1arm` suite** the orchestrator's one-arm Franka track draws from. The Franka
  survey (robotensor/ICIL-robotwin-benchmark#83) names it and has not run. Until the table names
  it, `provisional_franka_1arm` serves the one-arm tasks of the track's three categories.
"""

from __future__ import annotations

from typing import Any

from robotwin_icil import tasks

#: How many arms a task's expert uses, as the arms table spells it.
ONE, SWITCHING, TWO = "1", "switching", "2"

#: Where `PROVISIONAL_ARMS` was copied from.
PROVISIONAL_ARMS_SOURCE = "branch arms-and-embodiment, src/robotwin_icil/tasks.yml at 5510453f50d1"

#: PROVISIONAL: every task's arms, until `robotwin_icil.tasks.Task` carries them.
PROVISIONAL_ARMS: dict[str, str] = {
    "dump_bin_bigbin": TWO,
    "move_can_pot": ONE,
    "move_pillbottle_pad": ONE,
    "move_playingcard_away": ONE,
    "move_stapler_pad": ONE,
    "place_a2b_left": ONE,
    "place_a2b_right": ONE,
    "place_bread_basket": TWO,
    "place_bread_skillet": TWO,
    "place_burger_fries": TWO,
    "place_can_basket": TWO,
    "place_cans_plasticbox": TWO,
    "place_container_plate": ONE,
    "place_dual_shoes": TWO,
    "place_empty_cup": ONE,
    "place_fan": ONE,
    "place_mouse_pad": ONE,
    "place_object_basket": TWO,
    "place_object_scale": ONE,
    "place_object_stand": ONE,
    "place_shoe": ONE,
    "put_bottles_dustbin": TWO,
    "blocks_ranking_rgb": SWITCHING,
    "blocks_ranking_size": SWITCHING,
    "stack_blocks_three": SWITCHING,
    "stack_blocks_two": SWITCHING,
    "stack_bowls_three": SWITCHING,
    "stack_bowls_two": SWITCHING,
    "beat_block_hammer": ONE,
    "click_alarmclock": ONE,
    "click_bell": ONE,
    "press_stapler": ONE,
    "stamp_seal": ONE,
    "turn_switch": ONE,
    "open_laptop": ONE,
    "open_microwave": ONE,
    "put_object_cabinet": TWO,
    "hanging_mug": TWO,
    "place_phone_stand": ONE,
    "grab_roller": TWO,
    "handover_block": TWO,
    "handover_mic": TWO,
    "lift_pot": TWO,
    "pick_diverse_bottles": TWO,
    "pick_dual_bottles": TWO,
    "adjust_bottle": ONE,
    "rotate_qrcode": ONE,
    "scan_object": TWO,
    "shake_bottle": ONE,
    "shake_bottle_horizontally": ONE,
}

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
PROVISIONAL_ARMS_NOTE = (
    f"task arms are PROVISIONAL, copied from {PROVISIONAL_ARMS_SOURCE}, until robotwin_icil's "
    "task table records them"
)


def arms_of(task: tasks.Task) -> str:
    """How many arms `task`'s expert uses: the table's own value once it has one."""
    recorded = getattr(task, "arms", None)
    return str(recorded) if recorded is not None else PROVISIONAL_ARMS[task.name]


def provisional_franka_1arm(table: tasks.TaskTable) -> tuple[str, ...]:
    """PROVISIONAL `franka_1arm`, in table order: per category of the track, its one-arm tasks.

    A category with no one-arm task gives its arm-switching tasks instead, so that every skill of
    the track can draw a unit: that is stacking, where each expert picks the arm nearest the next
    object. Two-arm tasks are never in it.
    """
    members: list[str] = []
    for category in FRANKA_1ARM_CATEGORIES:
        in_category = [t for t in table.tasks.values() if t.category == category]
        one_arm = [t.name for t in in_category if arms_of(t) == ONE]
        members += one_arm or [t.name for t in in_category if arms_of(t) == SWITCHING]
    return tuple(members)


def suites(table: tasks.TaskTable) -> dict[str, tuple[str, ...]]:
    """Every suite units can be drawn from: the table's, and `franka_1arm` until it has one."""
    out = dict(table.suites)
    out.setdefault(FRANKA_1ARM, provisional_franka_1arm(table))
    return out


def provisional(table: tasks.TaskTable) -> list[str]:
    """What the catalogue serves that no survey or table has settled yet."""
    notes = []
    if FRANKA_1ARM not in table.suites:
        notes.append(PROVISIONAL_FRANKA_1ARM)
    if any(getattr(task, "arms", None) is None for task in table.tasks.values()):
        notes.append(PROVISIONAL_ARMS_NOTE)
    return notes


def embodiment_of(suite: str) -> str:
    """The robot a suite's units run on, as `robotwin-icil --embodiment` names it."""
    return FRANKA if suite == FRANKA_1ARM else DEFAULT_EMBODIMENT


def task_label(name: str) -> str:
    """A task's name for people: `click_bell` -> `Click bell`."""
    return name.replace("_", " ").capitalize()


def catalogue(table: tasks.TaskTable | None = None) -> dict[str, Any]:
    """`{"suites": {suite: [task]}, "categories": {id: label}, "tasks": {task: {"category",
    "arms"}}}`, plus what of it is provisional."""
    table = table or tasks.table()
    return {
        "suites": {name: list(members) for name, members in suites(table).items()},
        "categories": dict(table.categories),
        "tasks": {
            name: {"category": task.category, "arms": arms_of(task), "label": task_label(name)}
            for name, task in table.tasks.items()
        },
        "embodiments": {name: embodiment_of(name) for name in suites(table)},
        "provisional": provisional(table),
    }
