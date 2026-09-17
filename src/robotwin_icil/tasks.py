"""The task table: which RoboTwin tasks the benchmark runs, what skill each one exercises, and how
many arms its expert needs.

The benchmark reports by manipulation skill category and can restrict a run to one-arm tasks, so
both are data (``tasks.yml``) that every caller iterates, never a set of names spelled out in
code. ``check_against_robotwin`` and ``check_arms_against_robotwin`` are what keep the table
honest when the pinned submodule moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from . import arms as arms_
from .arms import ARMS, LABELS, NON_TASK_STEMS, ONE, TWO

TABLE_PATH = Path(__file__).with_name("tasks.yml")


class TaskTableError(ValueError):
    """The table is internally inconsistent, or disagrees with the pinned RoboTwin checkout."""


@dataclass(frozen=True)
class Task:
    """One RoboTwin task, the skill category it is scored under, and the arms its expert uses."""

    name: str
    category: str
    category_label: str
    # "1", "switching" or "2" (`arms.ARMS`), from a static read of the task's play_once.
    arms: str


@dataclass(frozen=True)
class TaskTable:
    tasks: dict[str, Task]
    categories: dict[str, str]

    def __getitem__(self, name: str) -> Task:
        try:
            return self.tasks[name]
        except KeyError:
            raise TaskTableError(f"unknown task {name!r}; it is not in {TABLE_PATH.name}") from None

    def select(self, *, task: str | None = None, arms: str = TWO) -> tuple[Task, ...]:
        """Select all tasks or one task, narrowed to one-arm tasks by ``arms="1"``.

        ``arms="2"`` — the default, a two-arm robot — changes nothing. A single task that is not
        one-arm is refused rather than silently run.
        """
        selected = (self[task],) if task is not None else tuple(self.tasks.values())
        if arms == TWO:
            return selected
        if arms != ONE:
            raise TaskTableError(f"a run asks for arms {ONE} or {TWO}, not {arms!r}")
        one_arm = tuple(member for member in selected if member.arms == ONE)
        if not one_arm:
            if task is not None:
                what = f"task {task!r} needs {LABELS[selected[0].arms]}"
            else:
                what = "the task catalog has no one-arm task"
            raise TaskTableError(f"{what}; --arms 1 runs only tasks whose expert uses one arm")
        return one_arm

    def by_category(self) -> dict[str, tuple[Task, ...]]:
        grouped: dict[str, list[Task]] = {category: [] for category in self.categories}
        for task in self.tasks.values():
            grouped[task.category].append(task)
        return {category: tuple(tasks) for category, tasks in grouped.items()}


def load_table(path: Path = TABLE_PATH) -> TaskTable:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    categories = dict(raw["categories"])
    tasks: dict[str, Task] = {}
    for name, entry in raw["tasks"].items():
        if not isinstance(entry, dict) or set(entry) != {"category", "arms"}:
            raise TaskTableError(f"task {name!r} needs exactly `category` and `arms`")
        category = entry["category"]
        if category not in categories:
            raise TaskTableError(f"task {name!r} has unknown category {category!r}")
        # YAML reads `arms: 1` as an int; the table's vocabulary is the strings in ARMS.
        arms = str(entry["arms"])
        if arms not in ARMS:
            raise TaskTableError(
                f"task {name!r} has arms {entry['arms']!r}; expected one of {', '.join(ARMS)}"
            )
        tasks[name] = Task(
            name=name, category=category, category_label=categories[category], arms=arms
        )

    return TaskTable(tasks=tasks, categories=categories)


@lru_cache(maxsize=1)
def table() -> TaskTable:
    return load_table()


def robotwin_task_names(robotwin_root: Path) -> frozenset[str]:
    """Every task class RoboTwin actually ships, read from its ``envs/`` directory."""
    envs = robotwin_root / "envs"
    if not envs.is_dir():
        raise TaskTableError(f"no RoboTwin checkout at {robotwin_root}; init the submodule")
    return frozenset(path.stem for path in envs.glob("*.py") if path.stem not in NON_TASK_STEMS)


def check_against_robotwin(robotwin_root: Path, table_: TaskTable | None = None) -> None:
    """Raise unless the table covers the pinned checkout exactly.

    An upstream rename would otherwise drop a task from every run silently, which would move the
    reported score without moving anything visible in this repository.
    """
    table_ = table_ or table()
    upstream = robotwin_task_names(robotwin_root)
    ours = frozenset(table_.tasks)
    if missing := sorted(upstream - ours):
        raise TaskTableError(f"RoboTwin tasks missing from {TABLE_PATH.name}: {', '.join(missing)}")
    if extra := sorted(ours - upstream):
        raise TaskTableError(
            f"{TABLE_PATH.name} names tasks RoboTwin does not ship: {', '.join(extra)}"
        )


def check_arms_against_robotwin(robotwin_root: Path, table_: TaskTable | None = None) -> None:
    """Raise, naming the task, unless every task's ``arms`` matches a static read of its expert.

    A one-arm run selects tasks by this value; a wrong one would score a two-arm expert as a
    one-arm task, or hide a one-arm task from the run.
    """
    table_ = table_ or table()
    check_against_robotwin(robotwin_root, table_)
    verdicts = arms_.classify_all(robotwin_root / "envs")
    disagreements = [
        f"{name}: {TABLE_PATH.name} says {task.arms}, play_once says {verdicts[name].arms} "
        f"({verdicts[name].evidence})"
        for name, task in table_.tasks.items()
        if task.arms != verdicts[name].arms
    ]
    if disagreements:
        raise TaskTableError("arms disagree with RoboTwin:\n  " + "\n  ".join(disagreements))
