"""The task table: which RoboTwin tasks the benchmark runs, and what skill each one exercises.

The benchmark reports by manipulation skill category, so the mapping is data (``tasks.yml``) that
every caller iterates, never a set of names spelled out in code. ``check_against_robotwin`` is what
keeps the table honest when the pinned submodule moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

TABLE_PATH = Path(__file__).with_name("tasks.yml")

# Upstream files in `envs/` that are not tasks.
_NON_TASK_STEMS = frozenset({"__init__", "_base_task", "_GLOBAL_CONFIGS"})


class TaskTableError(ValueError):
    """The table is internally inconsistent, or disagrees with the pinned RoboTwin checkout."""


@dataclass(frozen=True)
class Task:
    """One RoboTwin task and the skill category it is scored under."""

    name: str
    category: str
    category_label: str


@dataclass(frozen=True)
class TaskTable:
    tasks: dict[str, Task]
    categories: dict[str, str]
    suites: dict[str, tuple[str, ...]]

    def __getitem__(self, name: str) -> Task:
        try:
            return self.tasks[name]
        except KeyError:
            raise TaskTableError(f"unknown task {name!r}; it is not in {TABLE_PATH.name}") from None

    def suite(self, name: str) -> tuple[Task, ...]:
        """The tasks of a named suite, in table order so a run is reproducible from its name."""
        try:
            members = self.suites[name]
        except KeyError:
            known = ", ".join(sorted(self.suites))
            raise TaskTableError(f"unknown suite {name!r}; known suites: {known}") from None
        return tuple(self.tasks[task] for task in members)

    def by_category(self) -> dict[str, tuple[Task, ...]]:
        grouped: dict[str, list[Task]] = {category: [] for category in self.categories}
        for task in self.tasks.values():
            grouped[task.category].append(task)
        return {category: tuple(tasks) for category, tasks in grouped.items()}


def load_table(path: Path = TABLE_PATH) -> TaskTable:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    categories = dict(raw["categories"])
    tasks: dict[str, Task] = {}
    for name, category in raw["tasks"].items():
        if category not in categories:
            raise TaskTableError(f"task {name!r} has unknown category {category!r}")
        tasks[name] = Task(name=name, category=category, category_label=categories[category])

    suites: dict[str, tuple[str, ...]] = {}
    for suite_name, members in raw["suites"].items():
        if members == "*":
            suites[suite_name] = tuple(tasks)
            continue
        for member in members:
            if member not in tasks:
                raise TaskTableError(f"suite {suite_name!r} names unknown task {member!r}")
        if len(set(members)) != len(members):
            raise TaskTableError(f"suite {suite_name!r} lists a task twice")
        suites[suite_name] = tuple(members)

    return TaskTable(tasks=tasks, categories=categories, suites=suites)


@lru_cache(maxsize=1)
def table() -> TaskTable:
    return load_table()


def robotwin_task_names(robotwin_root: Path) -> frozenset[str]:
    """Every task class RoboTwin actually ships, read from its ``envs/`` directory."""
    envs = robotwin_root / "envs"
    if not envs.is_dir():
        raise TaskTableError(f"no RoboTwin checkout at {robotwin_root}; init the submodule")
    return frozenset(path.stem for path in envs.glob("*.py") if path.stem not in _NON_TASK_STEMS)


def check_against_robotwin(robotwin_root: Path, table_: TaskTable | None = None) -> None:
    """Raise unless the table covers the pinned checkout exactly.

    An upstream rename would otherwise drop a task from every suite silently, which would move the
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
