"""A duel's units, as a pure function of the seed material the orchestrator hands over.

A third party holding a published record must derive the same units, on any Python and any numpy,
so nothing here consults a library RNG or Python's hash: every number is drawn from `Sha256Counter`,
sha256 over the seed material and a counter, in a fixed order. The task order is drawn first and
then each unit's candidates in turn, so a smaller duel's units are the first units of a larger one.
"""

from __future__ import annotations

import hashlib
from typing import Any

from robotwin_icil import tasks

from . import catalogue

#: Candidate scene seeds per unit, tried in order by `materialize` until the expert solves one. A
#: unit is void for both sides only when all of them are rejected: at the lowest expert success
#: the v1 survey kept (75%), four candidates all fail 0.4% of the time, and at 50% about 6% (the
#: spec voids a duel past 20%). Every rejected candidate costs one scene and one expert run, and
#: four of the slowest experts still fit materialize's wall clock.
SCENE_SEED_CANDIDATES = 4

#: Scene seeds are drawn from [0, SEED_BOUND): RoboTwin feeds them to `np.random.seed`, and this is
#: the bound the benchmark's own seed streams use.
SEED_BOUND = 2**31 - 1

#: Hashed ahead of the seed material, so these draws are this derivation's and no other's. A change
#: to how units are drawn changes this label, and with it every unit.
DERIVATION = "robotwin-icil-competition/units/1"


class Sha256Counter:
    """Integers from sha256(label | material | counter), the counter counting up from 0."""

    def __init__(self, material: str) -> None:
        self._prefix = f"{DERIVATION}|{material}|".encode()
        self._counter = 0

    def next_u64(self) -> int:
        digest = hashlib.sha256(self._prefix + str(self._counter).encode("ascii")).digest()
        self._counter += 1
        return int.from_bytes(digest[:8], "big")

    def below(self, bound: int) -> int:
        """Uniform in [0, bound), by rejecting the draws that would bias a modulo."""
        if bound < 1:
            raise ValueError(f"bound must be at least 1, not {bound}")
        limit = (1 << 64) - (1 << 64) % bound
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound


def suite_tasks(table: tasks.TaskTable, suite: str, category: str | None) -> list[str]:
    """The tasks units of `suite` (within `category`) are drawn from, in table order."""
    known = catalogue.suites(table)
    if suite not in known:
        raise ValueError(f"unknown suite {suite!r}; the catalogue has {', '.join(sorted(known))}")
    members = list(known[suite])
    if category is not None:
        if category not in table.categories:
            raise ValueError(f"unknown category {category!r}")
        members = [name for name in members if table[name].category == category]
        if not members:
            raise ValueError(f"suite {suite!r} has no task in category {category!r}")
    return members


def derive_units(
    *,
    seed_material: str,
    count: int,
    suite: str,
    category: str | None = None,
    table: tasks.TaskTable | None = None,
) -> list[dict[str, Any]]:
    """`count` units of `suite` (within `category`), each a task and its candidate scene seeds.

    Tasks are spread evenly: the suite's tasks are shuffled once and dealt in turn, so no task gets
    two units more than another. Each unit carries `task`, `task_label`, `suite`, `category` and
    `instance_params` — `scene_seeds`, its `SCENE_SEED_CANDIDATES` candidates in the order they
    are tried; `embodiment`, the robot; and `scene_seed`, None, because which candidate becomes
    the unit's scene is known only once materialize has run (its `result.json` and the prompt's
    `meta` name it).
    """
    if not isinstance(seed_material, str):
        raise TypeError(f"seed_material must be a string, not {type(seed_material).__name__}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"count must be a non-negative integer, not {count!r}")
    table = table or tasks.table()
    members = suite_tasks(table, suite, category)
    embodiment = catalogue.embodiment_of(suite)

    draws = Sha256Counter(seed_material)
    order = list(members)
    for i in range(len(order) - 1, 0, -1):  # Fisher-Yates, on sha256 draws
        j = draws.below(i + 1)
        order[i], order[j] = order[j], order[i]

    units = []
    for index in range(count):
        name = order[index % len(order)]
        seeds: list[int] = []
        while len(seeds) < SCENE_SEED_CANDIDATES:
            seed = draws.below(SEED_BOUND)
            if seed not in seeds:
                seeds.append(seed)
        units.append(
            {
                "task": name,
                "task_label": catalogue.task_label(name),
                "suite": suite,
                "category": table[name].category,
                "instance_params": {
                    "scene_seed": None,
                    "scene_seeds": seeds,
                    "embodiment": embodiment,
                },
            }
        )
    return units
