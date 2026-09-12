"""Which units a duel runs, derived so a third party can reproduce them.

Only the benchmark knows what a unit of it means. Here it is a task and a **scene seed**: RoboTwin
builds a scene deterministically from `(task, seed, config)`, so the seed is the whole initial
state and a published unit list is enough to rebuild every scene it names.

Derivation is a sha256 counter, not numpy's RNG, for the same reason the competition's own
derivation is: a stream that depends on a library version is not reproducible by someone holding
the published record.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

#: Scene seeds are drawn from this range. RoboTwin seeds numpy and torch with the value, so it
#: has to be a non-negative 32-bit integer.
SEED_SPACE = 2**31


def _counter(*parts: Any) -> int:
    material = "|".join(str(p) for p in parts).encode()
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


@dataclass(frozen=True)
class BenchmarkUnit:
    """One scored episode: a task, the scene it happens in, and where it sits in the list."""

    unit_index: int
    task: str
    skill_category: str
    scene_seed: int
    suite: str
    #: How many times the expert may be re-drawn before the unit is substituted. Expert failure is
    #: a generation failure, never a model failure, so this is a materializing budget.
    max_expert_attempts: int = 20

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkUnit:
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


def derive_units(
    *,
    seed_material: str,
    count: int,
    suite: str = "v1",
    tasks: list[str] | None = None,
    max_expert_attempts: int = 20,
) -> list[BenchmarkUnit]:
    """`count` units over a suite, a pure function of its arguments.

    Tasks are assigned round-robin in table order, which is how the benchmark's own runner spreads
    episodes: a short list then covers every task before repeating any.
    """
    if count <= 0:
        raise ValueError("a duel runs at least one unit")
    members = tasks if tasks is not None else [task.name for task in _suite(suite)]
    if not members:
        raise ValueError(f"suite {suite!r} has no tasks")
    out: list[BenchmarkUnit] = []
    for index in range(count):
        name = members[index % len(members)]
        out.append(
            BenchmarkUnit(
                unit_index=index,
                task=name,
                skill_category=_category(name),
                scene_seed=_counter(seed_material, suite, index) % SEED_SPACE,
                suite=suite,
                max_expert_attempts=max_expert_attempts,
            )
        )
    return out


def _suite(suite: str):
    from robotwin_icil import tasks

    return tasks.table().suite(suite)


def _category(name: str) -> str:
    from robotwin_icil import tasks

    return tasks.table()[name].category
