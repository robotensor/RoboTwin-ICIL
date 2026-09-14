"""Measure RoboTwin's expert before trusting a suite with it.

A task belongs in a scored suite only if its expert reliably produces a demonstration: every
rejected seed costs a full expert run, and a task whose expert rarely succeeds dominates a run's
wall-clock and its rejection statistics. A survey runs the expert alone over a fixed seed stream
per task and reports how often it succeeds, and why it does not.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .demo import arm_displacements, arms_moved
from .generate import attempt
from .tasks import Task

# The outcome of a seed whose expert produced a demonstration; every other outcome is a rejection.
OK = "ok"


@dataclass(frozen=True)
class SeedRecord:
    """What one surveyed seed measured: its outcome and, for a success, what the expert did.

    `arms_moved` is measured from the demonstration's joints (`demo.arms_moved`), so a task's
    `arms` entry — a static read of the expert's source — is checked against the expert's actual
    behaviour on every scene it solved. `displacement` is the number that measurement
    thresholds, each arm's largest departure from its first-frame value, kept so a borderline
    seed can be told from an idle one after the run. All three are None for a rejected seed.
    """

    seed: int
    outcome: str
    frames: int | None
    arms_moved: tuple[str, ...] | None
    displacement: dict[str, float] | None
    seconds: float

    @property
    def ok(self) -> bool:
        return self.outcome == OK

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "outcome": self.outcome,
            "frames": self.frames,
            "arms_moved": None if self.arms_moved is None else list(self.arms_moved),
            "displacement": None if self.displacement is None else dict(self.displacement),
            "seconds": self.seconds,
        }


@dataclass
class TaskSurvey:
    """One task's survey: the per-seed records, and every tally derived from them."""

    task: Task
    records: list[SeedRecord] = field(default_factory=list)
    # The robot the expert ran on: its success rate is the pair's, not the task's alone.
    embodiment: str | None = None
    # Whether every frame rendered its cameras. The survey reads only joints, so by default none do.
    images: bool = False

    @property
    def seeds(self) -> int:
        return len(self.records)

    @property
    def successes(self) -> int:
        return sum(1 for record in self.records if record.ok)

    @property
    def rejections(self) -> Counter:
        return Counter(record.outcome for record in self.records if not record.ok)

    @property
    def frames(self) -> list[int]:
        return [record.frames for record in self.records if record.ok]

    @property
    def seconds(self) -> float:
        return sum(record.seconds for record in self.records)

    @property
    def success_rate(self) -> float | None:
        return self.successes / self.seeds if self.seeds else None

    @property
    def mean_frames(self) -> float | None:
        return sum(self.frames) / len(self.frames) if self.frames else None

    def demonstrations_moving(self, arms: int) -> int:
        """How many successful demonstrations moved exactly `arms` arms.

        Over 0, 1 and 2 these partition the successes. A demonstration that moved no arm — a
        scene the expert found already solved, or motion under the threshold — is neither one-arm
        nor two-arm evidence and is counted on its own rather than folded into either.
        """
        return sum(1 for record in self.records if record.ok and len(record.arms_moved) == arms)

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task.name,
            "skill_category": self.task.category,
            "embodiment": self.embodiment,
            "images": self.images,
            "seeds": self.seeds,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "rejections": dict(sorted(self.rejections.items())),
            "one_arm_demonstrations": self.demonstrations_moving(1),
            "two_arm_demonstrations": self.demonstrations_moving(2),
            "no_arm_demonstrations": self.demonstrations_moving(0),
            "mean_demonstration_frames": self.mean_frames,
            "seconds_per_seed": self.seconds / self.seeds if self.seeds else None,
            "seeds_detail": [record.to_json() for record in self.records],
        }


def survey_task(
    task_env, task: Task, seeds: list[int], config, attempt_fn=attempt, images: bool = False
) -> TaskSurvey:
    """Run the expert once per seed, exactly as an episode's generator would, and record each.

    Everything recorded is read from the joints, so by default no camera renders a frame
    (`robotwin.capture`): ray tracing every camera is most of a seed's cost, and no expert reads
    an image. `images=True` renders them as an episode does.
    """
    result = TaskSurvey(
        task=task, embodiment=str(config.resolve(task.name)["embodiment_name"]), images=images
    )
    for index, seed in enumerate(seeds):
        started = time.monotonic()
        # Resolved afresh per seed, as the generator does: nothing RoboTwin mutates while building
        # one scene leaks into the next.
        args = config.resolve(task.name)
        outcome, demonstration, _ = attempt_fn(
            task_env, seed, args, config.save_freq, index, images=images
        )
        seconds = time.monotonic() - started
        if outcome.rejection is None:
            record = SeedRecord(
                seed,
                OK,
                len(demonstration),
                arms_moved(demonstration),
                arm_displacements(demonstration),
                seconds,
            )
        else:
            record = SeedRecord(seed, outcome.rejection.value, None, None, None, seconds)
        result.records.append(record)
    return result


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.0f}%"


def render(results: list[TaskSurvey]) -> str:
    """The plain-text table, one row per task; a rate is the task's on the robot the row names.

    *one-arm* is how many of the successes moved exactly one arm.
    """
    # The task, robot and expert columns are as wide as their widest cell, so every column after
    # them stays put whatever the names and the seed count.
    width = max([len(r.task.name) for r in results] + [4])
    robot_width = max([len(r.embodiment or "?") for r in results] + [5])
    experts = [f"{_percent(r.success_rate)} ({r.successes}/{r.seeds})" for r in results]
    expert_width = max([len(expert) for expert in experts] + [6])
    lines = [
        f"{'task':<{width}}  {'category':<14}  {'robot':<{robot_width}}  {'expert':>{expert_width}}  "
        f"{'one-arm':>7}  {'frames':>6}  {'s/seed':>6}  rejections"
    ]
    for r, expert in zip(results, experts, strict=True):
        rejections = ", ".join(f"{k} {v}" for k, v in sorted(r.rejections.items())) or "—"
        frames = "—" if r.mean_frames is None else f"{r.mean_frames:.0f}"
        per_seed = f"{r.seconds / r.seeds:.1f}" if r.seeds else "—"
        one_arm = str(r.demonstrations_moving(1)) if r.successes else "—"
        lines.append(
            f"{r.task.name:<{width}}  {r.task.category:<14}  {r.embodiment or '?':<{robot_width}}  "
            f"{expert:>{expert_width}}  {one_arm:>7}  {frames:>6}  {per_seed:>6}  {rejections}"
        )
    return "\n".join(lines) + "\n"
