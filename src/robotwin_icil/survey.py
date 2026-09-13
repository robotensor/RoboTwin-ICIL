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

from .generate import attempt
from .tasks import Task


@dataclass
class TaskSurvey:
    task: Task
    # The robot the expert ran on: its success rate is the pair's, not the task's alone.
    embodiment: str | None = None
    seeds: int = 0
    successes: int = 0
    rejections: Counter = field(default_factory=Counter)
    frames: list[int] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def success_rate(self) -> float | None:
        return self.successes / self.seeds if self.seeds else None

    @property
    def mean_frames(self) -> float | None:
        return sum(self.frames) / len(self.frames) if self.frames else None

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task.name,
            "skill_category": self.task.category,
            "embodiment": self.embodiment,
            "seeds": self.seeds,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "rejections": dict(sorted(self.rejections.items())),
            "mean_demonstration_frames": self.mean_frames,
            "seconds_per_seed": self.seconds / self.seeds if self.seeds else None,
        }


def survey_task(task_env, task: Task, seeds: list[int], config, attempt_fn=attempt) -> TaskSurvey:
    """Run the expert once per seed, exactly as an episode's generator would, and tally."""
    result = TaskSurvey(task=task)
    for index, seed in enumerate(seeds):
        started = time.monotonic()
        args = config.resolve(task.name)
        result.embodiment = str(args["embodiment_name"])
        outcome, demonstration, _ = attempt_fn(task_env, seed, args, config.save_freq, index)
        result.seconds += time.monotonic() - started
        result.seeds += 1
        if outcome.rejection is None:
            result.successes += 1
            result.frames.append(len(demonstration))
        else:
            result.rejections[outcome.rejection.value] += 1
    return result


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.0f}%"


def render(results: list[TaskSurvey]) -> str:
    width = max([len(r.task.name) for r in results] + [4])
    lines = [
        f"{'task':<{width}}  {'category':<14}  {'expert':>13}  {'frames':>6}  {'s/seed':>6}  rejections"
    ]
    for r in results:
        rejections = ", ".join(f"{k} {v}" for k, v in sorted(r.rejections.items())) or "—"
        frames = "—" if r.mean_frames is None else f"{r.mean_frames:.0f}"
        per_seed = f"{r.seconds / r.seeds:.1f}" if r.seeds else "—"
        expert = f"{_percent(r.success_rate)} ({r.successes}/{r.seeds})"
        lines.append(
            f"{r.task.name:<{width}}  {r.task.category:<14}  {expert:>13}  {frames:>6}  {per_seed:>6}  {rejections}"
        )
    return "\n".join(lines) + "\n"
