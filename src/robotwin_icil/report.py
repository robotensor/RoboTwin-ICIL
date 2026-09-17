"""Aggregate episode records into a Same Scene result: overall, by skill category, by task.

The only model score is the Same Scene 1-Demo Success Rate over valid evaluated episodes. Rejected
episodes (the expert could not solve the scene) and invalid ones (the reset did not reproduce it)
never enter its denominator; they are reported as diagnostics of the benchmark, not of the model.

Rates are fractions in `[0, 1]` everywhere; `render` is the single place they become percentages.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .arms import ONE
from .records import EpisodeRecord, RunManifest, Status
from .robotwin import embodiment_name
from .tasks import TaskTable


@dataclass(frozen=True)
class Rate:
    successes: int
    episodes: int

    @property
    def value(self) -> float | None:
        """None, not 0.0, when nothing was scored: an empty row is not a failing row."""
        return self.successes / self.episodes if self.episodes else None

    def to_json(self) -> dict[str, Any]:
        return {"successes": self.successes, "episodes": self.episodes, "success_rate": self.value}


@dataclass(frozen=True)
class Diagnostics:
    """Benchmark-quality numbers. Never a model score."""

    episodes: int
    scored: int
    rejected: int
    invalid: int
    expert_attempts: int
    rejections_by_reason: dict[str, int]
    mean_rollout_steps: float | None
    mean_successful_rollout_steps: float | None
    # For each rejection reason, the detail seen most often: what the failure actually was.
    rejection_examples: dict[str, str] = field(default_factory=dict)

    @property
    def expert_rejection_rate(self) -> float | None:
        rejected = sum(self.rejections_by_reason.values())
        return rejected / self.expert_attempts if self.expert_attempts else None

    @property
    def simulation_failure_rate(self) -> float | None:
        failed = self.rejections_by_reason.get("unstable", 0) + self.rejections_by_reason.get(
            "expert_error", 0
        )
        return failed / self.expert_attempts if self.expert_attempts else None

    def to_json(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "scored": self.scored,
            "rejected": self.rejected,
            "invalid": self.invalid,
            "expert_attempts": self.expert_attempts,
            "expert_rejection_rate": self.expert_rejection_rate,
            "simulation_failure_rate": self.simulation_failure_rate,
            "rejections_by_reason": dict(sorted(self.rejections_by_reason.items())),
            "mean_rollout_steps": self.mean_rollout_steps,
            "mean_successful_rollout_steps": self.mean_successful_rollout_steps,
            "rejection_examples": dict(sorted(self.rejection_examples.items())),
        }


@dataclass(frozen=True)
class Report:
    overall: Rate
    by_category: dict[str, Rate]
    by_task: dict[str, dict[str, Rate]]
    diagnostics: Diagnostics

    def to_json(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_json(),
            "by_category": {c: r.to_json() for c, r in self.by_category.items()},
            "by_task": {
                c: {t: r.to_json() for t, r in tasks.items()} for c, tasks in self.by_task.items()
            },
            "diagnostics": self.diagnostics.to_json(),
        }


def _rate(records: list[EpisodeRecord]) -> Rate:
    scored = [r for r in records if r.scored]
    return Rate(successes=sum(1 for r in scored if r.success), episodes=len(scored))


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def build(records: list[EpisodeRecord], table: TaskTable) -> Report:
    """Aggregate in table order, so two reports over the same tasks line up row for row."""
    by_category: dict[str, Rate] = {}
    by_task: dict[str, dict[str, Rate]] = {}
    for category in table.categories:
        in_category = [r for r in records if r.skill_category == category]
        if not in_category:
            continue
        by_category[category] = _rate(in_category)
        tasks = [t.name for t in table.tasks.values() if t.category == category]
        by_task[category] = {
            task: _rate([r for r in in_category if r.task == task])
            for task in tasks
            if any(r.task == task for r in in_category)
        }

    scored = [r for r in records if r.scored]
    reasons: Counter[str] = Counter()
    for record in records:
        reasons.update(record.rejections)
    diagnostics = Diagnostics(
        episodes=len(records),
        scored=len(scored),
        rejected=sum(1 for r in records if r.status is Status.REJECTED),
        invalid=sum(1 for r in records if r.status is Status.INVALID),
        expert_attempts=sum(r.expert_generation_attempts for r in records),
        rejections_by_reason=dict(reasons),
        mean_rollout_steps=_mean([r.steps for r in scored]),
        mean_successful_rollout_steps=_mean([r.steps for r in scored if r.success]),
        rejection_examples=_examples(records),
    )
    return Report(
        overall=_rate(records), by_category=by_category, by_task=by_task, diagnostics=diagnostics
    )


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def render(report: Report, manifest: RunManifest | None, table: TaskTable) -> str:
    """The plain-text report: overall, then each category, then its tasks."""
    lines = ["RoboTwin-ICIL", "============", ""]
    if manifest is not None:
        lines += [
            f"Evaluation setting:          {manifest.evaluation_setting}",
            "Demonstrations per episode:  1",
            f"Policy:                      {manifest.policy.get('policy', '?')}",
            f"Embodiment:                  {_embodiment(manifest)}",
            f"Tasks:                       {', '.join(manifest.tasks)}"
            + (" (one-arm tasks only)" if manifest.arms == ONE else ""),
            f"Global seed:                 {manifest.global_seed}",
            "",
        ]
    overall = report.overall
    lines += [
        f"Overall Same Scene 1-Demo Success:  {_percent(overall.value)}  ({overall.successes}/{overall.episodes})",
        "",
        "By manipulation skill:",
    ]
    width = max([len(t) for tasks in report.by_task.values() for t in tasks] + [24])
    for category, rate in report.by_category.items():
        label = table.categories[category]
        lines.append(
            f"  {label:<{width}}  {_percent(rate.value):>7}  ({rate.successes}/{rate.episodes})"
        )
        for task, task_rate in report.by_task[category].items():
            lines.append(
                f"    {task:<{width - 2}}  {_percent(task_rate.value):>7}  ({task_rate.successes}/{task_rate.episodes})"
            )
    d = report.diagnostics
    lines += [
        "",
        "Diagnostics (benchmark quality, not a model score):",
        f"  episodes {d.episodes}: scored {d.scored}, rejected {d.rejected}, invalid {d.invalid}",
        f"  expert attempts {d.expert_attempts}, rejection rate {_percent(d.expert_rejection_rate)},"
        f" simulation failure rate {_percent(d.simulation_failure_rate)}",
        "  rejections: "
        + (", ".join(f"{k} {v}" for k, v in sorted(d.rejections_by_reason.items())) or "none"),
        f"  mean rollout steps {_fmt(d.mean_rollout_steps)}, successful {_fmt(d.mean_successful_rollout_steps)}",
    ]
    for reason, example in d.rejection_examples.items():
        lines.append(f"    e.g. {reason}: {example[:120]}")
    return "\n".join(lines) + "\n"


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def _embodiment(manifest: RunManifest) -> str:
    """The robot a run ran on, as `embodiment_name` names it.

    A manifest written before a run could choose its robot has no `embodiment_name`, but it does
    carry RoboTwin's `embodiment` list, which names the robot on its own; nothing is asserted that
    the manifest does not say.
    """
    config = manifest.robotwin_config
    if config.get("embodiment_name") is not None:
        return str(config["embodiment_name"])
    if config.get("embodiment"):
        return embodiment_name(config["embodiment"])
    return "?"


def _examples(records: list[EpisodeRecord]) -> dict[str, str]:
    """For each rejection reason, the detail seen most often across the run."""
    seen: dict[str, Counter[str]] = {}
    for record in records:
        for reason, detail in record.rejection_details.items():
            seen.setdefault(reason, Counter())[detail] += record.rejections.get(reason, 1)
    return {reason: counts.most_common(1)[0][0] for reason, counts in sorted(seen.items())}
