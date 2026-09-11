"""Aggregate episode records into the V1 result: overall, by skill category, by task.

The only model score is the Same Scene 1-Demo Success Rate over valid evaluated episodes. Rejected
episodes (the expert could not solve the scene) and invalid ones (the reset did not reproduce it)
never enter its denominator; they are reported as diagnostics of the benchmark, not of the model.

Rates are fractions in `[0, 1]` everywhere; `render` is the single place they become percentages.

Reference runs (`report --reference`), an oracle's or another model's, print in columns beside the
run's own rates. They are context, not a second score, and stand beside a run only when they
evaluated the same scenes: the same global seed, tasks, RoboTwin commit, configuration and camera
profile.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .records import EpisodeRecord, RunDir, RunManifest, Status
from .tasks import TaskTable

# What a reference must share with the reported run, besides its camera profile and configs: the
# scenes are drawn from the global seed per episode, task by task, under the same expert budget,
# and built by RoboTwin's own task code (`load_actors`, the expert) at the commit recorded.
_SAME = (
    "evaluation_setting",
    "global_seed",
    "suite",
    "tasks",
    "max_expert_attempts",
    "robotwin_commit",
)
_MISSING = object()


class ReportError(ValueError):
    """A reference run cannot stand beside the reported run: it did not evaluate the same scenes."""


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


@dataclass(frozen=True)
class Reference:
    """Another run of the same scenes, reported beside this one under its policy's name."""

    label: str
    run_dir: str
    manifest: RunManifest
    report: Report

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "run_dir": self.run_dir,
            "policy": self.manifest.policy,
            **self.report.to_json(),
        }


def policy_label(manifest: RunManifest) -> str:
    """How a report names a run's policy: its adapter, else its policy name."""
    return str(manifest.policy.get("adapter") or manifest.policy.get("policy") or "?")


def differences(run: RunManifest, reference: RunManifest) -> list[str]:
    """The fields in which `reference` did not evaluate the scenes `run` did; empty when it did.

    Compared as resuming compares them (`RunManifest.identity()`), so a manifest from before
    camera profiles reads as `stock`. The RoboTwin commit is compared as resuming compares it,
    `-dirty` included: RoboTwin's task code builds the scenes and plans the expert's moves. The
    policy, its arguments, the benchmark's own commit and the machine may differ: that is what a
    reference is for.
    """
    ours, theirs = run.identity(), reference.identity()
    found = [
        key if key == "tasks" else f"{key} ({theirs[key]!r}, not {ours[key]!r})"
        for key in _SAME
        if ours[key] != theirs[key]
    ]
    if run.camera_profile() != reference.camera_profile():
        found.append(
            f"camera profile ({_profile(reference.camera_profile())}, "
            f"not {_profile(run.camera_profile())})"
        )
    for section in ("benchmark_config", "robotwin_config"):
        mine, other = ours[section], theirs[section]
        for key in sorted(set(mine) | set(other)):
            if (section, key) == ("benchmark_config", "camera_profile"):
                continue
            if mine.get(key, _MISSING) != other.get(key, _MISSING):
                found.append(f"{section}.{key}")
    return found


def load_reference(path: Path, run: RunManifest, table: TaskTable) -> Reference:
    """Load a reference run directory, refusing one that differs from `run` in its scenes, or
    that failed the frozen-policy audit."""
    reference = RunDir(Path(path))
    reference.check_audit()
    manifest = reference.manifest()
    found = differences(run, manifest)
    if found:
        raise ReportError(
            f"reference {reference.path} differs from the reported run in {'; '.join(found)}: a "
            "reference must run the same global seed, tasks, RoboTwin commit, configuration and "
            "camera profile"
        )
    return Reference(
        label=policy_label(manifest),
        run_dir=str(reference.path),
        manifest=manifest,
        report=build(reference.records(), table),
    )


def _rate(records: list[EpisodeRecord]) -> Rate:
    scored = [r for r in records if r.scored]
    return Rate(successes=sum(1 for r in scored if r.success), episodes=len(scored))


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def build(records: list[EpisodeRecord], table: TaskTable) -> Report:
    """Aggregate in table order, so two reports over the same suite line up row for row."""
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


def render(
    report: Report,
    manifest: RunManifest | None,
    table: TaskTable,
    references: Sequence[Reference] = (),
) -> str:
    """The plain-text report: overall, then each category, then its tasks.

    Each reference adds a column beside the run's own rates, and its overall rate under the run's.
    """
    lines = ["RoboTwin ICIL Benchmark", "=======================", ""]
    if manifest is not None:
        lines += [
            f"Evaluation setting:          {manifest.evaluation_setting}",
            "Demonstrations per episode:  1",
            f"Policy:                      {manifest.policy.get('policy', '?')}",
            f"Adapter:                     {_adapter(manifest.policy)}",
            f"Training regime:             {_regime(seen_in_training(manifest))}",
            f"Camera profile:              {_profile(manifest.camera_profile())}",
            f"Suite:                       {manifest.suite or ', '.join(manifest.tasks)}",
            f"Global seed:                 {manifest.global_seed}",
            "",
        ]
    overall = report.overall
    lines.append(
        f"Overall Same Scene 1-Demo Success:  {_percent(overall.value)}  ({overall.successes}/{overall.episodes})"
    )
    labels = _unique(
        [policy_label(manifest) if manifest is not None else "this run"]
        + [reference.label for reference in references]
    )
    if references:
        lines.append(
            "Reference runs (same global seed, tasks, RoboTwin commit, configuration and "
            "camera profile; not scores of this run):"
        )
        label_width = max(len(label) for label in labels[1:])
        for label, reference in zip(labels[1:], references, strict=True):
            lines.append(
                f"  {label:<{label_width}}  {_cell(reference.report.overall)}  {reference.run_dir}"
            )
    lines += ["", "By manipulation skill:"]
    reports = [report, *(reference.report for reference in references)]
    rows = _rows(reports, table)
    cells = [[_cell(_lookup(r, category, task)) for r in reports] for category, task in rows]
    width = max([len(task) for _, task in rows if task is not None] + [24])
    # One column needs no heading and no padding: the report reads as it always has.
    column = max(len(text) for text in [*labels, *(c for row in cells for c in row)])
    column = column if references else 0
    if references:
        lines.append(
            (f"  {'':<{width}}  " + "  ".join(name.ljust(column) for name in labels)).rstrip()
        )
    for (category, task), row in zip(rows, cells, strict=True):
        name = (
            f"  {table.categories[category]:<{width}}"
            if task is None
            else f"    {task:<{width - 2}}"
        )
        lines.append((f"{name}  " + "  ".join(text.ljust(column) for text in row)).rstrip())
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


def _rows(reports: Sequence[Report], table: TaskTable) -> list[tuple[str, str | None]]:
    """(category, task) in table order over every report, task None for a category's own row."""
    rows: list[tuple[str, str | None]] = []
    for category in table.categories:
        if not any(category in r.by_category for r in reports):
            continue
        rows.append((category, None))
        for task in (t.name for t in table.tasks.values() if t.category == category):
            if any(task in r.by_task.get(category, {}) for r in reports):
                rows.append((category, task))
    return rows


def _lookup(report: Report, category: str, task: str | None) -> Rate | None:
    if task is None:
        return report.by_category.get(category)
    return report.by_task.get(category, {}).get(task)


def _cell(rate: Rate | None) -> str:
    """A rate as the report prints it; a row a run has no episodes in is a bare dash."""
    if rate is None:
        return f"{'—':>7}"
    return f"{_percent(rate.value):>7}  ({rate.successes}/{rate.episodes})"


def _unique(labels: list[str]) -> list[str]:
    """Labels with repeats numbered, so two runs of one policy stay two columns: a, a #2."""
    seen: Counter[str] = Counter()
    unique = []
    for label in labels:
        seen[label] += 1
        unique.append(label if seen[label] == 1 else f"{label} #{seen[label]}")
    return unique


def seen_in_training(manifest: RunManifest) -> tuple[int, int] | None:
    """(k, N): of the run's N evaluated tasks, the k its policy says it was trained on.

    None when the description's `training_tasks` is "unknown" or absent: no claim either way.
    """
    trained = manifest.policy.get("training_tasks")
    if not isinstance(trained, (list, tuple)):
        return None
    evaluated = set(manifest.tasks)
    return len(evaluated & set(trained)), len(evaluated)


def _regime(seen: tuple[int, int] | None) -> str:
    """Held out when no evaluated task was trained on: V1 then measures use of the demonstration
    rather than memory of the task."""
    if seen is None:
        return "unknown (evaluation tasks seen in training: unknown)"
    k, n = seen
    return f"{'held out' if k == 0 else 'seen tasks'} (evaluation tasks seen in training: {k}/{n})"


def _adapter(policy: dict[str, Any]) -> str:
    return (
        f"{policy.get('adapter') or '—'} (adapter_version {policy.get('adapter_version') or '—'})"
    )


def _profile(identity: dict[str, str]) -> str:
    return f"{identity.get('name', '?')} (sha256 {str(identity.get('sha256', '?'))[:12]})"


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def _examples(records: list[EpisodeRecord]) -> dict[str, str]:
    """For each rejection reason, the detail seen most often across the run."""
    seen: dict[str, Counter[str]] = {}
    for record in records:
        for reason, detail in record.rejection_details.items():
            seen.setdefault(reason, Counter())[detail] += record.rejections.get(reason, 1)
    return {reason: counts.most_common(1)[0][0] for reason, counts in sorted(seen.items())}
