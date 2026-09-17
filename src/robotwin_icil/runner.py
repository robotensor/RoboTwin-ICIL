"""The episode loop: selected tasks, N episodes, one run directory.

Episodes are assigned to tasks round-robin in selection order, so every task gets `N // len(tasks)`
episodes (the first `N % len(tasks)` get one more) and episode `i`'s task is a pure function of
the task list and `i`. Together with per-episode seed streams, that makes any single episode
reproducible without replaying the ones before it, and lets an interrupted run resume by skipping
what `episodes.jsonl` already holds.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .arms import LABELS, ONE, TWO
from .episode import EpisodeSpec, run_episode
from .generate import INDEPENDENT_SEED_STREAM, ROBOTWIN_SEED_STREAM, SEED_STREAMS, robotwin_seeds
from .policy import ICILPolicy
from .records import (
    SAME_SCENE,
    EpisodeRecord,
    RecordError,
    RunDir,
    RunManifest,
    Status,
    environment,
    git_commit,
)
from .tasks import Task
from .video import EpisodeVideo

DEFAULT_MAX_EXPERT_ATTEMPTS = 20

# The parts of RoboTwin's resolved args that decide what the expert and the policy see. The
# embodiment configs themselves are large and implied by `embodiment` plus the RoboTwin commit.
_ROBOTWIN_CONFIG_KEYS = (
    "task_config",
    "embodiment",
    "embodiment_name",
    "camera",
    "data_type",
    "domain_randomization",
    "save_freq",
)


@dataclass(frozen=True)
class RunSpec:
    run_dir: Path
    tasks: tuple[Task, ...]
    episodes: int
    global_seed: int
    max_expert_attempts: int = DEFAULT_MAX_EXPERT_ATTEMPTS
    clear_cache_every: int = 5
    # Clips per episode; off by default. Not part of the run's identity: it changes no result.
    video: bool = False
    # "1" when the run asked for one-arm tasks only; "2", the default, runs whatever was named.
    arms: str = TWO
    seed_stream: str = INDEPENDENT_SEED_STREAM
    standard: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # The CLI selects through TaskTable.select; any other caller is held to the same rule, so
        # no manifest can claim a one-arm run over a task whose expert needs more.
        if self.arms not in (ONE, TWO):
            raise ValueError(f"a run asks for arms {ONE} or {TWO}, not {self.arms!r}")
        if self.seed_stream not in SEED_STREAMS:
            raise ValueError(f"a run draws seeds from {' or '.join(SEED_STREAMS)}")
        if self.arms == ONE:
            for task in self.tasks:
                if task.arms != ONE:
                    raise ValueError(
                        f"task {task.name!r} needs {LABELS[task.arms]}; a one-arm run holds only "
                        "tasks whose expert uses one arm"
                    )


def assign(tasks: Sequence[Task], episodes: int) -> list[Task]:
    """Episode i runs tasks[i % len(tasks)]: balanced, ordered, and a pure function of i."""
    if not tasks:
        raise ValueError("a run needs at least one task")
    return [tasks[i % len(tasks)] for i in range(episodes)]


def manifest_for(spec: RunSpec, policy: ICILPolicy, config) -> RunManifest:
    from . import robotwin

    extra = {}
    if spec.standard is not None:
        from . import source_sha256
        from .standard import validate_run

        validate_run(spec, config)
        extra = {"standard": spec.standard, "source_sha256": source_sha256()}
    args = config.resolve()
    return RunManifest(
        global_seed=spec.global_seed,
        evaluation_setting=SAME_SCENE,
        tasks=tuple(task.name for task in spec.tasks),
        arms=spec.arms,
        seed_stream=spec.seed_stream,
        episodes=spec.episodes,
        max_expert_attempts=spec.max_expert_attempts,
        policy=policy.describe(),
        benchmark_commit=git_commit(robotwin.REPO_ROOT),
        robotwin_commit=git_commit(robotwin.ROBOTWIN_ROOT),
        benchmark_config={
            "task_config": config.task_config,
            "save_freq": config.save_freq,
            "head_camera": config.head_camera,
            "overrides": dict(config.overrides or {}),
            "embodiment": config.embodiment,
            **extra,
        },
        robotwin_config={key: args.get(key) for key in _ROBOTWIN_CONFIG_KEYS},
        environment=environment(),
    )


def run(
    spec: RunSpec,
    policy: ICILPolicy,
    config,
    log: Callable[[str], None] = print,
) -> list[EpisodeRecord]:
    """Run (or resume) every episode of `spec`, appending each record as it finishes."""
    from . import robotwin

    # Resolve before entering the RoboTwin seam, which moves the working directory.
    run_dir = RunDir(Path(spec.run_dir).resolve())
    run_dir.start(manifest_for(spec, policy, config))
    done = run_dir.completed()
    plan = assign(spec.tasks, spec.episodes)
    if done:
        log(f"resuming {run_dir.path}: {len(done)}/{len(plan)} episodes already recorded")

    # One RoboTwin env alive at a time. Each env builds two CuRobo planners on the GPU and keeps
    # them for its lifetime, so holding every task's env at once grows GPU memory with the task count.
    # Episodes run task by task — episode i still runs plan[i] on its own seed stream — and each
    # task's env is released before the next one is built.
    progress = len(done)
    tried = _seeds_tried(spec, plan, run_dir) if spec.seed_stream == ROBOTWIN_SEED_STREAM else {}
    for task in spec.tasks:
        pending = [
            i for i, assigned in enumerate(plan) if assigned.name == task.name and i not in done
        ]
        if not pending:
            continue
        task_env = robotwin.load_task(task.name)
        try:
            for episode in pending:
                seeds = None
                if spec.seed_stream == ROBOTWIN_SEED_STREAM:
                    seeds = tuple(
                        robotwin_seeds(spec.global_seed, tried[task.name], spec.max_expert_attempts)
                    )
                record = run_episode(
                    EpisodeSpec(
                        episode=episode,
                        task=task,
                        global_seed=spec.global_seed,
                        max_expert_attempts=spec.max_expert_attempts,
                        seeds=seeds,
                    ),
                    policy,
                    config,
                    task_env=task_env,
                    video=EpisodeVideo(run_dir.episode_dir(episode)) if spec.video else None,
                )
                run_dir.append(record)
                if seeds is not None:
                    tried[task.name] += record.expert_generation_attempts
                progress += 1
                log(_line(record, progress, len(plan)))
                if spec.clear_cache_every and progress % spec.clear_cache_every == 0:
                    robotwin.clear_render_cache()
        finally:
            robotwin.close(task_env)
            task_env = None
            robotwin.free_gpu()
    return sorted(run_dir.records(), key=lambda record: record.episode)


def _seeds_tried(spec: RunSpec, plan: list[Task], run_dir: RunDir) -> dict[str, int]:
    """How many seeds of RoboTwin's stream each task's recorded episodes tried.

    A task's episodes run in order, each taking the seeds after the last one tried before it, so
    a resumed run continues the stream exactly — provided no earlier episode of the task is missing.
    """
    tried = {task.name: 0 for task in spec.tasks}
    records = {record.episode: record for record in run_dir.records()}
    for task in spec.tasks:
        indices = [i for i, assigned in enumerate(plan) if assigned.name == task.name]
        recorded = [i for i in indices if i in records]
        if recorded != indices[: len(recorded)]:
            raise RecordError(
                f"{run_dir.path}: {task.name}'s episodes are not recorded in order, so RoboTwin's "
                "seed stream cannot be continued; choose a new --run-dir"
            )
        tried[task.name] = sum(records[i].expert_generation_attempts for i in recorded)
    return tried


def _line(record: EpisodeRecord, done: int, total: int) -> str:
    if record.status is Status.SCORED:
        outcome = "success" if record.success else "failure"
        tail = f"{record.steps} steps"
    else:
        outcome = record.status.value
        tail = record.detail
    return (
        f"[{done}/{total}] episode {record.episode} {record.task} seed={record.scene_seed} "
        f"attempts={record.expert_generation_attempts} {outcome} ({tail})"
    )
