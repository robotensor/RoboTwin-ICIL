"""The episode loop: a suite, N episodes, one run directory.

Episodes are assigned to tasks round-robin in suite order, so every task gets `N // len(tasks)`
episodes (the first `N % len(tasks)` get one more) and episode `i`'s task is a pure function of
the suite and `i`. Together with per-episode seed streams, that makes any single episode
reproducible without replaying the ones before it, and lets an interrupted run resume by skipping
what `episodes.jsonl` already holds.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .episode import EpisodeSpec, run_episode
from .policy import ICILPolicy
from .records import (
    SAME_SCENE,
    EpisodeRecord,
    RunDir,
    RunManifest,
    Status,
    environment,
    git_commit,
)
from .tasks import Task

DEFAULT_MAX_EXPERT_ATTEMPTS = 20

# The parts of RoboTwin's resolved args that decide what the expert and the policy see. The
# embodiment configs themselves are large and implied by `embodiment` plus the RoboTwin commit.
_ROBOTWIN_CONFIG_KEYS = (
    "task_config",
    "embodiment",
    "camera",
    "data_type",
    "domain_randomization",
    "save_freq",
)


@dataclass(frozen=True)
class RunSpec:
    run_dir: Path
    tasks: tuple[Task, ...]
    suite: str | None
    episodes: int
    global_seed: int
    max_expert_attempts: int = DEFAULT_MAX_EXPERT_ATTEMPTS
    clear_cache_every: int = 5


def assign(tasks: Sequence[Task], episodes: int) -> list[Task]:
    """Episode i runs tasks[i % len(tasks)]: balanced, ordered, and a pure function of i."""
    if not tasks:
        raise ValueError("a run needs at least one task")
    return [tasks[i % len(tasks)] for i in range(episodes)]


def manifest_for(spec: RunSpec, policy: ICILPolicy, config) -> RunManifest:
    from . import robotwin

    args = config.resolve()
    return RunManifest(
        global_seed=spec.global_seed,
        evaluation_setting=SAME_SCENE,
        suite=spec.suite,
        tasks=tuple(task.name for task in spec.tasks),
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

    envs: dict[str, Any] = {}
    for episode, task in enumerate(plan):
        if episode in done:
            continue
        task_env = envs.get(task.name)
        if task_env is None:
            task_env = envs[task.name] = robotwin.load_task(task.name)
        record = run_episode(
            EpisodeSpec(
                episode=episode,
                task=task,
                global_seed=spec.global_seed,
                max_expert_attempts=spec.max_expert_attempts,
            ),
            policy,
            config,
            task_env=task_env,
        )
        run_dir.append(record)
        log(_line(record, len(plan)))
        if spec.clear_cache_every and (episode + 1) % spec.clear_cache_every == 0:
            robotwin.clear_render_cache()
    return run_dir.records()


def _line(record: EpisodeRecord, total: int) -> str:
    if record.status is Status.SCORED:
        outcome = "success" if record.success else "failure"
        tail = f"{record.steps} steps"
    else:
        outcome = record.status.value
        tail = record.detail
    return (
        f"[{record.episode + 1}/{total}] {record.task} seed={record.scene_seed} "
        f"attempts={record.expert_generation_attempts} {outcome} ({tail})"
    )
