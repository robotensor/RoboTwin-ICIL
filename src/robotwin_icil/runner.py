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

from . import camera_profiles
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
from .video import EpisodeVideo

DEFAULT_MAX_EXPERT_ATTEMPTS = 20

# The parts of RoboTwin's resolved args that decide what the expert and the policy see. The
# embodiment configs themselves are large and implied by `embodiment` plus the RoboTwin commit;
# the static cameras a camera profile rewrites in them are recorded as a derived entry,
# `static_cameras`.
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
    # Clips per episode; off by default. Not part of the run's identity: it changes no result.
    video: bool = False


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
            "camera_profile": camera_profiles.get(config.camera_profile).identity(),
        },
        robotwin_config={
            **{key: args.get(key) for key in _ROBOTWIN_CONFIG_KEYS},
            "static_cameras": camera_profiles.static_cameras(args),
        },
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
    # them for its lifetime, so holding every task's env at once grows GPU memory with the suite.
    # Episodes run task by task — episode i still runs plan[i] on its own seed stream — and each
    # task's env is released before the next one is built.
    progress = len(done)
    for task in spec.tasks:
        pending = [
            i for i, assigned in enumerate(plan) if assigned.name == task.name and i not in done
        ]
        if not pending:
            continue
        task_env = robotwin.load_task(task.name)
        try:
            for episode in pending:
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
                    video=EpisodeVideo(run_dir.episode_dir(episode)) if spec.video else None,
                )
                run_dir.append(record)
                progress += 1
                log(_line(record, progress, len(plan)))
                if spec.clear_cache_every and progress % spec.clear_cache_every == 0:
                    robotwin.clear_render_cache()
        finally:
            robotwin.close(task_env)
            task_env = None
            robotwin.free_gpu()
    return sorted(run_dir.records(), key=lambda record: record.episode)


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
