"""The episode loop: a suite, N episodes, one run directory.

Episodes are assigned to tasks round-robin in suite order, so every task gets `N // len(tasks)`
episodes (the first `N % len(tasks)` get one more) and episode `i`'s task is a pure function of
the suite and `i`. Together with per-episode seed streams, that makes any single episode
reproducible without replaying the ones before it, and lets an interrupted run resume by skipping
what `episodes.jsonl` already holds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any

from . import camera_profiles
from .episode import EpisodeSpec, run_episode
from .policy import ICILPolicy, PolicyError, check_description, json_mapping
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
from .video import DEFAULT_CAMERA, EpisodeVideo

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
    # The camera the clips show: the camera profile's `video_camera`.
    video_camera: str = DEFAULT_CAMERA
    # The keyword arguments the policy was built with; recorded, and part of the run's identity.
    policy_config: Mapping[str, Any] = field(default_factory=dict)


def assign(tasks: Sequence[Task], episodes: int) -> list[Task]:
    """Episode i runs tasks[i % len(tasks)]: balanced, ordered, and a pure function of i."""
    if not tasks:
        raise ValueError("a run needs at least one task")
    return [tasks[i % len(tasks)] for i in range(episodes)]


def manifest_for(
    spec: RunSpec, policy: ICILPolicy, config, description: dict[str, Any] | None = None
) -> RunManifest:
    from . import robotwin

    args = config.resolve()
    return RunManifest(
        global_seed=spec.global_seed,
        evaluation_setting=SAME_SCENE,
        suite=spec.suite,
        tasks=tuple(task.name for task in spec.tasks),
        episodes=spec.episodes,
        max_expert_attempts=spec.max_expert_attempts,
        policy=policy.describe() if description is None else description,
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
        policy_environment=_policy_environment(policy),
        policy_config=_policy_config(spec.policy_config),
    )


def _policy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """The policy's keyword arguments as the manifest holds them: plain JSON, paths as strings."""
    stringified = {k: str(v) if isinstance(v, PurePath) else v for k, v in config.items()}
    return json_mapping(stringified, "policy config")


def _policy_environment(policy: ICILPolicy) -> dict[str, str]:
    """The policy's `environment()`, refused unless every value is a string."""
    what = f"{policy.name}: environment()"
    reported = json_mapping(policy.environment(), what)
    for key, value in reported.items():
        if not isinstance(value, str):
            raise PolicyError(f"{what}: {key!r} must be a string, not {type(value).__name__}")
    return reported


def _described(policy: ICILPolicy, config) -> dict[str, Any]:
    """The policy's description, held to the convention, before anything is generated for it.

    A policy that needs a camera profile other than the run's is refused here rather than handed
    images from cameras it was not built for.
    """
    description = policy.describe()
    check_description(description)
    required = description.get("camera_profile_required")
    if required is not None and required != config.camera_profile:
        raise PolicyError(
            f"{policy.name} requires camera profile {required!r} but the run uses "
            f"{config.camera_profile!r}; pass --camera-profile {required}"
        )
    # As JSON reads it back, so a resumed run's description compares equal to the manifest's.
    return json_mapping(description, f"{policy.name}: describe()")


def run(
    spec: RunSpec,
    policy: ICILPolicy,
    config,
    log: Callable[[str], None] = print,
) -> list[EpisodeRecord]:
    """Run (or resume) every episode of `spec`, appending each record as it finishes.

    The run owns the policy's end: `policy.close()` is called exactly once, however the run ends.
    Before that, the frozen-policy audit describes the policy again: when the run finished, and
    also when it raised, if the policy gave a `parameter_checksum` at the start. A failed audit is
    recorded in the run directory before it is raised, so the run neither resumes nor reports.
    """
    description: dict[str, Any] | None = None
    began = False
    finished = False
    try:
        # Resolve before entering the RoboTwin seam, which moves the working directory.
        run_dir = RunDir(Path(spec.run_dir).resolve())
        # Described at the start and the end, not per episode: an adapter's description may hash
        # its parameters, which is not free.
        description = _described(policy, config)
        run_dir.start(manifest_for(spec, policy, config, description))
        began = True
        done = run_dir.completed()
        plan = assign(spec.tasks, spec.episodes)
        if done:
            log(f"resuming {run_dir.path}: {len(done)}/{len(plan)} episodes already recorded")
        _run_pending(spec, policy, config, run_dir, plan, done, description, log)
        finished = True
        return sorted(run_dir.records(), key=lambda record: record.episode)
    finally:
        try:
            if description is not None and (
                finished or description.get("parameter_checksum") is not None
            ):
                # A run refused at the start may have been pointed at another run's directory:
                # only a run this call began or resumed is marked.
                _audit(policy, description, run_dir if began else None)
        finally:
            policy.close()


def _audit(policy: ICILPolicy, started: dict[str, Any], run_dir: RunDir | None) -> None:
    """The frozen-policy audit: the parameters a run ends with are the ones it started with.

    Only as strong as the adapter's `parameter_checksum`; a policy that gives none is not audited.
    A failure is written to `run_dir`, when given, before it is raised: a fresh copy of the policy
    would describe itself as the manifest does, and resume the run as if nothing had happened.
    """
    ended = policy.describe()
    check_description(ended)
    before = started.get("parameter_checksum")
    after = ended.get("parameter_checksum")
    if before is not None and after != before:
        if run_dir is not None:
            run_dir.fail_audit(
                {"policy": policy.name, "parameter_checksum": {"start": before, "end": after}}
            )
        raise PolicyError(
            f"{policy.name}: parameters changed during the run: the policy must be frozen "
            f"(parameter_checksum {before!r} at the start, {after!r} at the end)"
        )


def _run_pending(
    spec: RunSpec,
    policy: ICILPolicy,
    config,
    run_dir: RunDir,
    plan: list[Task],
    done: set[int],
    description: dict[str, Any],
    log: Callable[[str], None],
) -> None:
    """Run every episode of `plan` not in `done`, appending each record as it finishes."""
    from . import robotwin

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
                    video=(
                        EpisodeVideo(run_dir.episode_dir(episode), camera=spec.video_camera)
                        if spec.video
                        else None
                    ),
                    description=description,
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
