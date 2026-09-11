"""One benchmark episode: expert demonstration, exact reset, frozen rollout, RoboTwin's verdict.

    generate a successful expert demonstration   (scene seed S, fingerprint F)
    close the env                                (never continue from the expert's final state)
    setup_demo again with seed S                 (the Same Scene)
    fingerprint again, require == F              (or the episode is invalid, not failed)
    policy.reset(); policy.set_demonstration(D)
    roll out through take_action until eval_success or the task's step limit

This is `vendor/RoboTwin/scripts/eval_policy_xpolicylab.py:run_one_batch_episode` with the expert
trajectory kept and handed to the policy instead of thrown away.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .generate import Generated, generate, scene_seeds
from .policy import ICILPolicy, Observation, PolicyError
from .records import SAME_SCENE, EpisodeRecord, Status
from .scene import compare, max_error
from .tasks import Task
from .video import EpisodeVideo


@dataclass(frozen=True)
class EpisodeSpec:
    episode: int
    task: Task
    global_seed: int
    max_expert_attempts: int


def run_episode(
    spec: EpisodeSpec,
    policy: ICILPolicy,
    config,
    task_env=None,
    video: EpisodeVideo | None = None,
) -> EpisodeRecord:
    """Run one episode end to end and return its record. Never raises for a scene or model failure.

    A `PolicyError` does propagate: an adapter that breaks the protocol is a bug to fix, not a
    stream of zero scores to average over. With `video`, the demonstration and the evaluation are
    written as two clips; a clip that fails to write is noted in the record, never scored.
    """
    from . import robotwin

    started = time.monotonic()
    task_env = task_env if task_env is not None else robotwin.load_task(spec.task.name)
    seeds = scene_seeds(spec.global_seed, spec.episode, spec.max_expert_attempts)
    generated = generate(
        task_env, seeds, lambda: config.resolve(spec.task.name), config.save_freq, spec.episode
    )
    describe = policy.describe()

    def record(status: Status, **fields) -> EpisodeRecord:
        return EpisodeRecord(
            episode=spec.episode,
            evaluation_setting=SAME_SCENE,
            skill_category=spec.task.category,
            task=spec.task.name,
            scene_seed=generated.seed,
            status=status,
            expert_generation_attempts=len(generated.attempts),
            rejections=_rejections(generated),
            rejection_details=_rejection_details(generated),
            model=str(describe.get("model", describe["policy"])),
            checkpoint=describe.get("checkpoint"),
            duration_s=round(time.monotonic() - started, 3),
            **fields,
        )

    empty = dict(success=None, steps=0, step_limit=None, scene_max_error=0.0)
    if not generated.ok:
        return record(
            Status.REJECTED,
            demonstration_frames=0,
            detail=f"no successful expert demonstration in {len(generated.attempts)} seeds",
            **empty,
        )

    demonstration = generated.demonstration
    video_note = _film(video, lambda: video.demonstration(demonstration))
    try:
        task_env.setup_demo(
            now_ep_num=spec.episode,
            seed=generated.seed,
            is_test=True,
            **config.resolve(spec.task.name),
        )
    except Exception as exc:
        robotwin.close(task_env)
        return record(
            Status.INVALID,
            demonstration_frames=len(demonstration),
            detail=f"evaluation scene failed to build: {type(exc).__name__}: {exc}",
            **empty,
        )

    try:
        mismatches = compare(generated.initial, robotwin.fingerprint(task_env))
        if mismatches:
            # The evaluation's first frame is the evidence for a reset bug; the episode is
            # already invalid, so observing it cannot change anything that is scored.
            video_note += _film(video, lambda: _final_frame(video, task_env))
            return record(
                Status.INVALID,
                demonstration_frames=len(demonstration),
                detail="scene drift: " + "; ".join(str(m) for m in mismatches[:5]) + video_note,
                **{**empty, "scene_max_error": max_error(mismatches)},
            )
        policy.reset()
        policy.set_demonstration(demonstration)
        success, detail = rollout(
            task_env, policy, observe=video.observe if video is not None else None
        )
        video_note += _film(video, lambda: _final_frame(video, task_env))
        return record(
            Status.SCORED,
            success=success,
            steps=int(task_env.take_action_cnt),
            step_limit=task_env.step_lim,
            demonstration_frames=len(demonstration),
            scene_max_error=0.0,
            detail=detail + video_note,
        )
    finally:
        robotwin.close(task_env)


def rollout(
    task_env,
    policy: ICILPolicy,
    observe: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> tuple[bool, str]:
    """Drive the policy until RoboTwin reports success or the task's step limit is reached.

    Success is RoboTwin's own: `take_action` runs `check_success()` after every step and latches
    `eval_success`. An exception from the simulator mid-rollout ends the episode as a failure, as
    upstream's evaluator does, except a GPU that runs out of memory or is lost: that breaks the
    simulator, not the policy, so it stops the run and a resume replays the episode.
    """
    from . import robotwin

    try:
        while not robotwin.episode_over(task_env):
            raw = robotwin.observation(task_env)
            if observe is not None:
                observe(raw["images"])
            observation = Observation(
                step=int(task_env.take_action_cnt),
                images=raw["images"],
                qpos=raw["qpos"],
                endpose=raw["endpose"],
            )
            for action in policy.act(observation):
                task_env.take_action(action, action_type=policy.action_type)
                if robotwin.episode_over(task_env):
                    break
    except PolicyError:
        raise
    except Exception as exc:
        if robotwin.gpu_exhausted(exc) or robotwin.gpu_lost(exc):
            raise robotwin.RoboTwinError(
                f"the GPU failed during the rollout: {type(exc).__name__}: {exc}"
            ) from exc
        return bool(task_env.eval_success), f"rollout error: {type(exc).__name__}: {exc}"
    return bool(task_env.eval_success), ""


def _final_frame(video: EpisodeVideo, task_env) -> None:
    from . import robotwin

    video.observe(robotwin.observation(task_env)["images"])
    video.finish()


def _film(video: EpisodeVideo | None, write: Callable[[], None]) -> str:
    """Run a video step; a failure becomes a note in the record's detail, never an outcome."""
    if video is None:
        return ""
    try:
        write()
    except Exception as exc:
        return f" [video: {type(exc).__name__}: {exc}]"
    return ""


def _rejections(generated: Generated) -> dict[str, int]:
    counts = Counter(a.rejection.value for a in generated.attempts if a.rejection is not None)
    return dict(sorted(counts.items()))


def _rejection_details(generated: Generated) -> dict[str, str]:
    """The first detail recorded for each rejection reason, so a record says why, not just how often."""
    details: dict[str, str] = {}
    for attempt in generated.attempts:
        if attempt.rejection is not None and attempt.detail:
            details.setdefault(attempt.rejection.value, attempt.detail[:200])
    return dict(sorted(details.items()))
