"""One benchmark episode: expert demonstration, exact reset, frozen rollout, RoboTwin's verdict.

    generate a successful expert demonstration   (scene seed S, fingerprint F)
    close the env                                (never continue from the expert's final state)
    setup_demo again with seed S                 (the Same Scene)
    fingerprint again, require == F              (or the episode is invalid, not failed)
    policy.reset(); policy.set_demonstration(D)
    roll out through take_action until eval_success or the task's step limit

This is `vendor/RoboTwin/scripts/eval_policy_xpolicylab.py:run_one_batch_episode` with the expert
trajectory kept and handed to the policy instead of thrown away.

The two halves are separate functions because they run in separate processes in a competition:
`generate.attempt` builds the demonstration (and `unit.materialize` writes it to disk) and
`evaluate` scores one from its fingerprint, whether that came from memory or from a file.
`run_episode` is the two back to back.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .demo import Demonstration
from .generate import Generated, generate, scene_seeds
from .policy import ICILPolicy, Observation, PolicyError
from .records import SAME_SCENE, EpisodeRecord, Status
from .scene import SceneFingerprint, compare, max_error
from .tasks import Task
from .video import EpisodeVideo


@dataclass(frozen=True)
class EpisodeSpec:
    episode: int
    task: Task
    global_seed: int
    max_expert_attempts: int


@dataclass(frozen=True)
class Evaluation:
    """What evaluating one demonstration in its rebuilt scene found.

    `valid` means the scene was rebuilt as the demonstration's and the policy acted in it;
    `success` is then RoboTwin's verdict. Otherwise nothing was scored, `detail` says why, and
    `success` is None.
    """

    valid: bool
    success: bool | None
    steps: int
    step_limit: int | None
    scene_max_error: float
    detail: str


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
    embodiment = str(config.resolve(spec.task.name)["embodiment_name"])
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
            embodiment=embodiment,
            checkpoint=describe.get("checkpoint"),
            duration_s=round(time.monotonic() - started, 3),
            **fields,
        )

    if not generated.ok:
        return record(
            Status.REJECTED,
            success=None,
            steps=0,
            step_limit=None,
            scene_max_error=0.0,
            demonstration_frames=0,
            detail=f"no successful expert demonstration in {len(generated.attempts)} seeds",
        )

    demonstration = generated.demonstration
    video_note = _film(video, lambda: video.demonstration(demonstration))
    evaluation = evaluate(
        task_env,
        spec.task.name,
        generated.seed,
        config,
        demonstration,
        generated.initial,
        policy,
        video=video,
        episode=spec.episode,
        note=video_note,
    )
    return record(
        Status.SCORED if evaluation.valid else Status.INVALID,
        success=evaluation.success,
        steps=evaluation.steps,
        step_limit=evaluation.step_limit,
        scene_max_error=evaluation.scene_max_error,
        demonstration_frames=len(demonstration),
        detail=evaluation.detail,
    )


def evaluate(
    task_env,
    task_name: str,
    seed: int,
    config,
    demonstration: Demonstration,
    initial: SceneFingerprint,
    policy: ICILPolicy,
    video: EpisodeVideo | None = None,
    episode: int = 0,
    score_policy_faults: bool = False,
    note: str = "",
) -> Evaluation:
    """Rebuild the demonstration's scene, check it is the same one, and roll the policy out in it.

    `initial` is the fingerprint taken when the demonstration was recorded; the rebuilt scene must
    match it before the policy is even handed the demonstration. The env is closed on the way out
    whatever happened. `note` — in a benchmark run, what writing the demonstration clip said — is
    appended to the detail of a scene that was built, ahead of the evaluation clip's own note.

    A `PolicyError`, and anything `reset` or `set_demonstration` raises, propagates, as in
    `run_episode`: in a run of the benchmark an adapter at fault is a bug to fix. With
    `score_policy_faults` they are instead the policy's result — a failure, with the reason in
    `detail` — as a competition needs: an evaluation that is not scored is thrown out, and a
    policy must not be able to throw out the episodes it is losing.
    """
    from . import robotwin

    try:
        task_env.setup_demo(
            now_ep_num=episode, seed=seed, is_test=True, **config.resolve(task_name)
        )
    except Exception as exc:
        robotwin.close(task_env)
        return _not_scored(f"evaluation scene failed to build: {type(exc).__name__}: {exc}")

    try:
        mismatches = compare(initial, robotwin.fingerprint(task_env))
        if mismatches:
            # The evaluation's first frame is the evidence for a reset bug; the episode is
            # already invalid, so observing it cannot change anything that is scored.
            final_note = _film(video, lambda: _final_frame(video, task_env))
            return _not_scored(
                "scene drift: " + "; ".join(str(m) for m in mismatches[:5]) + note + final_note,
                scene_max_error=max_error(mismatches),
            )
        try:
            policy.reset()
            policy.set_demonstration(demonstration)
        except Exception as exc:
            if not score_policy_faults:
                raise
            success, detail = False, f"policy failed before acting: {type(exc).__name__}: {exc}"
        else:
            try:
                success, detail = rollout(
                    task_env, policy, observe=video.observe if video is not None else None
                )
            except PolicyError as exc:
                if not score_policy_faults:
                    raise
                success, detail = bool(task_env.eval_success), f"policy broke the protocol: {exc}"
        final_note = _film(video, lambda: _final_frame(video, task_env))
        return Evaluation(
            valid=True,
            success=success,
            steps=int(task_env.take_action_cnt),
            step_limit=task_env.step_lim,
            scene_max_error=0.0,
            detail=detail + note + final_note,
        )
    finally:
        robotwin.close(task_env)


def _not_scored(detail: str, scene_max_error: float = 0.0) -> Evaluation:
    return Evaluation(
        valid=False,
        success=None,
        steps=0,
        step_limit=None,
        scene_max_error=scene_max_error,
        detail=detail,
    )


def rollout(
    task_env,
    policy: ICILPolicy,
    observe: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> tuple[bool, str]:
    """Drive the policy until RoboTwin reports success or the task's step limit is reached.

    Success is RoboTwin's own: `take_action` runs `check_success()` after every step and latches
    `eval_success`. An exception from the simulator mid-rollout ends the episode as a failure, as
    upstream's evaluator does — except the GPU running out of memory or the renderer losing it,
    which say nothing about the policy and leave a process that cannot simulate: those raise
    `RoboTwinError`, as they do while the expert runs. The policy's actions are checked against
    the widths this robot takes, read off its arms here rather than assumed — outside that
    catch-all, because a seam that cannot read them is a harness bug, not a stream of failed
    rollouts.
    """
    from . import robotwin

    action_dims = robotwin.action_dims(task_env)
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
            for action in policy.act(observation, action_dims):
                task_env.take_action(action, action_type=policy.action_type)
                if robotwin.episode_over(task_env):
                    break
    except PolicyError:
        raise
    except Exception as exc:
        if robotwin.gpu_exhausted(exc):
            raise robotwin.RoboTwinError(
                f"the GPU ran out of memory during the rollout: {exc}"
            ) from exc
        if robotwin.gpu_lost(exc):
            raise robotwin.RoboTwinError(
                f"the renderer lost the GPU during the rollout: {exc}"
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
