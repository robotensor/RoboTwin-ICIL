"""On-demand expert demonstrations: RoboTwin's own expert, run once per episode, kept if it won.

A generated episode that the expert cannot solve says nothing about the policy. Unstable
placements, planning failures, an unsuccessful expert and exceptions out of `play_once()` are
rejections of the *scene*: recorded with a reason, never counted against a model, and replaced by
the next seed in the episode's stream.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .demo import Demonstration, DemonstrationError
from .scene import SceneFingerprint

# RoboTwin feeds the seed to `np.random.seed`, which takes [0, 2**32).
_SEED_BOUND = 2**31 - 1

# How a run draws scene seeds. `independent`: each episode its own stream from (global seed,
# episode). `robotwin`: RoboTwin's evaluation stream, one per task, counting up from
# `100000 * (1 + global seed)`; an episode takes the seeds after those the task's earlier episodes
# tried, so the scenes scored are the scenes RoboTwin's own evaluation scores.
INDEPENDENT_SEED_STREAM = "independent"
ROBOTWIN_SEED_STREAM = "robotwin"
SEED_STREAMS = (INDEPENDENT_SEED_STREAM, ROBOTWIN_SEED_STREAM)


class Rejection(str, enum.Enum):
    UNSTABLE = "unstable"
    PLAN_FAILED = "plan_failed"
    EXPERT_FAILED = "expert_failed"
    EXPERT_ERROR = "expert_error"
    NO_DEMONSTRATION = "no_demonstration"


@dataclass(frozen=True)
class Attempt:
    seed: int
    rejection: Rejection | None
    detail: str = ""


@dataclass(frozen=True)
class Generated:
    """The outcome of generating one episode's demonstration.

    `demonstration` and `initial` are None when every seed in the budget was rejected; the attempts
    say why, and the episode is recorded as rejected rather than scored.
    """

    seed: int | None
    demonstration: Demonstration | None
    initial: SceneFingerprint | None
    attempts: tuple[Attempt, ...]

    @property
    def ok(self) -> bool:
        return self.demonstration is not None


def scene_seeds(global_seed: int, episode: int, count: int) -> list[int]:
    """The seeds an episode may try, in order: a pure function of (global seed, episode index).

    Episodes draw independent streams, so the scene of episode 7 does not depend on how many
    seeds episodes 0-6 rejected — a run can be resumed or sharded without changing any scene.
    """
    rng = np.random.default_rng([global_seed, episode])
    return [int(seed) for seed in rng.integers(0, _SEED_BOUND, size=count)]


def robotwin_seeds(global_seed: int, tried: int, count: int) -> list[int]:
    """The next `count` seeds of RoboTwin's evaluation stream for one task, after `tried` seeds
    that task's earlier episodes already tried (`scripts/eval_policy_xpolicylab.py`: `st_seed`)."""
    start = 100000 * (1 + global_seed) + tried
    return list(range(start, start + count))


def attempt(
    task_env, seed: int, args: dict, save_freq: int, episode: int, images: bool = True
) -> tuple[Attempt, Demonstration | None, SceneFingerprint | None]:
    """Build the scene for one seed, run the expert once, and keep what it did if it succeeded.

    `images=False` records the demonstration without rendering a camera (`robotwin.capture`): the
    same joints, frames and outcome, fit for measuring the expert but not for a policy's context.
    """
    from . import robotwin

    try:
        task_env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **args)
    except robotwin.unstable_error() as exc:
        robotwin.close(task_env)
        return Attempt(seed, Rejection.UNSTABLE, str(exc)), None, None
    except Exception as exc:
        # A scene fails to build for reasons outside the scene — GPU memory, a missing asset, a
        # planner that did not construct — and RoboTwin can leave the env half-built after it, so
        # every later seed would fail the same way. That is a broken simulator, not a rejected
        # scene: stop, rather than record it as data.
        robotwin.close(task_env)
        raise robotwin.RoboTwinError(
            f"building the scene for seed {seed} failed: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        initial = robotwin.fingerprint(task_env)
        # Timed from the scene the fingerprint saw: RoboTwin's settle is already behind it, and
        # the clock is gone again before `close`.
        with (
            robotwin.clock(task_env) as ticks,
            robotwin.capture(task_env, save_freq, ticks, images=images) as frames,
        ):
            task_env.play_once()
        if not task_env.plan_success:
            return Attempt(seed, Rejection.PLAN_FAILED), None, None
        if not task_env.check_success():
            return Attempt(seed, Rejection.EXPERT_FAILED), None, None
        try:
            demonstration = robotwin.demonstration_from(
                frames, robotwin.frame_rate_hz(task_env, save_freq)
            )
        except DemonstrationError as exc:
            return Attempt(seed, Rejection.NO_DEMONSTRATION, str(exc)), None, None
        return Attempt(seed, None), demonstration, initial
    except robotwin.RoboTwinError:
        # The harness itself failed (a clock that cannot count this scene's steps): every seed
        # would fail the same way, so it stops generation rather than reject the seed.
        raise
    except Exception as exc:
        if robotwin.gpu_exhausted(exc):
            raise robotwin.RoboTwinError(
                f"the GPU ran out of memory while the expert ran seed {seed}: {exc}"
            ) from exc
        if robotwin.gpu_lost(exc):
            raise robotwin.RoboTwinError(
                f"the renderer lost the GPU while the expert ran seed {seed}: {exc}"
            ) from exc
        # Anything else the expert raises is a failed seed, as upstream's collector counts it:
        # "target_pose cannot be None" is RoboTwin finding no feasible grasp.
        return Attempt(seed, Rejection.EXPERT_ERROR, f"{type(exc).__name__}: {exc}"), None, None
    finally:
        robotwin.close(task_env)


def generate(
    task_env,
    seeds: list[int],
    args_factory: Callable[[], dict],
    save_freq: int,
    episode: int,
    attempt_fn: Callable[
        ..., tuple[Attempt, Demonstration | None, SceneFingerprint | None]
    ] = attempt,
) -> Generated:
    """Try seeds in order until the expert produces one successful demonstration.

    `args_factory` builds a fresh config per attempt, so nothing RoboTwin mutates while building one
    scene can leak into the next.
    """
    attempts: list[Attempt] = []
    for seed in seeds:
        result, demonstration, initial = attempt_fn(
            task_env, seed, args_factory(), save_freq, episode
        )
        attempts.append(result)
        if demonstration is not None:
            return Generated(seed, demonstration, initial, tuple(attempts))
    return Generated(None, None, None, tuple(attempts))
