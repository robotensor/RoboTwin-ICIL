import dataclasses

import numpy as np
import pytest

from robotwin_icil.demo import Demonstration, Frame
from robotwin_icil.policy import (
    NEUTRAL_INSTRUCTION,
    DummyPolicy,
    ICILPolicy,
    Observation,
    PolicyError,
    ReplayPolicy,
    make_policy,
)

# aloha-agilex's widths; a dual Franka is {"qpos": 16, "ee": 16}. The harness reads them off the
# live robot; here they are what the fixtures were built for.
QPOS_DIM = 14
DIMS = {"qpos": QPOS_DIM, "ee": 16}


def demonstration(n: int = 4, offset: float = 0.0, qpos_dim: int = QPOS_DIM) -> Demonstration:
    frames = tuple(
        Frame(
            index=i,
            images={"head_camera": np.zeros((4, 4, 3), dtype=np.uint8)},
            qpos=np.full(qpos_dim, offset + i),
            endpose={},
        )
        for i in range(n)
    )
    return Demonstration(frames=frames, frequency=15)


def observation(step: int = 0, value: float = 0.5, qpos_dim: int = QPOS_DIM) -> Observation:
    return Observation(step=step, images={}, qpos=np.full(qpos_dim, value))


def test_act_before_a_demonstration_is_an_error():
    policy = ReplayPolicy()
    policy.reset()
    with pytest.raises(PolicyError):
        policy.act(observation(), DIMS)


def test_demonstration_before_reset_is_an_error():
    with pytest.raises(PolicyError):
        ReplayPolicy().set_demonstration(demonstration())


def test_an_episode_gets_exactly_one_demonstration():
    policy = ReplayPolicy()
    policy.reset()
    policy.set_demonstration(demonstration())
    with pytest.raises(PolicyError):
        policy.set_demonstration(demonstration())


def test_replay_plays_actions_in_order_then_holds_the_last():
    policy = ReplayPolicy()
    policy.reset()
    policy.set_demonstration(demonstration(n=3))
    played = [policy.act(observation(step), DIMS)[0, 0] for step in range(4)]
    assert played == [1.0, 2.0, 2.0, 2.0]


def test_reset_leaves_nothing_from_the_previous_episode():
    policy = ReplayPolicy()
    policy.reset()
    policy.set_demonstration(demonstration(n=3))
    policy.act(observation(), DIMS)
    policy.reset()
    policy.set_demonstration(demonstration(n=3, offset=10.0))
    assert policy.act(observation(), DIMS)[0, 0] == 11.0


def test_dummy_holds_the_current_state():
    policy = DummyPolicy()
    policy.reset()
    policy.set_demonstration(demonstration())
    np.testing.assert_array_equal(
        policy.act(observation(value=0.25), DIMS), np.full((1, QPOS_DIM), 0.25)
    )


@pytest.mark.parametrize("qpos_dim", [14, 16])
def test_actions_are_checked_against_the_robots_width(qpos_dim):
    # The width is the robot's, not a constant: a dual Franka takes 16-wide qpos actions, and an
    # action of the other robot's width is refused rather than mis-split by `take_action`.
    dims = {"qpos": qpos_dim, "ee": 16}
    policy = ReplayPolicy()
    policy.reset()
    policy.set_demonstration(demonstration(qpos_dim=qpos_dim))
    assert policy.act(observation(qpos_dim=qpos_dim), dims).shape == (1, qpos_dim)
    with pytest.raises(PolicyError, match=f"expected \\(k, {30 - qpos_dim}\\)"):
        policy.act(observation(qpos_dim=qpos_dim), {"qpos": 30 - qpos_dim, "ee": 16})


class _WrongWidth(ICILPolicy):
    name = "wrong_width"

    def _act(self, observation):
        return np.zeros(7)


class _NotFinite(ICILPolicy):
    name = "not_finite"

    def _act(self, observation):
        return np.full(QPOS_DIM, np.nan)


@pytest.mark.parametrize("cls", [_WrongWidth, _NotFinite])
def test_malformed_actions_are_rejected(cls):
    policy = cls()
    policy.reset()
    policy.set_demonstration(demonstration())
    with pytest.raises(PolicyError):
        policy.act(observation(), DIMS)


class _Torque(ICILPolicy):
    name = "torque"
    action_type = "torque"

    def _act(self, observation):
        return np.zeros(QPOS_DIM)


def test_an_action_type_the_robot_does_not_take_is_an_error():
    policy = _Torque()
    policy.reset()
    policy.set_demonstration(demonstration())
    with pytest.raises(PolicyError, match="no action width for action_type 'torque'"):
        policy.act(observation(), DIMS)


def test_make_policy_resolves_builtins_and_module_paths():
    assert isinstance(make_policy("replay"), ReplayPolicy)
    assert isinstance(make_policy("robotwin_icil.policy:DummyPolicy"), DummyPolicy)
    with pytest.raises(PolicyError):
        make_policy("no_such_policy")
    with pytest.raises(PolicyError):
        make_policy("robotwin_icil.demo:Frame")


def test_nothing_privileged_reaches_the_policy():
    # The policy sees observations and one demonstration. Neither may name the task or the seed.
    privileged = {"task", "task_name", "seed", "scene_seed", "info", "success"}
    assert not privileged & {f.name for f in dataclasses.fields(Observation)}
    assert not privileged & {f.name for f in dataclasses.fields(Demonstration)}
    assert (
        Observation(step=0, images={}, qpos=np.zeros(QPOS_DIM)).instruction == NEUTRAL_INSTRUCTION
    )
