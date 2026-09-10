import dataclasses

import numpy as np
import pytest

from robotwin_icil.demo import BIMANUAL_QPOS_DIM, Demonstration, Frame
from robotwin_icil.policy import (
    NEUTRAL_INSTRUCTION,
    DummyPolicy,
    ICILPolicy,
    Observation,
    PolicyError,
    ReplayPolicy,
    make_policy,
)


def demonstration(n: int = 4, offset: float = 0.0) -> Demonstration:
    frames = tuple(
        Frame(
            index=i,
            images={"head_camera": np.zeros((4, 4, 3), dtype=np.uint8)},
            qpos=np.full(BIMANUAL_QPOS_DIM, offset + i),
            endpose={},
        )
        for i in range(n)
    )
    return Demonstration(frames=frames, frequency=15)


def observation(step: int = 0, value: float = 0.5) -> Observation:
    return Observation(step=step, images={}, qpos=np.full(BIMANUAL_QPOS_DIM, value))


def test_act_before_a_demonstration_is_an_error():
    policy = ReplayPolicy()
    policy.reset()
    with pytest.raises(PolicyError):
        policy.act(observation())


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
    played = [policy.act(observation(step))[0, 0] for step in range(4)]
    assert played == [1.0, 2.0, 2.0, 2.0]


def test_reset_leaves_nothing_from_the_previous_episode():
    policy = ReplayPolicy()
    policy.reset()
    policy.set_demonstration(demonstration(n=3))
    policy.act(observation())
    policy.reset()
    policy.set_demonstration(demonstration(n=3, offset=10.0))
    assert policy.act(observation())[0, 0] == 11.0


def test_dummy_holds_the_current_state():
    policy = DummyPolicy()
    policy.reset()
    policy.set_demonstration(demonstration())
    np.testing.assert_array_equal(
        policy.act(observation(value=0.25)), np.full((1, BIMANUAL_QPOS_DIM), 0.25)
    )


class _WrongWidth(ICILPolicy):
    name = "wrong_width"

    def _act(self, observation):
        return np.zeros(7)


class _NotFinite(ICILPolicy):
    name = "not_finite"

    def _act(self, observation):
        return np.full(BIMANUAL_QPOS_DIM, np.nan)


@pytest.mark.parametrize("cls", [_WrongWidth, _NotFinite])
def test_malformed_actions_are_rejected(cls):
    policy = cls()
    policy.reset()
    policy.set_demonstration(demonstration())
    with pytest.raises(PolicyError):
        policy.act(observation())


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
        Observation(step=0, images={}, qpos=np.zeros(BIMANUAL_QPOS_DIM)).instruction
        == NEUTRAL_INSTRUCTION
    )
