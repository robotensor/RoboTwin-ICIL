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
    format_policy_arg,
    make_policy,
    parse_policy_arg,
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
    bare = Observation(step=0, images={}, qpos=np.zeros(BIMANUAL_QPOS_DIM))
    assert bare.instruction == NEUTRAL_INSTRUCTION
    # Measured finger positions are proprioception a real robot has; they default for callers
    # that read none.
    assert bare.gripper_joints is None


def test_every_hook_has_a_default():
    """A policy overrides only the hooks it needs, so the served path can call all of them on a
    built-in that never heard of a server."""
    policy = ReplayPolicy()
    policy.seed(7)
    assert policy.episode_info() == {}
    assert policy.environment() == {}
    policy.close()


@pytest.mark.parametrize(
    "item, expected",
    [
        ("steps=3", ("steps", 3)),
        ("temperature=0.5", ("temperature", 0.5)),
        ("lr=1.0e-4", ("lr", 0.0001)),
        ("lr=1e-4", ("lr", "1e-4")),  # YAML 1.1: no dot, no float
        ("lr=1.0e4", ("lr", "1.0e4")),  # nor without a signed exponent
        ("deterministic=true", ("deterministic", True)),
        ("deterministic=False", ("deterministic", False)),
        ("checkpoint=null", ("checkpoint", None)),
        ("checkpoint=", ("checkpoint", None)),
        ("config=configs/bpp.yml", ("config", "configs/bpp.yml")),
        ("arm=left", ("arm", "left")),
        ("revision='0123'", ("revision", "0123")),  # quoted: the string, not octal 83
        ("date=2024-01-01", ("date", "2024-01-01")),
        ("tags=[a, b]", ("tags", "[a, b]")),
        ("tag=#1", ("tag", "#1")),
        ("expr=a=b", ("expr", "a=b")),
    ],
)
def test_a_policy_arg_is_read_as_yaml_and_anything_else_stays_a_string(item, expected):
    assert parse_policy_arg(item) == expected


@pytest.mark.parametrize(
    "item, message",
    [
        ("steps", "is not KEY=VALUE"),
        ("2steps=3", "is not a Python identifier"),
        ("steps=.nan", "must be finite"),
        ("steps=.inf", "must be finite"),
    ],
)
def test_what_is_not_a_policy_arg_says_why(item, message):
    with pytest.raises(ValueError, match=message):
        parse_policy_arg(item)


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -7,
        10**30,
        0.5,
        -1.25,
        1.0,
        1e-10,
        1e16,
        5e-324,
        "",
        "0123",
        "true",
        "null",
        "1e-4",
        "a=b",
        "[a, b]",
        "configs/bpp.yml",
        'say "hi"',
        "tab\there",
        "naïve ☃",
    ],
)
def test_a_formatted_policy_arg_reads_back_as_itself(value):
    """A spawned policy server gets its client's keyword arguments on a command line; each has to
    arrive as the value that was given."""
    key, parsed = parse_policy_arg(format_policy_arg("key", value))
    assert key == "key" and parsed == value and type(parsed) is type(value)


def test_a_path_or_a_numpy_scalar_is_formatted_as_the_value_it_holds(tmp_path):
    assert parse_policy_arg(format_policy_arg("config", tmp_path / "bpp.yml")) == (
        "config",
        str(tmp_path / "bpp.yml"),
    )
    assert parse_policy_arg(format_policy_arg("scale", np.float32(0.5))) == ("scale", 0.5)
    assert parse_policy_arg(format_policy_arg("steps", np.int64(3))) == ("steps", 3)


@pytest.mark.parametrize(
    "value, message",
    [
        ([1, 2], "list values cannot be passed"),
        ({"a": 1}, "dict values cannot be passed"),
        (float("nan"), "it reads back as 'nan'"),
        (float("inf"), "it reads back as 'inf'"),
        (object(), "object values cannot be passed"),
    ],
)
def test_what_a_policy_arg_cannot_express_is_refused(value, message):
    with pytest.raises(PolicyError, match=message):
        format_policy_arg("key", value)
