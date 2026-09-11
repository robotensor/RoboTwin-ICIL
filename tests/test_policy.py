import dataclasses

import numpy as np
import pytest

from robotwin_icil.demo import (
    BIMANUAL_EE_DIM,
    BIMANUAL_QPOS_DIM,
    Demonstration,
    DemonstrationError,
    Frame,
)
from robotwin_icil.policy import (
    BUILTIN,
    NEUTRAL_INSTRUCTION,
    DummyPolicy,
    ICILPolicy,
    Observation,
    PolicyError,
    ReplayEEPolicy,
    ReplayPolicy,
    check_description,
    format_policy_arg,
    json_mapping,
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


def ee_demonstration(n: int = 4, offset: float = 0.0) -> Demonstration:
    """A demonstration whose flanges rise by 1 cm a frame, and whose left gripper closes."""
    frames = tuple(
        Frame(
            index=i,
            images={"head_camera": np.zeros((4, 4, 3), dtype=np.uint8)},
            qpos=np.full(BIMANUAL_QPOS_DIM, offset + i),
            endpose={
                "left_endpose": [0.0, 0.0, offset + 0.01 * i, 1.0, 0.0, 0.0, 0.0],
                "left_gripper": 1.0 if i < n - 1 else 0.0,
                "right_endpose": [0.1, 0.0, offset + 0.01 * i, 1.0, 0.0, 0.0, 0.0],
                "right_gripper": 1.0,
            },
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


def test_replay_ee_plays_the_ee_actions_in_order_then_holds_the_last():
    demo = ee_demonstration(n=3)
    policy = ReplayEEPolicy()
    policy.reset()
    policy.set_demonstration(demo)
    played = np.concatenate([policy.act(observation(step)) for step in range(4)])
    assert played.shape == (4, BIMANUAL_EE_DIM)
    expected = demo.ee_actions()
    np.testing.assert_array_equal(played, [expected[0], expected[1], expected[1], expected[1]])
    assert played[-1, 7] == 0.0  # the last frame's commanded gripper: closed


def test_replay_ee_starts_each_episode_from_its_own_demonstration():
    policy = ReplayEEPolicy()
    policy.reset()
    policy.set_demonstration(ee_demonstration(n=3))
    policy.act(observation())
    policy.reset()
    with pytest.raises(PolicyError):
        policy.act(observation())  # nothing left from the last episode to act on
    policy.set_demonstration(ee_demonstration(n=3, offset=10.0))
    assert policy.act(observation())[0, 2] == pytest.approx(10.01)


def test_replay_ee_needs_the_demonstrations_endposes():
    policy = ReplayEEPolicy()
    policy.reset()
    with pytest.raises(DemonstrationError, match="endpose has no"):
        policy.set_demonstration(demonstration())
    # The refused demonstration is not kept: acting is still refused, and a usable one is taken.
    with pytest.raises(PolicyError, match="before set_demonstration"):
        policy.act(observation())
    policy.set_demonstration(ee_demonstration())
    assert policy.act(observation()).shape == (1, BIMANUAL_EE_DIM)


def test_replay_ee_is_a_builtin_ee_policy():
    policy = make_policy("replay_ee")
    assert isinstance(policy, ReplayEEPolicy)
    assert policy.action_type == "ee" and policy.describe()["action_type"] == "ee"


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


class _QposWidthForEE(ICILPolicy):
    name = "qpos_width_for_ee"
    action_type = "ee"

    def _act(self, observation):
        return np.zeros(BIMANUAL_QPOS_DIM)


@pytest.mark.parametrize("cls", [_WrongWidth, _NotFinite, _QposWidthForEE])
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


def test_json_mapping_returns_what_json_reads_back():
    assert json_mapping({"a": (1, 2.5), "b": {"c": None, "d": True}}, "info") == {
        "a": [1, 2.5],
        "b": {"c": None, "d": True},
    }
    assert json_mapping({}, "info") == {}


@pytest.mark.parametrize(
    "data, message",
    [
        ([("a", 1)], "info must be a mapping, not list"),
        ({1: "a"}, "info: key 1 is not a string"),
        ({"a": object()}, "info: 'a' is not JSON-serialisable"),
        ({"a": np.float32(0.5)}, "info: 'a' is not JSON-serialisable"),
        ({"a": [float("inf")]}, "info: 'a' is not JSON-serialisable"),
    ],
)
def test_json_mapping_refuses_what_a_json_file_cannot_hold(data, message):
    with pytest.raises(PolicyError, match=message):
        json_mapping(data, "info")


def test_every_hook_has_a_default():
    policy = ReplayPolicy()
    policy.seed(7)
    assert policy.episode_info() == {}
    assert policy.environment() == {}
    policy.close()


@pytest.mark.parametrize("name", sorted(BUILTIN))
def test_builtins_are_zero_argument_constructible_and_described_by_the_convention(name):
    description = BUILTIN[name]().describe()
    check_description(description)
    assert description["policy"] == name and description["action_type"] in ("qpos", "ee")


def test_a_description_may_carry_every_convention_key_and_keys_of_its_own():
    check_description(
        {
            "policy": "bpp",
            "action_type": "ee",
            "adapter": "robotwin_icil_policies.bpp",
            "adapter_version": "0.1.0",
            "checkpoint": "/ckpt/bpp.pt",
            "checkpoint_sha256": "ab" * 32,
            "training_tasks": ["libero_10/kitchen_scene3"],
            "camera_profile_required": "far_side",
            "parameter_checksum": "c0ffee",
            "prompt_chunks": 4,
        }
    )
    check_description(
        {"policy": "x", "checkpoint": None, "checkpoint_sha256": None, "training_tasks": "unknown"}
    )


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"adapter": None}, "'adapter' must be a string"),
        ({"adapter_version": 1}, "'adapter_version' must be a string"),
        ({"checkpoint": 3}, "'checkpoint' must be a string or None"),
        ({"checkpoint_sha256": "ab" * 31}, "'checkpoint_sha256' must be 64 hex digits"),
        ({"checkpoint_sha256": "zz" * 32}, "'checkpoint_sha256' must be 64 hex digits"),
        ({"training_tasks": "click_bell"}, "'training_tasks' must be a list of task names"),
        ({"training_tasks": ["click_bell", ""]}, "'training_tasks' must be a list"),
        ({"camera_profile_required": ["far_side"]}, "'camera_profile_required' must be a"),
        ({"camera_profile_required": "farside"}, "must be a camera profile's name \\(stock, "),
        ({"parameter_checksum": 12}, "'parameter_checksum' must be a string or None"),
        ({"weights": object()}, "'weights' is not JSON-serialisable"),
    ],
)
def test_a_description_off_the_convention_is_refused(fields, message):
    with pytest.raises(PolicyError, match=message):
        check_description({"policy": "adapter", **fields})


def test_a_description_names_its_policy():
    with pytest.raises(PolicyError, match="must name the policy"):
        check_description({"model": "bpp"})
    with pytest.raises(PolicyError, match="must be a mapping"):
        check_description(None)


class _Configured(ReplayPolicy):
    name = "configured"

    def __init__(self, config=None, temperature=1.0):
        super().__init__()
        self.config, self.temperature = config, temperature


def test_make_policy_passes_keyword_arguments_to_the_class():
    policy = make_policy("test_policy:_Configured", config="bpp.yml", temperature=0.5)
    assert (policy.config, policy.temperature) == ("bpp.yml", 0.5)
    assert make_policy("test_policy:_Configured").temperature == 1.0  # and needs none


@pytest.mark.parametrize(
    "spec, kwargs",
    [("replay", {"checkpoint": "x.pt"}), ("test_policy:_Configured", {"temprature": 0.5})],
)
def test_make_policy_refuses_arguments_the_class_does_not_take(spec, kwargs):
    with pytest.raises(PolicyError, match=f"policy {spec!r} does not take these arguments"):
        make_policy(spec, **kwargs)


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
