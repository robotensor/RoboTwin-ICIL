"""`BPPConversionReplay`: the conversion oracle drives the chain end to end, without a model."""

import json

import numpy as np
import pytest

from icil_policies.bpp import settings as bpp_settings
from icil_policies.bpp.oracle import BPPConversionReplay
from icil_policies.bpp.synthetic import demonstration, observation
from robotwin_icil.policy import PolicyError, check_description

CONFIG = bpp_settings.Path(__file__).resolve().parents[1] / "configs"


def _episode(policy, demo, steps=6):
    policy.seed(7)
    policy.reset()
    policy.set_demonstration(demo)
    return [policy.act(observation(demo, min(step, len(demo) - 1), step)) for step in range(steps)]


def test_the_oracle_is_a_policy_the_runner_would_accept():
    policy = BPPConversionReplay()
    check_description(policy.describe())
    assert policy.describe()["camera_profile_required"] == "far_side"
    assert policy.describe()["training_tasks"] == []
    assert policy.action_type == "ee"


def test_it_needs_a_demonstration_before_it_acts():
    policy = BPPConversionReplay()
    policy.reset()
    demo = demonstration(steps=4)
    with pytest.raises(PolicyError, match="act\\(\\) before set_demonstration"):
        policy.act(observation(demo, 0, 0))


def test_it_replays_the_prompt_through_the_execution_chain():
    demo = demonstration(steps=21, step_m=0.002)
    policy = BPPConversionReplay()
    actions = np.concatenate(_episode(policy, demo, steps=5))
    assert actions.shape == (5, 16)
    # Each call moves the tool one prompt step along the world's +y, the demonstration's own
    # direction; the idle right arm holds its first-observation pose.
    moves = np.diff(actions[:, 1])
    np.testing.assert_allclose(moves, 0.002, atol=1e-6)
    idle = np.asarray(demo.frames[0].endpose["right_endpose"])
    np.testing.assert_allclose(actions[:, 8:15], np.tile(idle, (5, 1)), atol=1e-12)


def test_past_the_end_it_holds_where_the_expert_succeeded():
    demo = demonstration(steps=4)
    policy = BPPConversionReplay()
    actions = np.concatenate(_episode(policy, demo, steps=8))
    np.testing.assert_allclose(np.diff(actions[4:, :3], axis=0), 0.0, atol=1e-12)


def test_it_reports_what_the_episode_cost():
    demo = demonstration(steps=45, grippers=np.r_[np.ones(20), np.zeros(25)])
    policy = BPPConversionReplay()
    _episode(policy, demo, steps=4)
    info = policy.episode_info()
    assert info["active_arm"] == "left" and info["demonstration_arms"] == ["left"]
    assert info["prompt_chunks"] == 3 and info["prompt_steps"] == 44
    assert info["clipped_action_fraction"] == 0.0
    assert info["calls"] == 4 and info["stall_events"] == 0
    assert info["reanchors"] == {"tracking": 0, "plan_failed": 0}
    json.dumps(info)


def test_an_episode_starts_from_nothing():
    policy = BPPConversionReplay()
    assert policy.episode_info() == {}
    _episode(policy, demonstration(steps=8), steps=3)
    policy.reset()
    assert policy.episode_info() == {}


def test_the_grouped_mode_spends_fewer_calls_on_the_same_motion():
    demo = demonstration(steps=25, step_m=0.002)
    stepwise = np.concatenate(_episode(BPPConversionReplay(), demo, steps=4))
    grouped = np.concatenate(_episode(BPPConversionReplay(mode="ee_grouped"), demo, steps=1))
    # One grouped call covers the first four steps of the step-wise mode.
    np.testing.assert_allclose(grouped[0, :3], stepwise[3, :3], atol=1e-9)


def test_the_oracle_reads_the_committed_config():
    policy = BPPConversionReplay(config=str(CONFIG / "bpp_liberogen_combination.yaml"))
    assert policy.describe()["settings"]["mode"] == "ee_step"
    assert policy.config.endswith("bpp_liberogen_combination.yaml")


def test_it_reports_the_proprio_out_of_range_when_a_normalizer_is_beside_the_checkpoint(tmp_path):
    # `icil-bpp slim` writes normalizer.json so this numpy oracle can report the same fraction
    # the model reports, without torch. ee_pos is normalized to [-1, 1] over LIBERO's own range.
    (tmp_path / "normalizer.json").write_text(
        json.dumps({"ee_pos": {"scale": [1.0, 1.0, 1.0], "offset": [0.0, 0.0, 0.0]}})
    )
    policy = BPPConversionReplay(checkpoint=str(tmp_path))
    _episode(policy, demonstration(steps=8), steps=2)
    fraction = policy.episode_info()["proprio_out_of_range"]["ee_pos"]
    assert 0.0 <= fraction <= 1.0


def test_a_demonstration_without_measured_fingers_says_so():
    from robotwin_icil.demo import Demonstration, Frame

    demo = demonstration(steps=6)
    bare = Demonstration(
        frames=tuple(
            Frame(
                index=frame.index,
                images=frame.images,
                qpos=frame.qpos,
                endpose=frame.endpose,
                time_s=frame.time_s,
            )
            for frame in demo.frames
        ),
        frequency=demo.frequency,
    )
    policy = BPPConversionReplay(checkpoint="")
    policy.reset()
    with pytest.raises(PolicyError, match="measured gripper joints"):
        policy.set_demonstration(bare)
