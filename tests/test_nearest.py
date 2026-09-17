import numpy as np
import pytest

from robotwin_icil.dataset import CAMERAS
from robotwin_icil.demo import Demonstration, Frame
from robotwin_icil.policies.nearest import NearestPolicy, features
from robotwin_icil.policy import Observation, PolicyError, make_policy


def images(value):
    return {camera: np.full((24, 32, 3), value, np.uint8) for camera in CAMERAS.values()}


def demonstration():
    return Demonstration(
        tuple(Frame(i, images(i * 50), np.full(14, i), {}) for i in range(4)), frequency=15
    )


def test_reference_uses_observations_and_emits_next_absolute_target():
    policy = make_policy("robotwin_icil.policies.nearest:NearestPolicy", window="4")
    policy.reset()
    policy.set_demonstration(demonstration())
    observation = Observation(0, images(50), np.ones(14))
    assert policy.act(observation, {"qpos": 14})[0, 0] == 2
    assert policy.describe()["training"].startswith("none")
    assert policy.act(Observation(1, images(100), np.full(14, 2)), {"qpos": 14})[0, 0] == 3
    policy.reset()
    policy.set_demonstration(demonstration())
    assert policy.act(Observation(0, images(0), np.zeros(14)), {"qpos": 14})[0, 0] == 1


def test_reference_requires_all_public_cameras_and_valid_parameters():
    with pytest.raises(PolicyError, match="requires camera"):
        features({}, np.zeros(14), 0.25)
    with pytest.raises(PolicyError):
        NearestPolicy(window=0)
    with pytest.raises(PolicyError):
        NearestPolicy(rgb_weight="nan")
