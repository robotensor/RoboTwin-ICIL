"""The BPP adapter's config: defaults, refusals and what reaches `describe()`."""

import math

import numpy as np
import pytest
import yaml

from icil_policies.bpp import settings as bpp_settings
from robotwin_icil.policy import PolicyError

CONFIG = bpp_settings.Path(__file__).resolve().parents[1] / "configs"


def test_the_defaults_describe_the_released_checkpoint():
    settings = bpp_settings.load()
    assert settings.mode == "ee_step" and settings.action_type == "ee"
    assert settings.source_sha256.startswith("74e0f841")
    assert settings.agentview_geometry() == (320, 180, 45.0)
    assert settings.wrist_camera("left") == "left_camera"
    assert settings.wrist_camera("right") == "right_camera"


def test_the_committed_config_loads_and_names_the_far_side_profile():
    settings = bpp_settings.load(CONFIG / "bpp_liberogen_combination.yaml")
    assert settings.camera_profile == "far_side"
    assert settings.profile().name == "far_side"
    assert settings.checkpoint.endswith("liberogen_spatial_combination")
    assert "~" not in settings.checkpoint


def test_the_tool_correction_takes_the_panda_approach_onto_alohas():
    # C's third column is the frame's +x: robosuite's eef approaches along +z, aloha along +x.
    correction = bpp_settings.load().tool_correction()
    np.testing.assert_allclose(correction[:, 2], [1.0, 0.0, 0.0], atol=1e-12)
    assert np.linalg.det(correction) == pytest.approx(1.0)


def test_a_mode_names_its_action_type():
    assert bpp_settings.load(mode="qpos_ik").action_type == "qpos"
    assert bpp_settings.load(mode="ee_grouped").action_type == "ee"
    with pytest.raises(PolicyError, match="mode must be one of"):
        bpp_settings.load(mode="teleport")


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("alpha_p", 0.0, "alpha_p must be positive"),
        ("stretch", math.nan, "stretch must be positive"),
        ("camera_profile", "nope", "is not in cameras.yml"),
        ("agentview_type", "webcam", "agentview_type must be one of"),
        ("crop", 400, "does not fit"),
        ("wrist_cameras", ("one",), "must name two"),
        ("max_position_error_m", 0.001, "must exceed one full-scale commanded step"),
        ("max_rotation_error_rad", 0.01, "must exceed one full-scale commanded step"),
    ],
)
def test_a_setting_that_cannot_run_is_refused(key, value, message):
    with pytest.raises(PolicyError, match=message):
        bpp_settings.load(**{key: value})


def test_an_unknown_key_is_a_typo_not_a_silent_default(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("alpha_pp: 0.3\n")
    with pytest.raises(PolicyError, match="unknown keys"):
        bpp_settings.load(path)
    with pytest.raises(PolicyError, match="no such file"):
        bpp_settings.load(tmp_path / "missing.yaml")


def test_a_config_round_trips_through_yaml(tmp_path):
    settings = bpp_settings.load(mode="ee_grouped", alpha_p=0.3)
    path = tmp_path / "written.yaml"
    path.write_text(bpp_settings.dump(settings))
    assert yaml.safe_load(path.read_text())["mode"] == "ee_grouped"
    assert bpp_settings.load(path) == settings


def test_describe_is_json_and_carries_every_constant():
    description = bpp_settings.load().describe()
    assert description["alpha_p"] > 0 and description["action_type"] == "ee"
    assert description["tool_correction_axis_angle"][1] == pytest.approx(math.pi / 2)
    yaml.safe_dump(description)


def test_the_model_policy_refuses_the_simulators_own_process(monkeypatch):
    """`seed()` seeds torch's global RNG, which RoboTwin reseeds with the scene seed."""
    import sys
    import types

    from icil_policies.bpp.policy import BPPPolicy

    monkeypatch.setitem(sys.modules, "sapien", types.ModuleType("sapien"))
    with pytest.raises(PolicyError, match="remote"):
        BPPPolicy(config=str(CONFIG / "bpp_liberogen_combination.yaml"))
