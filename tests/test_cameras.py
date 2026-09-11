"""Camera profiles keep every seed's scene: the replace-only guard, exercised without RoboTwin."""

import copy
import math

import pytest

from robotwin_icil import camera_profiles
from robotwin_icil.camera_profiles import ProfileError, apply, parse

# Copied from RoboTwin's `assets/embodiments/aloha-agilex/config.yml`, the V1 embodiment.
STATIC_CAMERA_LIST = [
    {
        "name": "head_camera",
        "type": "D435",
        "position": [-0.032, -0.45, 1.35],
        "forward": [0, 0.6, -0.8],
        "left": [-1, 0, 0],
    },
    {
        "name": "front_camera",
        "type": "D435",
        "position": [0, -0.45, 0.85],
        "forward": [0, 1, -0.1],
        "left": [-1, 0, 0],
    },
]

# The keys of RoboTwin's `env_cfg/task_config/_camera_config.yml`.
ROBOTWIN_CAMERA_TYPES = ("L515", "Large_L515", "D435", "Large_D435")

SIDE_CAMERA = {
    "name": "side_camera",
    "type": "D435",
    "position": [0.5, 0.0, 1.0],
    "forward": [-1.0, 0.0, 0.0],
    "left": [0.0, -1.0, 0.0],
}


@pytest.fixture
def static_camera_list():
    return copy.deepcopy(STATIC_CAMERA_LIST)


def resolved(static_camera_list, **camera):
    """RoboTwin's `args` as `SceneConfig.resolve` hands them over, trimmed to what matters here."""
    return {
        "camera": {
            "head_camera_type": "D435",
            "wrist_camera_type": "D435",
            "collect_head_camera": True,
            "collect_wrist_camera": True,
            **camera,
        },
        "left_embodiment_config": {"static_camera_list": static_camera_list, "planner": "curobo"},
        "right_embodiment_config": {"static_camera_list": copy.deepcopy(static_camera_list)},
        "save_freq": 15,
    }


def replacing(old, **entry):
    return {"replace": {old: {**SIDE_CAMERA, **entry}}}


def names(args):
    return [c["name"] for c in args["left_embodiment_config"]["static_camera_list"]]


def test_the_table_has_stock_and_far_side():
    assert camera_profiles.names()[:2] == ("stock", "far_side")
    assert (
        camera_profiles.get("stock").definition() == camera_profiles.Profile("stock").definition()
    )


def test_stock_changes_nothing_and_shares_nothing(static_camera_list):
    args = resolved(static_camera_list)
    out = apply(args, "stock")
    assert out == args
    assert out["left_embodiment_config"] is not args["left_embodiment_config"]
    assert out["camera"] is not args["camera"]


def test_far_side_replaces_front_camera_in_its_slot(static_camera_list):
    args = resolved(static_camera_list)
    original = copy.deepcopy(args)
    out = apply(args, "far_side", ROBOTWIN_CAMERA_TYPES)
    assert names(out) == ["head_camera", "far_side_camera"]
    head, far_side = out["left_embodiment_config"]["static_camera_list"]
    assert head == STATIC_CAMERA_LIST[0]
    assert far_side == {
        "name": "far_side_camera",
        "type": "L515",
        "position": [0.0, 0.36, 1.20],
        "forward": [0.0, -0.778, -0.628],
        "left": [1.0, 0.0, 0.0],
    }
    assert out["left_embodiment_config"]["planner"] == "curobo"
    # RoboTwin reads the left config only; the right one and the caller's args stay as they were.
    assert out["right_embodiment_config"] == original["right_embodiment_config"]
    assert args == original


def test_far_side_looks_38_9_degrees_down_from_across_the_table():
    camera = camera_profiles.get("far_side").replace["front_camera"]
    forward = camera["forward"]
    down = math.degrees(math.asin(-forward[2] / math.hypot(*forward)))
    assert down == pytest.approx(38.9, abs=0.05)
    assert forward[1] < 0 < camera["position"][1]  # across the table, looking back at the robot
    assert camera["left"] == [1.0, 0.0, 0.0]  # image left is the robot's right


def test_robotwin_mutating_the_result_cannot_reach_the_profile(static_camera_list):
    # RoboTwin writes into camera entries while it builds them (`type`, missing `forward`).
    out = apply(resolved(static_camera_list), "far_side")
    out["left_embodiment_config"]["static_camera_list"][1]["type"] = "Large_L515"
    again = apply(resolved(copy.deepcopy(STATIC_CAMERA_LIST)), "far_side")
    assert again["left_embodiment_config"]["static_camera_list"][1]["type"] == "L515"


def test_camera_args_are_merged(static_camera_list):
    args = resolved(static_camera_list)
    profile = parse("wide_wrists", {"camera": {"wrist_camera_type": "L515"}})
    out = apply(args, profile, ROBOTWIN_CAMERA_TYPES)
    assert out["camera"]["wrist_camera_type"] == "L515"
    assert out["camera"]["head_camera_type"] == "D435"
    assert args["camera"]["wrist_camera_type"] == "D435"


def test_a_camera_type_may_change_in_place(static_camera_list):
    profile = parse(
        "large_front", replacing("front_camera", name="front_camera", type="Large_D435")
    )
    out = apply(resolved(static_camera_list), profile, ROBOTWIN_CAMERA_TYPES)
    assert names(out) == ["head_camera", "front_camera"]
    assert out["left_embodiment_config"]["static_camera_list"][1]["type"] == "Large_D435"


def test_an_unknown_profile_is_refused(static_camera_list):
    with pytest.raises(ProfileError, match="unknown camera profile 'nope'.*stock"):
        apply(resolved(static_camera_list), "nope")


def test_replacing_a_camera_that_does_not_exist_is_refused(static_camera_list):
    with pytest.raises(ProfileError, match="replaces 'top_camera', which does not exist"):
        apply(resolved(static_camera_list), parse("p", replacing("top_camera")))


def test_far_side_is_refused_on_an_embodiment_without_front_camera(static_camera_list):
    # Every embodiment but aloha-agilex ships head_camera alone.
    with pytest.raises(ProfileError, match="does not exist"):
        apply(resolved(static_camera_list[:1]), "far_side")


def test_changing_the_number_of_created_cameras_is_refused(static_camera_list):
    # With collect_head_camera off, a camera renamed to head_camera is no longer built.
    args = resolved(static_camera_list, collect_head_camera=False)
    with pytest.raises(ProfileError, match="number of static cameras RoboTwin creates from 1 to 0"):
        apply(args, parse("p", replacing("front_camera", name="head_camera")))


def test_toggling_collect_head_camera_is_refused(static_camera_list):
    for start, profile in [(True, False), (False, True)]:
        args = resolved(copy.deepcopy(static_camera_list), collect_head_camera=start)
        with pytest.raises(ProfileError, match="toggles camera.collect_head_camera"):
            apply(args, parse("p", {"camera": {"collect_head_camera": profile}}))


def test_head_camera_may_not_be_replaced_or_renamed(static_camera_list):
    args = resolved(static_camera_list)
    for entry in [{"name": "head_camera", "type": "L515"}, {"name": "top_camera"}]:
        with pytest.raises(ProfileError, match="replaces head_camera"):
            apply(args, parse("p", replacing("head_camera", **entry)))


def test_a_second_head_camera_is_refused(static_camera_list):
    with pytest.raises(ProfileError, match="names another camera head_camera"):
        apply(
            resolved(static_camera_list), parse("p", replacing("front_camera", name="head_camera"))
        )


def test_duplicate_names_are_refused(static_camera_list):
    args = resolved([*static_camera_list, copy.deepcopy(SIDE_CAMERA)])
    with pytest.raises(ProfileError, match=r"two cameras named \['side_camera'\]"):
        apply(args, parse("p", replacing("front_camera")))


def test_a_static_camera_may_not_take_a_wrist_camera_name(static_camera_list):
    # RoboTwin stores the wrist cameras under these names first; a static one would overwrite it.
    for wrist in ("left_camera", "right_camera"):
        for collect_wrist_camera in (True, False):
            args = resolved(
                copy.deepcopy(static_camera_list), collect_wrist_camera=collect_wrist_camera
            )
            with pytest.raises(ProfileError, match=f"names a static camera '{wrist}'"):
                apply(args, parse("p", replacing("front_camera", name=wrist)))


def test_a_camera_type_robotwin_lacks_is_refused(static_camera_list):
    args = resolved(static_camera_list)
    with pytest.raises(ProfileError, match="camera type 'D455'"):
        apply(args, parse("p", replacing("front_camera", type="D455")), ROBOTWIN_CAMERA_TYPES)
    with pytest.raises(ProfileError, match="camera type 'Wide'"):
        apply(args, parse("p", {"camera": {"wrist_camera_type": "Wide"}}), ROBOTWIN_CAMERA_TYPES)


def test_a_profile_must_render_the_camera_it_films(static_camera_list):
    args = resolved(static_camera_list)
    assert apply(args, parse("p", {"video_camera": "left_camera"}))
    with pytest.raises(ProfileError, match="films 'front_camera', which it does not render"):
        apply(args, parse("p", {**replacing("front_camera"), "video_camera": "front_camera"}))


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"add": {}}, "unknown keys"),
        ({"replace": ["front_camera"]}, "'replace' must map"),
        ({"replace": {"front_camera": {"name": "x"}}}, "missing"),
        (replacing("front_camera", fov=45), "unknown keys"),
        (replacing("front_camera", position=[0, 0]), "three finite numbers"),
        (replacing("front_camera", position=[0, 0, float("nan")]), "three finite numbers"),
        (replacing("front_camera", forward=[0, 0, 0]), "must not be zero"),
        (replacing("front_camera", left=[-1.0, 0.1, 0.0]), "perpendicular"),
        (replacing("front_camera", name=""), "non-empty string"),
        ({"camera": ["wrist_camera_type"]}, "'camera' must be a mapping"),
        ({"video_camera": 3}, "'video_camera' must be a camera name"),
    ],
)
def test_malformed_profiles_are_refused(raw, match):
    with pytest.raises(ProfileError, match=match):
        parse("p", raw)


def test_stock_must_change_nothing(tmp_path):
    path = tmp_path / "cameras.yml"
    path.write_text("profiles:\n  stock:\n    video_camera: left_camera\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="changes nothing"):
        camera_profiles.load(path)
    path.write_text("profiles:\n  far_side: {}\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="no 'stock' profile"):
        camera_profiles.load(path)


def test_the_sha256_is_canonical():
    far_side = camera_profiles.get("far_side")
    respelled = parse(
        "far_side",
        {
            "replace": {
                "front_camera": {
                    "left": [1, 0, 0],
                    "forward": [0, -0.778, -0.628],
                    "position": [0, 0.36, 1.2],
                    "type": "L515",
                    "name": "far_side_camera",
                }
            },
            "video_camera": "head_camera",
        },
    )
    assert respelled.sha256 == far_side.sha256
    assert len(far_side.sha256) == 64
    assert far_side.sha256 != camera_profiles.get("stock").sha256
    moved = replacing("front_camera", name="far_side_camera", type="L515")
    assert parse("far_side", moved).sha256 != far_side.sha256
    assert far_side.identity() == {"name": "far_side", "sha256": far_side.sha256}


def test_static_cameras_lists_what_robotwin_builds(static_camera_list):
    args = resolved(static_camera_list, head_camera_type="Large_D435")
    cameras = camera_profiles.static_cameras(apply(args, "far_side"))
    assert [(c["name"], c["type"]) for c in cameras] == [
        ("head_camera", "Large_D435"),
        ("far_side_camera", "L515"),
    ]
    headless = resolved(copy.deepcopy(STATIC_CAMERA_LIST), collect_head_camera=False)
    assert [c["name"] for c in camera_profiles.static_cameras(headless)] == ["front_camera"]
    assert camera_profiles.static_cameras({"save_freq": 1}) is None
