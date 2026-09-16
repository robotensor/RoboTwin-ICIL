import dataclasses
import json

import numpy as np
import pytest

from robotwin_icil import scene

IDENTITY = [1.0, 0.0, 0.0, 0.0]
# aloha-agilex's joint vector: the fingerprint compares whatever width the robot reports.
QPOS_DIM = 14


def fingerprint(**overrides) -> scene.SceneFingerprint:
    base = scene.SceneFingerprint(
        actors={
            "table": np.array([0.0, 0.0, 0.74, *IDENTITY]),
            "057_toycar": np.array([-0.22, 0.03, 0.76, *IDENTITY]),
        },
        articulations={"aloha": np.zeros(16)},
        articulation_roots={"aloha": np.array([0.0, -0.65, 0.0, *IDENTITY])},
        cameras={"head_camera": np.eye(4)},
        robot_qpos=np.zeros(QPOS_DIM),
        extras={"embodiment": "aloha.urdf", "wall_texture": 3, "table_texture": 7},
    )
    return dataclasses.replace(base, **overrides)


def test_an_identical_rebuild_is_the_same_scene():
    assert scene.compare(fingerprint(), fingerprint()) == []
    assert scene.max_error([]) == 0.0


def test_a_moved_object_is_caught():
    moved = fingerprint()
    moved.actors["057_toycar"] = moved.actors["057_toycar"] + np.array([0.01, 0, 0, 0, 0, 0, 0])
    found = scene.compare(fingerprint(), moved)
    assert [(m.part, m.name) for m in found] == [("actor", "057_toycar")]
    assert found[0].error == pytest.approx(0.01)


def test_a_flipped_quaternion_is_the_same_rotation():
    flipped = fingerprint()
    flipped.actors["table"] = np.array([0.0, 0.0, 0.74, -1.0, 0.0, 0.0, 0.0])
    assert scene.compare(fingerprint(), flipped) == []


def test_a_rotated_object_is_caught():
    angle = 0.1
    rotated = fingerprint()
    rotated.actors["057_toycar"] = np.array(
        [-0.22, 0.03, 0.76, np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    )
    found = scene.compare(fingerprint(), rotated)
    assert found and found[0].error == pytest.approx(angle)


def test_a_different_object_instance_is_caught():
    swapped = fingerprint(
        actors={"table": fingerprint().actors["table"], "081_playingcards": np.zeros(7)}
    )
    parts = {(m.part, m.name) for m in scene.compare(fingerprint(), swapped)}
    assert ("actor", "057_toycar") in parts and ("actor", "081_playingcards") in parts


def test_robot_state_camera_and_texture_drift_are_caught():
    drifted = fingerprint(
        robot_qpos=np.full(QPOS_DIM, 1e-3),
        cameras={"head_camera": np.eye(4) * 1.01},
        extras={"embodiment": "aloha.urdf", "wall_texture": 4, "table_texture": 7},
    )
    parts = {m.part for m in scene.compare(fingerprint(), drifted)}
    assert parts == {"robot", "camera", "extra"}


def test_another_robot_is_another_scene():
    other = fingerprint(
        extras={"embodiment": "panda.urdf|panda.urdf", "wall_texture": 3, "table_texture": 7}
    )
    found = scene.compare(fingerprint(), other)
    assert [(m.part, m.name) for m in found] == [("extra", "embodiment")]


def test_float_noise_below_tolerance_passes():
    noisy = fingerprint(robot_qpos=np.full(QPOS_DIM, scene.JOINT_TOL / 10))
    assert scene.compare(fingerprint(), noisy) == []


def test_deviation_measures_what_the_tolerance_lets_through():
    assert scene.deviation(fingerprint(), fingerprint()) == 0.0
    noisy = fingerprint(robot_qpos=np.full(QPOS_DIM, 2e-6))
    assert scene.compare(fingerprint(), noisy) == []
    assert scene.deviation(fingerprint(), noisy) == pytest.approx(2e-6)
    assert scene.deviation(fingerprint(), fingerprint(actors={})) is None  # not one scene at all


def test_repeated_names_are_keyed_by_scene_order():
    assert scene.unique_names(["table", "block", "block", "wall"]) == [
        "table",
        "block#1",
        "block#2",
        "wall",
    ]


def test_a_fingerprint_round_trips_through_json():
    original = fingerprint()
    data = json.loads(json.dumps(original.to_json()))  # plain data: nothing numpy survives
    restored = scene.SceneFingerprint.from_json(data)
    assert scene.compare(original, restored) == []
    assert restored.extras == original.extras
    assert scene.digest(restored) == scene.digest(original)
    assert restored.to_json() == original.to_json()


def test_the_digest_is_exact_where_compare_is_tolerant():
    # A tenth of the position tolerance passes `compare`; the digest still changes, because a
    # digest says "this file was written from this scene", not "close enough".
    nudged = fingerprint()
    nudged.actors["table"] = nudged.actors["table"] + np.array([1e-6, 0, 0, 0, 0, 0, 0])
    assert scene.compare(fingerprint(), nudged) == []
    assert scene.digest(nudged) != scene.digest(fingerprint())
    assert len(scene.digest(fingerprint())) == 64


def test_the_digest_does_not_depend_on_dict_order():
    reordered = fingerprint(
        actors=dict(reversed(list(fingerprint().actors.items()))),
        extras={"table_texture": 7, "wall_texture": 3, "embodiment": "aloha.urdf"},
    )
    assert scene.digest(reordered) == scene.digest(fingerprint())


def test_numpy_scalars_in_extras_become_plain_json():
    data = fingerprint(
        extras={"wall_texture": np.int64(3), "table_z_bias": np.float32(0.5)}
    ).to_json()
    assert data["extras"] == {"wall_texture": 3, "table_z_bias": 0.5}
    assert type(data["extras"]["wall_texture"]) is int
    json.dumps(data)


def test_canonical_json_refuses_nan():
    with pytest.raises(ValueError):
        scene.canonical_json({"x": float("nan")})
