import dataclasses

import numpy as np
import pytest

from robotwin_icil import scene
from robotwin_icil.demo import BIMANUAL_QPOS_DIM

IDENTITY = [1.0, 0.0, 0.0, 0.0]


def fingerprint(**overrides) -> scene.SceneFingerprint:
    base = scene.SceneFingerprint(
        actors={
            "table": np.array([0.0, 0.0, 0.74, *IDENTITY]),
            "057_toycar": np.array([-0.22, 0.03, 0.76, *IDENTITY]),
        },
        articulations={"aloha": np.zeros(16)},
        articulation_roots={"aloha": np.array([0.0, -0.65, 0.0, *IDENTITY])},
        cameras={"head_camera": np.eye(4)},
        robot_qpos=np.zeros(BIMANUAL_QPOS_DIM),
        extras={"wall_texture": 3, "table_texture": 7},
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
        robot_qpos=np.full(BIMANUAL_QPOS_DIM, 1e-3),
        cameras={"head_camera": np.eye(4) * 1.01},
        extras={"wall_texture": 4, "table_texture": 7},
    )
    parts = {m.part for m in scene.compare(fingerprint(), drifted)}
    assert parts == {"robot", "camera", "extra"}


def test_float_noise_below_tolerance_passes():
    noisy = fingerprint(robot_qpos=np.full(BIMANUAL_QPOS_DIM, scene.JOINT_TOL / 10))
    assert scene.compare(fingerprint(), noisy) == []


def test_repeated_names_are_keyed_by_scene_order():
    assert scene.unique_names(["table", "block", "block", "wall"]) == [
        "table",
        "block#1",
        "block#2",
        "wall",
    ]
