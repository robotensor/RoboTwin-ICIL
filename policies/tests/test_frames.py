import numpy as np
import pytest

from icil_policies.common import frames
from icil_policies.common.rotations import axis_angle_to_quat, normalize_quat, quat_to_matrix

M = frames.LIBERO_TO_WORLD


def test_m_is_a_rotation_turning_libero_forward_into_world_plus_y():
    np.testing.assert_allclose(M @ M.T, np.eye(3))
    assert np.linalg.det(M) == pytest.approx(1.0)
    np.testing.assert_allclose(M @ [1, 0, 0], [0, 1, 0])  # the robot faces world +y
    np.testing.assert_allclose(M @ [0, 1, 0], [-1, 0, 0])  # LIBERO's left is the robot's left
    np.testing.assert_allclose(M @ [0, 0, 1], [0, 0, 1])


def test_m_is_the_robot_roots_own_rotation():
    np.testing.assert_allclose(frames.root_rotation(), M, atol=1e-15)


def test_the_arm_bases_sit_either_side_of_the_centre_line():
    # Plan 3.1: arm bases at about (+-0.30, -0.42, 0.78).
    left, right = frames.arm_base("left"), frames.arm_base("right")
    np.testing.assert_allclose(left, [-0.297, -0.4195, 0.782], atol=1e-12)
    np.testing.assert_allclose(right, [0.3063, -0.4185, 0.781], atol=1e-12)


def test_libero_and_world_vectors_round_trip():
    v = np.random.default_rng(0).normal(size=(5, 3))
    np.testing.assert_allclose(frames.libero_to_world(frames.world_to_libero(v)), v)
    np.testing.assert_allclose(frames.world_to_libero(v), (M.T @ v.T).T)
    r = quat_to_matrix(normalize_quat(np.random.default_rng(1).normal(size=(5, 4))))
    np.testing.assert_allclose(
        frames.rotation_libero_to_world(frames.rotation_world_to_libero(r)), r
    )


def test_an_arm_base_is_the_origin_of_its_arm_frame():
    for arm in ("left", "right"):
        np.testing.assert_allclose(
            frames.position_in_arm_frame(frames.arm_base(arm), arm), 0, atol=1e-15
        )
    # 10 cm in front of the left base (world +y) is 10 cm forward in LIBERO's axes.
    ahead = frames.arm_base("left") + [0, 0.1, 0]
    np.testing.assert_allclose(frames.position_in_arm_frame(ahead, "left"), [0.1, 0, 0], atol=1e-15)


def test_the_tool_centre_is_twelve_centimetres_along_the_flanges_x():
    flange = np.array([0.1, 0.2, 0.3, 1, 0, 0, 0])
    np.testing.assert_allclose(frames.tcp_from_flange(flange), [0.22, 0.2, 0.3, 1, 0, 0, 0])
    # A flange whose +x points straight down, as for a top-down grasp.
    down = np.concatenate([[0, 0, 1.0], axis_angle_to_quat([0, np.pi / 2, 0])])
    np.testing.assert_allclose(frames.tcp_from_flange(down)[:3], [0, 0, 0.88], atol=1e-15)


def test_tool_centre_and_flange_round_trip_in_batches():
    rng = np.random.default_rng(2)
    poses = np.concatenate(
        [rng.normal(size=(4, 3, 3)), normalize_quat(rng.normal(size=(4, 3, 4)))], axis=-1
    )
    np.testing.assert_allclose(
        frames.flange_from_tcp(frames.tcp_from_flange(poses)), poses, atol=1e-15
    )


def test_poses_and_matrices_round_trip():
    pose = np.concatenate([[1.0, 2, 3], axis_angle_to_quat([0.1, -0.2, 0.3])])
    np.testing.assert_allclose(frames.matrix_to_pose(frames.pose_to_matrix(pose)), pose, atol=1e-15)
    with pytest.raises(ValueError, match=r"\(\.\.\., 7\)"):
        frames.tcp_from_flange(np.zeros(6))


def test_arms_and_their_slots():
    assert frames.other_arm("left") == "right" and frames.other_arm("right") == "left"
    with pytest.raises(ValueError, match="arm must be one of"):
        frames.arm_base("middle")
    qpos, ee = np.arange(14), np.arange(16)
    assert list(qpos[frames.QPOS_SLICES["right"]]) == list(range(7, 14))
    assert list(ee[frames.EE_SLICES["left"]]) == list(range(8))
