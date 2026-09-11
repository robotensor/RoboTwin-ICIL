import math

import numpy as np
import pytest

from icil_policies.common import rotations as rot

RNG = np.random.default_rng(0)


def _random_quats(n: int) -> np.ndarray:
    return rot.normalize_quat(RNG.normal(size=(n, 4)))


def _random_matrices(n: int) -> np.ndarray:
    return rot.quat_to_matrix(_random_quats(n))


def test_a_quarter_turn_about_z():
    q = np.array([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
    expected = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(rot.quat_to_matrix(q), expected, atol=1e-15)
    np.testing.assert_allclose(rot.matrix_to_quat(expected), q, atol=1e-15)
    np.testing.assert_allclose(rot.matrix_to_axis_angle(expected), [0, 0, math.pi / 2])
    np.testing.assert_allclose(rot.axis_angle_to_matrix([0, 0, math.pi / 2]), expected, atol=1e-15)


def test_matrices_from_quaternions_are_rotations():
    m = _random_matrices(100)
    np.testing.assert_allclose(
        m @ np.swapaxes(m, -1, -2), np.broadcast_to(np.eye(3), m.shape), atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.det(m), 1.0, atol=1e-12)


def test_q_and_minus_q_are_one_rotation_and_the_way_back_has_w_non_negative():
    q = _random_quats(100)
    np.testing.assert_allclose(rot.quat_to_matrix(q), rot.quat_to_matrix(-q), atol=1e-15)
    back = rot.matrix_to_quat(rot.quat_to_matrix(q))
    assert np.all(back[:, 0] >= 0)
    np.testing.assert_allclose(back, rot.canonical_quat(q), atol=1e-12)


def test_a_non_unit_quaternion_is_normalized_and_a_zero_one_refused():
    q = np.array([2.0, 0, 0, 0])
    np.testing.assert_allclose(rot.quat_to_matrix(q), np.eye(3))
    with pytest.raises(ValueError, match="zero quaternion"):
        rot.quat_to_matrix(np.zeros(4))


@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -2, 3]])
def test_half_turns_survive_the_round_trip(axis):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    m = rot.axis_angle_to_matrix(axis * math.pi)
    q = rot.matrix_to_quat(m)
    assert q[0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(rot.quat_to_matrix(q), m, atol=1e-12)
    rotvec = rot.matrix_to_axis_angle(m)
    assert np.linalg.norm(rotvec) == pytest.approx(math.pi)
    assert abs(rotvec @ axis) == pytest.approx(math.pi)


def test_axis_angle_round_trips_below_a_half_turn():
    directions = RNG.normal(size=(200, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    angles = RNG.uniform(0, math.pi - 1e-6, size=(200, 1))
    rotvec = directions * angles
    np.testing.assert_allclose(
        rot.matrix_to_axis_angle(rot.axis_angle_to_matrix(rotvec)), rotvec, atol=1e-9
    )
    np.testing.assert_allclose(
        rot.quat_to_axis_angle(rot.axis_angle_to_quat(rotvec)), rotvec, atol=1e-12
    )


@pytest.mark.parametrize("size", [0.0, 1e-12, 1e-9, 1e-7])
def test_tiny_rotations_stay_first_order_exact(size):
    rotvec = np.array([size, -2 * size, 0.5 * size])
    q = rot.axis_angle_to_quat(rotvec)
    np.testing.assert_allclose(q, [1, *(rotvec / 2)], atol=1e-15)
    np.testing.assert_allclose(rot.quat_to_axis_angle(q), rotvec, atol=1e-15)
    np.testing.assert_allclose(
        rot.axis_angle_to_matrix(rotvec), np.eye(3) + _skew(rotvec), atol=1e-13
    )


def test_the_non_canonical_axis_angle_matches_robosuite():
    q = _random_quats(100)
    q = q[np.abs(q[:, 0]) < 0.999]
    q[:10] *= -1 if q[0, 0] > 0 else 1
    w, xyz = q[:, :1], q[:, 1:]
    # robosuite.utils.transform_utils.quat2axisangle, on wxyz instead of its xyzw.
    expected = xyz * 2 * np.arccos(w) / np.sqrt(1 - w * w)
    np.testing.assert_allclose(rot.quat_to_axis_angle(q, canonical=False), expected, atol=1e-10)
    canonical = rot.quat_to_axis_angle(q)
    assert np.all(np.linalg.norm(canonical, axis=-1) <= math.pi + 1e-12)
    np.testing.assert_allclose(
        rot.axis_angle_to_matrix(canonical), rot.quat_to_matrix(q), atol=1e-12
    )


def test_rot6d_is_the_first_two_rows():
    m = _random_matrices(10)
    d6 = rot.matrix_to_rot6d(m)
    np.testing.assert_array_equal(d6[:, :3], m[:, 0, :])
    np.testing.assert_array_equal(d6[:, 3:], m[:, 1, :])
    np.testing.assert_allclose(rot.rot6d_to_matrix(d6), m, atol=1e-12)


def test_rot6d_of_any_six_numbers_is_a_rotation_keeping_the_first_row_direction():
    d6 = RNG.normal(size=(50, 6))
    m = rot.rot6d_to_matrix(d6)
    np.testing.assert_allclose(
        m @ np.swapaxes(m, -1, -2), np.broadcast_to(np.eye(3), m.shape), atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.det(m), 1.0, atol=1e-12)
    first = d6[:, :3] / np.linalg.norm(d6[:, :3], axis=-1, keepdims=True)
    np.testing.assert_allclose(m[:, 0, :], first, atol=1e-12)


def test_rot6d_matches_bpp_formula():
    # behavior_prompting/common/pose_util.py rot6d_to_mat, copied: rows b1, b2, b1 x b2.
    d6 = RNG.normal(size=(20, 6))
    a1, a2 = d6[:, :3], d6[:, 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    np.testing.assert_allclose(
        rot.rot6d_to_matrix(d6), np.stack([b1, b2, np.cross(b1, b2)], axis=-2)
    )


def test_the_quaternion_product_composes_like_matrices():
    a, b = _random_quats(20), _random_quats(20)
    np.testing.assert_allclose(
        rot.quat_to_matrix(rot.quat_multiply(a, b)),
        rot.quat_to_matrix(a) @ rot.quat_to_matrix(b),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rot.quat_multiply(a, rot.quat_conjugate(a)), np.tile([1.0, 0, 0, 0], (20, 1)), atol=1e-12
    )


def test_angles():
    assert rot.quat_angle(rot.axis_angle_to_quat([0, 0.3, 0])) == pytest.approx(0.3)
    assert rot.quat_angle(-rot.axis_angle_to_quat([0, 0.3, 0])) == pytest.approx(0.3)
    a = rot.axis_angle_to_matrix([0.1, 0.2, 0.3])
    b = rot.axis_angle_to_matrix([0, 0, 0.25]) @ a
    assert rot.relative_angle(a, b) == pytest.approx(0.25)


def test_slerp_ends_at_its_inputs_and_moves_at_constant_speed():
    q0, q1 = rot.axis_angle_to_quat([0, 0, 0.2]), rot.axis_angle_to_quat([0, 0, 1.4])
    np.testing.assert_allclose(rot.slerp(q0, q1, 0.0), q0, atol=1e-15)
    np.testing.assert_allclose(rot.slerp(q0, q1, 1.0), q1, atol=1e-15)
    ts = np.linspace(0, 1, 7)
    path = rot.slerp(np.tile(q0, (7, 1)), np.tile(q1, (7, 1)), ts)
    np.testing.assert_allclose(rot.quat_to_axis_angle(path)[:, 2], 0.2 + 1.2 * ts, atol=1e-12)


def test_slerp_takes_the_short_way_whichever_sign_it_is_given():
    q0, q1 = rot.axis_angle_to_quat([0.3, 0, 0]), rot.axis_angle_to_quat([0.9, 0, 0])
    np.testing.assert_allclose(
        rot.slerp(q0, -q1, 0.5), rot.axis_angle_to_quat([0.6, 0, 0]), atol=1e-12
    )
    np.testing.assert_allclose(rot.slerp(q0, q0 * (1 + 1e-12), 0.5), q0, atol=1e-12)


def test_shapes_are_checked():
    with pytest.raises(ValueError, match=r"\(\.\.\., 4\)"):
        rot.quat_to_matrix(np.zeros(3))
    with pytest.raises(ValueError, match=r"\(\.\.\., 3, 3\)"):
        rot.matrix_to_quat(np.eye(4))


def test_against_scipy_when_importable():
    transform = pytest.importorskip("scipy.spatial.transform")
    q = _random_quats(50)
    reference = transform.Rotation.from_quat(q[:, [1, 2, 3, 0]])
    np.testing.assert_allclose(rot.quat_to_matrix(q), reference.as_matrix(), atol=1e-12)
    np.testing.assert_allclose(
        rot.matrix_to_axis_angle(reference.as_matrix()), reference.as_rotvec(), atol=1e-9
    )


def test_against_transforms3d_when_importable():
    t3d = pytest.importorskip("transforms3d")
    for q in _random_quats(50):
        m = t3d.quaternions.quat2mat(q)
        np.testing.assert_allclose(rot.quat_to_matrix(q), m, atol=1e-12)
        np.testing.assert_allclose(rot.matrix_to_quat(m), t3d.quaternions.mat2quat(m), atol=1e-9)


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
