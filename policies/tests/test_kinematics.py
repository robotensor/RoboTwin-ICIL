import math
from pathlib import Path

import numpy as np
import pytest

from icil_policies.common import frames
from icil_policies.common.kinematics import (
    AlohaArm,
    Chain,
    KinematicsError,
    load_urdf,
    parse_urdf,
    rpy_matrix,
    sapien_joint_anchor,
)
from icil_policies.common.rotations import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    quat_to_matrix,
    relative_angle,
)

URDF = Path(__file__).parent / "data" / "aloha_like.urdf"
RNG = np.random.default_rng(0)

PLANAR = """<robot name="planar">
  <link name="base"/><link name="upper"/><link name="lower"/><link name="hand"/><link name="slider"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="upper"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1"/>
  </joint>
  <joint name="elbow" type="continuous">
    <origin xyz="0.3 0 0"/><parent link="upper"/><child link="lower"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="wrist" type="fixed">
    <origin xyz="0.2 0 0"/><parent link="lower"/><child link="hand"/>
  </joint>
  <joint name="rail" type="prismatic">
    <parent link="hand"/><child link="slider"/><axis xyz="0 0 2"/>
    <limit lower="0" upper="0.1"/>
  </joint>
</robot>"""


def _planar(tip="hand"):
    return Chain.from_urdf(parse_urdf(PLANAR), "base", tip)


def test_rpy_is_roll_then_pitch_then_yaw_about_fixed_axes():
    roll, pitch, yaw = 0.3, -0.7, 1.1
    expected = (
        axis_angle_to_matrix([0, 0, yaw])
        @ axis_angle_to_matrix([0, pitch, 0])
        @ axis_angle_to_matrix([roll, 0, 0])
    )
    np.testing.assert_allclose(rpy_matrix(roll, pitch, yaw), expected)
    joints = parse_urdf(
        '<robot name="r"><joint name="j" type="fixed"><origin xyz="1 2 3" rpy="0.3 -0.7 1.1"/>'
        '<parent link="a"/><child link="b"/></joint></robot>'
    )
    np.testing.assert_allclose(joints["b"].origin[:3, :3], expected)
    np.testing.assert_allclose(joints["b"].origin[:3, 3], [1, 2, 3])


def test_a_planar_arm_matches_its_closed_form():
    chain = _planar()
    assert chain.names == ("shoulder", "elbow")
    for q1, q2 in [(0, 0), (0.5, -0.3), (-1, 2.5)]:
        tip = chain.forward([q1, q2])
        expected = [
            0.3 * math.cos(q1) + 0.2 * math.cos(q1 + q2),
            0.3 * math.sin(q1) + 0.2 * math.sin(q1 + q2),
            0,
        ]
        np.testing.assert_allclose(tip[:3, 3], expected, atol=1e-15)
        np.testing.assert_allclose(tip[:3, :3], axis_angle_to_matrix([0, 0, q1 + q2]), atol=1e-15)


def test_a_prismatic_joint_slides_along_its_unit_axis_within_limits():
    chain = _planar("slider")
    tip = chain.forward([0, 0, 0.05])
    np.testing.assert_allclose(tip[:3, 3], [0.5, 0, 0.05], atol=1e-15)
    np.testing.assert_allclose(chain.lower, [-1, -np.inf, 0])
    np.testing.assert_allclose(chain.upper, [1, np.inf, 0.1])


def test_the_chain_follows_only_the_path_to_its_tip():
    joints = load_urdf(URDF)
    chain = Chain.from_urdf(joints, "footprint", "fl_link6")
    assert chain.names == tuple(f"fl_joint{i}" for i in range(1, 7))
    assert [j.name for j in chain.joints][0] == "fl_base_joint"
    with pytest.raises(KinematicsError, match="no chain"):
        Chain.from_urdf(joints, "fr_base_link", "fl_link6")
    with pytest.raises(KinematicsError, match="expected 6 joint values"):
        chain.forward(np.zeros(5))


def test_unusable_urdfs_are_refused():
    with pytest.raises(KinematicsError, match="not a URDF"):
        parse_urdf("<robot>")
    with pytest.raises(KinematicsError, match="expected 3 numbers"):
        parse_urdf(
            '<robot><joint name="j" type="fixed"><origin xyz="1 2"/><parent link="a"/><child link="b"/></joint></robot>'
        )
    floating = parse_urdf(
        '<robot><joint name="j" type="floating"><parent link="a"/><child link="b"/></joint></robot>'
    )
    with pytest.raises(KinematicsError, match="not supported"):
        Chain.from_urdf(floating, "a", "b")


def test_the_jacobian_matches_finite_differences():
    chain = Chain.from_urdf(load_urdf(URDF), "footprint", "fl_link6")
    q = RNG.uniform(-1, 1, size=6)
    jacobian = chain.jacobian(q)
    tip, h = chain.forward(q), 1e-7
    for i in range(6):
        moved = chain.forward(q + h * np.eye(6)[i])
        np.testing.assert_allclose(jacobian[:3, i], (moved[:3, 3] - tip[:3, 3]) / h, atol=1e-6)
        turn = matrix_to_axis_angle(moved[:3, :3] @ tip[:3, :3].T) / h
        np.testing.assert_allclose(jacobian[3:, i], turn, atol=1e-6)


def test_inverse_kinematics_recovers_a_reachable_pose():
    chain = Chain.from_urdf(load_urdf(URDF), "footprint", "fr_link6")
    for _ in range(10):
        q_true = RNG.uniform(-1, 1, size=6)
        target = chain.forward(q_true)
        result = chain.inverse(target, q_true + RNG.normal(scale=0.3, size=6))
        assert result.converged, result
        reached = chain.forward(result.q)
        np.testing.assert_allclose(reached[:3, 3], target[:3, 3], atol=1e-6)
        assert relative_angle(reached[:3, :3], target[:3, :3]) < 1e-5


def test_inverse_kinematics_reports_an_unreachable_pose_and_keeps_to_limits():
    chain = _planar()
    far = np.eye(4)
    far[:3, 3] = [2.0, 0, 0]
    result = chain.inverse(far, [0.1, 0.1], max_iterations=50)
    assert not result.converged and result.iterations == 50
    assert result.position_error_m > 1.4  # the arm reaches 0.5 m of the 2 m asked
    behind = chain.forward([2.0, 0.0])  # the shoulder's limit is 1 rad
    result = chain.inverse(behind, [0.0, 0.0], max_iterations=100)
    assert -1 <= result.q[0] <= 1 and not result.converged


def test_aloha_endposes_use_robotwins_turn_and_the_robot_root():
    arm = AlohaArm(URDF, "left")
    q = RNG.uniform(-1, 1, size=6)
    link = arm.link_frame(q)
    chain = Chain.from_urdf(load_urdf(URDF), "footprint", "fl_link6").forward(q)
    np.testing.assert_allclose(link[:3, :3], frames.root_rotation() @ chain[:3, :3], atol=1e-15)
    np.testing.assert_allclose(
        link[:3, 3], frames.root_rotation() @ chain[:3, 3] + [0, -0.65, 0], atol=1e-15
    )
    pose = arm.endpose(q)
    np.testing.assert_allclose(pose[:3], link[:3, 3], atol=1e-15)  # gripper_bias cancels the 0.12
    # SAPIEN's anchor for joint 6's x axis and global_trans_matrix cancel: the link's own frame.
    np.testing.assert_allclose(quat_to_matrix(pose[3:]), link[:3, :3], atol=1e-12)
    assert pose[3] >= 0


def _loader_anchor(axis):
    """sapien/wrapper/urdf_loader.py (3.0.0b1), the joint's `t_axis2joint`, copied as it is."""
    axis = np.array(axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-3:
        axis = np.array([1, 0, 0])
    else:
        axis /= axis_norm
    if abs(axis @ [1, 0, 0]) > 0.9:
        axis1 = np.cross(axis, [0, 0, 1])
        axis1 /= np.linalg.norm(axis1)
    else:
        axis1 = np.cross(axis, [1, 0, 0])
        axis1 /= np.linalg.norm(axis1)
    axis2 = np.cross(axis, axis1)
    t_axis2joint = np.eye(4)
    t_axis2joint[:3, 0] = axis
    t_axis2joint[:3, 1] = axis1
    t_axis2joint[:3, 2] = axis2
    return t_axis2joint[:3, :3]


def test_sapien_joint_anchors_are_built_as_its_loader_builds_them():
    # aloha's joint 6 turns about x: its anchor is diag(1, -1, -1), which global_trans_matrix undoes.
    np.testing.assert_allclose(sapien_joint_anchor([1, 0, 0]), np.diag([1.0, -1, -1]))
    np.testing.assert_allclose(sapien_joint_anchor([0, 1, 0]), [[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    # A zero axis falls back to x; SAPIEN's own copy raises there (an int array divided in place).
    np.testing.assert_allclose(sapien_joint_anchor([0, 0, 0]), np.diag([1.0, -1, -1]))
    for axis in [[0, 0, 2], [0.95, 0.1, 0], [0, -1, 0], *RNG.normal(size=(20, 3))]:
        anchor = sapien_joint_anchor(axis)
        np.testing.assert_allclose(anchor, _loader_anchor(axis), atol=1e-15)
        np.testing.assert_allclose(anchor.T @ anchor, np.eye(3), atol=1e-12)
        assert np.linalg.det(anchor) == pytest.approx(1.0)


def test_aloha_inverse_kinematics_round_trips_an_endpose():
    for side in ("left", "right"):
        arm = AlohaArm(URDF, side)
        q_true = RNG.uniform(-0.8, 0.8, size=6)
        result = arm.inverse(arm.endpose(q_true), q_true + RNG.normal(scale=0.2, size=6))
        assert result.converged
        reached, target = arm.endpose(result.q), arm.endpose(q_true)
        np.testing.assert_allclose(reached[:3], target[:3], atol=1e-6)
        assert relative_angle(quat_to_matrix(reached[3:]), quat_to_matrix(target[3:])) < 1e-5


def test_an_arm_whose_joints_are_not_alohas_is_refused(tmp_path):
    renamed = tmp_path / "renamed.urdf"
    renamed.write_text(URDF.read_text().replace('name="fl_joint3"', 'name="elbow"'))
    with pytest.raises(KinematicsError, match="expected"):
        AlohaArm(renamed, "left")
    AlohaArm(renamed, "right")
