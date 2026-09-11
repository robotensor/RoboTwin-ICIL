"""aloha-agilex's frames: where its arms sit, its tool centre point, and LIBERO's axes.

RoboTwin places aloha-agilex with its root link (`footprint`) at `robot_pose`, a quarter turn
about z so the robot faces world +y, and hangs each arm's base link off that root by a fixed
joint. Every constant below cites the file or plan section it comes from; the vendor files are
`vendor/RoboTwin/assets/embodiments/aloha-agilex/config.yml` (CONFIG) and its
`urdf/arx5_description_isaac.urdf` (URDF), and `vendor/RoboTwin/envs/robot/robot.py` (ROBOT).

- LIBERO's world axes are its Panda base's: x forward, y left, z up. A LIBERO vector v is
  `LIBERO_TO_WORLD @ v` in RoboTwin's world (plan 3.4, Frames), and that matrix is the robot
  root's own rotation, so LIBERO's axes are aloha's base axes.
- `endpose` is the flange pose (`fl_link6` / `fr_link6`). The tool centre point is 0.12 m
  further along the flange's own +x axis (CONFIG `gripper_bias: 0.12`; ROBOT `_trans_endpose`
  adds `gripper_bias` along +x for the tool centre and `gripper_bias - 0.12`, zero, for the
  endpose). robosuite applies OSC deltas at its tool centre point, so an adapter integrates
  there and converts back to the flange (plan 3.4).

Poses are `[x, y, z, qw, qx, qy, qz]`, RoboTwin's layout, with any leading batch shape.
"""

from __future__ import annotations

import numpy as np

from robotwin_icil.demo import ARMS, EE_POSE_DIM

from .rotations import canonical_quat, matrix_to_quat, quat_to_matrix

# CONFIG `robot_pose: [[0, -0.65, 0.0, 0.707, 0, 0, 0.707]]`: position, then wxyz. 0.707 is the
# file's rounding of sqrt(1/2); the rotation is a quarter turn about z.
ROBOT_ROOT_POSITION = np.array([0.0, -0.65, 0.0])
ROBOT_ROOT_QUAT = canonical_quat(np.array([0.707, 0.0, 0.0, 0.707]))

# Plan 3.4, Frames: a LIBERO vector v is M v in the world.
LIBERO_TO_WORLD = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# URDF `fl_base_joint` and `fr_base_joint` origins: each arm's base link in the root frame. Their
# small yaws (0.02 and 0.01 rad) turn the base links, not these points.
ARM_BASE_IN_ROOT = {
    "left": np.array([0.2305, 0.297, 0.782]),
    "right": np.array([0.2315, -0.3063, 0.781]),
}

# CONFIG `gripper_bias: 0.12`: the tool centre point along the flange's own +x axis.
TCP_OFFSET_M = 0.12

# Each arm's entries in RoboTwin's 14-dim qpos (6 joints, then the gripper) and in the 16-dim
# `take_action('ee')` layout (flange pose, then the gripper): `robotwin_icil.demo`.
QPOS_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}
EE_SLICES = {
    "left": slice(0, EE_POSE_DIM + 1),
    "right": slice(EE_POSE_DIM + 1, 2 * EE_POSE_DIM + 2),
}


def check_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    return arm


def other_arm(arm: str) -> str:
    return ARMS[1 - ARMS.index(check_arm(arm))]


def root_rotation() -> np.ndarray:
    """(3, 3) the robot root's rotation in the world."""
    return quat_to_matrix(ROBOT_ROOT_QUAT)


def arm_base(arm: str) -> np.ndarray:
    """(3,) where an arm's base link sits in the world: about (-0.30, -0.42, 0.78) for the left."""
    return root_rotation() @ ARM_BASE_IN_ROOT[check_arm(arm)] + ROBOT_ROOT_POSITION


def world_to_libero(vectors: np.ndarray) -> np.ndarray:
    """(..., 3) world vectors in LIBERO's axes: `M.T @ v`."""
    return np.asarray(vectors, dtype=np.float64) @ LIBERO_TO_WORLD


def libero_to_world(vectors: np.ndarray) -> np.ndarray:
    """(..., 3) LIBERO vectors in the world: `M @ v`."""
    return np.asarray(vectors, dtype=np.float64) @ LIBERO_TO_WORLD.T


def rotation_world_to_libero(matrices: np.ndarray) -> np.ndarray:
    """(..., 3, 3) world orientations in LIBERO's axes: `M.T @ R`. No tool-frame correction."""
    return LIBERO_TO_WORLD.T @ np.asarray(matrices, dtype=np.float64)


def rotation_libero_to_world(matrices: np.ndarray) -> np.ndarray:
    """(..., 3, 3) LIBERO orientations in the world: `M @ R`."""
    return LIBERO_TO_WORLD @ np.asarray(matrices, dtype=np.float64)


def position_in_arm_frame(positions: np.ndarray, arm: str) -> np.ndarray:
    """(..., 3) world positions relative to an arm's base, in LIBERO's axes: `M.T (p - b_arm)`.

    The arm-relative half of plan 3.6's `p_L = M.T (p_world - b_arm) + b_Panda`; the adapter
    adds its model's own base.
    """
    return world_to_libero(np.asarray(positions, dtype=np.float64) - arm_base(arm))


def tcp_from_flange(pose: np.ndarray) -> np.ndarray:
    """(..., 7) flange pose -> tool centre pose: 0.12 m along the flange's +x, same orientation."""
    return _along_x(pose, TCP_OFFSET_M)


def flange_from_tcp(pose: np.ndarray) -> np.ndarray:
    """(..., 7) tool centre pose -> the flange pose `take_action('ee')` reads."""
    return _along_x(pose, -TCP_OFFSET_M)


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """(..., 7) -> (..., 4, 4) homogeneous transforms."""
    pose = _poses(pose)
    out = np.zeros(pose.shape[:-1] + (4, 4))
    out[..., :3, :3] = quat_to_matrix(pose[..., 3:])
    out[..., :3, 3] = pose[..., :3]
    out[..., 3, 3] = 1.0
    return out


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    """(..., 4, 4) -> (..., 7), the quaternion with w >= 0."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate([matrix[..., :3, 3], matrix_to_quat(matrix[..., :3, :3])], axis=-1)


def _along_x(pose: np.ndarray, distance: float) -> np.ndarray:
    pose = _poses(pose)
    x_axis = quat_to_matrix(pose[..., 3:])[..., :, 0]
    return np.concatenate([pose[..., :3] + distance * x_axis, pose[..., 3:]], axis=-1)


def _poses(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim == 0 or pose.shape[-1] != EE_POSE_DIM:
        raise ValueError(f"expected poses of shape (..., {EE_POSE_DIM}), got {pose.shape}")
    return pose
