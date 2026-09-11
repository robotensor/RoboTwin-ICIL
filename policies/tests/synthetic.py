"""Small synthetic demonstrations and observations for the toolkit's tests."""

from __future__ import annotations

import numpy as np

from robotwin_icil.demo import Demonstration, Frame
from robotwin_icil.policy import Observation

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
CAMERA = "cam"


def endpose_row(
    left_position=(0.0, 0.0, 0.0),
    right_position=(0.0, 0.0, 0.0),
    *,
    left_quat=IDENTITY,
    right_quat=IDENTITY,
    left_gripper=1.0,
    right_gripper=1.0,
) -> np.ndarray:
    """One (16,) row in `take_action('ee')` layout."""
    return np.concatenate(
        [left_position, left_quat, [left_gripper], right_position, right_quat, [right_gripper]]
    ).astype(np.float64)


def endpose_dict(row: np.ndarray) -> dict:
    return {
        "left_endpose": row[:7].tolist(),
        "left_gripper": float(row[7]),
        "right_endpose": row[8:15].tolist(),
        "right_gripper": float(row[15]),
    }


def demonstration(endposes, times=None, qpos=None, gripper_joints=None, frequency=250 / 15):
    """A demonstration from (T, 16) endposes; each frame's image is filled with its index."""
    endposes = np.asarray(endposes, dtype=np.float64)
    count = len(endposes)
    qpos = np.zeros((count, 14)) if qpos is None else np.asarray(qpos, dtype=np.float64)
    frames = []
    for i in range(count):
        frames.append(
            Frame(
                index=i,
                images={CAMERA: np.full((2, 3, 3), i % 256, dtype=np.uint8)},
                qpos=qpos[i],
                endpose=endpose_dict(endposes[i]),
                time_s=None if times is None else float(times[i]),
                gripper_joints=None
                if gripper_joints is None
                else {arm: np.asarray(gripper_joints[arm][i]) for arm in ("left", "right")},
            )
        )
    return Demonstration(frames=tuple(frames), frequency=frequency)


def observation(endposes_row, qpos=None, step=0) -> Observation:
    return Observation(
        step=step,
        images={CAMERA: np.zeros((2, 3, 3), dtype=np.uint8)},
        qpos=np.zeros(14) if qpos is None else np.asarray(qpos, dtype=np.float64),
        endpose=endpose_dict(np.asarray(endposes_row, dtype=np.float64)),
    )
