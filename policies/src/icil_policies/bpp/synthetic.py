"""A synthetic RoboTwin demonstration and observations shaped like the BPP adapter's inputs.

The `far_side` profile's cameras (`far_side_camera` plus both wrists), measured finger joints,
frame times and a recognisable joint vector, so the adapter's whole chain — its tests and
`icil-bpp preflight`'s action-parity gate alike — runs without a simulator. It is not
benchmark data: nothing here is scored, and no episode ever sees it.
"""

from __future__ import annotations

import numpy as np

from robotwin_icil.demo import Demonstration, Frame
from robotwin_icil.policy import Observation

from ..common.frames import TCP_OFFSET_M, arm_base, flange_from_tcp
from ..common.rotations import axis_angle_to_quat

AGENTVIEW = "far_side_camera"
WRISTS = ("left_camera", "right_camera")
# aloha's `gripper_scale`: the finger joint positions for a closed and an open gripper.
FINGER_CLOSED, FINGER_OPEN = -0.01, 0.045
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def images(index: int) -> dict[str, np.ndarray]:
    """One frame's cameras, each filled with a value that follows the frame index."""
    return {
        AGENTVIEW: np.full((180, 320, 3), index % 256, dtype=np.uint8),
        WRISTS[0]: np.full((240, 320, 3), (index + 7) % 256, dtype=np.uint8),
        WRISTS[1]: np.full((240, 320, 3), (index + 11) % 256, dtype=np.uint8),
    }


def flange_of(tool_position, quat=IDENTITY) -> np.ndarray:
    """The flange pose whose tool centre point is at `tool_position` with orientation `quat`."""
    return flange_from_tcp(np.concatenate([np.asarray(tool_position, dtype=np.float64), quat]))


def endpose(left_flange, right_flange, left_gripper=1.0, right_gripper=1.0) -> dict:
    return {
        "left_endpose": np.asarray(left_flange, dtype=np.float64),
        "left_gripper": float(left_gripper),
        "right_endpose": np.asarray(right_flange, dtype=np.float64),
        "right_gripper": float(right_gripper),
    }


def fingers(left_gripper=1.0, right_gripper=1.0) -> dict[str, np.ndarray]:
    """Measured finger joints for commanded gripper values, as RoboTwin's robot reports them."""

    def joint(value: float) -> np.ndarray:
        return np.full(2, FINGER_CLOSED + value * (FINGER_OPEN - FINGER_CLOSED))

    return {"left": joint(left_gripper), "right": joint(right_gripper)}


def tool_path(steps: int, arm: str = "left", step_m: float = 0.002) -> np.ndarray:
    """(steps, 3) the active arm's tool centre moving out from over its own base."""
    start = arm_base(arm) + np.array([0.0, 0.25, -0.04 + TCP_OFFSET_M])
    return start + np.outer(np.arange(steps), [0.0, step_m, 0.0])


def demonstration(
    steps: int = 24,
    arm: str = "left",
    grippers: np.ndarray | None = None,
    rate_hz: float = 20.0,
    turn_rad: float = 0.0,
    step_m: float = 0.002,
) -> Demonstration:
    """A one-arm demonstration at `rate_hz`: the tool slides, turns and works its gripper."""
    path = tool_path(steps, arm, step_m)
    idle = arm_base("right" if arm == "left" else "left") + np.array([0.0, 0.2, 0.1])
    angles = np.linspace(0.0, turn_rad, steps)
    values = np.ones(steps) if grippers is None else np.asarray(grippers, dtype=np.float64)
    joints = np.concatenate([np.full(7, 0.1), np.full(7, 0.2)])
    frames = []
    for i in range(steps):
        active = flange_of(path[i], axis_angle_to_quat([0.0, 0.0, angles[i]]))
        other = flange_of(idle)
        left, right = (active, other) if arm == "left" else (other, active)
        left_value, right_value = (values[i], 1.0) if arm == "left" else (1.0, values[i])
        frames.append(
            Frame(
                index=i,
                images=images(i),
                qpos=joints,
                endpose=endpose(left, right, left_value, right_value),
                time_s=i / rate_hz,
                gripper_joints=fingers(left_value, right_value),
            )
        )
    return Demonstration(frames=tuple(frames), frequency=rate_hz)


def observation(demonstration_: Demonstration, index: int = 0, step: int = 0) -> Observation:
    """The observation a rollout would report in the state of one demonstration frame."""
    frame = demonstration_.frames[index]
    return Observation(
        step=step,
        images=frame.images,
        qpos=frame.qpos,
        endpose=frame.endpose,
        time_s=frame.time_s,
        gripper_joints=frame.gripper_joints,
    )
