"""The demonstration handed to a policy: what the expert saw, where it was, and what it did.

Model-independent on purpose. RoboTwin's expert produces frames of per-camera rgb plus the robot's
joint and end-effector state; a policy adapter selects what it consumes from that. The benchmark
core never learns what any particular model wants.

What is *not* here matters as much as what is. A demonstration carries no task name, no scene
seed, no `info` dict from `play_once()`, no actor handles and no success condition — see the privileged
information rule in CLAUDE.md. Everything in a `Frame` is something a real robot could observe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# RoboTwin's bimanual `joint_action.vector`: 6 arm joints + 1 gripper, per arm.
BIMANUAL_QPOS_DIM = 14


class DemonstrationError(ValueError):
    """A captured demonstration is not usable as policy context."""


@dataclass(frozen=True)
class Frame:
    """One observation of the expert, at one control step.

    `qpos` is the robot's state when the frame was taken; the action that carried the robot from
    this frame to the next is `action`, which is the *next* frame's `qpos` for a position-controlled
    expert. The last frame has no action.
    """

    index: int
    images: dict[str, np.ndarray]
    qpos: np.ndarray
    endpose: dict[str, Any]

    def __post_init__(self) -> None:
        if self.qpos.shape != (BIMANUAL_QPOS_DIM,):
            raise DemonstrationError(
                f"frame {self.index}: qpos has shape {self.qpos.shape}, expected ({BIMANUAL_QPOS_DIM},)"
            )
        for name, image in self.images.items():
            if image.ndim != 3 or image.shape[2] != 3:
                raise DemonstrationError(
                    f"frame {self.index}: camera {name!r} image has shape {image.shape}, expected (h, w, 3)"
                )


@dataclass(frozen=True)
class Demonstration:
    """One successful expert trajectory, as context for a frozen policy."""

    frames: tuple[Frame, ...]
    frequency: int
    cameras: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if len(self.frames) < 2:
            raise DemonstrationError(
                f"a demonstration needs at least two frames, got {len(self.frames)}"
            )
        if self.frequency <= 0:
            raise DemonstrationError(f"frequency must be positive, got {self.frequency}")
        indices = [frame.index for frame in self.frames]
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            raise DemonstrationError("frame indices are not strictly increasing")
        first = set(self.frames[0].images)
        for frame in self.frames:
            if set(frame.images) != first:
                raise DemonstrationError(
                    f"frame {frame.index} has cameras {sorted(frame.images)}, "
                    f"expected {sorted(first)}"
                )
        if not self.cameras:
            object.__setattr__(self, "cameras", tuple(sorted(first)))

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        return len(self.frames) / self.frequency

    def qpos(self) -> np.ndarray:
        """(T, 14) robot state over the demonstration."""
        return np.stack([frame.qpos for frame in self.frames])

    def actions(self) -> np.ndarray:
        """(T-1, 14) position targets, one per transition.

        The expert is position-controlled through `take_dense_action`, so the action that produced a
        transition is the state it arrived at. A replay policy feeding these back through
        `take_action(action_type='qpos')` reproduces the trajectory from the same initial state.
        """
        return np.stack([frame.qpos for frame in self.frames[1:]])

    def images(self, camera: str) -> np.ndarray:
        """(T, h, w, 3) rgb from one camera."""
        if camera not in self.cameras:
            raise DemonstrationError(f"no camera {camera!r}; have {list(self.cameras)}")
        return np.stack([frame.images[camera] for frame in self.frames])
