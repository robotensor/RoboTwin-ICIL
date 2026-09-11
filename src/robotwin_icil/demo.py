"""The demonstration handed to a policy: what the expert saw, where it was, and what it did.

Model-independent on purpose. RoboTwin's expert produces frames of per-camera rgb plus the robot's
joint and end-effector state; a policy adapter selects what it consumes from that. The benchmark
core never learns what any particular model wants.

What is *not* here matters as much as what is. A demonstration carries no task name, no scene
seed, no `info` dict from `play_once()`, no actor handles and no success condition — see the privileged
information rule in CLAUDE.md. Everything in a `Frame` is something a real robot could observe.
"""

from __future__ import annotations

import math
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

    `time_s` is simulated seconds since the expert started, counted in physics steps; None in
    frames recorded before the benchmark kept time.
    """

    index: int
    images: dict[str, np.ndarray]
    qpos: np.ndarray
    endpose: dict[str, Any]
    time_s: float | None = None

    def __post_init__(self) -> None:
        if self.time_s is not None and not math.isfinite(self.time_s):
            raise DemonstrationError(f"frame {self.index}: time_s is {self.time_s}")
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
    """One successful expert trajectory, as context for a frozen policy.

    Its frames are not evenly spaced in time: RoboTwin records one frame before each motion
    primitive, one after every `save_freq`-th physics step from its first, and one after its
    last, so each primitive adds a frame one step in, a shorter remainder, and an exact duplicate
    where the next primitive starts. `times()` says when each frame was taken.
    """

    frames: tuple[Frame, ...]
    # Nominal frames per second: the sim rate over RoboTwin's `save_freq`, the spacing of the
    # frames within a primitive. Not the spacing of every frame; see `times()`.
    frequency: float
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
        timed = [frame.time_s is not None for frame in self.frames]
        if any(timed) and not all(timed):
            raise DemonstrationError("some frames have a time_s and some do not")
        if all(timed):
            for before, frame in zip(self.frames[:-1], self.frames[1:], strict=True):
                if frame.time_s < before.time_s:
                    raise DemonstrationError(
                        f"frame {frame.index} is timed at {frame.time_s} s, before frame "
                        f"{before.index} at {before.time_s} s"
                    )

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        """Simulated seconds from the first frame to the last, by `times()`.

        Not `len / frequency`: boundary duplicates, one-step gaps and remainders make the frame
        count over the nominal rate wrong for timed data.
        """
        times = self.times()
        return float(times[-1] - times[0])

    def times(self) -> np.ndarray:
        """(T,) simulated seconds of each frame since the expert started.

        Recorded times are returned as they are; a tie is two frames with no physics step between
        them. Frames recorded without times get an estimate: a frame identical to the one before it
        (qpos, endpose and every image) shares its time, and every other frame follows the one
        before it by 1 / `frequency`. It collapses the duplicates at primitive boundaries but not
        the one-step and remainder gaps, which only recorded times resolve.
        """
        if self.frames[0].time_s is not None:
            return np.array([frame.time_s for frame in self.frames], dtype=np.float64)
        times = np.zeros(len(self.frames))
        for i in range(1, len(self.frames)):
            gap = 0.0 if _same_frame(self.frames[i - 1], self.frames[i]) else 1.0 / self.frequency
            times[i] = times[i - 1] + gap
        return times

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


def _same_frame(a: Frame, b: Frame) -> bool:
    """Whether two frames observed the same state: equal qpos, endpose and images."""
    return (
        np.array_equal(a.qpos, b.qpos)
        and _equal(a.endpose, b.endpose)
        and a.images.keys() == b.images.keys()
        and all(np.array_equal(a.images[name], b.images[name]) for name in a.images)
    )


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, dict) or isinstance(b, dict):
        return (
            isinstance(a, dict)
            and isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_equal(a[key], b[key]) for key in a)
        )
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))
