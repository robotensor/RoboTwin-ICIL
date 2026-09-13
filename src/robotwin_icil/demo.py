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

# The order of the two halves of a qpos row, and of what `arms_moved` reports.
ARMS = ("left", "right")
# An arm moved if any of its values departs from its first-frame value by more than this. One
# number gates two units — radians for an arm joint, a fraction of full travel for the gripper's
# normalised [0, 1] opening — and 0.05 is small against either: about 3 degrees of a joint, a
# twentieth of a gripper swing.
MOVED_THRESHOLD = 0.05


class DemonstrationError(ValueError):
    """A captured demonstration is not usable as policy context."""


@dataclass(frozen=True)
class Frame:
    """One observation of the expert, at one control step.

    `qpos` is the robot's state when the frame was taken; the action that carried the robot from
    this frame to the next is `action`, which is the *next* frame's `qpos` for a position-controlled
    expert. The last frame has no action.

    `qpos` is RoboTwin's `joint_action.vector`: the left arm's joints and gripper, then the right's.
    Its width is the robot's — 14 on aloha-agilex, 16 on two Franka arms — so a frame only requires
    a flat vector; the demonstration requires every frame to have the same one.

    `time_s` is simulated seconds since the expert started, counted in physics steps; None in a
    frame recorded without a clock, when `Demonstration.times()` falls back to the nominal rate.
    """

    index: int
    images: dict[str, np.ndarray]
    qpos: np.ndarray
    endpose: dict[str, Any]
    time_s: float | None = None

    def __post_init__(self) -> None:
        if self.time_s is not None and not math.isfinite(self.time_s):
            raise DemonstrationError(f"frame {self.index}: time_s is {self.time_s}")
        if self.qpos.ndim != 1 or self.qpos.shape[0] == 0:
            raise DemonstrationError(
                f"frame {self.index}: qpos has shape {self.qpos.shape}, expected a flat joint vector"
            )
        if not np.isfinite(self.qpos).all():
            # No joint reads as NaN or inf; and a NaN compares false against every threshold, so
            # `arms_moved` would count such an arm as still rather than refuse it.
            raise DemonstrationError(f"frame {self.index}: qpos has a non-finite value")
        for name, image in self.images.items():
            if image.ndim != 3 or image.shape[2] != 3:
                raise DemonstrationError(
                    f"frame {self.index}: camera {name!r} image has shape {image.shape}, expected (h, w, 3)"
                )


@dataclass(frozen=True)
class Demonstration:
    """One successful expert trajectory, as context for a frozen policy.

    Its frames are not evenly spaced in time. RoboTwin's `take_dense_action` records one frame
    before its first physics step, one after every `save_freq`-th step counted from that first
    step, and one after its last; `together_move_to_pose` does the same in a loop of its own. So
    every motion primitive adds a frame one step in, a shorter remainder at its end, and a
    duplicate where the next primitive starts. `times()` says when each frame was taken.
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
            if frame.qpos.shape[0] != self.qpos_dim:
                raise DemonstrationError(
                    f"frame {frame.index} has a {frame.qpos.shape[0]}-wide qpos, "
                    f"expected {self.qpos_dim} like frame {self.frames[0].index}"
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
    def timed(self) -> bool:
        """Whether the frames carry the simulated time they were taken at."""
        return self.frames[0].time_s is not None

    def times(self) -> np.ndarray:
        """(T,) seconds since the expert started, one per frame.

        The recorded `time_s` of every frame when the demonstration was captured under a clock:
        real, and uneven, as the class docstring says. Without one, the nominal `index /
        frequency`, which is only right within a single motion primitive.
        """
        if self.timed:
            return np.asarray([frame.time_s for frame in self.frames], dtype=np.float64)
        return np.arange(len(self.frames), dtype=np.float64) / self.frequency

    @property
    def duration_s(self) -> float:
        return len(self.frames) / self.frequency

    @property
    def qpos_dim(self) -> int:
        """Width of every frame's `qpos`: the robot's joint vector, 14 on aloha-agilex, 16 on Franka."""
        return int(self.frames[0].qpos.shape[0])

    def qpos(self) -> np.ndarray:
        """(T, qpos_dim) robot state over the demonstration."""
        return np.stack([frame.qpos for frame in self.frames])

    def actions(self) -> np.ndarray:
        """(T-1, qpos_dim) position targets, one per transition.

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


def arm_displacements(demonstration: Demonstration) -> dict[str, float]:
    """Each arm's largest departure from its first-frame value, over every joint and every frame.

    Assumes RoboTwin's `joint_action.vector` layout, which holds for every embodiment it ships:
    the left arm's joints then its gripper, followed by the right arm's joints then its gripper,
    both halves the same width (7+7 on aloha-agilex, 8+8 on two Franka Pandas). The row is split
    at half of `Demonstration.qpos_dim`, the width the robot's frames carry, so the split follows
    the robot the run chose; an odd width is refused rather than guessed at. The value is in
    radians when a joint set it and a fraction of full travel when the gripper did, since the
    gripper is upstream's normalised [0, 1] opening; the survey records it per seed so the
    `arms_moved` threshold can be judged from the json without re-running the expert.
    """
    qpos = demonstration.qpos()
    width = demonstration.qpos_dim
    if width % 2:
        raise DemonstrationError(f"qpos width {width} does not split into two equal arms")
    excursion = np.abs(qpos - qpos[0]).max(axis=0)
    halves = np.split(excursion, 2)
    return {arm: float(half.max()) for arm, half in zip(ARMS, halves, strict=True)}


def arms_moved(demonstration: Demonstration, threshold: float = MOVED_THRESHOLD) -> tuple[str, ...]:
    """Which arms the expert drove, in `ARMS` order, read from the joint trajectory.

    An arm moved if its `arm_displacements` entry exceeds `threshold`: any of its values, the
    gripper included, departed from its first-frame value by more than that at some frame. The
    one threshold reads in two units, radians for an arm joint and a fraction of full travel for
    the gripper, and an open-close swing clears it by a wide margin either way.

    The survey uses this to measure a task's `arms` entry against what its expert actually did.
    """
    displacements = arm_displacements(demonstration)
    return tuple(arm for arm in ARMS if displacements[arm] > threshold)
