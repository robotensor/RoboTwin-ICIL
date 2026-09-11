"""A demonstration on a fixed-rate time grid (plan 3.5).

RoboTwin records a frame before each motion primitive, after every `save_freq`-th physics step
from its first, and after its last, so frames are unevenly spaced and some share a time; a
gripper opening or closing is many frames of a still arm. A model trained at a fixed control
rate, 20 Hz for BPP and UniSkill, therefore needs the demonstration resampled on time, never on
frame index: index times 1/frequency distorts a 20 Hz resampling by up to 0.2 s per primitive.

Samples fall on `t0 + k / rate_hz` from the first frame's time. Between the two frames around a
sample:

- positions (flange positions, arm joints) are interpolated linearly,
- orientations by slerp,
- grippers are held: every gripper value, commanded or measured, is the last frame's at or
  before the sample (zero-order hold), so a gripper command is never half-issued,
- images are the nearest frame's, the earlier one on an exact tie in distance.

Times may repeat (`Demonstration` requires them non-decreasing): frames with no physics step
between them. A sample at such a time takes the last of those frames, which carries every
command issued at that instant; a sample between two times interpolates from the last frame of
the earlier time to the first of the later one. Untimed demonstrations use
`Demonstration.times()`'s estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from robotwin_icil.demo import ARMS, EE_POSE_DIM, Demonstration

from .frames import EE_SLICES, QPOS_SLICES
from .rotations import slerp

DEFAULT_RATE_HZ = 20.0


@dataclass(frozen=True)
class Resampled:
    """A demonstration's state at each sample of a fixed-rate grid."""

    rate_hz: float
    times: np.ndarray  # (N,) seconds, on the grid t0 + k / rate_hz
    source: np.ndarray  # (N,) the frame whose image each sample shows
    held: np.ndarray  # (N,) the last frame at or before each sample, whose grippers it holds
    endposes: np.ndarray  # (N, 16) `take_action('ee')` layout
    qpos: np.ndarray  # (N, 14) RoboTwin's joint vector
    gripper_joints: dict[str, np.ndarray] | None  # arm -> (N, joints), None if never measured
    demonstration: Demonstration = field(repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.times)

    def images(self, camera: str) -> np.ndarray:
        """(N, h, w, 3) one camera's image at each sample."""
        self.demonstration.images(camera)[:0]  # an unknown camera raises here, with the list
        frames = self.demonstration.frames
        return np.stack([frames[i].images[camera] for i in self.source])


def sample_times(times: np.ndarray, rate_hz: float, include_end: bool = True) -> np.ndarray:
    """The grid `t0 + k / rate_hz` covering `times`.

    It runs to the last grid point at or before the final frame; with `include_end`, one more
    point when that falls short of the final frame's time, so the final frame, where the expert
    succeeded, is always sampled (the extra sample holds it).
    """
    if not (math.isfinite(rate_hz) and rate_hz > 0):
        raise ValueError(f"rate_hz must be positive and finite, got {rate_hz}")
    times = np.asarray(times, dtype=np.float64)
    start, end = float(times[0]), float(times[-1])
    count = math.floor((end - start) * rate_hz + 1e-9) + 1
    if include_end and start + (count - 1) / rate_hz < end - 1e-12:
        count += 1
    return start + np.arange(count) / rate_hz


def resample(
    demonstration: Demonstration, rate_hz: float = DEFAULT_RATE_HZ, include_end: bool = True
) -> Resampled:
    """The demonstration on a `rate_hz` grid over its own times."""
    times = demonstration.times()
    grid = sample_times(times, rate_hz, include_end)
    last = len(times) - 1
    before = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, last)
    after = np.minimum(before + 1, last)
    span = times[after] - times[before]
    moving = (after > before) & (span > 0)
    fraction = np.where(moving, (grid - times[before]) / np.where(moving, span, 1.0), 0.0)
    nearest = np.where(grid - times[before] <= times[after] - grid, before, after)

    def linear(values: np.ndarray) -> np.ndarray:
        return values[before] + fraction[:, None] * (values[after] - values[before])

    poses = demonstration.endposes()
    endposes = poses[before].copy()  # grippers held; poses overwritten below
    for arm in ARMS:
        start = EE_SLICES[arm].start
        position, quat = slice(start, start + 3), slice(start + 3, start + EE_POSE_DIM)
        endposes[:, position] = linear(poses[:, position])
        endposes[:, quat] = slerp(poses[before, quat], poses[after, quat], fraction)

    joints = demonstration.qpos()
    qpos = joints[before].copy()  # grippers held; arm joints overwritten below
    for arm in ARMS:
        arm_joints = slice(QPOS_SLICES[arm].start, QPOS_SLICES[arm].stop - 1)
        qpos[:, arm_joints] = linear(joints[:, arm_joints])

    measured = None
    if demonstration.frames[0].gripper_joints is not None:
        measured = {
            arm: np.stack([np.asarray(f.gripper_joints[arm]) for f in demonstration.frames])[before]
            for arm in ARMS
        }

    return Resampled(
        rate_hz=float(rate_hz),
        times=grid,
        source=nearest,
        held=before,
        endposes=endposes,
        qpos=qpos,
        gripper_joints=measured,
        demonstration=demonstration,
    )
