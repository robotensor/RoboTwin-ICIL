"""An end-effector target that keeps sub-tolerance motion, and a stall detector (plan 3.4).

A LIBERO-trained model commands about 1 mm of tool motion per 20 Hz step, under CuRobo's 5 mm
goal tolerance. Re-basing every `ee` call on the measured pose could therefore stall: each plan
may count the arm as already there. `VirtualTarget` integrates the model's deltas on a target
of its own instead, so residuals below the tolerance add up, and re-anchors that target to the
measured pose only when the tracking error exceeds a bound or a plan fails.

The default bounds are twice CuRobo's goal tolerance (5 mm, 0.05 rad). They are provisional:
plan 3.4 sizes them from the #36 probe of what CuRobo does with 1-20 mm targets.

Deltas are applied as robosuite applies OSC deltas: the translation added to the position, the
rotation multiplied on the left in the world frame. Positions are the tool centre point's; the
adapter converts to and from the flange (`frames.tcp_from_flange`).

`StallDetector` flags calls where the tool was told to move and did not: across `window` calls
it moved less than `min_motion_m` while the commanded motion added up to at least that. It
reports; what to do about a stall is the adapter's, and the count goes into `episode_info()`.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .rotations import relative_angle

# Twice CuRobo's goal tolerance (plan 3.4); provisional until sized from the #36 probe.
DEFAULT_MAX_POSITION_ERROR_M = 0.010
DEFAULT_MAX_ROTATION_ERROR_RAD = 0.10
# Provisional: ten calls without 2 mm of tool motion while at least 2 mm were commanded.
DEFAULT_STALL_WINDOW = 10
DEFAULT_STALL_MOTION_M = 0.002


class VirtualTarget:
    """A tool-centre target integrated from deltas, re-anchored only when tracking breaks down."""

    def __init__(
        self,
        max_position_error_m: float = DEFAULT_MAX_POSITION_ERROR_M,
        max_rotation_error_rad: float = DEFAULT_MAX_ROTATION_ERROR_RAD,
    ) -> None:
        if not (max_position_error_m > 0 and max_rotation_error_rad > 0):
            raise ValueError("the re-anchoring bounds must be positive")
        self.max_position_error_m = max_position_error_m
        self.max_rotation_error_rad = max_rotation_error_rad
        self.reset()

    def reset(self) -> None:
        """Forget the target and the re-anchor counts: a new episode anchors afresh."""
        self.position: np.ndarray | None = None
        self.rotation: np.ndarray | None = None
        self.reanchors = {"tracking": 0, "plan_failed": 0}

    def tracking_error(
        self, measured_position: np.ndarray, measured_rotation: np.ndarray
    ) -> tuple[float, float]:
        """(metres, radians) between the target and the measured pose."""
        if self.position is None or self.rotation is None:
            raise ValueError("no target yet: step() anchors the first one")
        distance = float(np.linalg.norm(self.position - np.asarray(measured_position)))
        return distance, float(relative_angle(self.rotation, measured_rotation))

    def step(
        self,
        measured_position: np.ndarray,
        measured_rotation: np.ndarray,
        delta_position: np.ndarray,
        delta_rotation: np.ndarray | None = None,
        plan_failed: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply one delta and return the new target (position, rotation matrix), copies.

        The target is re-anchored to the measured pose first if there is none yet, if the last
        call's plan failed, or if either tracking error exceeds its bound.
        """
        measured_position = np.asarray(measured_position, dtype=np.float64)
        measured_rotation = np.asarray(measured_rotation, dtype=np.float64)
        if self.position is None or self.rotation is None:
            self._anchor(measured_position, measured_rotation)
        elif plan_failed:
            self._anchor(measured_position, measured_rotation)
            self.reanchors["plan_failed"] += 1
        else:
            distance, angle = self.tracking_error(measured_position, measured_rotation)
            if distance > self.max_position_error_m or angle > self.max_rotation_error_rad:
                self._anchor(measured_position, measured_rotation)
                self.reanchors["tracking"] += 1
        self.position = self.position + np.asarray(delta_position, dtype=np.float64)
        if delta_rotation is not None:
            self.rotation = np.asarray(delta_rotation, dtype=np.float64) @ self.rotation
        return self.position.copy(), self.rotation.copy()

    def info(self) -> dict[str, Any]:
        """JSON-ready, for `episode_info()`."""
        return {"reanchors": dict(self.reanchors)}

    def _anchor(self, position: np.ndarray, rotation: np.ndarray) -> None:
        self.position, self.rotation = position.copy(), rotation.copy()


class StallDetector:
    """Counts stalls: `window` calls in which the tool, told to move, did not."""

    def __init__(
        self, window: int = DEFAULT_STALL_WINDOW, min_motion_m: float = DEFAULT_STALL_MOTION_M
    ) -> None:
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window!r}")
        if not min_motion_m > 0:
            raise ValueError(f"min_motion_m must be positive, got {min_motion_m!r}")
        self.window = window
        self.min_motion_m = min_motion_m
        self.reset()

    def reset(self) -> None:
        self._positions: deque[np.ndarray] = deque(maxlen=self.window + 1)
        self._commanded: deque[float] = deque(maxlen=self.window)
        self.stalled = False
        self.events = 0

    def update(self, position: np.ndarray, commanded_m: float = 0.0) -> bool:
        """Record the tool's measured position and the motion commanded since the last call.

        Returns whether the tool is stalled now. Consecutive stalled calls are one event.
        """
        self._positions.append(np.asarray(position, dtype=np.float64))
        if len(self._positions) > 1:
            self._commanded.append(float(commanded_m))
        stalled = False
        if len(self._positions) == self.window + 1:
            start = self._positions[0]
            moved = max(float(np.linalg.norm(p - start)) for p in self._positions)
            stalled = moved < self.min_motion_m <= sum(self._commanded)
        if stalled and not self.stalled:
            self.events += 1
        self.stalled = stalled
        return stalled

    def info(self) -> dict[str, Any]:
        return {"stall_events": self.events}
