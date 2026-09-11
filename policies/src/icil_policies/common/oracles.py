"""The conversion oracle for models that act in resampled qpos (plan A3, B2).

BPP-RoboTwin and UniSkill-RoboTwin are trained at a fixed control rate, 20 Hz, on the
demonstration resampled by time (`resample`), and predict each step's next absolute qpos. Their
target convention is `resampled_qpos_actions`: one action per resampled step, the next step's
qpos, the final step holding its own. One action per resampled step keeps a model's step count
aligned with anything else indexed by resampled step, such as UniSkill's skill rows.

`ResampledQposReplay` plays those targets back through `take_action('qpos')`, one per call,
holding the last. It is what a perfect imitator of the resampled trajectory scores, so it is the
ceiling of this conversion for any such model. Where it falls below the native `replay`, which
plays every recorded frame, the loss is in the resampling: fewer and coarser targets, linearly
interpolated arm joints, gripper commands held between samples.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from robotwin_icil.demo import Demonstration
from robotwin_icil.policy import ReplayPolicy

from .resample import DEFAULT_RATE_HZ, resample

ADAPTER_VERSION = "1"


def next_targets(states: np.ndarray) -> np.ndarray:
    """(N, d): row i is state i + 1, and the final row holds the final state."""
    states = np.asarray(states)
    if len(states) == 0:
        raise ValueError("no states to take targets from")
    return np.concatenate([states[1:], states[-1:]])


def resampled_qpos_actions(
    demonstration: Demonstration, rate_hz: float = DEFAULT_RATE_HZ
) -> np.ndarray:
    """(N, 14) the qpos target of each of the demonstration's N steps at `rate_hz`."""
    return next_targets(resample(demonstration, rate_hz).qpos)


class ResampledQposReplay(ReplayPolicy):
    """Plays the demonstration's resampled qpos targets back, one per call, then holds the last.

    Numpy only, so it runs in the simulator env:
    `--policy icil_policies.common.oracles:ResampledQposReplay`.
    """

    name = "resampled_qpos_replay"
    action_type = "qpos"

    def __init__(self, rate_hz: float = DEFAULT_RATE_HZ) -> None:
        super().__init__()
        rate_hz = float(rate_hz)
        if not (math.isfinite(rate_hz) and rate_hz > 0):
            raise ValueError(f"rate_hz must be positive and finite, got {rate_hz}")
        self.rate_hz = rate_hz

    def _played(self, demonstration: Demonstration) -> np.ndarray:
        return resampled_qpos_actions(demonstration, self.rate_hz)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "adapter": self.name,
            "adapter_version": ADAPTER_VERSION,
            "rate_hz": self.rate_hz,
            "checkpoint": None,
            "training_tasks": [],
        }

    def episode_info(self) -> dict[str, Any]:
        """The resampled steps and how many calls played them: past the end, calls hold."""
        steps = 0 if self._actions is None else len(self._actions)
        return {"resampled_steps": steps, "calls": self._cursor}
