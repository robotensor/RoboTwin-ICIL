"""Which arm a single-arm model drives, and how the other one keeps still (plan 3.3).

RoboTwin's robot is bimanual everywhere, while BPP and UniSkill as released drive one arm. All
nine V1 experts pick the arm by the object's x over symmetric spawn ranges; `stack_blocks_two`
and `stack_bowls_two` use both when the objects fall on opposite sides.

**Arm choice.** The arm whose tool centre travels further over the demonstration. Paths within
`tie_m` of each other are a tie, broken by whichever arm's tool centre first moves more than
`move_m` from where it started; if neither ever does, or both first do so at the same time, the
left arm, the arm RoboTwin's V1 experts give an object at x = 0 (`envs/click_bell.py`:
`"right" if ... p[0] > 0 else "left"`). The choice reads only the demonstration's end-effector
poses and times, which a real robot observes; nothing privileged. The chosen arm and the
demonstration's arm set go into an adapter's `episode_info()` (#37) through `ArmChoice.info()`.

**Idle arm.** Held for the whole episode at a fixed target taken from the episode's first
observation: its joint targets for `qpos` actions, its flange pose for `ee` actions, and its
commanded gripper either way (open at the start, which six V1 success checks require). Fixed
rather than echoing the current pose: every successful `ee` plan re-bases on the measured joints
within CuRobo's tolerance, which could let a loaded arm creep (plan 3.3; the #36 simulator test
measures both holds, and the fixed one stays under 1e-3 rad).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robotwin_icil.demo import ARMS, EE_POSE_DIM, Demonstration

from .frames import EE_SLICES, QPOS_SLICES, check_arm, other_arm, tcp_from_flange

# Provisional thresholds, not yet measured against the jitter of an arm the expert holds still.
DEFAULT_TIE_M = 0.01
DEFAULT_MOVE_M = 0.005


@dataclass(frozen=True)
class ArmChoice:
    """The arm a single-arm adapter drives for one demonstration, and why."""

    arm: str
    rule: str  # "path", "first_move" or "default": which step of the rule decided
    path_m: dict[str, float]  # each arm's tool-centre path length
    first_move_s: dict[str, float | None]  # when each tool centre first left its start by move_m
    moved: tuple[str, ...]  # `Demonstration.arms_moved()`: the demonstration's arm set

    @property
    def idle(self) -> str:
        return other_arm(self.arm)

    def info(self) -> dict[str, Any]:
        """JSON-ready, for `episode_info()`."""
        return {
            "active_arm": self.arm,
            "arm_rule": self.rule,
            "tcp_path_m": {arm: round(self.path_m[arm], 6) for arm in ARMS},
            "first_move_s": self.first_move_s,
            "demonstration_arms": list(self.moved),
        }


def tcp_positions(demonstration: Demonstration) -> dict[str, np.ndarray]:
    """arm -> (T, 3) the tool centre point's world position in every frame."""
    poses = demonstration.endposes()
    return {
        arm: tcp_from_flange(poses[:, EE_SLICES[arm].start : EE_SLICES[arm].start + EE_POSE_DIM])[
            :, :3
        ]
        for arm in ARMS
    }


def choose_arm(
    demonstration: Demonstration, tie_m: float = DEFAULT_TIE_M, move_m: float = DEFAULT_MOVE_M
) -> ArmChoice:
    """The arm whose tool centre travels further; see the module docstring for ties."""
    times = demonstration.times()
    paths, first_move = {}, {}
    for arm, tcp in tcp_positions(demonstration).items():
        paths[arm] = float(np.linalg.norm(np.diff(tcp, axis=0), axis=1).sum())
        away = np.flatnonzero(np.linalg.norm(tcp - tcp[0], axis=1) > move_m)
        first_move[arm] = float(times[away[0]]) if away.size else None

    left, right = ARMS
    if abs(paths[left] - paths[right]) > tie_m:
        arm, rule = max(ARMS, key=paths.__getitem__), "path"
    elif first_move[left] != first_move[right] and (first_move[left], first_move[right]) != (
        None,
        None,
    ):
        started = {a: t for a, t in first_move.items() if t is not None}
        arm, rule = min(started, key=started.__getitem__), "first_move"
    else:
        arm, rule = left, "default"
    return ArmChoice(
        arm=arm, rule=rule, path_m=paths, first_move_s=first_move, moved=demonstration.arms_moved()
    )


@dataclass(frozen=True)
class IdleArmHold:
    """The idle arm's fixed target for a whole episode; see the module docstring."""

    arm: str
    qpos: np.ndarray  # (7,) joint targets then gripper, from the first observation's qpos
    ee: np.ndarray | None  # (8,) flange pose then gripper, None if it reported no endpose

    @classmethod
    def from_observation(cls, observation: Any, arm: str) -> IdleArmHold:
        """From the episode's first `Observation` (or a demonstration `Frame`: same fields)."""
        check_arm(arm)
        qpos = np.asarray(observation.qpos, dtype=np.float64)[QPOS_SLICES[arm]].copy()
        endpose = observation.endpose or {}
        ee = None
        if f"{arm}_endpose" in endpose and f"{arm}_gripper" in endpose:
            pose = np.asarray(endpose[f"{arm}_endpose"], dtype=np.float64)
            if pose.shape != (EE_POSE_DIM,):
                raise ValueError(f"{arm}_endpose has shape {pose.shape}, expected ({EE_POSE_DIM},)")
            ee = np.concatenate([pose, [float(endpose[f"{arm}_gripper"])]])
        return cls(arm=arm, qpos=qpos, ee=ee)

    def apply_qpos(self, actions: np.ndarray) -> np.ndarray:
        """(..., 14) qpos actions with the idle arm's slots set to the hold; a copy."""
        return _apply(actions, 14, QPOS_SLICES[self.arm], self.qpos)

    def apply_ee(self, actions: np.ndarray) -> np.ndarray:
        """(..., 16) `ee` actions with the idle arm's slots set to the hold; a copy."""
        if self.ee is None:
            raise ValueError(f"no endpose for the {self.arm} arm in the first observation")
        return _apply(actions, 2 * (EE_POSE_DIM + 1), EE_SLICES[self.arm], self.ee)


def _apply(actions: np.ndarray, width: int, slots: slice, values: np.ndarray) -> np.ndarray:
    out = np.array(actions, dtype=np.float64)
    if out.ndim == 0 or out.shape[-1] != width:
        raise ValueError(f"expected actions of shape (..., {width}), got {out.shape}")
    out[..., slots] = values
    return out
