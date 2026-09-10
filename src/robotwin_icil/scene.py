"""Same Scene verification: fingerprint an initial state, and compare two of them.

Scene generation is deterministic in (task, seed, config) because RoboTwin seeds numpy and torch
before it builds anything. The whole metric rests on that, so it is checked on every episode
rather than trusted: the demonstration's initial scene and the evaluation's initial scene are
fingerprinted right after `setup_demo`, before anyone acts, and compared here.

Pure: the fingerprint is plain arrays, so the comparison is tested without a simulator. Reading a
fingerprint off a live env is `robotwin.fingerprint`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Tight enough that a different object placement cannot pass, loose enough for float noise when
# the same scene is rebuilt. Same-seed rebuilds on one machine are expected to be bit-identical;
# the tolerance only exists so a platform's last-ulp differences are not reported as drift.
POSITION_TOL_M = 1e-5
ROTATION_TOL_RAD = 1e-4
JOINT_TOL = 1e-5
CAMERA_TOL = 1e-5


@dataclass(frozen=True)
class SceneFingerprint:
    """Everything that makes one initial scene this scene and not another.

    Poses are (7,) arrays of position then wxyz quaternion. Names with duplicates in a scene are
    disambiguated by scene order (`name#1`, `name#2`), which is itself deterministic.
    """

    actors: dict[str, np.ndarray]
    articulations: dict[str, np.ndarray]
    articulation_roots: dict[str, np.ndarray]
    cameras: dict[str, np.ndarray]
    robot_qpos: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Mismatch:
    """One way two fingerprints differ: which part, which entity, and by how much."""

    part: str
    name: str
    detail: str
    error: float = float("inf")

    def __str__(self) -> str:
        return f"{self.part}[{self.name}]: {self.detail}"


def rotation_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angle between two unit quaternions, in radians. q and -q are the same rotation."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    dot = abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
    return 2.0 * float(np.arccos(min(1.0, dot)))


def _compare_names(part: str, a: dict, b: dict) -> list[Mismatch]:
    found = []
    for name in sorted(set(a) - set(b)):
        found.append(Mismatch(part, name, "present in the demonstration scene only"))
    for name in sorted(set(b) - set(a)):
        found.append(Mismatch(part, name, "present in the evaluation scene only"))
    return found


def _compare_poses(
    part: str, a: dict, b: dict, position_tol: float, rotation_tol: float
) -> list[Mismatch]:
    found = _compare_names(part, a, b)
    for name in sorted(set(a) & set(b)):
        pa, pb = np.asarray(a[name], dtype=np.float64), np.asarray(b[name], dtype=np.float64)
        offset = float(np.linalg.norm(pa[:3] - pb[:3]))
        if offset > position_tol:
            found.append(Mismatch(part, name, f"position differs by {offset:.3e} m", offset))
        angle = rotation_angle(pa[3:7], pb[3:7])
        if angle > rotation_tol:
            found.append(Mismatch(part, name, f"rotation differs by {angle:.3e} rad", angle))
    return found


def _compare_arrays(part: str, a: dict, b: dict, tol: float) -> list[Mismatch]:
    found = _compare_names(part, a, b)
    for name in sorted(set(a) & set(b)):
        va, vb = np.asarray(a[name], dtype=np.float64), np.asarray(b[name], dtype=np.float64)
        if va.shape != vb.shape:
            found.append(Mismatch(part, name, f"shape {va.shape} vs {vb.shape}"))
            continue
        error = float(np.max(np.abs(va - vb))) if va.size else 0.0
        if error > tol:
            found.append(Mismatch(part, name, f"differs by up to {error:.3e}", error))
    return found


def compare(
    demonstration: SceneFingerprint,
    evaluation: SceneFingerprint,
    *,
    position_tol: float = POSITION_TOL_M,
    rotation_tol: float = ROTATION_TOL_RAD,
    joint_tol: float = JOINT_TOL,
    camera_tol: float = CAMERA_TOL,
) -> list[Mismatch]:
    """Every difference between two initial scenes; empty means they are the Same Scene."""
    found: list[Mismatch] = []
    found += _compare_poses(
        "actor", demonstration.actors, evaluation.actors, position_tol, rotation_tol
    )
    found += _compare_poses(
        "articulation_root",
        demonstration.articulation_roots,
        evaluation.articulation_roots,
        position_tol,
        rotation_tol,
    )
    found += _compare_arrays(
        "articulation", demonstration.articulations, evaluation.articulations, joint_tol
    )
    found += _compare_arrays("camera", demonstration.cameras, evaluation.cameras, camera_tol)
    found += _compare_arrays(
        "robot", {"qpos": demonstration.robot_qpos}, {"qpos": evaluation.robot_qpos}, joint_tol
    )
    for key in sorted(set(demonstration.extras) | set(evaluation.extras)):
        if demonstration.extras.get(key) != evaluation.extras.get(key):
            found.append(
                Mismatch(
                    "extra",
                    key,
                    f"{demonstration.extras.get(key)!r} vs {evaluation.extras.get(key)!r}",
                )
            )
    return found


def max_error(mismatches: list[Mismatch]) -> float:
    """The largest measured deviation, for the episode record; 0.0 for an exact match."""
    return max((m.error for m in mismatches), default=0.0)


def unique_names(names: list[str]) -> list[str]:
    """Disambiguate repeated entity names by scene order, so fingerprints key on something stable."""
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    seen: dict[str, int] = {}
    keyed = []
    for name in names:
        if counts[name] == 1:
            keyed.append(name)
            continue
        seen[name] = seen.get(name, 0) + 1
        keyed.append(f"{name}#{seen[name]}")
    return keyed
