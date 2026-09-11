"""Rotations as numpy arrays: wxyz quaternions, matrices, axis-angle and rot6d.

Conventions, fixed here once for every adapter:

- Quaternions are scalar-first, `[w, x, y, z]`, as RoboTwin reports them (`endpose`, through
  transforms3d). q and -q are one rotation; a function that returns a quaternion returns the one
  with w >= 0, as transforms3d's `mat2quat` does, unless it says otherwise. The runner's xyzw
  reorder would corrupt RoboTwin's quaternions (plan 3.4, Rotations).
- Matrices rotate column vectors: `v_world = R @ v_body`.
- Axis-angle is a 3-vector, the unit axis times the angle in radians, the angle in [0, pi].
- rot6d is the first two ROWS of the matrix, `[R00, R01, R02, R10, R11, R12]`: BPP's
  convention (plan 3.4; `behavior_prompting/common/pose_util.py` `mat_to_rot6d`, and
  pytorch3d's `matrix_to_rotation_6d` in `train_network/model/common/rotation_conversions.py`).
  The way back is Gram-Schmidt on those rows, as both do.

Every function takes any leading batch shape and returns float64.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12
# Below this angle (radians), sin(x)/x style ratios switch to their Taylor expansions.
_SMALL = 1e-8


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """q scaled to unit norm; a zero quaternion is an error, not a rotation."""
    q = _vectors(q, 4)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < _EPS):
        raise ValueError("a zero quaternion is not a rotation")
    return q / norm


def canonical_quat(q: np.ndarray) -> np.ndarray:
    """The unit quaternion of q's rotation with w >= 0 (of q and -q, the one transforms3d gives)."""
    q = normalize_quat(q)
    return np.where(q[..., :1] < 0, -q, q)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3). q need not be unit; it is normalized first."""
    w, x, y, z = np.moveaxis(normalize_quat(q), -1, 0)
    rows = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) wxyz with w >= 0.

    Shepperd's method: of 4w², 4x², 4y² and 4z², read from the diagonal, the largest picks the
    division, so the result stays exact for rotations near 180 degrees, where w is near 0.
    """
    m = _matrices(matrix)
    r00, r11, r22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    squares = np.stack(
        [1 + r00 + r11 + r22, 1 + r00 - r11 - r22, 1 - r00 + r11 - r22, 1 - r00 - r11 + r22],
        axis=-1,
    )
    s = 2 * np.sqrt(np.maximum(squares, _EPS))  # 4w, 4x, 4y, 4z in the four cases
    d21, d02, d10 = (
        m[..., 2, 1] - m[..., 1, 2],
        m[..., 0, 2] - m[..., 2, 0],
        m[..., 1, 0] - m[..., 0, 1],
    )
    s01, s02, s12 = (
        m[..., 0, 1] + m[..., 1, 0],
        m[..., 0, 2] + m[..., 2, 0],
        m[..., 1, 2] + m[..., 2, 1],
    )
    cases = np.stack(
        [
            np.stack([s[..., 0] / 4, d21 / s[..., 0], d02 / s[..., 0], d10 / s[..., 0]], axis=-1),
            np.stack([d21 / s[..., 1], s[..., 1] / 4, s01 / s[..., 1], s02 / s[..., 1]], axis=-1),
            np.stack([d02 / s[..., 2], s01 / s[..., 2], s[..., 2] / 4, s12 / s[..., 2]], axis=-1),
            np.stack([d10 / s[..., 3], s02 / s[..., 3], s12 / s[..., 3], s[..., 3] / 4], axis=-1),
        ],
        axis=-2,
    )
    pick = np.argmax(squares, axis=-1)[..., None, None]
    return canonical_quat(np.take_along_axis(cases, pick, axis=-2)[..., 0, :])


def axis_angle_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """(..., 3) axis times angle -> (..., 4) wxyz, w >= 0 for angles up to pi."""
    v = _vectors(rotvec, 3)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    safe = np.where(angle > _SMALL, angle, 1.0)
    # sin(angle / 2) / angle, which tends to 1/2 - angle² / 48 at 0.
    scale = np.where(angle > _SMALL, np.sin(angle / 2) / safe, 0.5 - angle**2 / 48)
    return np.concatenate([np.cos(angle / 2), v * scale], axis=-1)


def quat_to_axis_angle(q: np.ndarray, canonical: bool = True) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3) axis times angle.

    With `canonical`, q is first turned to w >= 0, so the angle is in [0, pi]. Without it, q's
    own sign is kept and the angle is 2 acos(w), in [0, 2 pi]: robosuite's `quat2axisangle`,
    which LIBERO's recorded `ee_ori` went through, does that, so matching recorded data can
    need it.
    """
    q = canonical_quat(q) if canonical else normalize_quat(q)
    w, xyz = q[..., :1], q[..., 1:]
    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(sin_half, w)
    safe = np.where(sin_half > _SMALL, sin_half, 1.0)
    # angle / sin(angle / 2) tends to 2 / w as the rotation vanishes (w -> +-1).
    scale = np.where(sin_half > _SMALL, angle / safe, 2 / np.where(w != 0, w, 1.0))
    return xyz * scale


def axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """(..., 3) -> (..., 3, 3)."""
    return quat_to_matrix(axis_angle_to_quat(rotvec))


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 3), the angle in [0, pi]; at exactly pi either axis sign is right."""
    return quat_to_axis_angle(matrix_to_quat(matrix))


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): the first two rows, BPP's convention."""
    m = _matrices(matrix)
    return m[..., :2, :].reshape(m.shape[:-2] + (6,))


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3): Gram-Schmidt on the two rows, the third their cross product.

    Any 6 numbers with independent halves give a rotation, as a network's outputs need; norms
    are floored at 1e-12 as BPP's `normalize` floors them.
    """
    d6 = _vectors(rot6d, 6)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), _EPS)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), _EPS)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-2)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The Hamilton product a b: the rotation b, then a (`quat_to_matrix(a) @ quat_to_matrix(b)`)."""
    aw, ax, ay, az = np.moveaxis(_vectors(a, 4), -1, 0)
    bw, bx, by, bz = np.moveaxis(_vectors(b, 4), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """The inverse rotation of a unit quaternion."""
    return _vectors(q, 4) * np.array([1.0, -1.0, -1.0, -1.0])


def quat_angle(q: np.ndarray) -> np.ndarray:
    """(..., 4) -> (...,) the rotation angle of q, in [0, pi]."""
    q = canonical_quat(q)
    return 2 * np.arctan2(np.linalg.norm(q[..., 1:], axis=-1), q[..., 0])


def relative_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(..., 3, 3) twice -> (...,) the angle of the rotation taking a to b, in [0, pi]."""
    a, b = _matrices(a), _matrices(b)
    return quat_angle(matrix_to_quat(b @ np.swapaxes(a, -1, -2)))


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Spherical interpolation from q0 (t = 0) to q1 (t = 1) along the shorter arc, w >= 0.

    q1 is negated where it lies more than 90 degrees from q0 in quaternion space, so the path is
    the rotation's shortest one whichever sign q1 was given with. Nearly equal quaternions are
    interpolated linearly and renormalized, which is exact to within 1e-8.
    """
    q0, q1 = normalize_quat(q0), normalize_quat(q1)
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    close = sin_theta < 1e-6
    safe = np.where(close, 1.0, sin_theta)
    w0 = np.where(close, 1 - t, np.sin((1 - t) * theta) / safe)
    w1 = np.where(close, t, np.sin(t * theta) / safe)
    return canonical_quat(w0 * q0 + w1 * q1)


def _vectors(x: np.ndarray, size: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or x.shape[-1] != size:
        raise ValueError(f"expected shape (..., {size}), got {x.shape}")
    return x


def _matrices(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2 or x.shape[-2:] != (3, 3):
        raise ValueError(f"expected shape (..., 3, 3), got {x.shape}")
    return x
