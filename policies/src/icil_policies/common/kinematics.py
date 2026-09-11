"""Forward and inverse kinematics from a URDF in numpy, and aloha-agilex's arms in RoboTwin's terms.

Used only by an adapter's `qpos_ik` execution mode (plan 3.4), which turns end-effector targets
into joint targets for `take_action('qpos')`, the path the replay oracle validates. RoboTwin
already owns aloha's kinematics (SAPIEN, CuRobo), so this duplicates it on purpose, small and
dependency-free, and a simulator test (`policies/tests/sim`) checks it against RoboTwin's own
endposes. `qpos_ik` is the fallback mode for that reason.

The URDF is read with `xml.etree`: each joint's origin (xyz, then roll-pitch-yaw about fixed
axes), axis, type and limits, along the chain from a root link to a tip link. A child link's
frame is `parent @ origin @ motion(q)`, the motion a rotation about the joint axis (revolute,
continuous) or a translation along it (prismatic). That is URDF's convention, and SAPIEN's
loader keeps it: it aligns each joint's axis with x through a pose applied on both the parent and
the child side of the joint, so the link frame stays the URDF child frame
(`sapien/wrapper/urdf_loader.py`, `t_axis2parent` and `t_axis2joint`).

Inverse kinematics is Levenberg-Marquardt on the geometric Jacobian with Sugihara's damping,
which shrinks with the error, so it keeps converging near the wrist's singular configurations
where a fixed damping stalls; joint limits are clipped after every step.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frames import ROBOT_ROOT_POSITION, TCP_OFFSET_M, check_arm, pose_to_matrix, root_rotation
from .rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quat

MOVABLE = ("revolute", "continuous", "prismatic")


class KinematicsError(ValueError):
    """A URDF or chain the kinematics cannot use."""


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str  # revolute, continuous, prismatic, fixed, floating or planar
    parent: str
    child: str
    origin: np.ndarray  # (4, 4) the joint frame in the parent link's frame
    axis: np.ndarray  # (3,) unit, in the joint frame
    lower: float
    upper: float


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    converged: bool
    iterations: int
    position_error_m: float
    rotation_error_rad: float


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF's rpy: roll about x, then pitch about y, then yaw about z, all fixed axes."""
    return (
        axis_angle_to_matrix([0.0, 0.0, yaw])
        @ axis_angle_to_matrix([0.0, pitch, 0.0])
        @ axis_angle_to_matrix([roll, 0.0, 0.0])
    )


def parse_urdf(text: str) -> dict[str, Joint]:
    """Every joint of a URDF document, keyed by its child link (a link has one parent joint)."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise KinematicsError(f"not a URDF document: {exc}") from None
    joints: dict[str, Joint] = {}
    for element in root.iter("joint"):
        parent, child = element.find("parent"), element.find("child")
        if parent is None or child is None:
            continue  # a transmission's <joint name=...> reference, not a joint
        name = element.get("name", "")
        origin = element.find("origin")
        xyz = _floats(origin.get("xyz") if origin is not None else None, 3, f"joint {name} xyz")
        rpy = _floats(origin.get("rpy") if origin is not None else None, 3, f"joint {name} rpy")
        transform = np.eye(4)
        transform[:3, :3] = rpy_matrix(*rpy)
        transform[:3, 3] = xyz
        axis_element = element.find("axis")
        axis = _floats(
            axis_element.get("xyz") if axis_element is not None else "1 0 0",
            3,
            f"joint {name} axis",
        )
        norm = np.linalg.norm(axis)
        limit = element.find("limit")
        joints[child.get("link", "")] = Joint(
            name=name,
            kind=element.get("type", ""),
            parent=parent.get("link", ""),
            child=child.get("link", ""),
            origin=transform,
            axis=axis / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0]),
            lower=float(limit.get("lower", -np.inf)) if limit is not None else -np.inf,
            upper=float(limit.get("upper", np.inf)) if limit is not None else np.inf,
        )
    return joints


def load_urdf(path: str | Path) -> dict[str, Joint]:
    return parse_urdf(Path(path).read_text(encoding="utf-8"))


class Chain:
    """The joints from a root link to a tip link; `q` holds one value per movable joint."""

    def __init__(self, joints: Sequence[Joint]) -> None:
        for joint in joints:
            if joint.kind not in MOVABLE + ("fixed",):
                raise KinematicsError(f"joint {joint.name!r} is {joint.kind!r}, not supported")
        self.joints = tuple(joints)
        movable = [j for j in self.joints if j.kind in MOVABLE]
        self.names = tuple(j.name for j in movable)
        continuous = np.array([j.kind == "continuous" for j in movable], dtype=bool)
        self.lower = np.where(continuous, -np.inf, [j.lower for j in movable])
        self.upper = np.where(continuous, np.inf, [j.upper for j in movable])

    @classmethod
    def from_urdf(cls, joints: Mapping[str, Joint], root: str, tip: str) -> Chain:
        path, link = [], tip
        while link != root:
            joint = joints.get(link)
            if joint is None or len(path) > len(joints):
                raise KinematicsError(f"no chain of joints leads from link {root!r} to {tip!r}")
            path.append(joint)
            link = joint.parent
        return cls(path[::-1])

    def __len__(self) -> int:
        return len(self.names)

    def forward(self, q: np.ndarray) -> np.ndarray:
        """(4, 4) the tip link's frame in the root link's frame."""
        return self._walk(self._q(q))[0]

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """(6, n) geometric Jacobian at the tip's origin, in the root frame: linear rows first."""
        tip, axes = self._walk(self._q(q))
        return _jacobian(tip, axes)

    def inverse(
        self,
        target: np.ndarray,
        q0: np.ndarray,
        max_iterations: int = 200,
        position_tolerance_m: float = 1e-6,
        rotation_tolerance_rad: float = 1e-5,
        min_damping: float = 1e-6,
        max_step_rad: float = 0.2,
    ) -> IKResult:
        """Joint values placing the tip at `target` (4, 4, root frame), from the seed `q0`.

        Levenberg-Marquardt, `(J.T J + (e.e / 2 + min_damping) I) dq = J.T e`, with e the
        position error and the rotation error as an axis-angle, both in the root frame
        (T. Sugihara, "Solvability-unconcerned inverse kinematics by the Levenberg-Marquardt
        method", IEEE T-RO 27(5), 2011). Each step is at most `max_step_rad` long, then clipped
        to the joint limits. Not converging is reported, not raised: the caller decides what an
        unreachable target means.
        """
        target = np.asarray(target, dtype=np.float64)
        q = np.clip(self._q(q0), self.lower, self.upper)
        for iteration in range(max_iterations + 1):
            tip, axes = self._walk(q)
            error = np.concatenate(
                [target[:3, 3] - tip[:3, 3], matrix_to_axis_angle(target[:3, :3] @ tip[:3, :3].T)]
            )
            position_error, rotation_error = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
            if position_error <= position_tolerance_m and rotation_error <= rotation_tolerance_rad:
                return IKResult(q, True, iteration, float(position_error), float(rotation_error))
            if iteration == max_iterations:
                break
            jacobian = _jacobian(tip, axes)
            damping = 0.5 * float(error @ error) + min_damping
            step = np.linalg.solve(
                jacobian.T @ jacobian + damping * np.eye(len(q)), jacobian.T @ error
            )
            length = np.linalg.norm(step)
            if length > max_step_rad:
                step *= max_step_rad / length
            q = np.clip(q + step, self.lower, self.upper)
        return IKResult(q, False, max_iterations, float(position_error), float(rotation_error))

    def _q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (len(self),):
            raise KinematicsError(f"expected {len(self)} joint values {self.names}, got {q.shape}")
        return q

    def _walk(self, q: np.ndarray) -> tuple[np.ndarray, list[tuple[str, np.ndarray, np.ndarray]]]:
        """The tip frame, and each movable joint's (kind, world axis, world origin)."""
        frame, axes, i = np.eye(4), [], 0
        for joint in self.joints:
            frame = frame @ joint.origin
            if joint.kind not in MOVABLE:
                continue
            axis = frame[:3, :3] @ joint.axis
            axes.append((joint.kind, axis, frame[:3, 3].copy()))
            motion = np.eye(4)
            if joint.kind == "prismatic":
                motion[:3, 3] = joint.axis * q[i]
            else:
                motion[:3, :3] = axis_angle_to_matrix(joint.axis * q[i])
            frame = frame @ motion
            i += 1
        return frame, axes


def _jacobian(tip: np.ndarray, axes: list[tuple[str, np.ndarray, np.ndarray]]) -> np.ndarray:
    columns = []
    for kind, axis, origin in axes:
        if kind == "prismatic":
            columns.append(np.concatenate([axis, np.zeros(3)]))
        else:
            columns.append(np.concatenate([np.cross(axis, tip[:3, 3] - origin), axis]))
    return np.stack(columns, axis=1)


def sapien_joint_anchor(axis: np.ndarray) -> np.ndarray:
    """(3, 3) the rotation SAPIEN's URDF loader puts between a link and its parent joint's frame.

    SAPIEN's joints move along their own x axis, so the loader builds a frame whose x is the
    URDF axis and hands it to the joint on both sides (`sapien/wrapper/urdf_loader.py`,
    `t_axis2joint`). A joint's `global_pose` is its child link's pose times this rotation. Built
    here exactly as the loader builds it.
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    axis = np.array([1.0, 0.0, 0.0]) if norm < 1e-3 else axis / norm
    other = [0.0, 0.0, 1.0] if abs(axis @ [1.0, 0.0, 0.0]) > 0.9 else [1.0, 0.0, 0.0]
    second = np.cross(axis, other)
    second /= np.linalg.norm(second)
    return np.stack([axis, second, np.cross(axis, second)], axis=1)


# aloha-agilex, from `assets/embodiments/aloha-agilex/config.yml` (CONFIG) and its URDF.
# The URDF's root link, placed at CONFIG `robot_pose` (`frames.ROBOT_ROOT_*`).
ALOHA_ROOT_LINK = "footprint"
# CONFIG `move_group`: the last link of each arm.
ALOHA_TIP_LINKS = {"left": "fl_link6", "right": "fr_link6"}
# CONFIG `ee_joints`: RoboTwin reads the endpose off this joint's `global_pose`, not the link's.
ALOHA_EE_JOINTS = {"left": "fl_joint6", "right": "fr_joint6"}
# CONFIG `arm_joints_name`.
ALOHA_ARM_JOINTS = {
    "left": tuple(f"fl_joint{i}" for i in range(1, 7)),
    "right": tuple(f"fr_joint{i}" for i in range(1, 7)),
}
# CONFIG `global_trans_matrix` and `delta_matrix`: `envs/robot/robot.py` `_trans_endpose` turns
# the ee joint's frame by R_joint @ global_trans_matrix @ delta_matrix and moves it by
# `gripper_bias - 0.12` (zero: CONFIG `gripper_bias: 0.12`) along the turned +x. The joint's
# frame is the link's turned by SAPIEN's anchor, diag(1, -1, -1) for joint 6's x axis, which
# global_trans_matrix undoes: the endpose's orientation is `fl_link6`'s own.
ALOHA_GLOBAL_TRANS_MATRIX = np.diag([1.0, -1.0, -1.0])
ALOHA_DELTA_MATRIX = np.eye(3)
ALOHA_ENDPOSE_OFFSET_M = TCP_OFFSET_M - 0.12


class AlohaArm:
    """One aloha-agilex arm: its 6 joints to and from RoboTwin's world-frame endpose.

    The endpose is `_trans_endpose` of the ee joint's global pose: the tip link's frame, turned
    by SAPIEN's joint anchor, then by `global_trans_matrix @ delta_matrix`.
    """

    def __init__(self, urdf_path: str | Path, arm: str) -> None:
        self.arm = check_arm(arm)
        self.chain = Chain.from_urdf(load_urdf(urdf_path), ALOHA_ROOT_LINK, ALOHA_TIP_LINKS[arm])
        if self.chain.names != ALOHA_ARM_JOINTS[arm]:
            raise KinematicsError(
                f"{urdf_path}: the {arm} arm's joints are {self.chain.names}, "
                f"expected {ALOHA_ARM_JOINTS[arm]}"
            )
        ee_joint = self.chain.joints[-1]
        if ee_joint.name != ALOHA_EE_JOINTS[arm]:
            raise KinematicsError(
                f"{urdf_path}: the {arm} arm ends in joint {ee_joint.name!r}, "
                f"expected {ALOHA_EE_JOINTS[arm]!r}"
            )
        self._root = np.eye(4)
        self._root[:3, :3] = root_rotation()
        self._root[:3, 3] = ROBOT_ROOT_POSITION
        self._turn = (
            sapien_joint_anchor(ee_joint.axis) @ ALOHA_GLOBAL_TRANS_MATRIX @ ALOHA_DELTA_MATRIX
        )

    def link_frame(self, q: np.ndarray) -> np.ndarray:
        """(4, 4) the arm's last link (`f?_link6`) in the world."""
        return self._root @ self.chain.forward(q)

    def endpose(self, q: np.ndarray) -> np.ndarray:
        """(7,) `[x, y, z, qw, qx, qy, qz]` as RoboTwin's `get_obs()` reports this arm's endpose."""
        link = self.link_frame(q)
        rotation = link[:3, :3] @ self._turn
        position = link[:3, 3] + rotation[:, 0] * ALOHA_ENDPOSE_OFFSET_M
        return np.concatenate([position, matrix_to_quat(rotation)])

    def inverse(self, endpose: np.ndarray, q0: np.ndarray, **options) -> IKResult:
        """Joint values reaching a world-frame endpose, from the seed `q0` (the measured joints).

        `options` go to `Chain.inverse`.
        """
        target = pose_to_matrix(endpose)
        link = np.eye(4)
        link[:3, :3] = target[:3, :3] @ self._turn.T
        link[:3, 3] = target[:3, 3] - target[:3, 0] * ALOHA_ENDPOSE_OFFSET_M
        in_root = np.linalg.inv(self._root) @ link
        return self.chain.inverse(in_root, q0, **options)


def _floats(text: str | None, count: int, where: str) -> np.ndarray:
    if text is None:
        return np.zeros(count)
    try:
        values = np.array([float(v) for v in text.split()])
    except ValueError:
        raise KinematicsError(f"{where}: cannot read {text!r}") from None
    if values.shape != (count,):
        raise KinematicsError(f"{where}: expected {count} numbers, got {text!r}")
    return values
