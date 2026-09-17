"""Training trajectories from the released dataset, without privileged policy inputs.

Every trajectory is training data: the standard profile scores policies in simulation on
RoboTwin's own evaluation scenes (scene seeds from 100000), not on a stored split.

Archived rows are N transitions, not an N+1-frame materialized evaluation prompt. Physical
timestamps and terminal wrist images are unavailable: do not synthesize prompt.npz from them.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .policy import NEUTRAL_INSTRUCTION

TASKS = 50
EPISODES_PER_TASK = 50

CAMERAS = {
    "cam_head": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
JOINT_KEYS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)
POSE_KEYS = ("left_ee_poses", "left_ee_joint_states", "right_ee_poses", "right_ee_joint_states")


class DatasetError(ValueError):
    """The dataset is incomplete or violates the canonical representation."""


def dependencies():
    try:
        import h5py
        from PIL import Image
    except ImportError as exc:
        raise DatasetError('install dataset support: pip install -e ".[dataset]"') from exc
    return h5py, Image


def decode_rgb(encoded) -> np.ndarray:
    _, image = dependencies()
    with image.open(io.BytesIO(bytes(encoded))) as decoded:
        return np.asarray(decoded.convert("RGB"))


def packed(h5, group: str, keys: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([h5[f"{group}/{key}"][:] for key in keys], axis=1).astype(np.float32)


def inspect(h5) -> dict[str, Any]:
    try:
        qpos = packed(h5, "state", JOINT_KEYS)
        actions = packed(h5, "action", JOINT_KEYS)
        poses = packed(h5, "state", POSE_KEYS)
        action_poses = packed(h5, "action", POSE_KEYS)
        n = len(qpos)
        if n < 2 or qpos.shape != (n, 14) or actions.shape != (n, 14):
            raise DatasetError("expected at least two aligned 14-wide Aloha rows")
        if poses.shape != (n, 16) or action_poses.shape != (n, 16):
            raise DatasetError("expected aligned 16-wide end-effector pose rows")
        if not all(np.isfinite(a).all() for a in (qpos, actions, poses, action_poses)):
            raise DatasetError("non-finite state or action")
        if not np.array_equal(actions[:-1], qpos[1:]):
            raise DatasetError("actions are not the source's absolute next-row joint targets")
        for camera in CAMERAS:
            base = h5[f"vision/{camera}"]
            if len(base["colors"]) != n or tuple(base["shape"][()]) != (240, 320, 3):
                raise DatasetError(f"camera {camera}: row count or resolution mismatch")
            for name, shape in (("intrinsic_matrix", (n, 3, 3)), ("extrinsics_matrix", (n, 4, 4))):
                if base[name].shape != shape or not np.isfinite(base[name][:]).all():
                    raise DatasetError(f"camera {camera}: invalid {name}")
        return {"rows": n, "source_frequency_value": int(h5["additional_info/frequency"][()])}
    except (KeyError, ValueError, OSError) as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"unsupported source HDF5: {exc}") from exc


@dataclass(frozen=True)
class Trajectory:
    rgb: dict[str, np.ndarray]
    qpos: np.ndarray
    endpose: np.ndarray
    actions: np.ndarray
    action_endpose: np.ndarray
    source_frequency_value: int
    simulated_frame_times: None = None

    def context(self) -> dict[str, Any]:
        return {
            "rgb": self.rgb,
            "qpos": self.qpos,
            "endpose": self.endpose,
            "actions": self.actions,
            "action_endpose": self.action_endpose,
            "simulated_frame_times": None,
        }

    def query(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self.qpos):
            raise DatasetError("query index outside the trajectory")
        return {
            "rgb": {k: v[: index + 1] for k, v in self.rgb.items()},
            "qpos": self.qpos[: index + 1],
            "endpose": self.endpose[: index + 1],
            "instruction": NEUTRAL_INSTRUCTION,
        }


def load_trajectory(path: Path, cameras: tuple[str, ...] = tuple(CAMERAS.values())) -> Trajectory:
    h5py, _ = dependencies()
    if not set(cameras) <= set(CAMERAS.values()):
        raise DatasetError("only the three public RGB cameras are policy inputs")
    try:
        with h5py.File(path, "r") as h5:
            info = inspect(h5)
            images = {}
            for source, public in CAMERAS.items():
                if public in cameras:
                    frames = [decode_rgb(value) for value in h5[f"vision/{source}/colors"]]
                    if any(f.shape != (240, 320, 3) for f in frames):
                        raise DatasetError(f"{source}: decoded image resolution mismatch")
                    images[public] = np.stack(frames)
            return Trajectory(
                rgb=images,
                qpos=packed(h5, "state", JOINT_KEYS),
                endpose=packed(h5, "state", POSE_KEYS),
                actions=packed(h5, "action", JOINT_KEYS),
                action_endpose=packed(h5, "action", POSE_KEYS),
                source_frequency_value=info["source_frequency_value"],
            )
    except (OSError, KeyError) as exc:
        raise DatasetError(f"cannot load trajectory {path}: {exc}") from exc


class ReleasedDataset:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        try:
            with (self.root / "episodes.jsonl").open() as file:
                self.episodes = [json.loads(line) for line in file if line.strip()]
            ids = {(row["task"], row["episode_index"]) for row in self.episodes}
            tasks = {task for task, _ in ids}
            if len(ids) != len(self.episodes) or len(ids) != TASKS * EPISODES_PER_TASK:
                raise DatasetError("release must contain 50 unique episodes for each of 50 tasks")
            if len(tasks) != TASKS or any(not 0 <= index < EPISODES_PER_TASK for _, index in ids):
                raise DatasetError("release must index episodes 0-49 of each of 50 tasks")
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise DatasetError(f"cannot open release at {self.root}: {exc}") from exc

    def __len__(self) -> int:
        return len(self.episodes)

    def path(self, index: int) -> Path:
        if not -len(self.episodes) <= index < len(self.episodes):
            raise DatasetError("episode index outside the release")
        path = (self.root / self.episodes[index]["hdf5"]).resolve()
        if not path.is_relative_to(self.root):
            raise DatasetError("episode path escapes the dataset root")
        return path

    def __getitem__(self, index: int) -> Trajectory:
        return load_trajectory(self.path(index))

    def training_pair(self, context_index: int, query_index: int, row: int) -> dict[str, Any]:
        self.path(context_index)
        self.path(query_index)
        context = self.episodes[context_index]
        query = self.episodes[query_index]
        if context_index == query_index or context["task"] != query["task"]:
            raise DatasetError("use distinct context/query episodes from the same task")
        demonstration, target = self[context_index], self[query_index]
        inputs = {"demonstration": demonstration.context(), "observation": target.query(row)}
        return {"inputs": inputs, "target_actions": target.actions[row : row + 1]}

    def training_sample(self, index: int, row: int) -> dict[str, Any]:
        """Same-trajectory context plus a causal query prefix, a Same Scene training proxy.

        Future demonstration actions are intentionally context. The query branch itself never
        contains future observations or its supervised target action.
        """
        trajectory = self[index]
        return {
            "inputs": {"demonstration": trajectory.context(), "observation": trajectory.query(row)},
            "target_actions": trajectory.actions[row : row + 1],
        }
