"""A demonstration on disk: `prompt.npz`, the file both sides of a duel are handed.

A duel hands both policies the same demonstration, and a third party checks it by hash, so a
demonstration has to exist as bytes rather than only in memory. `write_prompt` and `read_prompt`
are that round trip: named arrays in one `npz`, plus a JSON string `meta`.

The arrays, grouped by the channel a consumer allows or drops as a unit:

| channel      | arrays                                                        |
| ------------ | ------------------------------------------------------------- |
| `video`      | `frames_<camera>`, one `(T, h, w, 3)` uint8 block per camera  |
| `proprio`    | `qpos` `(T, D)` float64, `endpose` `(T, 16)` float64          |
| `actions`    | `actions` `(T-1, D)` float64                                  |
| metadata     | `times` `(T,)` float64 seconds, `frequency` `()` float64      |
| privileged   | `meta`                                                        |

`CHANNELS` is the published map, in the shape the orchestrator reads: `video`'s entries are
prefixes (it cannot enumerate a benchmark's cameras), the others name arrays exactly. `times` and
`frequency` are in no modality channel — they say when a frame was taken, never what the robot did
— so every view keeps them. `meta` is in no channel at all: it holds the task, the scene seed and
the initial scene's fingerprint, and a policy never sees it.

`endpose` flattens RoboTwin's endpose dict once, here: per arm, left then right, the end-effector
pose `[x, y, z, qw, qx, qy, qz]` as RoboTwin's `get_arm_pose` reports it, then the gripper value,
16 wide — the layout `take_action` reads an `ee` action in.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .demo import Demonstration, DemonstrationError, Frame

PROMPT_FILE = "prompt.npz"
PROMPT_SCHEMA = 1

#: Arrays whose name starts with this are one camera's frames each.
FRAMES_PREFIX = "frames_"

#: channel -> the arrays carrying it. Entries under a channel in `PREFIX_CHANNELS` are prefixes.
CHANNELS: dict[str, tuple[str, ...]] = {
    "video": (FRAMES_PREFIX,),
    "proprio": ("qpos", "endpose"),
    "actions": ("actions",),
}
PREFIX_CHANNELS = ("video",)
#: Timing, kept by every view: in no modality channel.
METADATA = ("times", "frequency")
#: The one member of the file a policy never receives.
PRIVILEGED = "meta"

ARMS = ("left", "right")
POSE_DIM = 7
ENDPOSE_DIM = len(ARMS) * (POSE_DIM + 1)


class PromptError(ValueError):
    """A prompt file cannot be written from this demonstration, or read back as one."""


def channel_of(name: str) -> str | None:
    """Which channel an array belongs to; "metadata", "privileged", or None for a name no channel claims."""
    if name == PRIVILEGED:
        return "privileged"
    if name in METADATA:
        return "metadata"
    for channel, members in CHANNELS.items():
        for member in members:
            if channel in PREFIX_CHANNELS and name.startswith(member):
                return channel
            if name == member:
                return channel
    return None


def policy_arrays(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a prompt a policy may be handed: every array, never `meta`."""
    return {name: value for name, value in arrays.items() if name != PRIVILEGED}


def flatten_endpose(endpose: Mapping[str, Any]) -> np.ndarray:
    """RoboTwin's endpose dict as one (16,) float64 row: left pose, left gripper, right pose, right gripper."""
    row = []
    for arm in ARMS:
        pose = endpose.get(f"{arm}_endpose")
        gripper = endpose.get(f"{arm}_gripper")
        if pose is None or gripper is None:
            raise PromptError(
                f"endpose has no {arm}_endpose or {arm}_gripper; RoboTwin records endposes only "
                "when the task config's data_type.endpose is on"
            )
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (POSE_DIM,):
            raise PromptError(f"{arm}_endpose has shape {pose.shape}, expected ({POSE_DIM},)")
        row.extend(pose.tolist())
        row.append(float(gripper))
    return np.asarray(row, dtype=np.float64)


def unflatten_endpose(row: np.ndarray) -> dict[str, Any]:
    """The dict `flatten_endpose` came from, keys and values as RoboTwin's `get_obs()` gives them."""
    row = np.asarray(row, dtype=np.float64)
    if row.shape != (ENDPOSE_DIM,):
        raise PromptError(f"endpose row has shape {row.shape}, expected ({ENDPOSE_DIM},)")
    endpose: dict[str, Any] = {}
    for k, arm in enumerate(ARMS):
        start = k * (POSE_DIM + 1)
        endpose[f"{arm}_endpose"] = row[start : start + POSE_DIM].tolist()
        endpose[f"{arm}_gripper"] = float(row[start + POSE_DIM])
    return endpose


def arrays_from(demonstration: Demonstration) -> dict[str, np.ndarray]:
    """The prompt's arrays, without `meta`, from an in-memory demonstration."""
    arrays: dict[str, np.ndarray] = {}
    for camera in demonstration.cameras:
        images = demonstration.images(camera)
        if images.dtype != np.uint8:
            raise PromptError(f"camera {camera!r} frames are {images.dtype}, expected uint8")
        arrays[f"{FRAMES_PREFIX}{camera}"] = images
    arrays["qpos"] = np.asarray(demonstration.qpos(), dtype=np.float64)
    arrays["endpose"] = np.stack([flatten_endpose(frame.endpose) for frame in demonstration.frames])
    arrays["actions"] = np.asarray(demonstration.actions(), dtype=np.float64)
    arrays["times"] = np.asarray(demonstration.times(), dtype=np.float64)
    arrays["frequency"] = np.asarray(demonstration.frequency, dtype=np.float64)
    return arrays


def demonstration_from(arrays: Mapping[str, np.ndarray]) -> Demonstration:
    """A demonstration from a prompt's arrays; `meta`, present or not, is never read.

    Every array must have the published dtype and shape: a reader that cast a string or a
    float32 block into place would score a prompt no writer produced.
    """
    cameras = sorted(
        name[len(FRAMES_PREFIX) :] for name in arrays if name.startswith(FRAMES_PREFIX)
    )
    for name in ("qpos", "endpose", "actions", "times", "frequency"):
        if name not in arrays:
            raise PromptError(f"prompt has no {name!r} array")
    if not cameras:
        raise PromptError(f"prompt has no {FRAMES_PREFIX}<camera> array")
    qpos, endpose, actions, times, frequency = (
        _float64(arrays, name) for name in ("qpos", "endpose", "actions", "times", "frequency")
    )
    if qpos.ndim != 2:
        raise PromptError(f"qpos has shape {qpos.shape}, expected (T, D)")
    count = qpos.shape[0]
    for name, array, shape in (
        ("endpose", endpose, (count, ENDPOSE_DIM)),
        ("actions", actions, (count - 1, qpos.shape[1])),
        ("times", times, (count,)),
        ("frequency", frequency, ()),
    ):
        if array.shape != shape:
            raise PromptError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.isfinite(frequency):
        raise PromptError(f"frequency is {float(frequency)}, expected a finite rate")
    if not np.array_equal(actions, qpos[1:]):
        raise PromptError(
            "actions are not the next frame's qpos, as a position-controlled expert's are"
        )
    images = {camera: np.asarray(arrays[f"{FRAMES_PREFIX}{camera}"]) for camera in cameras}
    for camera, block in images.items():
        if block.dtype != np.uint8:
            raise PromptError(f"camera {camera!r} frames are {block.dtype}, expected uint8")
        if block.ndim != 4 or block.shape[0] != count or block.shape[3] != 3:
            raise PromptError(
                f"camera {camera!r} frames have shape {block.shape}, expected ({count}, h, w, 3)"
            )
    try:
        frames = tuple(
            Frame(
                index=i,
                images={camera: images[camera][i] for camera in cameras},
                qpos=qpos[i],
                endpose=unflatten_endpose(endpose[i]),
                time_s=float(times[i]),
            )
            for i in range(count)
        )
        return Demonstration(frames=frames, frequency=float(frequency), cameras=tuple(cameras))
    except DemonstrationError as exc:
        raise PromptError(f"prompt does not hold a demonstration: {exc}") from exc


def _float64(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    array = np.asarray(arrays[name])
    if array.dtype != np.float64:
        raise PromptError(f"{name} is {array.dtype}, expected float64")
    return array


def write_prompt(path: str | Path, demonstration: Demonstration, meta: Mapping[str, Any]) -> str:
    """Write the demonstration and its privileged `meta` to `path`; returns the file's sha256.

    Written beside `path` and moved into place, so an interrupted write never leaves a truncated
    prompt where a reader looks for one.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = arrays_from(demonstration)
    partial = target.with_name(target.name + ".partial")
    try:
        # A file object, not a path: given a path, numpy appends ".npz" to one that lacks it.
        with open(partial, "wb") as handle:
            np.savez_compressed(handle, **arrays, meta=json.dumps(dict(meta), sort_keys=True))
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return sha256_of(target)


def read_raw(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The arrays (never including `meta`) and the parsed `meta`, as the file holds them."""
    try:
        with np.load(Path(path), allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files if name != PRIVILEGED}
            meta_text = loaded[PRIVILEGED].item() if PRIVILEGED in loaded.files else None
    except Exception as exc:
        # Bytes that are not an npz fail in numpy, zipfile or zlib, each with its own exception
        # (EOFError, BadZipFile, zlib.error, ...); to the caller they are one unreadable prompt.
        raise PromptError(f"cannot read prompt {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(meta_text, str):
        raise PromptError(f"prompt {path} has no meta")
    try:
        meta = json.loads(meta_text)
    except ValueError as exc:
        raise PromptError(f"prompt {path} has unreadable meta: {exc}") from exc
    if not isinstance(meta, dict):
        raise PromptError(f"prompt {path} meta is {type(meta).__name__}, not an object")
    return arrays, meta


def read_prompt(path: str | Path) -> tuple[Demonstration, dict[str, Any]]:
    """`write_prompt`'s inverse: the demonstration a policy is handed, and the meta it is not."""
    arrays, meta = read_raw(path)
    return demonstration_from(arrays), meta


def sha256_of(path: str | Path) -> str:
    """The content address of a prompt: the sha256 of its bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
