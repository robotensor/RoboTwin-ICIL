"""One demonstration, on disk, in the convention the orchestrator already reads.

A duel needs both sides to see identical bytes and a third party to check them, and this benchmark
generates demonstrations in memory and never serialises them. So one is written as an `npz` of
named arrays plus a JSON `meta` string - the same shape the competition's other benchmarks write
and its loader already reads.

Deliberately **not** a new file format, and deliberately no change to `Demonstration` or `Frame`.
The orchestrator decides what a policy may see of a demonstration: it withholds the channels its
field withholds, over arrays like these. Keeping that decision there means it holds for every
benchmark and survives this one changing underneath it.

The arrays are grouped by channel so the orchestrator's view can allow or drop them as a unit:

| channel | arrays |
| --- | --- |
| `video` | `frames_*`, one `(T, h, w, 3)` uint8 block per camera |
| `proprio` | `qpos` `(T, 14)`, `endpose` `(T, 16)`, `gripper_joints` `(T, 2, J)` |
| `actions` | `actions` `(T-1, 14)` |
| `metadata` | `times` `(T,)` |

The spellings are the orchestrator's, not ours: an entry ending in `*` is a prefix, which is how a
family the orchestrator cannot enumerate - it does not know this benchmark's camera list - stays
inside an allow-list. `metadata` is the channel every view keeps, and frame times are the case it
exists for: they say when a frame was taken, never what the robot did, so no field's modality list
would ever claim them and a view that dropped them would leave the demonstration untimed.

`meta` is in no channel. The scene seed and the initial-state fingerprint travel in it so Same
Scene can be re-verified against a published artifact - they are privileged, and a policy never
sees `meta`.

`gripper_joints` carries the *measured* finger positions, left arm then right, in metres. The
gripper value inside `qpos` and `endpose` is RoboTwin's command, which reads closed while the
fingers rest on an object; a model that was trained on where the fingers actually are (BPP's
`gripper_states`, say) needs the measurement, and a prompt that dropped it could not serve one.
It is absent from a demonstration recorded before the benchmark read them, and from a prompt
written before this, which reads back with no measurement rather than failing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from robotwin_icil.demo import Demonstration

PROMPT_SCHEMA = 1

#: An entry ending in this is a **prefix**, matching every array whose name starts with the rest
#: of it. The orchestrator's allow-list cannot enumerate one array per camera, because it does not
#: know a benchmark's camera list, so `frames_*` is how that family is declared without opening
#: the list up.
PREFIX_MARK = "*"

#: The channel holding arrays every view keeps. Not a modality: no field asks for it, and none
#: may drop it.
METADATA_CHANNEL = "metadata"

#: channel -> the arrays that carry it. The orchestrator is told this through `info()`; it is
#: repeated here so a prompt can be checked on its own.
CHANNELS = {
    "video": (f"frames_{PREFIX_MARK}",),
    "proprio": ("qpos", "endpose", "gripper_joints"),
    "actions": ("actions",),
    METADATA_CHANNEL: ("times",),
}


def dump(
    demonstration: Demonstration,
    path: str | Path,
    *,
    task: str,
    scene_seed: int,
    fingerprint_sha256: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> str:
    """Write one demonstration and return its sha256."""
    from robotwin_icil.demo import ARMS

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "times": np.asarray(demonstration.times(), dtype=np.float64),
        "qpos": np.asarray(demonstration.qpos(), dtype=np.float32),
        "endpose": np.asarray(demonstration.endposes(), dtype=np.float32),
        "actions": np.asarray(demonstration.actions(), dtype=np.float32),
    }
    # All frames carry them or none do, which `Demonstration` itself checks.
    if demonstration.frames[0].gripper_joints is not None:
        arrays["gripper_joints"] = np.asarray(
            [[frame.gripper_joints[arm] for arm in ARMS] for frame in demonstration.frames],
            dtype=np.float32,
        )
    for camera in demonstration.cameras:
        arrays[f"frames_{camera}"] = np.asarray(demonstration.images(camera), dtype=np.uint8)
    meta = {
        "schema": PROMPT_SCHEMA,
        "task": task,
        "scene_seed": int(scene_seed),
        "setting": "same_scene",
        "cameras": list(demonstration.cameras),
        "frequency": float(demonstration.frequency),
        "steps": int(len(demonstration) - 1),
        "fingerprint_sha256": fingerprint_sha256,
        **(provenance or {}),
    }
    np.savez_compressed(target, meta=json.dumps(meta, sort_keys=True), **arrays)
    return prompt_sha256(target)


def load(path: str | Path) -> dict[str, Any]:
    """Every array in the file, and its metadata. The orchestrator applies its field's view."""
    with np.load(Path(path), allow_pickle=False) as z:
        out: dict[str, Any] = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out


def prompt_sha256(path: str | Path) -> str:
    """The content address of a prompt: the sha256 of the bytes, as the store addresses clips."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def channel_of(name: str) -> str | None:
    """Which channel an array belongs to, or None for one no channel claims.

    `meta` is the only member of the file that is deliberately in none: it is a JSON string rather
    than an array, and it is privileged. Anything else landing here is an oversight, and
    `verify_prompt` reports it rather than letting a view silently drop it.
    """
    if name == "meta":
        return None
    for channel, members in CHANNELS.items():
        for member in members:
            if member.endswith(PREFIX_MARK) and name.startswith(member[: -len(PREFIX_MARK)]):
                return channel
            if name == member:
                return channel
    return None
