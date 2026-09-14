"""Checking a materialized prompt against its unit, and reading either command's result.

Both read files a subprocess wrote, and neither needs the simulator: `verify_prompt` is what lets
anyone holding a published prompt confirm it is the one its unit asked for, and `read_result` is
how the orchestrator learns what a command did.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import robotwin_icil
from robotwin_icil import prompt as prompt_
from robotwin_icil.records import SAME_SCENE
from robotwin_icil.scene import SceneFingerprint, digest

from . import commands

#: The files the two commands write, as `robotwin_icil.unit` names them.
PROMPT_FILE = prompt_.PROMPT_FILE
RESULT_FILE = "result.json"

#: `qpos` width per robot: joints then a gripper per arm, six joints an arm on aloha-agilex and
#: seven on a Franka. An `ee` action is 16 on either (`robotwin.EE_ACTION_DIM`).
QPOS_DIMS = {"aloha-agilex": 14, "franka-panda": 16}

#: Whose a void is, in a result: `void_cause` as `robotwin-icil` writes it.
VOID_CAUSES = ("harness", "policy")

#: The cameras a policy observes under the task config every unit uses (`commands.TASK_CONFIG`):
#: RoboTwin's `demo_clean` collects the head camera and both wrist cameras, named as
#: `envs/camera/camera.py` names them. A prompt and an observation hold `frames_<camera>` of each.
CAMERAS = ("head_camera", "left_camera", "right_camera")

#: The scene config every unit is materialized with, as a prompt's `meta` records it: the argv's
#: task config and frame spacing, no head camera or override, the Same Scene setting.
SCENE_CONFIG: dict[str, Any] = {
    "task_config": commands.TASK_CONFIG,
    "save_freq": commands.SAVE_FREQ,
    "head_camera": None,
    "overrides": {},
    "evaluation_setting": SAME_SCENE,
}


def sha256_of(path: Path) -> str:
    digest_ = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest_.update(block)
    return digest_.hexdigest()


def verify_prompt(*, path: str, unit: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the prompt at `path` is the one `unit` asked for; every problem, not the first.

    Reads the file itself, never a manifest, and never unpickles it (`allow_pickle=False`): its
    `meta` must name the unit's task, one of its candidate scene seeds and its robot, as that robot
    was chosen, and the scene config every unit is built with (`SCENE_CONFIG`), which run-unit
    rebuilds the scene from; every array must belong to the published channel map and together
    hold a demonstration of the robot's width, from no camera but `CAMERAS`; and the recorded scene
    fingerprint must digest to the digest beside it. Returns `{"ok", "sha256", "problems"}`, where
    `sha256` is the file's bytes' (None if it cannot be read), plus the `task`, `scene_seed` and
    `embodiment` it found.
    """
    file = Path(path)
    problems: list[str] = []
    found: dict[str, Any] = {"task": None, "scene_seed": None, "embodiment": None}

    def verdict(sha256: str | None) -> dict[str, Any]:
        return {"ok": not problems, "sha256": sha256, "problems": problems, **found}

    try:
        sha256 = sha256_of(file)
    except OSError as exc:
        problems.append(f"cannot read {file}: {exc}")
        return verdict(None)
    try:
        arrays, meta = prompt_.read_raw(file)
    except prompt_.PromptError as exc:
        problems.append(str(exc))
        return verdict(sha256)

    params = unit.get("instance_params")
    params = params if isinstance(params, Mapping) else {}
    embodiment = meta.get("embodiment") if isinstance(meta.get("embodiment"), dict) else {}
    found.update(
        task=meta.get("task"), scene_seed=meta.get("scene_seed"), embodiment=embodiment.get("name")
    )

    if meta.get("schema") != prompt_.PROMPT_SCHEMA:
        problems.append(f"meta schema is {meta.get('schema')!r}, not {prompt_.PROMPT_SCHEMA}")
    if meta.get("task") != unit.get("task"):
        problems.append(
            f"the prompt is of task {meta.get('task')!r}; the unit asks for {unit.get('task')!r}"
        )

    seed, candidates = meta.get("scene_seed"), params.get("scene_seeds")
    if not isinstance(candidates, list) or not candidates:
        problems.append("the unit names no candidate scene_seeds")
    elif isinstance(seed, bool) or not isinstance(seed, int) or seed not in candidates:
        problems.append(
            f"the prompt's scene seed {seed!r} is not one of the unit's candidates {candidates}"
        )

    wanted = params.get("embodiment")
    name, choice = embodiment.get("name"), embodiment.get("choice")
    if name != wanted or choice != wanted:
        problems.append(
            f"the prompt was built on {name!r} (chosen as {choice!r}); the unit asks for {wanted!r}"
        )

    # run-unit rebuilds and scores the scene from exactly these, so a prompt built under another
    # config is another unit however its task, seed and robot read.
    for key, value in SCENE_CONFIG.items():
        if key not in meta or meta[key] != value or type(meta[key]) is not type(value):
            problems.append(
                f"the prompt's {key} is {meta.get(key)!r}; every unit is built with {value!r}"
            )

    unclaimed = sorted(a for a in arrays if prompt_.channel_of(a) in (None, "privileged"))
    if unclaimed:
        problems.append(f"arrays in no published channel: {', '.join(unclaimed)}")
    try:
        demonstration = prompt_.demonstration_from(arrays)
    except prompt_.PromptError as exc:
        problems.append(f"the arrays do not hold a demonstration: {exc}")
    else:
        if wanted in QPOS_DIMS and demonstration.qpos_dim != QPOS_DIMS[wanted]:
            problems.append(
                f"qpos is {demonstration.qpos_dim} wide; {wanted} is {QPOS_DIMS[wanted]}"
            )
        if meta.get("frames") != len(demonstration):
            problems.append(
                f"meta says {meta.get('frames')!r} frames; the arrays hold {len(demonstration)}"
            )
        if meta.get("cameras") != list(demonstration.cameras):
            problems.append(
                f"meta names cameras {meta.get('cameras')!r}; the arrays hold "
                f"{list(demonstration.cameras)}"
            )
        # run-unit hands the policy every camera the prompt holds, so one no unit collects would
        # reach the policy as if the benchmark observed it.
        unobserved = sorted(set(demonstration.cameras) - set(CAMERAS))
        if unobserved:
            problems.append(
                f"frames of camera(s) {', '.join(unobserved)}, which no unit observes; the "
                f"benchmark's cameras are {', '.join(CAMERAS)}"
            )

    scene = meta.get("scene")
    try:
        recorded = digest(SceneFingerprint.from_json(scene["fingerprint"]))
        if recorded != scene["sha256"]:
            problems.append(
                f"the recorded scene digests to {recorded[:12]}, meta says "
                f"{str(scene['sha256'])[:12]}: meta was edited after it was written"
            )
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"meta has no readable scene fingerprint: {type(exc).__name__}: {exc}")
    return verdict(sha256)


def read_result(*, out_dir: str) -> dict[str, Any]:
    """The `result.json` either command wrote in `out_dir`, with the ABI's fields made consistent.

    Returns everything the file holds, with `success` (a bool, or None exactly when void), `void`,
    `steps` (an int or None), `error` (a reason whenever void) and `void_cause` ("policy",
    "harness", or None when not void; `RoboTwinBenchmark.read_result` says what each means). A
    result that says neither void nor a boolean success, cannot be read, or was written by other
    benchmark source than this plugin runs (`source_sha256`) is void on the harness.
    """
    path = Path(out_dir) / RESULT_FILE
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        doc = None
        reason = f"unreadable {RESULT_FILE}: {type(exc).__name__}: {exc}"
    if not isinstance(doc, dict):
        if doc is not None:
            reason = f"{RESULT_FILE} holds {type(doc).__name__}, not an object"
        return {
            "success": None,
            "void": True,
            "void_cause": "harness",
            "steps": None,
            "error": reason,
        }

    source, mine = doc.get("source_sha256"), robotwin_icil.source_sha256()
    if source != mine:
        # Written by other benchmark code than this plugin describes and built the argv for: its
        # fields cannot be read as this benchmark's.
        return {
            **doc,
            "success": None,
            "void": True,
            "void_cause": "harness",
            "steps": None,
            "error": f"{RESULT_FILE} was written by benchmark source {source!r}, not the {mine} "
            "this plugin runs",
        }

    success = doc.get("success")
    void = doc.get("void") is not False or not isinstance(success, bool)
    error = doc.get("error") if isinstance(doc.get("error"), str) and doc.get("error") else None
    steps = doc.get("steps")
    steps = int(steps) if _finite_int(steps) and not void else None
    cause = doc.get("void_cause")
    if void:
        cause = cause if cause in VOID_CAUSES else "harness"
        error = error or "the benchmark wrote a void result without a reason"
    return {
        **doc,
        "success": None if void else success,
        "void": void,
        "void_cause": cause if void else None,
        "steps": steps,
        "error": error,
    }


def _finite_int(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and float(value).is_integer()
    )
