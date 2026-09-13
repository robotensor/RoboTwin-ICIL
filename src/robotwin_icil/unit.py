"""One evaluation unit in two files: a demonstration built once and saved, an episode run from it.

A duel hands both policies the same demonstration, and an expert's trajectory is not reproducible
from its seed (CuRobo's IK seeds live for the env's lifetime, and planning retries stop on a wall
clock), so the demonstration is built once and saved, and every evaluation reads the file:

    materialize   build the scene for one seed, run the expert once, and write
                  prompt.npz, demonstration.mp4 and result.json
    run_unit      read prompt.npz, rebuild the scene from its meta, check the live fingerprint
                  against the recorded one, roll the policy out, and write result.json and
                  evaluation.mp4

Both are `robotwin-icil` commands, and both are the halves `episode.run_episode` runs in one
process: `generate.attempt` and `episode.evaluate`. What the files hold is `prompt`'s business;
this module decides what a rejected seed, a tampered prompt and a drifted scene write.

`meta` is privileged. `run_unit` reads it to rebuild and verify the scene; the policy is handed
the `Demonstration` and nothing else.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import generate
from .demo import Demonstration
from .episode import evaluate
from .policy import ICILPolicy, PolicyError
from .prompt import PROMPT_FILE, PROMPT_SCHEMA, PromptError, read_prompt, sha256_of, write_prompt
from .records import SAME_SCENE, git_commit
from .scene import SceneFingerprint, digest
from .video import DEMONSTRATION_CLIP, EpisodeVideo

#: Written by both commands into their `--out` directory.
RESULT_FILE = "result.json"
#: The rollout clip `run_unit` writes; a run directory's episode keeps its own name.
EVALUATION_CLIP = "evaluation.mp4"


@dataclass(frozen=True)
class Materialized:
    """What `materialize` did: a prompt written, or a seed rejected, and the result it recorded."""

    ok: bool
    attempt: generate.Attempt
    result: dict[str, Any]
    demonstration: Demonstration | None = None
    initial: SceneFingerprint | None = None


def materialize(
    task: str,
    scene_seed: int,
    config,
    out_dir: str | Path,
    *,
    task_env=None,
    video: bool = True,
) -> Materialized:
    """Build the scene for `scene_seed`, run the expert once, and save what it did.

    A rejected seed is a legitimate outcome, not an error: `result.json` says why and no prompt
    is written. A simulator that cannot build any scene raises `RoboTwinError`, as `generate`
    does. Stale files from an earlier command in the same directory are removed first, so what
    the directory holds afterwards is this call's.
    """
    from . import robotwin

    started = time.monotonic()
    # Absolute before the RoboTwin seam moves the working directory into the checkout.
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for stale in (PROMPT_FILE, DEMONSTRATION_CLIP, RESULT_FILE):
        (out / stale).unlink(missing_ok=True)

    args = config.resolve(task)
    embodiment = str(args["embodiment_name"])
    task_env = task_env if task_env is not None else robotwin.load_task(task)
    try:
        attempt, demonstration, initial = generate.attempt(
            task_env, int(scene_seed), args, config.save_freq, 0
        )
    finally:
        robotwin.free_gpu()

    def finish(**fields: Any) -> dict[str, Any]:
        result = {
            "task": task,
            "scene_seed": int(scene_seed),
            "embodiment": embodiment,
            "attempts": 1,
            **fields,
            "duration_s": round(time.monotonic() - started, 3),
        }
        _write_json(out / RESULT_FILE, result)
        return result

    if demonstration is None or initial is None:
        assert attempt.rejection is not None
        result = finish(ok=False, rejection=attempt.rejection.value, detail=attempt.detail)
        return Materialized(ok=False, attempt=attempt, result=result)

    meta = build_meta(task, int(scene_seed), config, args, demonstration, initial)
    prompt_sha256 = write_prompt(out / PROMPT_FILE, demonstration, meta)
    note = ""
    if video:
        note = _film(lambda: EpisodeVideo(out).demonstration(demonstration))
    result = finish(
        ok=True,
        frames=len(demonstration),
        cameras=list(demonstration.cameras),
        prompt_sha256=prompt_sha256,
        scene_sha256=meta["scene"]["sha256"],
        rejections={},
        video=DEMONSTRATION_CLIP if (out / DEMONSTRATION_CLIP).is_file() else None,
        detail=note.strip(),
    )
    return Materialized(
        ok=True, attempt=attempt, result=result, demonstration=demonstration, initial=initial
    )


def build_meta(
    task: str,
    scene_seed: int,
    config,
    args: dict[str, Any],
    demonstration: Demonstration,
    initial: SceneFingerprint,
) -> dict[str, Any]:
    """The privileged `meta` of a prompt: enough to rebuild its scene and to prove it is the same.

    `config` and `args` are the `SceneConfig` the scene was built with and what it resolved to;
    `initial` is the fingerprint taken before the expert acted.
    """
    from . import robotwin

    return {
        "schema": PROMPT_SCHEMA,
        "evaluation_setting": SAME_SCENE,
        "task": task,
        "scene_seed": int(scene_seed),
        "embodiment": {
            "name": str(args["embodiment_name"]),
            "robotwin": list(args["embodiment"]),
        },
        "task_config": config.task_config,
        "save_freq": int(config.save_freq),
        "head_camera": config.head_camera,
        "overrides": dict(config.overrides or {}),
        "scene": {"fingerprint": initial.to_json(), "sha256": digest(initial)},
        "benchmark_commit": git_commit(robotwin.REPO_ROOT),
        "robotwin_commit": git_commit(robotwin.ROBOTWIN_ROOT),
        "expert": {"attempts": 1, "rejections": {}, "rejection_details": {}},
        "frames": len(demonstration),
        "cameras": list(demonstration.cameras),
    }


def recorded_scene(meta: dict[str, Any]) -> SceneFingerprint:
    """The fingerprint `meta` carries, once its digest has been shown to be that fingerprint's.

    A meta edited after it was written — a seed, a pose, the digest itself — no longer digests
    to what it claims, and is refused before any scene is built.
    """
    for key in ("task", "scene_seed", "task_config", "save_freq", "embodiment", "scene"):
        if key not in meta:
            raise PromptError(f"prompt meta has no {key!r}")
    if not isinstance(meta["embodiment"], dict) or "name" not in meta["embodiment"]:
        raise PromptError("prompt meta's embodiment has no name")
    if isinstance(meta["scene_seed"], bool) or not isinstance(meta["scene_seed"], int):
        raise PromptError(f"prompt meta's scene_seed is {meta['scene_seed']!r}, not an integer")
    scene = meta["scene"]
    if not isinstance(scene, dict) or "fingerprint" not in scene or "sha256" not in scene:
        raise PromptError("prompt meta has no scene fingerprint and digest")
    try:
        fingerprint = SceneFingerprint.from_json(scene["fingerprint"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PromptError(f"prompt meta's scene fingerprint is malformed: {exc}") from exc
    actual = digest(fingerprint)
    if actual != scene["sha256"]:
        raise PromptError(
            f"prompt meta is inconsistent: its scene fingerprint digests to {actual[:12]}, "
            f"meta says {str(scene['sha256'])[:12]}; it was edited after it was written"
        )
    return fingerprint


def scene_config_from(meta: dict[str, Any]):
    """The `SceneConfig` that rebuilds a prompt's scene: the same task config, rate and robot."""
    from . import robotwin

    return robotwin.SceneConfig(
        task_config=str(meta["task_config"]),
        save_freq=int(meta["save_freq"]),
        head_camera=meta.get("head_camera"),
        overrides=dict(meta.get("overrides") or {}) or None,
        embodiment=str(meta["embodiment"]["name"]),
    )


def run_unit(
    prompt_path: str | Path,
    policy: ICILPolicy,
    out_dir: str | Path,
    *,
    task_env=None,
    video: bool = True,
) -> dict[str, Any]:
    """Evaluate `policy` on the prompt at `prompt_path`, writing `result.json` and `evaluation.mp4`.

    Never raises for what the policy did, nor for a prompt that cannot be trusted: `success` is
    the verdict when the scene was rebuilt as recorded and the policy acted, and the unit is
    `void`, with the reason in `error`, when it was not — an unreadable or tampered prompt, a
    scene that drifted or would not build, a policy that broke the protocol. `success` and
    `steps` are None exactly when the unit is void. A simulator that cannot load the task raises
    `RoboTwinError`.
    """
    from . import robotwin

    started = time.monotonic()
    # Absolute before the RoboTwin seam moves the working directory into the checkout.
    prompt_path = Path(prompt_path).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for stale in (RESULT_FILE, EVALUATION_CLIP):
        (out / stale).unlink(missing_ok=True)
    describe = policy.describe()
    result: dict[str, Any] = {
        "success": None,
        "void": True,
        "steps": None,
        "step_limit": None,
        "error": None,
        "detail": "",
        "scene_max_error": 0.0,
        "model": str(describe.get("model", describe["policy"])),
        "embodiment": None,
        "task": None,
        "scene_seed": None,
        "evaluation_setting": SAME_SCENE,
        "prompt_sha256": None,
        "video": None,
    }

    def finish(**fields: Any) -> dict[str, Any]:
        result.update(fields)
        result["video"] = EVALUATION_CLIP if (out / EVALUATION_CLIP).is_file() else None
        result["duration_s"] = round(time.monotonic() - started, 3)
        _write_json(out / RESULT_FILE, result)
        return result

    def void(error: str, **fields: Any) -> dict[str, Any]:
        return finish(success=None, void=True, steps=None, step_limit=None, error=error, **fields)

    try:
        demonstration, meta = read_prompt(prompt_path)
    except PromptError as exc:
        return void(f"unreadable prompt: {exc}")
    embodiment = meta.get("embodiment")
    result.update(
        task=meta.get("task"),
        scene_seed=meta.get("scene_seed"),
        embodiment=embodiment.get("name") if isinstance(embodiment, dict) else None,
        prompt_sha256=sha256_of(prompt_path),
    )
    try:
        initial = recorded_scene(meta)
        config = scene_config_from(meta)
    except PromptError as exc:
        return void(str(exc))
    except (KeyError, TypeError, ValueError) as exc:
        return void(f"prompt meta is malformed: {type(exc).__name__}: {exc}")

    task_env = task_env if task_env is not None else robotwin.load_task(str(meta["task"]))
    clip = (
        EpisodeVideo(out, evaluation_clip=EVALUATION_CLIP, fps=demonstration.frequency)
        if video
        else None
    )
    try:
        evaluation = evaluate(
            task_env,
            str(meta["task"]),
            int(meta["scene_seed"]),
            config,
            demonstration,
            initial,
            policy,
            video=clip,
        )
    except PolicyError as exc:
        return void(f"policy broke the protocol: {exc}")
    finally:
        robotwin.free_gpu()

    if not evaluation.valid:
        return void(evaluation.detail, scene_max_error=evaluation.scene_max_error)
    return finish(
        success=bool(evaluation.success),
        void=False,
        steps=int(evaluation.steps),
        step_limit=evaluation.step_limit,
        error=None,
        detail=evaluation.detail,
        scene_max_error=0.0,
    )


def read_result(out_dir: str | Path) -> dict[str, Any]:
    """The `result.json` a command left in `out_dir`."""
    return json.loads((Path(out_dir) / RESULT_FILE).read_text(encoding="utf-8"))


def _film(write: Callable[[], None]) -> str:
    """A clip that fails to write is a note in the result, never an outcome."""
    try:
        write()
    except Exception as exc:
        return f" [video: {type(exc).__name__}: {exc}]"
    return ""


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
