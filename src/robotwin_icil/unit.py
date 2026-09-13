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

from . import generate, tasks
from .demo import Demonstration
from .episode import evaluate
from .policy import ICILPolicy
from .prompt import PROMPT_FILE, PROMPT_SCHEMA, PromptError, read_prompt, sha256_of, write_prompt
from .records import SAME_SCENE, git_commit
from .scene import SceneFingerprint, deviation, digest
from .video import DEMONSTRATION_CLIP, EpisodeVideo

#: Written by both commands into their `--out` directory.
RESULT_FILE = "result.json"
#: The rollout clip `run_unit` writes; a run directory's episode keeps its own name.
EVALUATION_CLIP = "evaluation.mp4"
#: What each command writes into its `--out` directory, and so clears there before anything else.
MATERIALIZE_OUTPUTS = (PROMPT_FILE, DEMONSTRATION_CLIP, RESULT_FILE)
RUN_UNIT_OUTPUTS = (RESULT_FILE, EVALUATION_CLIP)


class UnitError(ValueError):
    """A command was pointed at a directory it must not write into."""


def clear_outputs(
    out_dir: str | Path, names: tuple[str, ...], prompt: str | Path | None = None
) -> Path:
    """`out_dir`, absolute and created, with no file left in it that this command writes.

    Called first, before anything that can fail, so a directory reused for a second command
    never holds the first one's result as if it were the second's. `prompt` is run-unit's: a
    unit's result is never written over the result materialize left beside its prompt.
    """
    # Absolute before the RoboTwin seam moves the working directory into the checkout.
    out = Path(out_dir).resolve()
    if prompt is not None and Path(prompt).resolve().parent == out:
        raise UnitError(
            f"--out {out} holds the prompt; run-unit writes its result into a directory of its own"
        )
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        (out / name).unlink(missing_ok=True)
    return out


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
    is written. The result carries what the orchestrator's `read_result` reads from either
    command — `success`, `void`, `steps`, `error` — next to `ok`: a written prompt is a success
    whose `steps` are the demonstration's actions, a rejected seed is void with the rejection as
    its `error`. A simulator that cannot build any scene raises `RoboTwinError`, as `generate`
    does. Stale files from an earlier command in the same directory are removed first, so what
    the directory holds afterwards is this call's.
    """
    from . import robotwin

    started = time.monotonic()
    out = clear_outputs(out_dir, MATERIALIZE_OUTPUTS)

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
        reason = attempt.rejection.value + (f": {attempt.detail}" if attempt.detail else "")
        result = finish(
            ok=False,
            success=None,
            void=True,
            steps=None,
            error=f"expert rejected the seed: {reason}",
            rejection=attempt.rejection.value,
            detail=attempt.detail,
        )
        return Materialized(ok=False, attempt=attempt, result=result)

    meta = build_meta(task, int(scene_seed), config, args, demonstration, initial)
    prompt_sha256 = write_prompt(out / PROMPT_FILE, demonstration, meta)
    note = ""
    if video:
        note = _film(lambda: EpisodeVideo(out).demonstration(demonstration))
    result = finish(
        ok=True,
        success=True,
        void=False,
        steps=len(demonstration) - 1,
        error=None,
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
        # `choice` is how the robot was picked — an `EMBODIMENTS` name, or None for the task
        # config's own list — and is what rebuilds it: a name alone cannot say which arm distance,
        # or which robot the task config gave.
        "embodiment": {
            "name": str(args["embodiment_name"]),
            "robotwin": list(args["embodiment"]),
            "choice": config.embodiment,
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

    The digest covers the fingerprint alone: an edited pose, fingerprint or digest no longer
    matches and is refused here, before any scene is built. An edited seed, task config or robot
    leaves the digest intact and is caught instead by the rebuilt scene's fingerprint, which then
    differs from the recorded one. Neither is tamper-proof against someone who rewrites both;
    the prompt's sha256, published before either side runs, is what a third party checks.
    """
    for key in ("task", "scene_seed", "task_config", "save_freq", "embodiment", "scene"):
        if key not in meta:
            raise PromptError(f"prompt meta has no {key!r}")
    try:
        tasks.table()[meta["task"]]
    except (tasks.TaskTableError, TypeError):
        raise PromptError(
            f"prompt meta's task {meta['task']!r} is not one this benchmark has"
        ) from None
    if not isinstance(meta["embodiment"], dict) or "name" not in meta["embodiment"]:
        raise PromptError("prompt meta's embodiment has no name")
    if "choice" not in meta["embodiment"]:
        raise PromptError("prompt meta's embodiment has no choice")
    choice = meta["embodiment"]["choice"]
    if choice is not None and not isinstance(choice, str):
        raise PromptError(f"prompt meta's embodiment choice is {choice!r}, not a name or null")
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
    """The `SceneConfig` that rebuilds a prompt's scene: the same task config, rate and robot.

    The robot is chosen as it was when the prompt was written, not by its recorded name.
    """
    from . import robotwin

    return robotwin.SceneConfig(
        task_config=str(meta["task_config"]),
        save_freq=int(meta["save_freq"]),
        head_camera=meta.get("head_camera"),
        overrides=dict(meta.get("overrides") or {}) or None,
        embodiment=meta["embodiment"]["choice"],
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

    Never raises for what the policy did, nor for a prompt that cannot be trusted. `success` is
    the verdict once the scene was rebuilt as recorded and the policy was handed it, and a policy
    at fault — raising from `reset` or `set_demonstration`, or breaking the protocol mid-rollout
    with a wrong-width or non-finite action — has failed, with the reason in `detail`: a void unit
    leaves the score, and a policy must not be able to void the units it is losing. The unit is
    `void`, with the reason in `error`, only when the harness could not give the policy a fair
    episode — an unreadable or tampered prompt, a scene that drifted or would not build, a GPU
    that failed during the rollout.
    `success` and `steps` are None exactly when the unit is void. A simulator that cannot load
    the task raises `RoboTwinError`; an `out_dir` holding the prompt raises `UnitError`.
    """
    from . import robotwin

    started = time.monotonic()
    # Absolute before the RoboTwin seam moves the working directory into the checkout.
    prompt_path = Path(prompt_path).resolve()
    out = clear_outputs(out_dir, RUN_UNIT_OUTPUTS, prompt=prompt_path)
    describe = policy.describe()
    result: dict[str, Any] = {
        "success": None,
        "void": True,
        "steps": None,
        "step_limit": None,
        "error": None,
        "detail": "",
        "scene_max_error": None,
        "model": str(describe.get("model", describe["policy"])),
        "checkpoint": describe.get("checkpoint"),
        "benchmark_commit": git_commit(robotwin.REPO_ROOT),
        "robotwin_commit": git_commit(robotwin.ROBOTWIN_ROOT),
        "embodiment": None,
        "task": None,
        "scene_seed": None,
        "evaluation_setting": SAME_SCENE,
        "prompt_sha256": None,
        "scene_sha256": None,
        "live_scene_sha256": None,
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
        result["scene_sha256"] = digest(initial)
        config = scene_config_from(meta)
    except PromptError as exc:
        return void(str(exc))
    except robotwin.RoboTwinError as exc:
        return void(f"prompt meta's scene config is refused: {exc}")
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
            score_policy_faults=True,
        )
    except robotwin.RoboTwinError as exc:
        # The simulator failed under the policy (the GPU lost or full): nothing to score.
        return void(f"simulator failed: {exc}")
    finally:
        robotwin.free_gpu()

    # What the rebuilt scene was, on the result itself: two runs of one prompt can then be shown
    # to have started from the identical scene, not only to have passed the tolerant check.
    seen: dict[str, Any] = {}
    if evaluation.live is not None:
        seen["live_scene_sha256"] = digest(evaluation.live)
        seen["scene_max_error"] = deviation(initial, evaluation.live)
    if not evaluation.valid:
        return void(evaluation.detail, **seen)
    return finish(
        success=bool(evaluation.success),
        void=False,
        steps=int(evaluation.steps),
        step_limit=evaluation.step_limit,
        error=None,
        detail=evaluation.detail,
        **seen,
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
