"""Running one unit against a policy the orchestrator is already serving. Needs the simulator.

The entrant's weights run in the competition's own container, behind its own server; the simulator
runs here, outside it. The two meet over an address the orchestrator hands both sides, so the
untrusted container never gets SAPIEN, CuRobo or the asset tree - and the simulator never gets the
entrant's code.

That out-of-process path needs a policy server in the benchmark core (`robotwin_icil.serve` and
`RemotePolicy`), which is built on another branch and not on `main`. Rather than duplicate it here
or pretend, `--policy-address` says exactly what it needs and what to install; `--policy`, an
importable `module:Class`, is the in-process path and works today. Wiring the served path is
tracked as its own issue.

Same Scene is checked against the **published prompt**, not against something rebuilt alongside it:
the prompt carries the seed it was recorded at and a digest of that scene, so a rollout that drifts
is caught against the artifact a third party can also check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .prompt import load


def run_unit(
    *,
    task: str,
    scene_seed: int,
    prompt: Path,
    out_dir: Path,
    policy_address: str | None = None,
    policy_spec: str | None = None,
    authkey_file: str | None = None,
    view: str = "video_only",
    task_config: str = "demo_clean",
    record_video: bool = True,
) -> dict[str, Any]:
    """Rebuild the prompt's scene, hand the policy its demonstration, roll out, score.

    Never raises for a scene or model failure: a duel needs a verdict for every unit, and a fault
    that is the harness's rather than the model's comes back as `void` so it can be excluded from
    the score instead of counted as a loss.
    """
    from robotwin_icil import robotwin, scene
    from robotwin_icil.episode import rollout
    from robotwin_icil.video import EpisodeVideo

    doc = load(prompt)
    meta = doc["meta"]
    if meta.get("task") != task:
        return _void(f"prompt is for {meta.get('task')!r}, the unit is {task!r}")
    recorded_seed = int(meta.get("scene_seed", scene_seed))

    config = robotwin.SceneConfig(task_config=task_config, save_freq=int(meta.get("save_freq", 15)))
    demonstration = _demonstration(doc)
    clip = EpisodeVideo(out_dir) if record_video else None

    try:
        policy = _policy(policy_address, policy_spec, authkey_file)
    except Exception as exc:  # noqa: BLE001
        return _void(f"{type(exc).__name__}: {exc}")

    task_env = robotwin.load_task(task)
    try:
        task_env.setup_demo(now_ep_num=0, seed=recorded_seed, is_test=True, **config.resolve(task))
        initial = robotwin.fingerprint(task_env)
        drift = _drift(scene, initial, meta.get("fingerprint_sha256"))
        if drift is not None:
            return _void(drift, scene_max_error=None)

        policy.seed(recorded_seed ^ 0x706F6C)
        policy.reset()
        policy.set_demonstration(demonstration)
        success, detail = rollout(task_env, policy, observe=clip.observe if clip else None)
        result: dict[str, Any] = {
            "task": task,
            "scene_seed": recorded_seed,
            "success": bool(success),
            "void": False,
            "detail": detail,
            "view": view,
            "prompt_sha256": meta.get("sha256"),
        }
    except Exception as exc:  # noqa: BLE001 - the duel needs a verdict, not a traceback
        return _void(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            policy.close()
        except Exception:  # noqa: BLE001
            pass
        robotwin.close(task_env)
        robotwin.free_gpu()

    if clip is not None:
        try:
            clip.finish()
            result["video"] = "evaluation_same_scene.mp4"
        except Exception:  # noqa: BLE001 - a missing clip is a gap in the record, not a score
            result["video"] = None
    return result


def _policy(address: str | None, spec: str | None, authkey_file: str | None):
    """The policy this unit runs: one the orchestrator is serving, or one imported in process.

    A served policy is how a competition keeps the entrant's code out of the simulator. It needs
    `robotwin_icil.remote`, which lives on the branch that builds the policy server; until that
    reaches `main`, this says so rather than failing somewhere less legible.
    """
    from robotwin_icil.policy import make_policy

    if address:
        try:
            from robotwin_icil.remote import RemotePolicy
        except ImportError as exc:
            raise RuntimeError(
                "--policy-address needs the benchmark's policy server (robotwin_icil.remote), "
                "which is not in this checkout. Use --policy module:Class to run in process, or "
                "install a benchmark revision that has it."
            ) from exc
        return RemotePolicy(address=address, authkey_file=authkey_file)
    if not spec:
        raise RuntimeError("one of --policy-address or --policy is required")
    return make_policy(spec)


def _demonstration(doc: dict[str, Any]):
    """Rebuild a `Demonstration` from the prompt's arrays.

    Everything the file holds is rebuilt here. What a policy may *see* of it is the orchestrator's
    decision, applied before this is ever called - which is why this module has no view logic and
    no list of withheld channels.
    """
    import numpy as np

    from robotwin_icil.demo import Demonstration, Frame

    meta = doc["meta"]
    cameras = tuple(meta["cameras"])
    times = np.asarray(doc["times"], dtype=np.float64)
    qpos = np.asarray(doc["qpos"], dtype=np.float64)
    endpose = np.asarray(doc["endpose"], dtype=np.float64)
    images = {camera: np.asarray(doc[f"frames_{camera}"]) for camera in cameras}
    frames = tuple(
        Frame(
            index=i,
            images={camera: images[camera][i] for camera in cameras},
            qpos=qpos[i],
            endpose=_endpose(endpose[i]),
        )
        for i in range(len(times))
    )
    return Demonstration(frames=frames, frequency=float(meta["frequency"]), cameras=cameras)


def _endpose(row: Any) -> dict[str, Any]:
    """The 16-wide `ee` row back into RoboTwin's own dict."""
    return {
        "left_endpose": row[0:7],
        "left_gripper": float(row[7]),
        "right_endpose": row[8:15],
        "right_gripper": float(row[15]),
    }


def _drift(scene: Any, initial: Any, expected: str | None) -> str | None:
    """Whether the scene we rebuilt is the one the prompt was recorded in."""
    if not expected:
        return None
    from .materialize import fingerprint_sha256

    got = fingerprint_sha256(initial)
    if got == expected:
        return None
    return f"scene drift: rebuilt {got[:12] if got else 'nothing'}, prompt says {expected[:12]}"


def _void(error: str, **extra: Any) -> dict[str, Any]:
    return {"success": None, "void": True, "steps": None, "error": error, **extra}
