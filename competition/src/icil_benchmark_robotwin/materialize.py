"""Producing one unit's prompt. Needs the simulator.

Reuses the benchmark's own generator unchanged: RoboTwin's expert, its rejection rules, its
fingerprint. A seed the expert fails on is redrawn, because expert failure is a generation
failure and never a model failure - and doing that here, before either side of the duel runs, is
the whole reason the competition materializes prompts up front rather than mid-duel.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .plugin import PROMPT_NAME
from .prompt import dump


def seed_stream(scene_seed: int, attempts: int) -> list[int]:
    """The seeds this unit may try, in order.

    The first is the unit's own, so a published unit list names the scene it got. The rest are
    derived from it, so a retry is still reproducible from the record.
    """
    from .units import SEED_SPACE

    out = [int(scene_seed)]
    for i in range(1, max(1, attempts)):
        material = f"{scene_seed}|retry|{i}".encode()
        out.append(int(hashlib.sha256(material).hexdigest()[:8], 16) % SEED_SPACE)
    return out


def materialize_unit(
    *,
    task: str,
    scene_seed: int,
    out_dir: Path,
    max_expert_attempts: int = 20,
    task_config: str = "demo_clean",
    save_freq: int = 15,
    record_video: bool = True,
) -> dict[str, Any]:
    """Generate this unit's demonstration, write it, and report what it took."""
    from robotwin_icil import robotwin
    from robotwin_icil.generate import generate

    config = robotwin.SceneConfig(task_config=task_config, save_freq=save_freq)
    task_env = robotwin.load_task(task)
    try:
        generated = generate(
            task_env,
            seed_stream(scene_seed, max_expert_attempts),
            lambda: config.resolve(task),
            save_freq,
            episode=0,
        )
    finally:
        robotwin.close(task_env)
        robotwin.free_gpu()

    if generated.demonstration is None:
        raise RuntimeError(
            f"{task}@{scene_seed}: the expert never succeeded in {max_expert_attempts} attempts; "
            f"{_rejections(generated)}"
        )

    sha = dump(
        generated.demonstration,
        out_dir / PROMPT_NAME,
        task=task,
        scene_seed=int(generated.seed),
        fingerprint_sha256=fingerprint_sha256(generated.initial),
        provenance={
            "task_config": task_config,
            "save_freq": save_freq,
            "expert_generation_attempts": len(generated.attempts),
            # The unit asked for this seed; `scene_seed` is the one the expert actually solved.
            # They differ only when the first seeds were rejected, and the record says so.
            "requested_scene_seed": int(scene_seed),
        },
    )
    report: dict[str, Any] = {
        "task": task,
        "scene_seed": int(generated.seed),
        "requested_scene_seed": int(scene_seed),
        "sha256": sha,
        "frames": len(generated.demonstration),
        "attempts": len(generated.attempts),
        "rejections": _rejections(generated),
    }
    if record_video:
        report["video"] = _clip(generated.demonstration, out_dir)
    return report


def fingerprint_sha256(initial: Any) -> str | None:
    """A digest of the initial scene, so Same Scene can be re-checked against what was published.

    The fingerprint itself stays out of the prompt's arrays: actor poses are privileged, and a
    policy never sees `meta`.
    """
    if initial is None:
        return None
    import numpy as np

    h = hashlib.sha256()
    for part in ("actors", "articulations", "articulation_roots", "cameras"):
        for name, value in sorted(getattr(initial, part).items()):
            h.update(f"{part}|{name}|".encode())
            h.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(initial.robot_qpos, dtype=np.float64).tobytes())
    return h.hexdigest()


def _rejections(generated: Any) -> dict[str, int]:
    """Why the expert was redrawn, by reason. Generation statistics, never a model's score."""
    counts: dict[str, int] = {}
    for attempt in generated.attempts:
        rejection = attempt.rejection
        reason = getattr(rejection, "value", rejection) if rejection is not None else "ok"
        key = str(reason)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _clip(demonstration: Any, out_dir: Path) -> str | None:
    """The demonstration as an mp4, for the record. Best effort: a missing encoder is not a reason
    to fail a duel, and what is scored on is the prompt itself."""
    try:
        from robotwin_icil.video import EpisodeVideo

        clip = EpisodeVideo(out_dir)
        clip.demonstration(demonstration)
        written = out_dir / "demonstration.mp4"
        return written.name if written.exists() else None
    except Exception:  # noqa: BLE001
        return None


def read_report(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / "materialize.json").read_text())
