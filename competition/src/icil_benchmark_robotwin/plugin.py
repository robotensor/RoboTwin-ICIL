"""The object the competition discovers, and what it may ask of this benchmark.

The surface splits in two, and the split is the whole design.

**Pure** - `info`, `catalogue`, `derive_units`, `verify_prompt`, `read_result` - must import and
run with no simulator, no assets and no GPU. The validator host reads a catalogue, CI checks a
unit list and a third party verifies a published prompt; none of them can build a scene.

**Commands** - `materialize_command`, `run_command` - return an **argv**, not a result. Everything
that needs the simulator happens in a subprocess the orchestrator launches, so the orchestrator
never imports SAPIEN and can run the simulator side in another image, or on another host, without
a line changing here.

Nothing in this module imports `icilval`.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .prompt import CHANNELS
from .units import derive_units

#: The competition ABI this plugin speaks. Version 1 is provisional: it was designed against one
#: benchmark, and a mismatch is refused rather than half-honoured.
BENCHMARK_API_VERSION = 1

#: What the orchestrator calls this benchmark, and what `skills.<skill>.simulator` carries.
BENCHMARK_ID = "robotwin"

#: The only protocol this benchmark implements: one demonstration, and the rollout starts in the
#: identical scene it was recorded in.
PROTOCOL = "same_scene_1demo"

#: The demonstration views this benchmark can serve. Same Scene makes the sensorimotor view
#: degenerate - replaying the demonstration's own actions solves the episode - so it is not
#: offered, and a field asking for it is refused rather than quietly scored.
VIEWS = ("video_only",)

#: The prompt file one materialized unit produces.
PROMPT_NAME = "prompt.npz"
RESULT_NAME = "result.json"


class Benchmark:
    """RoboTwin ICIL, as the competition sees it."""

    id = BENCHMARK_ID
    api_version = BENCHMARK_API_VERSION

    # -- pure ---------------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        from robotwin_icil import __version__ as version

        return {
            "id": self.id,
            "api_version": self.api_version,
            "version": version,
            "title": "RoboTwin 2.0, Same Scene, one demonstration",
            "protocol": PROTOCOL,
            "views": list(VIEWS),
            "action_space": "bimanual_qpos_14",
            "embodiment": "aloha-agilex",
            "prompt_name": PROMPT_NAME,
            # Which array carries which channel, so the orchestrator's demonstration view can
            # allow or drop them as a unit. Only the benchmark knows what its arrays mean; the
            # decision about what a policy may see stays the orchestrator's.
            "demo_channels": {k: list(v) for k, v in CHANNELS.items()},
            "commits": _commits(),
        }

    def catalogue(self) -> dict[str, Any]:
        """Suites, tasks and skill categories. The menu, not the meal: no demonstrations."""
        from robotwin_icil import tasks

        table = tasks.table()
        return {
            "id": self.id,
            "suites": {name: list(members) for name, members in table.suites.items()},
            "categories": dict(table.categories),
            "tasks": {
                name: {"category": task.category, "category_label": task.category_label}
                for name, task in table.tasks.items()
            },
        }

    def derive_units(
        self, *, seed_material: str, count: int, suite: str = "v1", **extra: Any
    ) -> list[dict[str, Any]]:
        units = derive_units(seed_material=seed_material, count=count, suite=suite, **extra)
        return [u.as_dict() for u in units]

    def verify_prompt(self, *, path: str, unit: Mapping[str, Any]) -> dict[str, Any]:
        """Check a materialized prompt is the one this unit asked for, with no simulator.

        Reads the file rather than trusting a manifest: this is what lets anyone holding the
        published prompt confirm what a duel actually ran on.
        """
        import numpy as np

        from .prompt import PROMPT_SCHEMA, prompt_sha256

        target = Path(path)
        if target.is_dir():
            target = target / PROMPT_NAME
        problems: list[str] = []
        if not target.exists():
            return {"ok": False, "sha256": "", "problems": [f"{target.name}: missing"]}
        try:
            with np.load(target, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                frames = int(z["times"].shape[0]) if "times" in z.files else 0
                cameras = list(meta.get("cameras", []))
        except Exception as exc:  # noqa: BLE001 - a corrupt prompt is a duel-stopping fault
            return {"ok": False, "sha256": "", "problems": [f"unreadable: {exc}"]}

        if meta.get("schema") != PROMPT_SCHEMA:
            problems.append(f"schema {meta.get('schema')} is not {PROMPT_SCHEMA}")
        for key in ("task", "scene_seed"):
            if key in unit and meta.get(key) != unit[key]:
                problems.append(f"{key} is {meta.get(key)!r}, the unit asked for {unit[key]!r}")
        if frames < 2:
            problems.append(f"{frames} frames: a demonstration needs at least two")
        if not cameras:
            problems.append("no cameras recorded")
        return {
            "ok": not problems,
            "sha256": prompt_sha256(target),
            "task": meta.get("task"),
            "scene_seed": meta.get("scene_seed"),
            "frames": frames,
            "cameras": cameras,
            "problems": problems,
        }

    def read_result(self, *, out_dir: str) -> dict[str, Any]:
        """One unit's result, as `run_command`'s subprocess wrote it."""
        path = Path(out_dir)
        if path.is_dir():
            path = path / RESULT_NAME
        if not path.exists():
            return {"success": None, "void": True, "steps": None, "error": f"{path.name}: missing"}
        try:
            doc = json.loads(path.read_text())
        except ValueError as exc:
            return {"success": None, "void": True, "steps": None, "error": f"unreadable: {exc}"}
        void = bool(doc.get("void"))
        return {
            "success": None if void else bool(doc.get("success")),
            "void": void,
            "steps": doc.get("steps"),
            "error": doc.get("error"),
            "physics_steps": doc.get("physics_steps"),
            "scene_max_error": doc.get("scene_max_error"),
            "video": doc.get("video"),
        }

    # -- commands -----------------------------------------------------------------------

    def materialize_command(
        self, *, unit: Mapping[str, Any], out_dir: str, **extra: Any
    ) -> Sequence[str]:
        """The argv that produces this unit's prompt, its clip and its hash."""
        argv = [
            "robotwin-icil-competition",
            "materialize",
            "--task",
            str(unit["task"]),
            "--scene-seed",
            str(unit["scene_seed"]),
            "--out",
            str(out_dir),
            "--max-expert-attempts",
            str(unit.get("max_expert_attempts", 20)),
        ]
        return _with_extra(argv, extra)

    def run_command(
        self,
        *,
        unit: Mapping[str, Any],
        prompt: str,
        out_dir: str,
        policy_address: str,
        **extra: Any,
    ) -> Sequence[str]:
        """The argv that runs this unit against a policy already served at `policy_address`."""
        argv = [
            "robotwin-icil-competition",
            "run-unit",
            "--task",
            str(unit["task"]),
            "--scene-seed",
            str(unit["scene_seed"]),
            "--prompt",
            str(prompt),
            "--out",
            str(out_dir),
            "--policy-address",
            str(policy_address),
            "--view",
            str(extra.pop("view", VIEWS[0])),
        ]
        return _with_extra(argv, extra)


def _with_extra(argv: list[str], extra: Mapping[str, Any]) -> list[str]:
    """Pass-through options, so the orchestrator can add one without a change here."""
    for key, value in sorted(extra.items()):
        if value is None or value is False:
            continue
        flag = f"--{key.replace('_', '-')}"
        argv.append(flag)
        if value is not True:
            argv.append(str(value))
    return argv


def _commits() -> dict[str, str | None]:
    """The benchmark and RoboTwin revisions a run is pinned to, when this is a git checkout."""
    root = Path(__file__).resolve().parents[4]
    return {
        "benchmark": _git(root),
        "robotwin": _git(root / "vendor" / "RoboTwin"),
    }


def _git(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


#: What the entry point resolves to.
BENCHMARK = Benchmark()
