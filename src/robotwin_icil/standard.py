"""Robotensor's standard profile: RoboTwin 2.0's own evaluation, with the demonstration as input.

RoboTwin scores a policy on every task, 100 episodes each, from scene seed `100000 * (1 + seed)`
upwards, skipping each seed its expert cannot solve, under `demo_clean` (Easy) and
`demo_randomized` (Hard). This profile keeps all of that — the tasks, the seed stream, the
settings, the step limits and `check_success()` — and changes one thing: the expert's successful
trajectory on the scored scene is handed to the frozen policy as its one demonstration.
Independent of duel unit derivation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import source_sha256
from .generate import ROBOTWIN_SEED_STREAM
from .records import SAME_SCENE, RunDir, write_json
from .report import build
from .tasks import table

PROFILE = "robotwin-icil-standard"
# RoboTwin's two evaluation settings, by the name its leaderboard uses and its task config.
SETTINGS = {"clean": "demo_clean", "randomized": "demo_randomized"}
SEEDS = (0,)
EPISODES_PER_TASK = 100
# RoboTwin itself has no per-episode cap; this bounds a task whose expert stops solving scenes.
MAX_EXPERT_ATTEMPTS = 50
EMBODIMENT = "aloha-agilex"
SAVE_FREQ = 15


class StandardError(ValueError):
    """A result cannot claim the standard protocol."""


def task_names() -> tuple[str, ...]:
    names = tuple(table().tasks)
    if len(names) != 50:
        raise StandardError("the standard profile requires the original 50-task RoboTwin catalogue")
    return names


def metadata() -> dict[str, Any]:
    return {
        "profile": PROFILE,
        "tasks": list(task_names()),
        "evaluation_setting": SAME_SCENE,
        "demonstrations_per_episode": 1,
        "seed_stream": ROBOTWIN_SEED_STREAM,
        "max_expert_attempts": MAX_EXPERT_ATTEMPTS,
        "embodiment": EMBODIMENT,
        "save_freq": SAVE_FREQ,
    }


def validate_run(spec, config) -> None:
    if spec.standard != metadata():
        raise StandardError("standard profile metadata does not match this implementation")
    tasks = tuple(t.name for t in spec.tasks)
    if tasks != task_names() or spec.episodes < 1 or spec.episodes % len(tasks):
        raise StandardError("standard evaluation requires all 50 tasks, equally allocated")
    if (
        spec.arms != "2"
        or spec.max_expert_attempts != MAX_EXPERT_ATTEMPTS
        or spec.seed_stream != ROBOTWIN_SEED_STREAM
    ):
        raise StandardError(
            "standard evaluation requires all arm counts, RoboTwin's seed stream and "
            f"{MAX_EXPERT_ATTEMPTS} expert attempts"
        )
    if (
        config.task_config not in SETTINGS.values()
        or config.embodiment != EMBODIMENT
        or config.save_freq != SAVE_FREQ
        or config.head_camera is not None
        or config.overrides
    ):
        raise StandardError(
            "standard evaluation requires unmodified demo_clean or demo_randomized on "
            f"{EMBODIMENT}, save_freq={SAVE_FREQ}"
        )


def start(root: Path, plan: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "standard.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise StandardError("this directory belongs to a different standard evaluation plan")
    write_json(path, plan)


def _valid_seeds(seeds) -> bool:
    return (
        isinstance(seeds, list)
        and bool(seeds)
        and all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds)
        and len(seeds) == len(set(seeds))
    )


def plan(policy: dict, settings: list[str], seeds: list[int], episodes_per_task: int) -> dict:
    if not settings or len(settings) != len(set(settings)) or not set(settings) <= set(SETTINGS):
        raise StandardError(f"choose distinct settings from {', '.join(SETTINGS)}")
    if not _valid_seeds(seeds) or episodes_per_task < 1:
        raise StandardError("provide distinct seeds and a positive episodes-per-task count")
    return {
        "standard": metadata(),
        "source_sha256": source_sha256(),
        "policy": policy,
        "settings": settings,
        "seeds": seeds,
        "episodes_per_task": episodes_per_task,
    }


def run_dir(root: Path, setting: str, seed: int) -> Path:
    return root / setting / f"seed_{seed}"


def score(root: Path) -> dict[str, Any]:
    try:
        return _score(Path(root).resolve())
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise StandardError(f"cannot read standard results: {exc}") from exc


def _score(root: Path) -> dict[str, Any]:
    path = root / "standard.json"
    if not path.is_file():
        raise StandardError(f"no standard evaluation plan at {path}")
    p = json.loads(path.read_text())
    if p["standard"] != metadata():
        raise StandardError("standard plan uses another profile")
    settings, seeds, budget = p["settings"], p["seeds"], p["episodes_per_task"]
    if (
        not isinstance(settings, list)
        or not settings
        or not set(settings) <= set(SETTINGS)
        or not _valid_seeds(seeds)
        or not isinstance(budget, int)
        or budget < 1
    ):
        raise StandardError("malformed standard settings, seed groups or episode budget")
    baseline = None
    results = {}
    for setting in settings:
        result, identity = _score_setting(root, p, setting)
        if baseline is not None and identity is not None and identity != baseline:
            raise StandardError("settings use different benchmark or simulator revisions")
        baseline = baseline or identity
        results[setting] = result
    return {**p, "complete": all(r["complete"] for r in results.values()), "settings": results}


def _score_setting(root: Path, p: dict, setting: str) -> tuple[dict[str, Any], tuple | None]:
    names = task_names()
    seeds, budget = p["seeds"], p["episodes_per_task"]
    records = []
    per_seed = {}
    baseline = None
    for seed in seeds:
        rd = RunDir(run_dir(root, setting, seed))
        rs = rd.records()
        if not rd.manifest_path.is_file():
            if rs:
                raise StandardError(f"{setting} seed {seed}: records without a manifest")
            per_seed[str(seed)] = {"recorded": 0, "scored": 0}
            continue
        manifest = rd.manifest()
        if (
            manifest.global_seed != seed
            or manifest.tasks != names
            or manifest.episodes != budget * len(names)
            or manifest.evaluation_setting != SAME_SCENE
            or manifest.arms != "2"
            or manifest.max_expert_attempts != MAX_EXPERT_ATTEMPTS
            or manifest.seed_stream != ROBOTWIN_SEED_STREAM
            or manifest.policy != p["policy"]
            or manifest.benchmark_config.get("standard") != p["standard"]
            or manifest.benchmark_config.get("source_sha256") != p["source_sha256"]
        ):
            raise StandardError(
                f"{setting} seed {seed}: manifest is inconsistent with the standard plan"
            )
        config = manifest.benchmark_config
        if any(
            config.get(k) != v
            for k, v in {
                "task_config": SETTINGS[setting],
                "embodiment": EMBODIMENT,
                "save_freq": SAVE_FREQ,
                "head_camera": None,
                "overrides": {},
            }.items()
        ):
            raise StandardError(f"{setting} seed {seed}: nonstandard scene configuration")
        identity = (manifest.benchmark_commit, manifest.robotwin_commit)
        if baseline is not None and identity != baseline:
            raise StandardError(f"{setting}: seed groups use different simulator revisions")
        baseline = identity
        ids = set()
        for r in rs:
            if (
                r.episode in ids
                or not 0 <= r.episode < manifest.episodes
                or r.task != names[r.episode % len(names)]
                or r.evaluation_setting != SAME_SCENE
                or r.embodiment != EMBODIMENT
                or r.skill_category != table()[r.task].category
                or r.model != str(p["policy"].get("model", p["policy"]["policy"]))
                or r.checkpoint != p["policy"].get("checkpoint")
            ):
                raise StandardError(f"{setting} seed {seed}: duplicate or out-of-profile episode")
            ids.add(r.episode)
        per_seed[str(seed)] = {"recorded": len(rs), "scored": sum(r.scored for r in rs)}
        records.extend(rs)
    tasks = {}
    for task in names:
        rs = [r for r in records if r.task == task]
        valid = [r for r in rs if r.scored]
        successes = sum(bool(r.success) for r in valid)
        tasks[task] = {
            "category": table()[task].category,
            "planned": budget * len(seeds),
            "recorded": len(rs),
            "scored": len(valid),
            "successes": successes,
            "success_rate": successes / len(valid) if valid else None,
        }
    complete = all(t["scored"] == t["planned"] for t in tasks.values())
    rates = [t["success_rate"] for t in tasks.values()]
    macro = sum(rates) / len(rates) if all(r is not None for r in rates) else None
    categories = {}
    for category in table().categories:
        members = [t["success_rate"] for t in tasks.values() if t["category"] == category]
        if members:
            categories[category] = (
                sum(members) / len(members) if all(r is not None for r in members) else None
            )
    official = complete and tuple(seeds) == SEEDS and budget == EPISODES_PER_TASK
    return {
        "task_config": SETTINGS[setting],
        "complete": complete,
        "official_protocol": official,
        "success_rate": macro,
        "official_score": macro if official else None,
        "by_category": categories,
        "by_task": tasks,
        "by_seed": per_seed,
        "diagnostics": build(records, table()).diagnostics.to_json(),
    }, baseline
