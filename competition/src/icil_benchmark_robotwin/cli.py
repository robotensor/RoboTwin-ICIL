"""`robotwin-icil-competition`: what the plugin's argv builders name.

Two halves, matching the plugin's own split. `catalogue` and `units` need no simulator and are
covered in CI. `materialize` and `run-unit` build a scene, so they import RoboTwin inside the
function that needs it and are exercised against the real simulator separately.

The orchestrator calls these rather than importing this package's simulator half, which is what
lets the simulator side run in another image, or on another host, with nothing changing here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .plugin import BENCHMARK, RESULT_NAME, VIEWS
from .units import derive_units


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="robotwin-icil-competition", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    info = sub.add_parser("info", help="what this benchmark is, and what it is pinned to")
    info.set_defaults(func=cmd_info)

    cat = sub.add_parser("catalogue", help="suites, tasks and skill categories")
    cat.set_defaults(func=cmd_catalogue)

    units = sub.add_parser("units", help="derive a duel's unit list")
    units.add_argument("--seed-material", required=True, help="the duel id, or any stable string")
    units.add_argument("--count", type=int, required=True)
    units.add_argument("--suite", default="v1")
    units.add_argument("--out", default=None)
    units.set_defaults(func=cmd_units)

    mat = sub.add_parser("materialize", help="produce one unit's prompt (needs the simulator)")
    mat.add_argument("--task", required=True)
    mat.add_argument("--scene-seed", type=int, required=True)
    mat.add_argument("--out", required=True)
    mat.add_argument("--max-expert-attempts", type=int, default=20)
    mat.add_argument("--task-config", default="demo_clean")
    mat.add_argument("--save-freq", type=int, default=15)
    mat.add_argument("--no-video", action="store_true")
    mat.set_defaults(func=cmd_materialize)

    run = sub.add_parser(
        "run-unit", help="run one unit against a served policy (needs the simulator)"
    )
    run.add_argument("--task", required=True)
    run.add_argument("--scene-seed", type=int, required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--policy-address", default=None, help="a policy the orchestrator is serving")
    run.add_argument("--policy", default=None, help="module:Class, run in this process")
    run.add_argument("--authkey-file", default=None)
    run.add_argument("--view", default=VIEWS[0], choices=list(VIEWS))
    run.add_argument("--task-config", default="demo_clean")
    run.add_argument("--no-video", action="store_true")
    run.set_defaults(func=cmd_run_unit)

    verify = sub.add_parser("verify", help="check a materialized prompt, with no simulator")
    verify.add_argument("--prompt", required=True)
    verify.add_argument("--task", default=None)
    verify.add_argument("--scene-seed", type=int, default=None)
    verify.set_defaults(func=cmd_verify)

    return p


def cmd_info(args: Any) -> int:
    print(json.dumps(BENCHMARK.info(), indent=2, sort_keys=True))
    return 0


def cmd_catalogue(args: Any) -> int:
    print(json.dumps(BENCHMARK.catalogue(), indent=2, sort_keys=True))
    return 0


def cmd_units(args: Any) -> int:
    units = derive_units(seed_material=args.seed_material, count=args.count, suite=args.suite)
    doc = [u.as_dict() for u in units]
    text = json.dumps(doc, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)
    return 0


def cmd_verify(args: Any) -> int:
    unit: dict[str, Any] = {}
    if args.task:
        unit["task"] = args.task
    if args.scene_seed is not None:
        unit["scene_seed"] = args.scene_seed
    verdict = BENCHMARK.verify_prompt(path=args.prompt, unit=unit)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


def cmd_materialize(args: Any) -> int:
    """Generate one unit's demonstration and write it where the orchestrator asked.

    The expert is RoboTwin's own, and a seed it fails on is redrawn - expert failure is a
    generation failure, never a model failure. A unit whose expert never succeeds fails here,
    before either side of the duel has started, which is the point of materializing at all.
    """
    from .materialize import materialize_unit

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = materialize_unit(
            task=args.task,
            scene_seed=args.scene_seed,
            out_dir=out_dir,
            max_expert_attempts=args.max_expert_attempts,
            task_config=args.task_config,
            save_freq=args.save_freq,
            record_video=not args.no_video,
        )
    except Exception as exc:  # noqa: BLE001 - reported to the orchestrator, which substitutes
        print(f"materialize failed: {exc}", file=sys.stderr)
        return 1
    (out_dir / "materialize.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_run_unit(args: Any) -> int:
    """Rebuild the prompt's scene, hand the policy its demonstration, roll out, score."""
    from .run import run_unit

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_unit(
        task=args.task,
        scene_seed=args.scene_seed,
        prompt=Path(args.prompt),
        out_dir=out_dir,
        policy_address=args.policy_address,
        policy_spec=args.policy,
        authkey_file=args.authkey_file,
        view=args.view,
        task_config=args.task_config,
        record_video=not args.no_video,
    )
    (out_dir / RESULT_NAME).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    # A void unit is not a failure of this process: the orchestrator decides what to do with it.
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
