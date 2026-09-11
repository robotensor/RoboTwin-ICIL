"""`robotwin-icil`: run the benchmark, report a run, list the task table, look through the cameras.

`report` and `tasks` never import the simulator, so a finished run directory can be re-reported
anywhere the package installs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

from . import camera_profiles
from . import report as report_
from . import tasks as tasks_
from .policy import PolicyError, make_policy
from .records import RecordError, RunDir
from .robotwin import RoboTwinError


def _eval(args: argparse.Namespace) -> int:
    from .robotwin import SceneConfig
    from .runner import RunSpec, run

    table = tasks_.table()
    if args.task:
        selected, suite = (table[args.task],), None
    else:
        selected, suite = table.suite(args.suite), args.suite
    spec = RunSpec(
        run_dir=Path(args.run_dir).resolve(),
        tasks=selected,
        suite=suite,
        episodes=args.episodes,
        global_seed=args.seed,
        max_expert_attempts=args.max_expert_attempts,
        video=args.video,
        video_camera=camera_profiles.get(args.camera_profile).video_camera,
        policy_config=args.policy_args,
    )
    config = SceneConfig(
        task_config=args.task_config,
        save_freq=args.save_freq,
        camera_profile=args.camera_profile,
    )
    try:
        records = run(spec, make_policy(args.policy, **args.policy_args), config)
    except RoboTwinError as exc:
        print(f"robotwin-icil: {exc}", file=sys.stderr)
        return 1
    built = report_.build(records, table)
    print()
    print(report_.render(built, RunDir(spec.run_dir).manifest(), table), end="")
    return 0


def _report(args: argparse.Namespace) -> int:
    run_dir = RunDir(Path(args.run_dir))
    table = tasks_.table()
    built = report_.build(run_dir.records(), table)
    if args.json:
        print(json.dumps(built.to_json(), indent=2))
    else:
        print(report_.render(built, run_dir.manifest(), table), end="")
    return 0


def _survey(args: argparse.Namespace) -> int:
    from . import robotwin
    from .generate import scene_seeds
    from .survey import render, survey_task

    table = tasks_.table()
    selected = (table[args.task],) if args.task else table.suite(args.suite)
    # Resolve before entering the RoboTwin seam, which moves the working directory.
    out = Path(args.json).resolve() if args.json else None
    config = robotwin.SceneConfig(
        task_config=args.task_config,
        save_freq=args.save_freq,
        camera_profile=args.camera_profile,
    )
    seeds = scene_seeds(args.seed, 0, args.seeds)
    profile = camera_profiles.get(args.camera_profile).identity()
    print(f"camera profile {profile['name']} (sha256 {profile['sha256'][:12]})", flush=True)
    results = []
    for task in selected:
        result = survey_task(robotwin.load_task(task.name), task, seeds, config)
        results.append(result)
        print(f"{task.name}: expert solved {result.successes}/{result.seeds}", flush=True)
        if out is not None:
            # Rewritten after every task: an interrupted survey keeps what it has measured.
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps([r.to_json() for r in results], indent=2) + "\n"
            out.write_text(payload, encoding="utf-8")
    print()
    print(render(results), end="")
    return 0


def _cameras(args: argparse.Namespace) -> int:
    from . import robotwin
    from .png import write_png

    task = tasks_.table()[args.task]
    # Resolve before entering the RoboTwin seam, which moves the working directory.
    out = Path(args.out).resolve()
    config = robotwin.SceneConfig(task_config=args.task_config, camera_profile=args.camera_profile)
    for name, image in sorted(robotwin.snapshot(task.name, args.seed, config).items()):
        path = write_png(out / f"{name}.png", image)
        print(f"{name}: {image.shape[1]}x{image.shape[0]}  {path}")
    return 0


def _tasks(args: argparse.Namespace) -> int:
    table = tasks_.table()
    suites = {name: set(members) for name, members in table.suites.items() if name != "all"}
    for category, members in table.by_category().items():
        print(f"{table.categories[category]} ({category})")
        for task in members:
            tags = ", ".join(sorted(name for name, names in suites.items() if task.name in names))
            print(f"  {task.name}" + (f"  [{tags}]" if tags else ""))
    return 0


_NULLS = ("", "~", "null", "Null", "NULL")


def parse_policy_arg(item: str) -> tuple[str, Any]:
    """One `--policy-arg KEY=VALUE`, or `ValueError` saying why it is not one.

    The value is read as YAML, so numbers, booleans and null arrive typed; a quoted value is the
    string inside the quotes, and anything else stays the string it was. KEY must be a Python
    identifier, since it becomes a keyword argument.
    """
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"{item!r} is not KEY=VALUE")
    if not key.isidentifier():
        raise ValueError(f"{key!r} is not a Python identifier")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        return key, raw
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{key}={raw}: a number must be finite")
    if isinstance(value, (bool, int, float)):
        return key, value
    if value is None and raw.strip() in _NULLS:
        return key, None
    if isinstance(value, str) and raw.strip()[:1] in ("'", '"'):
        return key, value
    return key, raw


class _PolicyArgs(argparse.Action):
    """Collects every `--policy-arg KEY=VALUE` into one mapping; a KEY given twice is refused."""

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            key, value = parse_policy_arg(values)
        except ValueError as exc:
            parser.error(f"{option_string}: {exc}")
        collected = dict(getattr(namespace, self.dest) or {})
        if key in collected:
            parser.error(f"{option_string}: {key!r} is given twice")
        collected[key] = value
        setattr(namespace, self.dest, collected)


def _add_camera_profile(parser: argparse.ArgumentParser, flag: str = "--camera-profile") -> None:
    parser.add_argument(
        flag,
        dest="camera_profile",
        default=camera_profiles.STOCK,
        choices=camera_profiles.names(),
        help="a camera profile from cameras.yml; every profile keeps every seed's scene",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robotwin-icil",
        description="Same Scene one-demonstration ICIL benchmark on RoboTwin 2.0.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("eval", help="run episodes and record them in a run directory")
    run.add_argument(
        "--policy", required=True, help="built-in name (replay, dummy) or module:Class"
    )
    run.add_argument(
        "--policy-arg",
        dest="policy_args",
        action=_PolicyArgs,
        default={},
        metavar="KEY=VALUE",
        help="a keyword argument for the policy, its value read as YAML; repeatable "
        "(an adapter's many settings go in its own YAML: config=PATH)",
    )
    which = run.add_mutually_exclusive_group(required=True)
    which.add_argument("--suite", help="a suite from tasks.yml, e.g. v1")
    which.add_argument("--task", help="a single RoboTwin task")
    run.add_argument("--episodes", type=int, required=True)
    run.add_argument("--seed", type=int, default=0, help="global benchmark seed")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--max-expert-attempts", type=int, default=20)
    run.add_argument(
        "--task-config", default="demo_clean", help="RoboTwin env_cfg/task_config name"
    )
    run.add_argument(
        "--save-freq", type=int, default=15, help="control steps per demonstration frame"
    )
    run.add_argument(
        "--video",
        action="store_true",
        help="write demonstration.mp4 and evaluation_same_scene.mp4 per episode",
    )
    _add_camera_profile(run)
    run.set_defaults(handler=_eval)

    rep = commands.add_parser("report", help="report a run directory; needs no simulator")
    rep.add_argument("run_dir")
    rep.add_argument(
        "--json", action="store_true", help="fractions as JSON instead of the text report"
    )
    rep.set_defaults(handler=_report)

    sur = commands.add_parser(
        "survey", help="run RoboTwin's expert alone and report how often it succeeds per task"
    )
    which = sur.add_mutually_exclusive_group(required=True)
    which.add_argument("--suite", help="a suite from tasks.yml, e.g. v1 or all")
    which.add_argument("--task", help="a single RoboTwin task")
    sur.add_argument("--seeds", type=int, default=20, help="seeds per task")
    sur.add_argument("--seed", type=int, default=0, help="global seed for the seed stream")
    sur.add_argument("--json", help="also write the per-task results to this file")
    sur.add_argument("--task-config", default="demo_clean")
    sur.add_argument("--save-freq", type=int, default=15)
    _add_camera_profile(sur)
    sur.set_defaults(handler=_survey)

    cam = commands.add_parser(
        "cameras",
        help="write one PNG per camera of one scene; how a camera profile is checked by eye",
    )
    _add_camera_profile(cam, "--profile")
    cam.add_argument("--task", required=True, help="a RoboTwin task")
    cam.add_argument("--seed", type=int, default=0, help="the scene seed itself")
    cam.add_argument("--out", required=True, help="directory for <camera>.png")
    cam.add_argument("--task-config", default="demo_clean")
    cam.set_defaults(handler=_cameras)

    lst = commands.add_parser("tasks", help="list the task table and suite membership")
    lst.set_defaults(handler=_tasks)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_parser()
    except camera_profiles.ProfileError as exc:
        print(f"robotwin-icil: {exc}", file=sys.stderr)
        return 1
    args = parser.parse_args(argv)
    if getattr(args, "seeds", 1) < 1:
        print("robotwin-icil: --seeds must be at least 1", file=sys.stderr)
        return 2
    if getattr(args, "episodes", 1) < 1:
        print("robotwin-icil: --episodes must be at least 1", file=sys.stderr)
        return 2
    try:
        return args.handler(args)
    except (
        RecordError,
        PolicyError,
        RoboTwinError,
        tasks_.TaskTableError,
        camera_profiles.ProfileError,
    ) as exc:
        print(f"robotwin-icil: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
