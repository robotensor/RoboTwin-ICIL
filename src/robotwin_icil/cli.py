"""`robotwin-icil`: run the benchmark, report a run, list the task table.

`report` and `tasks` never import the simulator, so a finished run directory can be re-reported
anywhere the package installs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import report as report_
from . import tasks as tasks_
from .policy import PolicyError, make_policy
from .records import RecordError, RunDir


def _eval(args: argparse.Namespace) -> int:
    from .robotwin import RoboTwinError, SceneConfig
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
    )
    config = SceneConfig(task_config=args.task_config, save_freq=args.save_freq)
    try:
        records = run(spec, make_policy(args.policy), config)
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


def _tasks(args: argparse.Namespace) -> int:
    table = tasks_.table()
    suites = {name: set(members) for name, members in table.suites.items() if name != "all"}
    for category, members in table.by_category().items():
        print(f"{table.categories[category]} ({category})")
        for task in members:
            tags = ", ".join(sorted(name for name, names in suites.items() if task.name in names))
            print(f"  {task.name}" + (f"  [{tags}]" if tags else ""))
    return 0


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
    run.set_defaults(handler=_eval)

    rep = commands.add_parser("report", help="report a run directory; needs no simulator")
    rep.add_argument("run_dir")
    rep.add_argument(
        "--json", action="store_true", help="fractions as JSON instead of the text report"
    )
    rep.set_defaults(handler=_report)

    lst = commands.add_parser("tasks", help="list the task table and suite membership")
    lst.set_defaults(handler=_tasks)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "episodes", 1) < 1:
        print("robotwin-icil: --episodes must be at least 1", file=sys.stderr)
        return 2
    try:
        return args.handler(args)
    except (RecordError, PolicyError, tasks_.TaskTableError) as exc:
        print(f"robotwin-icil: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
