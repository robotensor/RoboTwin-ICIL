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
from .arms import LABELS, ONE, TWO
from .demo import DemonstrationError
from .policy import PolicyError, make_policy
from .records import RecordError, RunDir
from .robotwin import EMBODIMENTS, RoboTwinError


def _eval(args: argparse.Namespace) -> int:
    from .robotwin import SceneConfig
    from .runner import RunSpec, run

    table = tasks_.table()
    selected = table.select(suite=args.suite, task=args.task, arms=args.arms)
    spec = RunSpec(
        run_dir=Path(args.run_dir).resolve(),
        tasks=selected,
        suite=args.suite,
        episodes=args.episodes,
        global_seed=args.seed,
        max_expert_attempts=args.max_expert_attempts,
        video=args.video,
        arms=args.arms,
    )
    config = SceneConfig(
        task_config=args.task_config, save_freq=args.save_freq, embodiment=args.embodiment
    )
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


def _survey(args: argparse.Namespace) -> int:
    from . import robotwin
    from .generate import scene_seeds
    from .survey import render, survey_task

    table = tasks_.table()
    selected = table.select(suite=args.suite, task=args.task, arms=args.arms)
    # Resolve before entering the RoboTwin seam, which moves the working directory.
    out = Path(args.json).resolve() if args.json else None
    config = robotwin.SceneConfig(
        task_config=args.task_config, save_freq=args.save_freq, embodiment=args.embodiment
    )
    seeds = scene_seeds(args.seed, 0, args.seeds)
    results = []
    for task in selected:
        result = survey_task(robotwin.load_task(task.name), task, seeds, config)
        results.append(result)
        print(
            f"{task.name}: expert solved {result.successes}/{result.seeds}, "
            f"{result.demonstrations_moving(1)} with one arm",
            flush=True,
        )
        if out is not None:
            # Rewritten after every task: an interrupted survey keeps what it has measured.
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps([r.to_json() for r in results], indent=2) + "\n"
            out.write_text(payload, encoding="utf-8")
    print()
    print(render(results), end="")
    return 0


def _tasks(args: argparse.Namespace) -> int:
    table = tasks_.table()
    suites = {name: set(members) for name, members in table.suites.items() if name != "all"}
    for category, members in table.by_category().items():
        if args.arms == ONE:
            members = tuple(task for task in members if task.arms == ONE)
        if not members:
            continue
        print(f"{table.categories[category]} ({category})")
        for task in members:
            tags = ", ".join(sorted(name for name, names in suites.items() if task.name in names))
            print(f"  {task.name}  ({LABELS[task.arms]})" + (f"  [{tags}]" if tags else ""))
    return 0


def _add_arms(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--arms",
        choices=[ONE, TWO],
        default=TWO,
        help="1 selects only tasks whose expert uses one arm; 2 (the default) changes nothing",
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
    _add_embodiment(run)
    run.add_argument(
        "--save-freq", type=int, default=15, help="control steps per demonstration frame"
    )
    run.add_argument(
        "--video",
        action="store_true",
        help="write demonstration.mp4 and evaluation_same_scene.mp4 per episode",
    )
    _add_arms(run)
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
    _add_embodiment(sur)
    sur.add_argument("--save-freq", type=int, default=15)
    _add_arms(sur)
    sur.set_defaults(handler=_survey)

    lst = commands.add_parser(
        "tasks", help="list the task table: skill category, arms and suite membership"
    )
    _add_arms(lst)
    lst.set_defaults(handler=_tasks)
    return parser


def _add_embodiment(parser: argparse.ArgumentParser) -> None:
    # No default: without the flag the task config's own robot runs, as `--task-config` chose it.
    # Every shipped config names aloha-agilex, so a plain run is unchanged.
    parser.add_argument(
        "--embodiment",
        choices=sorted(EMBODIMENTS),
        help="the robot: aloha-agilex (one dual-arm URDF, 14-wide qpos) or franka-panda "
        "(two Franka arms 0.8 m apart, 16-wide qpos); without it, the task config's own, "
        "aloha-agilex in every shipped config",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        DemonstrationError,
        tasks_.TaskTableError,
    ) as exc:
        print(f"robotwin-icil: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
