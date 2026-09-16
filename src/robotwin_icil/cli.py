"""`robotwin-icil`: run the benchmark, report a run, list the task table.

`materialize` and `run-unit` are the benchmark as a competition runs it: one command builds and
saves a demonstration, another evaluates a policy from the saved file, each in a process of its
own. `report` and `tasks` never import the simulator, so a finished run directory can be
re-reported anywhere the package installs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import report as report_
from . import source_sha256
from . import tasks as tasks_
from .arms import LABELS, ONE, TWO
from .demo import DemonstrationError
from .policy import PolicyError, make_policy
from .prompt import PromptError
from .records import RecordError, RunDir, write_json
from .remote import ACT_TIMEOUT_S, POLICY_BUDGET_S, RESULT_RESERVE_S
from .robotwin import DENOISER_ENV, DENOISERS, EMBODIMENTS, RoboTwinError
from .unit import UnitError


def _eval(args: argparse.Namespace) -> int:
    from .robotwin import SceneConfig
    from .runner import RunSpec, run

    table = tasks_.table()
    selected = table.select(task=args.task, arms=args.arms)
    spec = RunSpec(
        run_dir=Path(args.run_dir).resolve(),
        tasks=selected,
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
    selected = table.select(task=args.task, arms=args.arms)
    # Resolve before entering the RoboTwin seam, which moves the working directory.
    out = Path(args.json).resolve() if args.json else None
    config = robotwin.SceneConfig(
        task_config=args.task_config, save_freq=args.save_freq, embodiment=args.embodiment
    )
    seeds = scene_seeds(args.seed, 0, args.seeds)
    results = []
    for task in selected:
        result = survey_task(robotwin.load_task(task.name), task, seeds, config, images=args.images)
        results.append(result)
        print(
            f"{task.name}: expert solved {result.successes}/{result.seeds}, "
            f"{result.demonstrations_moving(1)} with one arm",
            flush=True,
        )
        if out is not None:
            # Rewritten whole after every task: an interrupted survey keeps what it has measured,
            # and a kill that lands mid-write leaves the previous file rather than a torn one.
            # `images` heads the file as well as each task, so no reader takes it for a rendered run.
            out.parent.mkdir(parents=True, exist_ok=True)
            write_json(out, {"images": args.images, "tasks": [r.to_json() for r in results]})
    print()
    print(render(results), end="")
    return 0


def _materialize(args: argparse.Namespace) -> int:
    from . import robotwin
    from .unit import MATERIALIZE_OUTPUTS, clear_outputs, materialize

    # First, so a command that fails from here on leaves nothing an earlier one wrote.
    out = clear_outputs(args.out, MATERIALIZE_OUTPUTS)
    task = tasks_.table()[args.task]
    config = robotwin.SceneConfig(
        task_config=args.task_config, save_freq=args.save_freq, embodiment=args.embodiment
    )
    done = materialize(task.name, args.scene_seeds, config, out)
    print(json.dumps(done.result, indent=2, sort_keys=True))
    # A rejected seed is a result, written to result.json like a prompt; only a harness error
    # exits 1, and then no result.json holds a reason the caller would read past a non-zero exit.
    return 0


def _run_unit(args: argparse.Namespace) -> int:
    from .unit import RUN_UNIT_OUTPUTS, clear_outputs, run_unit

    started = time.monotonic()
    # First, so a command that fails from here on leaves nothing an earlier one wrote.
    out = clear_outputs(args.out, RUN_UNIT_OUTPUTS, prompt=args.prompt)
    if args.policy_address is not None:
        from . import remote

        deadline = None
        if args.unit_timeout_s is not None:
            # No call to the policy runs into the caller's kill: run-unit keeps time to write why.
            deadline = started + args.unit_timeout_s - remote.RESULT_RESERVE_S
        # The key comes from the environment, never argv; nothing connects until the unit resets
        # the policy, so a policy that cannot be reached is the unit's result, not an exit 1.
        policy = remote.RemotePolicy(
            args.policy_address,
            remote.authkey_from_env(args.authkey_env),
            act_timeout_s=(
                remote.ACT_TIMEOUT_S if args.act_timeout_s is None else args.act_timeout_s
            ),
            policy_budget_s=(
                remote.POLICY_BUDGET_S if args.policy_budget_s is None else args.policy_budget_s
            ),
            deadline=deadline,
            log_file=args.policy_log,
        )
    else:
        policy = make_policy(args.policy, **_policy_kwargs(args.policy_arg or []))
    try:
        # Resolve before entering the RoboTwin seam, which moves the working directory.
        result = run_unit(Path(args.prompt).resolve(), policy, out)
    finally:
        # However the unit ended: a served policy's server exits once its client has gone.
        policy.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    # A failed or void unit is a result, written to result.json; only a harness error exits 1.
    return 0


def _policy_kwargs(pairs: list[str]) -> dict[str, str]:
    kwargs: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise PolicyError(f"--policy-arg takes key=value, not {pair!r}")
        kwargs[key] = value
    return kwargs


def _tasks(args: argparse.Namespace) -> int:
    table = tasks_.table()
    for category, members in table.by_category().items():
        if args.arms == ONE:
            members = tuple(task for task in members if task.arms == ONE)
        if not members:
            continue
        print(f"{table.categories[category]} ({category})")
        for task in members:
            print(f"  {task.name}  ({LABELS[task.arms]})")
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
    run.add_argument("--task", help="a single RoboTwin task; without it, all cataloged tasks")
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
    sur.add_argument("--task", help="a single RoboTwin task; without it, all cataloged tasks")
    sur.add_argument("--seeds", type=int, default=20, help="seeds per task")
    sur.add_argument("--seed", type=int, default=0, help="global seed for the seed stream")
    sur.add_argument("--json", help="also write the per-task results to this file")
    sur.add_argument("--task-config", default="demo_clean")
    _add_embodiment(sur)
    sur.add_argument("--save-freq", type=int, default=15)
    sur.add_argument(
        "--images",
        action="store_true",
        help="render every camera at every frame, as eval does; the survey reads only joints, "
        "so without this no camera renders",
    )
    _add_arms(sur)
    sur.set_defaults(handler=_survey)

    mat = commands.add_parser(
        "materialize",
        help="build one demonstration and save it: prompt.npz, demonstration.mp4, result.json; "
        "exits 0 whenever result.json was written, every candidate seed rejected included",
    )
    mat.add_argument("--task", required=True, help="a single RoboTwin task")
    mat.add_argument(
        "--scene-seed",
        dest="scene_seeds",
        type=int,
        action="append",
        required=True,
        metavar="SEED",
        help="a candidate scene; repeat it to give more, tried in order until the expert solves one",
    )
    mat.add_argument("--out", required=True, help="directory the three files are written into")
    mat.add_argument(
        "--task-config", default="demo_clean", help="RoboTwin env_cfg/task_config name"
    )
    _add_embodiment(mat)
    mat.add_argument(
        "--save-freq", type=int, default=15, help="control steps per demonstration frame"
    )
    _add_denoiser(mat)
    _add_expect_source(mat)
    mat.set_defaults(handler=_materialize)

    unit = commands.add_parser(
        "run-unit",
        help="evaluate a policy from a saved prompt: rebuilds and verifies its scene, writes "
        "result.json and evaluation.mp4; exits 0 whether the policy succeeded, failed or could "
        "not be reached",
    )
    unit.add_argument("--prompt", required=True, help="a prompt.npz written by materialize")
    policy = unit.add_mutually_exclusive_group(required=True)
    policy.add_argument(
        "--policy",
        help="a policy run in this process: built-in name (replay, dummy) or module:Class",
    )
    policy.add_argument(
        "--policy-address",
        metavar="ADDR",
        help="a policy served by `python -m icil_policy.serve`: a Unix socket path or host:port; "
        "needs --authkey-env",
    )
    unit.add_argument(
        "--policy-arg",
        action="append",
        metavar="KEY=VALUE",
        help="a keyword argument for --policy's constructor; repeatable",
    )
    unit.add_argument(
        "--authkey-env",
        metavar="NAME",
        help="the environment variable holding the served policy's key as hex, at least 16 "
        "bytes; the key itself is never an argument",
    )
    unit.add_argument(
        "--act-timeout-s",
        type=_positive_seconds,
        metavar="S",
        help=f"how long one act of the served policy may take (default {ACT_TIMEOUT_S:g})",
    )
    unit.add_argument(
        "--policy-budget-s",
        type=_positive_seconds,
        metavar="S",
        help="how long all the calls to the served policy may take together in the unit; the "
        f"call it runs out in voids the unit on the policy (default {POLICY_BUDGET_S:g})",
    )
    unit.add_argument(
        "--unit-timeout-s",
        type=_positive_seconds,
        metavar="S",
        help="how long run-unit has before whoever started it kills it: no call to the served "
        f"policy runs within {RESULT_RESERVE_S:g}s of that, and one cut short there while the "
        "policy is within its budget voids the unit on the harness",
    )
    unit.add_argument(
        "--policy-log",
        metavar="PATH",
        help="the served policy's log file, whose tail ends the error of a unit it voided",
    )
    unit.add_argument("--out", required=True, help="directory the result and clip are written into")
    _add_denoiser(unit)
    _add_expect_source(unit)
    unit.set_defaults(handler=_run_unit)

    lst = commands.add_parser("tasks", help="list the task table: skill category and arms")
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


def _add_expect_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expect-source-sha256",
        metavar="HEX",
        help="refuse to run, exit 1, unless this benchmark's source digests to HEX "
        "(robotwin_icil.source_sha256); the competition plugin passes its own",
    )


def _add_denoiser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--denoiser",
        choices=DENOISERS,
        help=f"the ray-tracing denoiser, as ${DENOISER_ENV} sets it, for a caller that passes "
        "the command no such variable (default: none on GPUs OIDN cannot run on, else RoboTwin's)",
    )


def _positive_seconds(text: str) -> float:
    try:
        seconds = float(text)
    except ValueError:
        seconds = float("nan")
    if not 0 < seconds < float("inf"):
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive number of seconds")
    return seconds


def _run_unit_usage(args: argparse.Namespace) -> str | None:
    """Why run-unit's policy flags do not go together, or None: the parser keeps --policy and
    --policy-address apart, and this keeps each one's own flags with it."""
    if args.policy_address is not None:
        if not args.authkey_env:
            return "--policy-address needs --authkey-env, the variable holding the policy's key"
        if args.policy_arg:
            return "--policy-arg configures a policy run in this process, not a served one"
        return None
    given = [
        flag
        for flag, value in (
            ("--authkey-env", args.authkey_env),
            ("--act-timeout-s", args.act_timeout_s),
            ("--policy-budget-s", args.policy_budget_s),
            ("--unit-timeout-s", args.unit_timeout_s),
            ("--policy-log", args.policy_log),
        )
        if value is not None
    ]
    if given:
        return f"{', '.join(given)} go with --policy-address, not --policy"
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "seeds", 1) < 1:
        print("robotwin-icil: --seeds must be at least 1", file=sys.stderr)
        return 2
    if getattr(args, "episodes", 1) < 1:
        print("robotwin-icil: --episodes must be at least 1", file=sys.stderr)
        return 2
    usage = _run_unit_usage(args) if args.command == "run-unit" else None
    if usage is not None:
        print(f"robotwin-icil: {usage}", file=sys.stderr)
        return 2
    expected = getattr(args, "expect_source_sha256", None)
    if expected is not None and expected != source_sha256():
        # Before anything is cleared or written: the command line was built for other code, and
        # its flags may not mean here what its builder meant.
        print(
            f"robotwin-icil: this benchmark's source digests to {source_sha256()}, not the "
            f"{expected} the command was built for: the interpreter runs another robotwin_icil",
            file=sys.stderr,
        )
        return 1
    if getattr(args, "denoiser", None):
        # Read by the RoboTwin seam when it sets a scene up, which is after this.
        os.environ[DENOISER_ENV] = args.denoiser
    try:
        return args.handler(args)
    except (
        RecordError,
        PolicyError,
        PromptError,
        RoboTwinError,
        DemonstrationError,
        tasks_.TaskTableError,
        UnitError,
    ) as exc:
        print(f"robotwin-icil: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
