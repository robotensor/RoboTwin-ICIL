"""The argv of the benchmark's two commands, run in the simulator's environment.

The orchestrator runs these; it never imports the simulator, and the plugin builds them without
one. Every argv is `[python, "-m", "robotwin_icil.cli", ...]`: `python` is the simulator
environment's interpreter, and every path is absolute, since the orchestrator chooses the working
directory and RoboTwin moves its own.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from robotwin_icil import tasks
from robotwin_icil.robotwin import EMBODIMENTS

#: The environment variable naming the simulator environment's interpreter.
PYTHON_ENV = "ROBOTWIN_ICIL_PYTHON"
#: Where `scripts/install_robotwin.sh` puts it.
DEFAULT_PYTHON = "/root/miniforge3/envs/robotwin/bin/python"
CLI_MODULE = "robotwin_icil.cli"

#: RoboTwin's task config and frame spacing every unit is materialized with, written into the argv
#: rather than left to the command's defaults, so a changed default cannot change a unit.
TASK_CONFIG = "demo_clean"
SAVE_FREQ = 15


def simulator_python() -> str:
    """The interpreter the commands run under: `$ROBOTWIN_ICIL_PYTHON`, or `DEFAULT_PYTHON`."""
    return os.environ.get(PYTHON_ENV) or DEFAULT_PYTHON


def materialize_command(*, unit: Mapping[str, Any], out_dir: str) -> list[str]:
    """`materialize` for `unit`: its task on its robot, every candidate seed in order, into
    `out_dir`."""
    task, embodiment, seeds = _unit(unit)
    argv = [simulator_python(), "-m", CLI_MODULE, "materialize", "--task", task]
    argv += [
        "--embodiment",
        embodiment,
        "--task-config",
        TASK_CONFIG,
        "--save-freq",
        str(SAVE_FREQ),
    ]
    for seed in seeds:
        argv += ["--scene-seed", str(seed)]
    return [*argv, "--out", os.path.abspath(out_dir)]


def run_command(
    *,
    unit: Mapping[str, Any],
    prompt: str,
    out_dir: str,
    policy_address: str,
    authkey_env: str,
    **extra: Any,
) -> list[str]:
    """`run-unit` from `prompt` against the policy served at `policy_address`, into `out_dir`.

    The robot, task and scene come from the prompt's own meta, which `verify_prompt` has held to
    the unit; `unit` is checked all the same, so a unit that cannot be run is refused here. The key
    travels as the name `authkey_env`, never as a value. `extra` may carry `act_timeout_s` and
    `policy_log` (the served policy's log file); keywords a later orchestrator adds are ignored.
    """
    _unit(unit)
    if not isinstance(authkey_env, str) or not authkey_env or "=" in authkey_env:
        raise ValueError(f"authkey_env must name an environment variable, not {authkey_env!r}")
    argv = [simulator_python(), "-m", CLI_MODULE, "run-unit", "--prompt", os.path.abspath(prompt)]
    argv += ["--policy-address", _address(policy_address), "--authkey-env", authkey_env]
    act_timeout_s = extra.get("act_timeout_s")
    if act_timeout_s is not None:
        argv += ["--act-timeout-s", repr(float(act_timeout_s))]
    policy_log = extra.get("policy_log")
    if policy_log:
        argv += ["--policy-log", os.path.abspath(str(policy_log))]
    return [*argv, "--out", os.path.abspath(out_dir)]


def _unit(unit: Mapping[str, Any]) -> tuple[str, str, list[int]]:
    """A unit's task, robot and candidate seeds, or `ValueError` naming what it lacks."""
    task = unit.get("task")
    if not isinstance(task, str) or task not in tasks.table().tasks:
        raise ValueError(f"unit task {task!r} is not a task of this benchmark")
    params = unit.get("instance_params")
    params = params if isinstance(params, Mapping) else {}
    embodiment = params.get("embodiment")
    if embodiment not in EMBODIMENTS:
        raise ValueError(f"unit embodiment {embodiment!r} is not one of {sorted(EMBODIMENTS)}")
    seeds = params.get("scene_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(f"unit scene_seeds {seeds!r} are not distinct integers in [0, 2**32)")
    return task, str(embodiment), list(seeds)


def _address(address: str) -> str:
    """`host:port` as given; a Unix socket path made absolute. The policy wire's own rule: an
    address with no `/` whose part after the last `:` is a port number is `host:port`."""
    if not isinstance(address, str) or not address:
        raise ValueError(f"policy_address {address!r} is neither a socket path nor host:port")
    host, sep, port = address.rpartition(":")
    if sep and host and port.isdigit() and "/" not in address:
        return address
    return os.path.abspath(address)
