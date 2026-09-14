"""`BENCHMARK`: the RoboTwin ICIL benchmark, as the orchestrator's `icil.benchmarks` entry point
loads it.

The orchestrator checks a plugin structurally, so nothing here subclasses or imports anything of
the orchestrator's: `RoboTwinBenchmark` simply has the id, the ABI version and the seven methods
`icil_orchestrator.benchmarks.api.Benchmark` names. The five pure ones run with no simulator, no
assets and no GPU (they read `robotwin_icil`'s task table, prompt format and results); the two
command builders return the argv of `robotwin-icil` in the simulator's environment.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import robotwin_icil
from robotwin_icil import prompt as prompt_
from robotwin_icil import remote, tasks
from robotwin_icil.records import git_commit, is_checkout_top
from robotwin_icil.robotwin import EE_ACTION_DIM, EMBODIMENTS, REPO_ROOT

from . import catalogue as catalogue_
from . import commands, prompts, units

#: This distribution's version, as its pyproject.toml gives it.
VERSION = "0.1.0.dev0"
BENCHMARK_ID = "robotwin"
#: The orchestrator ABI this plugin was written against (`BENCHMARK_API_VERSION`).
API_VERSION = 1

#: The cameras a policy observes, which `verify_prompt` holds a prompt's frames to.
CAMERAS = prompts.CAMERAS
#: The demonstration starts in the very scene the rollout is scored in.
PROTOCOL = "same_initial_state"
#: What a demonstration shows: frames, the action trajectory and proprioception.
VIEWS = ("sensorimotor",)
ACTION_TYPES = ("qpos", "ee")


def robotwin_commit(repo_root: Path = REPO_ROOT) -> str | None:
    """The RoboTwin commit the benchmark pins: its submodule's gitlink, readable without the
    submodule checked out. None unless `repo_root` is the top of a checkout: a wheel installed
    inside some other repository must not report that repository's gitlink."""
    if not is_checkout_top(repo_root):
        return None
    try:
        done = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD:vendor/RoboTwin"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip() or None


class RoboTwinBenchmark:
    """The RoboTwin ICIL benchmark, as one competition benchmark."""

    id = BENCHMARK_ID
    api_version = API_VERSION

    # -- pure ---------------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Identity and shape: the robots and their action widths, the cameras, the protocol, the
        prompt's channel map, the commits, the command line, and what is still provisional."""
        table = tasks.table()
        return {
            "id": self.id,
            "api_version": self.api_version,
            # The source is what identifies the benchmark code: an install from a wheel has no
            # commit, and the commands refuse to run under a source other than this one.
            "benchmark": {
                "distribution": "robotwin-icil",
                "version": robotwin_icil.__version__,
                "source_sha256": robotwin_icil.source_sha256(),
            },
            "plugin": {"distribution": "robotwin-icil-competition", "version": VERSION},
            "commits": {"benchmark": git_commit(REPO_ROOT), "robotwin": robotwin_commit()},
            "embodiments": {
                name: {
                    "robotwin": list(EMBODIMENTS[name]),
                    "action_dims": {"qpos": prompts.QPOS_DIMS[name], "ee": EE_ACTION_DIM},
                }
                for name in sorted(EMBODIMENTS)
            },
            "embodiment_of_suite": {
                name: catalogue_.embodiment_of(name) for name in catalogue_.suites(table)
            },
            "action_types": list(ACTION_TYPES),
            "cameras": list(CAMERAS),
            "protocol": PROTOCOL,
            "views": list(VIEWS),
            "channels": {name: list(arrays) for name, arrays in prompt_.CHANNELS.items()},
            "prefix_channels": list(prompt_.PREFIX_CHANNELS),
            "metadata": list(prompt_.METADATA),
            "task_config": commands.TASK_CONFIG,
            "save_freq": commands.SAVE_FREQ,
            "scene_seed_candidates": units.SCENE_SEED_CANDIDATES,
            # What units are drawn from, and what it was when the derivation was pinned: a
            # different table derives no units.
            "catalogue_sha256": units.catalogue_sha256(table),
            "pinned_catalogue_sha256": units.CATALOGUE_SHA256,
            "derivation": units.DERIVATION,
            "void_causes": list(prompts.VOID_CAUSES),
            # What run-unit gives a served policy unless `extra` says otherwise. A unit's
            # `unit_timeout_s` must leave the harness its own time beyond the policy's budget.
            "limits": {
                "connect_timeout_s": remote.CONNECT_TIMEOUT_S,
                "setup_timeout_s": remote.SETUP_TIMEOUT_S,
                "act_timeout_s": remote.ACT_TIMEOUT_S,
                "close_timeout_s": remote.CLOSE_TIMEOUT_S,
                "policy_budget_s": remote.POLICY_BUDGET_S,
                "result_reserve_s": remote.RESULT_RESERVE_S,
            },
            "run_extra": list(commands.RUN_EXTRA),
            "cli": {
                "python": commands.simulator_python(),
                "python_env": commands.PYTHON_ENV,
                "module": commands.CLI_MODULE,
                "materialize": "materialize --task T --embodiment E --task-config C "
                "--save-freq F --scene-seed S [--scene-seed S ...] "
                "--expect-source-sha256 HEX --out DIR",
                "run": "run-unit --prompt PROMPT --policy-address ADDR --authkey-env NAME "
                "[--act-timeout-s S] [--policy-budget-s S] [--unit-timeout-s S] "
                "[--policy-log PATH] --expect-source-sha256 HEX --out DIR",
            },
            "provisional": catalogue_.provisional(table),
        }

    def catalogue(self) -> dict[str, Any]:
        """`{"suites": {suite: [task]}, "categories": {id: label}, "tasks": {task: {"category",
        "arms", "label"}}, "embodiments": {suite: robot}, "provisional": [note]}`."""
        return catalogue_.catalogue()

    def derive_units(
        self, *, seed_material: str, count: int, suite: str, category: str | None = None
    ) -> list[dict[str, Any]]:
        """`count` units of `suite` within `category`, from a sha256 counter over `seed_material`:
        see `units.derive_units`."""
        return units.derive_units(
            seed_material=seed_material, count=count, suite=suite, category=category
        )

    def verify_prompt(self, *, path: str, unit: Mapping[str, Any]) -> dict[str, Any]:
        """Whether the prompt at `path` is the one `unit` asked for, read from the file:
        see `prompts.verify_prompt`."""
        return prompts.verify_prompt(path=path, unit=unit)

    def read_result(self, *, out_dir: str) -> dict[str, Any]:
        """The `result.json` either command wrote in `out_dir`.

        At least the ABI's `success` (None exactly when void), `void`, `steps` and `error`, and
        `void_cause`, a field this benchmark adds to the ABI's result:

        - None: the unit was scored, or the prompt written. A policy that answered `reset`,
          `prompt` or `act` with an error (it raised), or returned an action of the wrong width or
          a non-finite one, has failed: `success` false, `void` false.
        - "policy": the policy could not be spoken to — nothing listened, `hello` was refused or
          not answered, a call ran past its timeout, the connection dropped, a reply was
          malformed. `error` ends with the policy server's log tail when run-unit had its log.
        - "harness": every other void — every materialize candidate rejected by the expert, an
          unreadable or tampered prompt, scene drift, a simulator or GPU failure, any other fault
          of the harness, and a `result.json` that is missing or unreadable.
        """
        return prompts.read_result(out_dir=out_dir)

    # -- commands -----------------------------------------------------------------------

    def materialize_command(self, *, unit: Mapping[str, Any], out_dir: str) -> Sequence[str]:
        """`materialize` of the unit's task on its robot, every candidate seed in order."""
        return commands.materialize_command(unit=unit, out_dir=out_dir)

    def run_command(
        self,
        *,
        unit: Mapping[str, Any],
        prompt: str,
        out_dir: str,
        policy_address: str,
        authkey_env: str,
        **extra: Any,
    ) -> Sequence[str]:
        """`run-unit --policy-address`, with `--act-timeout-s`, `--policy-budget-s`,
        `--unit-timeout-s` and `--policy-log` when `extra` carries `act_timeout_s`,
        `policy_budget_s`, `unit_timeout_s` and `policy_log`: see `commands.run_command`."""
        return commands.run_command(
            unit=unit,
            prompt=prompt,
            out_dir=out_dir,
            policy_address=policy_address,
            authkey_env=authkey_env,
            **extra,
        )


#: What the `icil.benchmarks` entry point `robotwin` loads.
BENCHMARK = RoboTwinBenchmark()
