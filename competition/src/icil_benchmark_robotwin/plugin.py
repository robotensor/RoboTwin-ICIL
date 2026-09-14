"""`BENCHMARK`: the RoboTwin ICIL benchmark, as the orchestrator's `icil.benchmarks` entry point
loads it.

The orchestrator checks a plugin structurally, so nothing here subclasses or imports anything of
the orchestrator's: `RoboTwinBenchmark` simply has the id, the ABI version and the methods
`icil_orchestrator.benchmarks.api.Benchmark` names. The pure ones run with no simulator, no assets
and no GPU: they read `robotwin_icil`'s task table, prompt format and results.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import robotwin_icil
from robotwin_icil import prompt as prompt_
from robotwin_icil import tasks
from robotwin_icil.records import git_commit
from robotwin_icil.robotwin import EE_ACTION_DIM, EMBODIMENTS, REPO_ROOT

from . import catalogue as catalogue_
from . import prompts, units

#: This distribution's version, as its pyproject.toml gives it.
VERSION = "0.1.0.dev0"
BENCHMARK_ID = "robotwin"
#: The orchestrator ABI this plugin was written against (`BENCHMARK_API_VERSION`).
API_VERSION = 1

#: The cameras a policy observes under the task config every unit uses (`demo_clean`): RoboTwin
#: collects the head camera and both wrist cameras, named as `envs/camera/camera.py` names them. A
#: prompt and an observation hold `frames_<camera>` of each.
CAMERAS = ("head_camera", "left_camera", "right_camera")
#: The demonstration starts in the very scene the rollout is scored in.
PROTOCOL = "same_initial_state"
#: What a demonstration shows: frames, the action trajectory and proprioception.
VIEWS = ("sensorimotor",)
ACTION_TYPES = ("qpos", "ee")


def robotwin_commit(repo_root: Path = REPO_ROOT) -> str | None:
    """The RoboTwin commit the benchmark pins: its submodule's gitlink, readable without the
    submodule checked out. None outside a git checkout of the benchmark."""
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
        prompt's channel map, the commits, and what is still provisional."""
        table = tasks.table()
        return {
            "id": self.id,
            "api_version": self.api_version,
            "benchmark": {"distribution": "robotwin-icil", "version": robotwin_icil.__version__},
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
            "scene_seed_candidates": units.SCENE_SEED_CANDIDATES,
            "void_causes": list(prompts.VOID_CAUSES),
            "provisional": catalogue_.provisional(table),
        }

    def catalogue(self) -> dict[str, Any]:
        """`{"suites": {suite: [task]}, "categories": {id: label}, "tasks": {task: {"category",
        "arms", "label"}}}`, the robot of each suite, and what of it is provisional."""
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


#: What the `icil.benchmarks` entry point `robotwin` loads.
BENCHMARK = RoboTwinBenchmark()
