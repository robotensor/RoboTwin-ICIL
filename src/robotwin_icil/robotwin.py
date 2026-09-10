"""The only module that imports RoboTwin.

Everything else in the package stays importable without SAPIEN, assets or a GPU, so the protocol,
records and reporting are covered by tests that run in CI. Import this module and you have taken a
dependency on `vendor/RoboTwin`.

The mechanisms reused here, all from `vendor/RoboTwin`:

- `envs/<task>.py` — one class per task over `Base_Task`; `play_once()` is the scripted expert.
- `Base_Task.setup_demo(now_ep_num=…, seed=S, **args)` — seeds numpy and torch, then builds the
  scene, so generation is deterministic in (task, seed, config).
- `Base_Task._take_picture()` — called from `take_dense_action` every `save_freq` control steps
  while `save_data` is set. Upstream pickles the frame to a cache directory; we intercept it and
  keep frames in memory instead. The submodule is never patched.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .demo import Demonstration, Frame

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = REPO_ROOT / "vendor" / "RoboTwin"


class RoboTwinError(RuntimeError):
    """RoboTwin is missing, misconfigured, or refused to build a scene."""


def _ensure_importable() -> None:
    """RoboTwin is a checkout, not a package: it imports from, and loads assets relative to, its root.

    `envs/utils/create_actor.py` and `_base_task.py` open `./assets/...` against the working
    directory, so the process runs from `vendor/RoboTwin` once it enters this seam. Callers resolve
    their own paths (run directories, checkpoints) to absolute before calling in.
    """
    if not (ROBOTWIN_ROOT / "envs" / "_base_task.py").is_file():
        raise RoboTwinError(
            f"no RoboTwin checkout at {ROBOTWIN_ROOT}; run "
            "`git submodule update --init --recursive`"
        )
    root = str(ROBOTWIN_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    if Path.cwd() != ROBOTWIN_ROOT:
        os.chdir(ROBOTWIN_ROOT)


def load_task(task_name: str):
    """Instantiate RoboTwin's env class for a task. No scene exists until `setup_demo`."""
    _ensure_importable()
    try:
        module = importlib.import_module(f"envs.{task_name}")
        return getattr(module, task_name)()
    except (ImportError, AttributeError) as exc:
        raise RoboTwinError(f"RoboTwin has no task {task_name!r}") from exc


def unstable_error() -> type[Exception]:
    """RoboTwin's `UnStableError`, raised when generated object placement will not settle."""
    _ensure_importable()
    from envs.utils import UnStableError

    return UnStableError


@dataclass(frozen=True)
class SceneConfig:
    """The RoboTwin config a run holds fixed across every episode.

    Recorded in the run manifest: two runs with different configs are not comparable, because the
    config decides the cameras, the embodiment and the domain randomization the expert and the
    policy both see.
    """

    task_config: str = "demo_clean"
    save_freq: int = 15
    head_camera: str | None = None
    overrides: dict[str, Any] | None = None

    def resolve(self) -> dict[str, Any]:
        """Build the `args` dict `setup_demo` takes, from RoboTwin's own config files."""
        _ensure_importable()
        config_dir = ROBOTWIN_ROOT / "env_cfg" / "task_config"
        args = yaml.safe_load((config_dir / f"{self.task_config}.yml").read_text(encoding="utf-8"))

        embodiments = yaml.safe_load(
            (config_dir / "_embodiment_config.yml").read_text(encoding="utf-8")
        )
        embodiment = args["embodiment"]
        if len(embodiment) == 1:
            left = right = embodiments[embodiment[0]]["file_path"]
            args["dual_arm_embodied"] = True
            embodiment_name = str(embodiment[0])
        elif len(embodiment) == 3:
            left = embodiments[embodiment[0]]["file_path"]
            right = embodiments[embodiment[1]]["file_path"]
            args["embodiment_dis"] = embodiment[2]
            args["dual_arm_embodied"] = False
            embodiment_name = f"{embodiment[0]}+{embodiment[1]}"
        else:
            raise RoboTwinError(f"embodiment config must have 1 or 3 entries, got {embodiment}")

        args["left_robot_file"] = left
        args["right_robot_file"] = right
        args["left_embodiment_config"] = _embodiment_config(left)
        args["right_embodiment_config"] = _embodiment_config(right)
        args["embodiment_name"] = embodiment_name
        args["task_config"] = self.task_config

        if self.head_camera:
            args["camera"]["head_camera_type"] = self.head_camera

        # The benchmark never writes RoboTwin's pickle cache or hdf5 dataset: demonstrations are
        # captured in memory, and a run's only outputs are its records and optional videos.
        args["save_freq"] = self.save_freq
        args["save_data"] = False
        args["render_freq"] = 0
        args["eval_mode"] = True
        args["need_plan"] = True
        args["collect_data"] = False
        args["eval_video_save_dir"] = None
        for key, value in (self.overrides or {}).items():
            args[key] = value
        return args


def _embodiment_config(robot_file: str) -> dict[str, Any]:
    # `_embodiment_config.yml` names embodiments as `./assets/embodiments/...`, relative to the root.
    path = ROBOTWIN_ROOT / robot_file / "config.yml"
    if not path.is_file():
        raise RoboTwinError(
            f"embodiment config {path} is missing; run `bash scripts/install_robotwin.sh` "
            "to download RoboTwin's assets"
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def observation(env) -> dict[str, Any]:
    """One observation of the live scene, in the shape a `Frame` carries."""
    raw = env.get_obs()
    return {
        "images": _images(raw),
        "qpos": np.asarray(raw["joint_action"]["vector"], dtype=np.float64),
        "endpose": copy.deepcopy(raw.get("endpose", {})),
    }


def _images(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    images = {}
    for name, camera in raw.get("observation", {}).items():
        rgb = camera.get("rgb") if isinstance(camera, dict) else None
        if rgb is not None:
            images[name] = np.asarray(rgb)
    return images


@contextlib.contextmanager
def capture(env, frequency: int) -> Iterator[list[Frame]]:
    """Record every frame the expert's `_take_picture` would have pickled, in memory.

    RoboTwin drives recording from inside `take_dense_action`, which calls `_take_picture()` every
    `save_freq` control steps when `save_data` is set. Overriding the method on the instance keeps
    the expert, its timing and the submodule untouched — the frames are simply kept rather than
    written to a cache directory that would then be read back and deleted.
    """
    frames: list[Frame] = []
    original_take_picture = env._take_picture
    original_save_data = env.save_data
    original_save_freq = env.save_freq

    def _capture() -> None:
        obs = observation(env)
        frames.append(
            Frame(
                index=len(frames),
                images=obs["images"],
                qpos=obs["qpos"],
                endpose=obs["endpose"],
            )
        )

    env._take_picture = _capture
    env.save_data = True
    env.save_freq = frequency
    try:
        yield frames
    finally:
        env._take_picture = original_take_picture
        env.save_data = original_save_data
        env.save_freq = original_save_freq


def demonstration_from(frames: list[Frame], frequency: int) -> Demonstration:
    return Demonstration(frames=tuple(frames), frequency=frequency)
