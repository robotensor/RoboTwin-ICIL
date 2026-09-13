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
import gc
import importlib
import os
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .demo import Demonstration, Frame
from .scene import SceneFingerprint, unique_names

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = REPO_ROOT / "vendor" / "RoboTwin"


class RoboTwinError(RuntimeError):
    """RoboTwin is missing, misconfigured, or refused to build a scene."""


def use_env_render_manifests() -> None:
    """Point SAPIEN at the NVIDIA render manifests `install_robotwin.sh` left in the env, if any.

    Without root the installer cannot put the Vulkan ICD and EGL vendor manifests where the loaders
    look, so it writes them under the env's `share/robotwin-icil`, where SAPIEN finds them only
    through these variables. Benchmark commands run the env's interpreter directly, so no
    activation script sets them. Explicit settings win.
    """
    share = Path(sys.prefix) / "share" / "robotwin-icil"
    for variable, manifest in (
        ("VK_ICD_FILENAMES", share / "vulkan" / "icd.d" / "nvidia_icd.json"),
        ("__EGL_VENDOR_LIBRARY_FILENAMES", share / "glvnd" / "egl_vendor.d" / "10_nvidia.json"),
    ):
        if manifest.is_file():
            os.environ.setdefault(variable, str(manifest))


def denoiser_for(capability: tuple[int, int] | None, override: str | None) -> str | None:
    """The ray-tracing denoiser to use where RoboTwin asks for "oidn", or None to keep its request.

    SAPIEN 3.0.0b1's OIDN has no CUDA device for compute capability 10.0 and above (Blackwell): it
    logs "unsupported device type: CUDA" and leaves every image as rendered, so turning it off there
    changes no pixel (checked on an RTX 5090). Its failing path also hangs a camera read for good
    while another process loads the GPU. `ROBOTWIN_ICIL_DENOISER` (`oidn` or `none`) overrides.
    """
    if override:
        if override not in ("oidn", "none"):
            raise RoboTwinError(
                f"ROBOTWIN_ICIL_DENOISER must be 'oidn' or 'none', not {override!r}"
            )
        return None if override == "oidn" else "none"
    if capability is not None and capability[0] >= 10:
        return "none"
    return None


def _gpu_capability() -> tuple[int, int] | None:
    """The first GPU's compute capability from `nvidia-smi`, without initialising CUDA here."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        major, minor = out.splitlines()[0].strip().split(".")
        return int(major), int(minor)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def use_supported_denoiser() -> None:
    """Swap RoboTwin's "oidn" request for a denoiser this GPU can run (see `denoiser_for`).

    RoboTwin sets the denoiser in `setup_scene` through `sapien.render.set_ray_tracing_denoiser`,
    looked up at call time, so wrapping that function once per process is enough.
    """
    try:
        import sapien.render as render
    except ImportError:
        return
    current = render.set_ray_tracing_denoiser
    if getattr(current, "robotwin_icil_denoiser", None) is not None:
        return
    choice = denoiser_for(_gpu_capability(), os.environ.get("ROBOTWIN_ICIL_DENOISER"))
    if choice is None:
        return

    def set_ray_tracing_denoiser(name: str) -> None:
        current(choice if name == "oidn" else name)

    set_ray_tracing_denoiser.robotwin_icil_denoiser = choice
    render.set_ray_tracing_denoiser = set_ray_tracing_denoiser


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
    use_env_render_manifests()
    use_supported_denoiser()
    root = str(ROBOTWIN_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    if Path.cwd() != ROBOTWIN_ROOT:
        os.chdir(ROBOTWIN_ROOT)


def load_task(task_name: str):
    """Instantiate RoboTwin's env class for a task. No scene exists until `setup_demo`.

    A task that does not exist and a task whose imports fail are different problems — the second
    is a broken install (a missing planner, a bad pin) — so they are reported differently.
    """
    _ensure_importable()
    module_name = f"envs.{task_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise RoboTwinError(f"RoboTwin has no task {task_name!r}") from exc
        raise RoboTwinError(
            f"importing RoboTwin task {task_name!r} failed: {exc}; see docs/install.md"
        ) from exc
    except ImportError as exc:
        raise RoboTwinError(
            f"importing RoboTwin task {task_name!r} failed: {exc}; see docs/install.md"
        ) from exc
    task_class = getattr(module, task_name, None)
    if task_class is None:
        raise RoboTwinError(f"RoboTwin module {module_name} defines no class {task_name!r}")
    return task_class()


def unstable_error() -> type[Exception]:
    """RoboTwin's `UnStableError`, raised when generated object placement will not settle."""
    _ensure_importable()
    from envs.utils import UnStableError

    return UnStableError


# The distance between two Franka arms' bases. RoboTwin's configuration guide gives
# `embodiment: [franka-panda, franka-panda, 0.8]` as its dual-Franka example and says the interval
# is "typically 0.6–0.8 meters": https://robotwin-platform.github.io/doc/usage/configurations.html
FRANKA_ARM_DISTANCE_M = 0.8

# The robots a run can choose, in the form RoboTwin's `embodiment` config takes. RoboTwin always
# builds a left and a right arm (`envs/robot/robot.py:_init_robot_`): a dual-arm robot is one entry,
# one URDF holding both arms; two single-arm robots are `[left, right, distance]`, one URDF each,
# their bases `distance` metres apart along x. The names are `_embodiment_config.yml`'s.
EMBODIMENTS: dict[str, tuple[Any, ...]] = {
    "aloha-agilex": ("aloha-agilex",),
    "franka-panda": ("franka-panda", "franka-panda", FRANKA_ARM_DISTANCE_M),
}


def embodiment_name(embodiment: Sequence[Any]) -> str:
    """The name a run records for a RoboTwin `embodiment` list: what `--embodiment` takes.

    `[x]` and `[x, x, d]` are both robot `x`, two arms either way; two different arms are named
    `left+right`, as upstream's scripts file them.
    """
    names = [str(name) for name in embodiment[:2]]
    return names[0] if len(set(names)) == 1 else "+".join(names)


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
    # A name from `EMBODIMENTS`, or None for the task config's own `embodiment` list.
    embodiment: str | None = None

    def resolve(self, task_name: str | None = None) -> dict[str, Any]:
        """Build the `args` dict `setup_demo` takes, from RoboTwin's own config files.

        Pass `task_name` for any scene that is built: RoboTwin reads it back as `self.task_name` to
        look up the task's evaluation step limit, and silently allows 1000 steps without it.
        `embodiment_name` in the result is the benchmark's name for the robot (`embodiment_name`);
        upstream's envs never read that key, only its collection scripts do.
        """
        _ensure_importable()
        config_dir = ROBOTWIN_ROOT / "env_cfg" / "task_config"
        args = yaml.safe_load((config_dir / f"{self.task_config}.yml").read_text(encoding="utf-8"))

        embodiments = yaml.safe_load(
            (config_dir / "_embodiment_config.yml").read_text(encoding="utf-8")
        )
        if self.embodiment is not None:
            if self.embodiment not in EMBODIMENTS:
                raise RoboTwinError(
                    f"unknown embodiment {self.embodiment!r}; the benchmark runs "
                    f"{', '.join(sorted(EMBODIMENTS))} (RoboTwin ships {', '.join(sorted(embodiments))})"
                )
            args["embodiment"] = list(EMBODIMENTS[self.embodiment])
        embodiment = args["embodiment"]
        if len(embodiment) not in (1, 3):
            raise RoboTwinError(f"embodiment config must have 1 or 3 entries, got {embodiment}")
        for name in embodiment[:2]:
            if name not in embodiments:
                raise RoboTwinError(
                    f"embodiment {name!r} is not in _embodiment_config.yml, which has "
                    f"{', '.join(sorted(embodiments))}"
                )
        if len(embodiment) == 1:
            left = right = embodiments[embodiment[0]]["file_path"]
            args["dual_arm_embodied"] = True
        else:
            left = embodiments[embodiment[0]]["file_path"]
            right = embodiments[embodiment[1]]["file_path"]
            args["embodiment_dis"] = embodiment[2]
            args["dual_arm_embodied"] = False

        args["left_robot_file"] = left
        args["right_robot_file"] = right
        args["left_embodiment_config"] = _embodiment_config(left)
        args["right_embodiment_config"] = _embodiment_config(right)
        args["embodiment_name"] = embodiment_name(embodiment)
        args["task_config"] = self.task_config
        if task_name is not None:
            args["task_name"] = task_name

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


def close(env, clear_cache: bool = False) -> None:
    """Tear a scene down, swallowing errors: a scene that failed half-built must not end the run.

    `clear_cache` also drops SAPIEN's render cache; upstream does that every few episodes to bound
    memory over a long run.
    """
    try:
        env.close_env(clear_cache=clear_cache)
    except Exception:
        pass


def gpu_exhausted(exc: BaseException) -> bool:
    """Whether an exception is the GPU running out of memory, recognised without importing torch."""
    return type(exc).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(exc)


def gpu_lost(exc: BaseException) -> bool:
    """Whether an exception is the renderer losing the GPU: Vulkan's device-lost error.

    SAPIEN raises it from a camera read (`vk::Device::waitForFences: ErrorDeviceLost`), for
    instance when another process has filled the GPU's memory. The process cannot render again,
    so no later seed or step in it means anything.
    """
    text = str(exc)
    return "ErrorDeviceLost" in text or "VK_ERROR_DEVICE_LOST" in text


def free_gpu() -> None:
    """Hand the memory of released envs — CuRobo's planners live on the GPU — back to the driver."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_render_cache() -> None:
    """Drop SAPIEN's render cache; upstream does this every few episodes to bound memory."""
    _ensure_importable()
    from sapien.render import clear_cache

    clear_cache()


def episode_over(env) -> bool:
    """Upstream's own end condition: success latched by `take_action`, or the task's step limit."""
    step_lim = getattr(env, "step_lim", None)
    return bool(env.eval_success) or (step_lim is not None and env.take_action_cnt >= step_lim)


# `take_action(action_type='ee')` reads a pose (7) and a gripper per arm, whatever the arm.
EE_ACTION_DIM = 2 * (7 + 1)


def action_dims(env) -> dict[str, int]:
    """The width `take_action` expects per action type, read off the live robot's arms.

    A `qpos` action has the layout of `get_obs()['joint_action']['vector']`: the left arm's joints
    and gripper, then the right's — 14 on aloha-agilex's six-joint arms, 16 on two seven-joint
    Frankas. `take_action` splits a qpos action by these same lengths (`len(jointstate) - 1` per
    arm), so the harness reads them from the same place rather than assuming a robot.
    """
    left = len(env.robot.get_left_arm_jointState())
    right = len(env.robot.get_right_arm_jointState())
    return {"qpos": left + right, "ee": EE_ACTION_DIM}


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
def capture(env, save_freq: int) -> Iterator[list[Frame]]:
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
    env.save_freq = save_freq
    try:
        yield frames
    finally:
        env._take_picture = original_take_picture
        env.save_data = original_save_data
        env.save_freq = original_save_freq


def frame_rate_hz(env, save_freq: int) -> float:
    """Frames per second of a recording taken every `save_freq` control steps.

    RoboTwin steps physics every `scene.get_timestep()` seconds (1/250 by default) and
    `take_dense_action` calls `_take_picture` every `save_freq` of those steps. The rate is not
    `save_freq` itself, although upstream passes `save_freq` where a frame rate is meant.
    """
    return 1.0 / (float(env.scene.get_timestep()) * save_freq)


def demonstration_from(frames: list[Frame], frequency: float) -> Demonstration:
    return Demonstration(frames=tuple(frames), frequency=frequency)


def fingerprint(env) -> SceneFingerprint:
    """The live scene's initial state: take it right after `setup_demo`, before anyone acts.

    Covers what makes a scene this scene: every actor's pose (object instances, placements, the
    table), every articulation's root pose and joints (the robot, articulated objects), camera
    extrinsics, the robot's commanded qpos, which robot was built, and the texture and lighting
    draws recorded in `env.info`. Harness-only: none of it is ever handed to a policy.
    """
    actors = env.scene.get_all_actors()
    articulations = env.scene.get_all_articulations()
    actor_keys = unique_names([actor.get_name() for actor in actors])
    articulation_keys = unique_names([articulation.get_name() for articulation in articulations])
    textures = (
        env.info.get("texture_info", {}) if isinstance(getattr(env, "info", None), dict) else {}
    )
    return SceneFingerprint(
        actors={
            key: _pose7(actor.get_pose()) for key, actor in zip(actor_keys, actors, strict=True)
        },
        articulations={
            key: np.asarray(articulation.get_qpos(), dtype=np.float64)
            for key, articulation in zip(articulation_keys, articulations, strict=True)
        },
        articulation_roots={
            key: _pose7(articulation.get_root_pose())
            for key, articulation in zip(articulation_keys, articulations, strict=True)
        },
        cameras={
            name: np.asarray(config["extrinsic_cv"], dtype=np.float64)
            for name, config in env.cameras.get_config().items()
        },
        robot_qpos=np.asarray(
            env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(),
            dtype=np.float64,
        ),
        extras={
            "embodiment": _embodiment_built(env.robot),
            "wall_texture": textures.get("wall_texture"),
            "table_texture": textures.get("table_texture"),
            "crazy_random_light": bool(getattr(env, "crazy_random_light", False)),
            "table_z_bias": float(getattr(env, "table_z_bias", 0.0)),
        },
    )


def _embodiment_built(robot) -> str:
    """The robot as `_init_robot_` built it: one URDF holding both arms, or one URDF per arm."""
    if robot.is_dual_arm:
        return str(robot.left_urdf_path)
    return f"{robot.left_urdf_path}|{robot.right_urdf_path}"


def _pose7(pose) -> np.ndarray:
    return np.concatenate(
        [np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64)]
    )
