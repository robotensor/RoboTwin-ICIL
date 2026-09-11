"""Every constant the BPP adapter runs on, as one YAML file (issue #42; plan 3.1, 3.4, 3.6).

`BPPPolicy(config=PATH)` and `BPPConversionReplay(config=PATH)` read the same file, so the
conversion oracle and the model run the same conversion: the checkpoint and its digests, the
tracking gains, the time stretch, the tool-frame correction, the wrist roll, the crop, the
execution mode, the re-anchoring bounds and the chunk budget. What the file does not say takes
the defaults below, and every value reaches the run manifest through `describe()`.

The fixed numbers are BPP's and robosuite's own, not settings:

- One action unit is 0.05 m of position goal and 0.5 rad of rotation goal: robosuite 1.4's
  `controllers/config/osc_pose.json`, `output_max` `[0.05, 0.05, 0.05, 0.5, 0.5, 0.5]` (and
  `controllers/osc.py:116`), the controller LIBERO's `ControlEnvForRunner` loads
  (`train_network/env/libero/env_wrapper.py:53`).
- 20 Hz is LIBERO's control rate (`env_wrapper.py:31` `control_freq=20`, and the runner's
  `robosuite_fps = 20`, `env_runner/libero_image_runner.py:130`).
- Chunking: `prompt_chunk_n_actions` 20, action horizon 16, `exec_action_horizon` 12,
  observation horizons 2, action shape [10] with rep `delta`, images 224
  (`train_network/config/task/libero_defaults.yaml`, `libero_policy_dunetp.yaml`).
- LIBERO's `gripper_states` is `[finger, -finger]` in metres over roughly [0, 0.04]
  (`libero_defaults.yaml`); aloha's finger joints run -0.01 to 0.045 m (the embodiment's
  `gripper_scale`).
- LIBERO's Panda base stands at `(-0.16 - table_length / 2, 0, table_offset_z)` =
  `(-0.66, 0, 0.90)`: robosuite's `PandaRobot.base_xpos_offset["table"]` applied by LIBERO's
  `BDDLBaseDomain._load_model` to `table_full_size` `(1.0, 1.2, 0.05)` and `table_offset`
  `(0, 0, 0.90)` (`libero/libero/envs/problems/libero_tabletop_manipulation.py:18`). The z is
  the table top rather than the mount's own frame: an **assumption**, cross-checked against the
  LIBERO-Gen demonstrations, whose recorded `ee_pos` sits at z 0.915-1.203 m, and against the
  checkpoint's normalizer by the proprio out-of-range fraction every episode reports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robotwin_icil import camera_profiles
from robotwin_icil.policy import PolicyError

from ..common.rotations import axis_angle_to_matrix

# robosuite 1.4 OSC_POSE `output_max`: one action unit of position and of rotation.
OSC_POSITION_SCALE_M = 0.05
OSC_ROTATION_SCALE_RAD = 0.5
# LIBERO's control rate, the rate a demonstration is resampled to.
RATE_HZ = 20.0
# BPP's shape_meta (`task/libero_defaults.yaml`).
PROMPT_CHUNK_N_ACTIONS = 20
ACTION_HORIZON = 16
EXEC_ACTION_HORIZON = 12
OBS_HORIZON = 2
ACTION_DIM = 10
IMAGE_SIZE = 224
# LIBERO's `gripper_states` and aloha-agilex's `gripper_scale`, both in metres.
LIBERO_FINGER_RANGE_M = (0.0, 0.04)
ALOHA_FINGER_RANGE_M = (-0.01, 0.045)
# RoboTwin's camera types (`env_cfg/task_config/_camera_config.yml`): width, height, vertical fov.
CAMERA_GEOMETRY = {
    "L515": (320, 180, 45.0),
    "Large_L515": (640, 360, 45.0),
    "D435": (320, 240, 37.0),
    "Large_D435": (640, 480, 37.0),
}
# How a chunk's executed deltas are grouped by `ee_grouped` (plan 3.4): the trailing single
# delta keeps the last two observations one 20 Hz step apart.
GROUPED_SPLITS = (4, 4, 3, 1)
MODES = ("ee_step", "ee_grouped", "qpos_ik")
# The action types each mode drives, as `ICILPolicy.action_type`.
MODE_ACTION_TYPE = {"ee_step": "ee", "ee_grouped": "ee", "qpos_ik": "qpos"}


def _vector(value: Any, size: int, where: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise PolicyError(f"bpp config: {where} must be {size} finite numbers, got {value!r}")
    return tuple(float(v) for v in array)


@dataclass(frozen=True)
class Settings:
    """The adapter's constants. Built by `load`; never mutated afterwards."""

    # The slimmed checkpoint (`icil-bpp slim`) and what it came from.
    checkpoint: str = ""
    checkpoint_sha256: str = ""
    source_checkpoint: str = (
        "austinpatel/liberogen_spatial_combination/"
        "liberogen_spatial_combination_behavior_prompting.ckpt"
        "@0e4c1fdf496592583acfbde51ce82880006aa0f3"
    )
    source_sha256: str = "74e0f841f3b86f1383adda23dab59fddee8806a4bcb5685ed7f826afb320c097"

    # Tracking gains (plan 3.4): an OSC delta is a goal offset, and the arm covers `alpha` of it
    # in one 20 Hz step. Fitted by `icil-bpp calibrate` on the checkpoint's own LIBERO-Gen data.
    alpha_p: float = 0.241057
    alpha_r: float = 0.203565
    # Prompt speed (plan 3.4): samples per demonstration second are `stretch * 20`.
    stretch: float = 1.0

    # The tool-frame correction C of `R_L = M.T R_world C` (plan 3.6): aloha approaches along the
    # flange's +x, robosuite's eef along +z, so the default is a quarter turn about the flange's
    # y. To be confirmed from renders in #47.
    tool_correction_axis_angle: tuple[float, float, float] = (0.0, math.pi / 2, 0.0)
    # Quarter turns applied to the wrist image so it reaches BPP the way LIBERO's does (#47).
    wrist_roll_quarter_turns: int = 0

    # Cameras. The profile in `cameras.yml` is the source of truth for pose; the type's geometry
    # comes from `CAMERA_GEOMETRY`, and the crop follows plan 3.1.
    camera_profile: str = "far_side"
    agentview_camera: str = "far_side_camera"
    agentview_type: str = "L515"
    wrist_cameras: tuple[str, str] = ("left_camera", "right_camera")
    wrist_type: str = "D435"
    crop: int = 180
    wrist_crop: int = 240
    # Where the active arm's workspace centre sits relative to its base, in world axes: 0.25 m
    # towards the table and down to the table top. Its projection centres the crop (#47).
    workspace_centre_offset_m: tuple[float, float, float] = (0.0, 0.25, -0.04)
    # Set to pin the crop's centre column instead of projecting the workspace centre.
    crop_column: dict[str, float] | None = None

    # Execution (plan 3.4).
    mode: str = "ee_step"
    # The virtual target is re-anchored to the measured pose once tracking is this far off. It
    # must exceed one full-scale commanded step (`alpha_p * 0.05` = 12 mm of tool motion at the
    # fitted gain), or every call would re-anchor and sub-tolerance motion could never add up,
    # which is the whole point of the target (plan 3.4). Twice that step, against CuRobo's 5 mm
    # goal tolerance and the #36 probe, where 1 mm targets already moved the arm.
    max_position_error_m: float = 0.025
    max_rotation_error_rad: float = 0.25
    stall_window: int = 10
    stall_motion_m: float = 0.002
    # At most this many prompt chunks, or the demonstration is refused (V1 needs 5-20).
    max_prompt_chunks: int = 50

    # LIBERO's Panda base, the anchor of `p_L = M.T (p_world - b_arm) + b_Panda`.
    libero_base_m: tuple[float, float, float] = (-0.66, 0.0, 0.90)

    # The checkpoint's normalizer, as `icil-bpp slim` writes it: plain JSON, so the numpy
    # conversion oracle can report the proprio out-of-range fraction without torch. Empty means
    # `normalizer.json` beside the checkpoint.
    normalizer: str = ""
    # aloha-agilex's URDF, for `qpos_ik` only (`icil_policies.common.kinematics`).
    urdf_path: str = ""

    # Arm choice (`icil_policies.common.arms`).
    arm_tie_m: float = 0.01
    arm_move_m: float = 0.005

    # Free-form provenance: where each number came from, copied into `describe()`.
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise PolicyError(f"bpp config: mode must be one of {list(MODES)}, got {self.mode!r}")
        for name in ("alpha_p", "alpha_r", "stretch"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise PolicyError(f"bpp config: {name} must be positive and finite, got {value!r}")
        if self.camera_profile not in camera_profiles.names():
            raise PolicyError(
                f"bpp config: camera_profile {self.camera_profile!r} is not in cameras.yml "
                f"({', '.join(camera_profiles.names())})"
            )
        for name in ("agentview_type", "wrist_type"):
            if getattr(self, name) not in CAMERA_GEOMETRY:
                raise PolicyError(
                    f"bpp config: {name} must be one of {sorted(CAMERA_GEOMETRY)}, "
                    f"got {getattr(self, name)!r}"
                )
        step_m = self.alpha_p * OSC_POSITION_SCALE_M
        step_rad = self.alpha_r * OSC_ROTATION_SCALE_RAD
        if self.max_position_error_m <= step_m or self.max_rotation_error_rad <= step_rad:
            raise PolicyError(
                "bpp config: the re-anchoring bounds must exceed one full-scale commanded step "
                f"({step_m:.4f} m, {step_rad:.4f} rad at these gains), or the virtual target is "
                "re-anchored at every call and sub-tolerance motion is lost"
            )
        width, height, _ = CAMERA_GEOMETRY[self.agentview_type]
        if not 0 < self.crop <= min(width, height):
            raise PolicyError(f"bpp config: crop {self.crop} does not fit a {width}x{height} frame")
        wrist_width, wrist_height, _ = CAMERA_GEOMETRY[self.wrist_type]
        if not 0 < self.wrist_crop <= min(wrist_width, wrist_height):
            raise PolicyError(f"bpp config: wrist_crop {self.wrist_crop} does not fit the wrist")
        object.__setattr__(
            self, "tool_correction_axis_angle", _vector(self.tool_correction_axis_angle, 3, "C")
        )
        object.__setattr__(self, "libero_base_m", _vector(self.libero_base_m, 3, "libero_base_m"))
        object.__setattr__(
            self,
            "workspace_centre_offset_m",
            _vector(self.workspace_centre_offset_m, 3, "workspace_centre_offset_m"),
        )
        object.__setattr__(self, "wrist_cameras", tuple(self.wrist_cameras))
        if len(self.wrist_cameras) != 2:
            raise PolicyError(f"bpp config: wrist_cameras must name two, got {self.wrist_cameras}")

    @property
    def action_type(self) -> str:
        """The `ICILPolicy.action_type` this mode drives."""
        return MODE_ACTION_TYPE[self.mode]

    def tool_correction(self) -> np.ndarray:
        """(3, 3) C, the fixed tool-frame correction of plan 3.6."""
        return axis_angle_to_matrix(np.asarray(self.tool_correction_axis_angle))

    def wrist_camera(self, arm: str) -> str:
        return self.wrist_cameras[0] if arm == "left" else self.wrist_cameras[1]

    def profile(self) -> camera_profiles.Profile:
        return camera_profiles.get(self.camera_profile)

    def agentview_geometry(self) -> tuple[int, int, float]:
        return CAMERA_GEOMETRY[self.agentview_type]

    def describe(self) -> dict[str, Any]:
        """Every constant, JSON-ready, for `describe()` and the run manifest."""
        plain: dict[str, Any] = {}
        for entry in fields(self):
            value = getattr(self, entry.name)
            plain[entry.name] = list(value) if isinstance(value, tuple) else value
        plain["action_type"] = self.action_type
        return plain


def load(config: str | Path | None = None, **overrides: Any) -> Settings:
    """The settings in a YAML file, with `overrides` on top; no file gives the defaults.

    A key the adapter does not know is refused rather than ignored, so a typo in a config is not
    a silently different conversion. Paths inside the file are resolved against its directory,
    since the harness moves into `vendor/RoboTwin` before a policy is used.
    """
    data: dict[str, Any] = {}
    if config is not None:
        path = Path(config).expanduser().resolve()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PolicyError(f"bpp config: no such file: {path}") from exc
        except yaml.YAMLError as exc:
            raise PolicyError(f"bpp config {path}: {exc}") from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise PolicyError(f"bpp config {path}: must be a mapping, got {type(raw).__name__}")
        data = dict(raw)
        for key in ("checkpoint", "normalizer", "urdf_path"):
            if isinstance(data.get(key), str) and data[key]:
                data[key] = str((path.parent / Path(data[key]).expanduser()).resolve())
    data.update(overrides)
    known = {entry.name for entry in fields(Settings)}
    if unknown := sorted(set(data) - known):
        raise PolicyError(f"bpp config: unknown keys {unknown}; known keys are {sorted(known)}")
    try:
        return Settings(**data)
    except TypeError as exc:
        raise PolicyError(f"bpp config: {exc}") from exc


def dump(settings: Settings) -> str:
    """The YAML that `load` reads back as these settings; how `calibrate` writes a config."""
    return yaml.safe_dump(asdict(settings), sort_keys=True)
