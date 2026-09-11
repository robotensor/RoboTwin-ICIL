"""RoboTwin's demonstration and observations as BPP's LIBERO quantities, and back (plan 3.1-3.6).

numpy only, so the conversion oracle and the training-free checks run in the simulator's
environment, where neither torch nor BPP is installed. Everything model-side (the network, its
normalizer, its chunker) lives in `model.py` and `policy.py`.

What BPP consumes, per its `shape_meta` (`train_network/config/task/libero_defaults.yaml`):

| key | shape | what |
| --- | --- | --- |
| `agentview_rgb` | (3, 224, 224) | the third-person view, float 0..1 |
| `eye_in_hand_rgb` | (3, 224, 224) | the wrist view |
| `ee_pos` | (3,) | the tool centre point in LIBERO's world frame |
| `ee_ori` | (6,) | rot6d of the tool orientation |
| `gripper_states` | (2,) | `[finger, -finger]` in metres |
| `action` | (10,) | `[dx, dy, dz, rot6d(6), gripper]`, `rep: delta` |

An action is a 20 Hz robosuite OSC command in [-1, 1]: position in units of 0.05 m of goal
offset, rotation as an axis-angle in units of 0.5 rad, encoded as the rot6d of that axis-angle
exactly as BPP's own dataset encodes it (`dataset/libero_replay_image_dataset.py:353`
`_convert_actions`, through `RotationTransformer(axis_angle -> rotation_6d)`), and a gripper
command that is negative to open (`env_runner/libero_image_runner.py:628`, where the runner's
own "open the gripper" action sets `action[..., 6] = -1`).

An OSC delta is a *goal offset*, not a displacement: the arm covers about `alpha` of it in one
control step, so prompt deltas are divided by `alpha` and executed deltas multiplied by it
(plan 3.4; `icil-bpp calibrate` fits both on the checkpoint's own LIBERO-Gen data).

Frames (plan 3.4): a LIBERO vector v is `M v` in RoboTwin's world, and robosuite applies
position deltas at the tool centre point and rotations on the left in the world frame, so a
rotation delta in LIBERO's axes is `M.T dR M` in the world. Orientations are corrected by a
fixed tool-frame rotation C (`Settings.tool_correction`), since aloha approaches along the
flange's +x and robosuite's eef along +z.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robotwin_icil.demo import EE_POSE_DIM, Demonstration
from robotwin_icil.policy import PolicyError

from ..common import images as image_path
from ..common.arms import ArmChoice, IdleArmHold, choose_arm
from ..common.chunking import ChunkExecutor
from ..common.frames import (
    EE_SLICES,
    LIBERO_TO_WORLD,
    QPOS_SLICES,
    arm_base,
    check_arm,
    flange_from_tcp,
    matrix_to_pose,
    tcp_from_flange,
    world_to_libero,
)
from ..common.resample import Resampled, resample
from ..common.rotations import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rot6d,
    quat_to_matrix,
    rot6d_to_matrix,
)
from ..common.virtual_target import StallDetector, VirtualTarget
from .settings import (
    ACTION_DIM,
    GROUPED_SPLITS,
    LIBERO_FINGER_RANGE_M,
    OSC_POSITION_SCALE_M,
    OSC_ROTATION_SCALE_RAD,
    PROMPT_CHUNK_N_ACTIONS,
    RATE_HZ,
    Settings,
)
from .settings import ALOHA_FINGER_RANGE_M as ALOHA_FINGERS

# A commanded gripper value moves by more than this between samples to count as rising or
# falling; below it the value is flat and the label follows whether it is open or closed.
GRIPPER_FLAT = 1e-6
# RoboTwin's commanded gripper is 0 (closed) to 1 (open); the halfway point splits the two.
GRIPPER_CLOSED_BELOW = 0.5
# BPP's proprioception keys, in the order `describe()` and the tests report them.
PROPRIO_KEYS = ("ee_pos", "ee_ori", "gripper_states")


# ----------------------------------------------------------------------------- cameras


def camera_pose(settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of the third-person camera in the world, from the profile in `cameras.yml`.

    RoboTwin builds a static camera's pose from `forward`, `left` and their cross product
    (`envs/camera/camera.py:101-106`: `mat44[:3, :3] = stack([forward, left, up], axis=1)`), so
    the camera's own axes are x forward, y left, z up, and R's columns are exactly those.
    """
    profile = settings.profile()
    for entry in profile.replace.values():
        if entry.get("name") == settings.agentview_camera:
            if entry.get("type") != settings.agentview_type:
                raise PolicyError(
                    f"camera profile {profile.name!r} renders {settings.agentview_camera} as a "
                    f"{entry.get('type')}, but the config says {settings.agentview_type}"
                )
            forward = np.asarray(entry["forward"], dtype=np.float64)
            left = np.asarray(entry["left"], dtype=np.float64)
            forward = forward / np.linalg.norm(forward)
            left = left / np.linalg.norm(left)
            rotation = np.stack([forward, left, np.cross(forward, left)], axis=1)
            return rotation, np.asarray(entry["position"], dtype=np.float64)
    raise PolicyError(
        f"camera profile {profile.name!r} does not render {settings.agentview_camera!r}"
    )


def project(point: np.ndarray, settings: Settings) -> tuple[float, float]:
    """(column, row) where a world point lands in the third-person frame.

    A pinhole camera with SAPIEN's `fovy` (the vertical field): the focal length in pixels is
    `(height / 2) / tan(fovy / 2)`, 217.3 px for the 45-degree L515 at 320x180. The camera looks
    along its own +x, with +y to image left and +z to image top.
    """
    width, height, fovy_deg = settings.agentview_geometry()
    rotation, position = camera_pose(settings)
    camera = rotation.T @ (np.asarray(point, dtype=np.float64) - position)
    if camera[0] <= 1e-6:
        raise PolicyError(f"the point {list(point)} is not in front of {settings.agentview_camera}")
    focal = (height / 2) / math.tan(math.radians(fovy_deg) / 2)
    return width / 2 - focal * camera[1] / camera[0], height / 2 - focal * camera[2] / camera[0]


def workspace_centre(settings: Settings, arm: str) -> np.ndarray:
    """(3,) the world point the crop is centred on: the arm's base, offset towards the table.

    A constant of the camera profile and the embodiment, never scene information.
    """
    return arm_base(check_arm(arm)) + np.asarray(settings.workspace_centre_offset_m)


def crop_column(settings: Settings, arm: str) -> float:
    """The column the arm-centred crop is centred on (plan 3.1), before clamping.

    `Settings.crop_column` pins it; otherwise it is the projection of the arm's workspace
    centre. `images.crop_start` then clamps the window inside the frame, which for the 180-wide
    crop of a 320-wide frame holds the centre within columns 90-230.
    """
    check_arm(arm)
    if settings.crop_column is not None:
        try:
            return float(settings.crop_column[arm])
        except (KeyError, TypeError) as exc:
            raise PolicyError(f"bpp config: crop_column has no {arm!r} entry") from exc
    return project(workspace_centre(settings, arm), settings)[0]


def agentview_view(image: np.ndarray, settings: Settings, arm: str) -> np.ndarray:
    """(224, 224, 3) float32 from the third-person frame: arm-centred crop, 128, then 224."""
    return image_path.arm_centred_view(image, crop_column(settings, arm), crop=settings.crop)


def wrist_view(image: np.ndarray, settings: Settings) -> np.ndarray:
    """(224, 224, 3) float32 from a wrist frame: centre square, the fixed roll, 128, then 224.

    The roll is a multiple of 90 degrees (`wrist_roll_quarter_turns`), chosen from renders in
    #47: aloha's wrist camera is mounted on link 6, LIBERO's on the Panda hand, and the two need
    not agree on which way is up.
    """
    image = np.asarray(image)
    square = image_path.crop_square(image, settings.wrist_crop, centre_col=image.shape[1] / 2)
    turned = np.ascontiguousarray(np.rot90(square, k=settings.wrist_roll_quarter_turns))
    return image_path.arm_centred_view(
        turned, centre_col=settings.wrist_crop / 2, crop=settings.wrist_crop
    )


def views(images: Mapping[str, np.ndarray], settings: Settings, arm: str) -> dict[str, np.ndarray]:
    """BPP's two image keys, each (3, 224, 224) float32, as its loader hands them to the model.

    BPP's loader moves the channel axis first and flips the rows of its stored LIBERO frames,
    which are upside down (`dataset/libero_replay_image_dataset.py:39-52`), and its LIBERO
    runner flips the simulator's frames the same way before the policy sees them
    (`env/libero/env_wrapper.py:146`). Both leave the model an upright image, which is what
    RoboTwin renders, so nothing is flipped here.
    """
    missing = [
        name
        for name in (settings.agentview_camera, settings.wrist_camera(arm))
        if name not in images
    ]
    if missing:
        raise PolicyError(
            f"the {settings.camera_profile!r} profile must render {missing}; "
            f"this observation has {sorted(images)}"
        )
    return {
        "agentview_rgb": _chw(agentview_view(images[settings.agentview_camera], settings, arm)),
        "eye_in_hand_rgb": _chw(wrist_view(images[settings.wrist_camera(arm)], settings)),
    }


def _chw(view: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.moveaxis(view, -1, 0))


# ----------------------------------------------------------------- proprioception (plan 3.6)


def gripper_states(finger_m: float, settings: Settings) -> np.ndarray:
    """(2,) LIBERO's `[finger, -finger]` from one measured aloha finger joint, in metres.

    Affine from aloha's `gripper_scale` range onto LIBERO's, so a closed aloha gripper reads as
    a closed Panda gripper. The measurement, not RoboTwin's commanded value, which reads fully
    closed while the fingers rest on an object.
    """
    low, high = ALOHA_FINGERS
    libero_low, libero_high = LIBERO_FINGER_RANGE_M
    finger = libero_low + (float(finger_m) - low) * (libero_high - libero_low) / (high - low)
    return np.array([finger, -finger], dtype=np.float64)


def libero_tool_pose(
    flange: np.ndarray, settings: Settings, arm: str
) -> tuple[np.ndarray, np.ndarray]:
    """(position (3,), rotation (3, 3)) of the tool centre point in LIBERO's frame."""
    pose = np.asarray(flange, dtype=np.float64)
    if pose.shape != (EE_POSE_DIM,):
        raise PolicyError(f"expected a flange pose of shape ({EE_POSE_DIM},), got {pose.shape}")
    tool = tcp_from_flange(pose)
    position = world_to_libero(tool[:3] - arm_base(check_arm(arm))) + np.asarray(
        settings.libero_base_m
    )
    rotation = LIBERO_TO_WORLD.T @ quat_to_matrix(tool[3:]) @ settings.tool_correction()
    return position, rotation


def proprioception(
    flange: np.ndarray, finger_m: float, settings: Settings, arm: str
) -> dict[str, np.ndarray]:
    """BPP's three proprioception keys for one arm at one instant."""
    position, rotation = libero_tool_pose(flange, settings, arm)
    return {
        "ee_pos": position,
        "ee_ori": matrix_to_rot6d(rotation),
        "gripper_states": gripper_states(finger_m, settings),
    }


def finger_of(gripper_joints: Mapping[str, Any] | None, arm: str, where: str) -> float:
    """The measured base finger joint of one arm, in metres (`Frame.gripper_joints`, #36)."""
    if not gripper_joints or arm not in gripper_joints:
        raise PolicyError(
            f"{where}: the BPP adapter needs measured gripper joints; this one has none. "
            "Record demonstrations and observations with a benchmark that reads them (#36)."
        )
    joints = np.asarray(gripper_joints[arm], dtype=np.float64)
    if joints.ndim != 1 or joints.size == 0:
        raise PolicyError(f"{where}: {arm} gripper joints have shape {joints.shape}")
    return float(joints[0])


def normalize(values: np.ndarray, params: Mapping[str, Sequence[float]]) -> np.ndarray:
    """BPP's normalizer applied by hand: `x * scale + offset` (`model/common/normalizer.py`)."""
    scale = np.asarray(params["scale"], dtype=np.float64)
    offset = np.asarray(params["offset"], dtype=np.float64)
    return np.asarray(values, dtype=np.float64) * scale + offset


def out_of_range_fraction(values: np.ndarray, params: Mapping[str, Sequence[float]]) -> float:
    """The fraction of normalized entries outside [-1, 1]: how far the robot is out of LIBERO."""
    normalized = normalize(values, params)
    return float(np.mean(np.abs(normalized) > 1.0)) if normalized.size else 0.0


# ------------------------------------------------------------------ one BPP observation


def observation_flange(observation: Any, arm: str) -> np.ndarray:
    """(7,) the arm's flange pose out of an `Observation`'s or `Frame`'s `endpose`."""
    endpose = observation.endpose or {}
    try:
        pose = np.asarray(endpose[f"{arm}_endpose"], dtype=np.float64)
    except KeyError as exc:
        raise PolicyError(
            f"the observation has no {arm}_endpose; it has {sorted(endpose)}"
        ) from exc
    if pose.shape != (EE_POSE_DIM,):
        raise PolicyError(f"{arm}_endpose has shape {pose.shape}, expected ({EE_POSE_DIM},)")
    return pose


def observation_state(observation: Any, settings: Settings, arm: str) -> dict[str, np.ndarray]:
    """BPP's five observation keys for one `Observation`: two images and three proprio arrays."""
    finger = finger_of(observation.gripper_joints, arm, f"observation {observation.step}")
    return {
        **views(observation.images, settings, arm),
        **proprioception(observation_flange(observation, arm), finger, settings, arm),
    }


def prompt_state(prompt: Prompt, settings: Settings) -> dict[str, np.ndarray]:
    """The prompt's observations, stacked: (L, 3, 224, 224) images and (L, d) proprioception.

    One entry per chunk, taken at the resampled steps `prompt.frames`, which is where BPP's own
    sampler reads them (`common/sampler.py:936-945`, `downsample_obs_by_chunk=True`).
    """
    resampled, arm = prompt.resampled, prompt.arm
    start = EE_SLICES[arm].start
    measured = resampled.gripper_joints
    if measured is None:
        raise PolicyError(
            "the BPP adapter needs measured gripper joints; this demonstration has none (#36)"
        )
    frames = [resampled.demonstration.frames[resampled.source[i]] for i in prompt.frames]
    rows = []
    for step, frame in zip(prompt.frames, frames, strict=True):
        state = {
            **views(frame.images, settings, arm),
            **proprioception(
                resampled.endposes[step, start : start + EE_POSE_DIM],
                float(measured[arm][step][0]),
                settings,
                arm,
            ),
        }
        rows.append(state)
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


# ------------------------------------------------------------------- the prompt (plan 3.4, 3.5)


@dataclass(frozen=True)
class Prompt:
    """One demonstration as BPP's prompt: a row per 20 Hz step, and observations per chunk.

    `actions` is (T, 10) at `rate_hz`; `observations` holds one entry per chunk, the state at
    steps 0, 20, 40, ... , which is where BPP's own sampler reads them
    (`common/sampler.py:939` with `downsample_obs_by_chunk=True`, then
    `utils/prompt_util.py:90-110` chunking the actions).
    """

    arm: str
    rate_hz: float
    actions: np.ndarray  # (T, 10)
    frames: tuple[int, ...]  # the resampled step each prompt observation comes from
    clipped: float  # the fraction of action entries that hit +-1
    resampled: Resampled = field(repr=False, compare=False)

    @property
    def chunks(self) -> int:
        return len(self.frames)


def prompt_actions(resampled: Resampled, arm: str, settings: Settings) -> tuple[np.ndarray, float]:
    """((T, 10), clipped fraction) the demonstration's own actions in BPP's representation.

    Tool-centre deltas between consecutive samples, in LIBERO's frame, divided by the gains and
    clipped to [-1, 1]; the rotation is then re-encoded as the rot6d of the clipped axis-angle,
    as BPP's dataset does. The gripper label follows the onset of the commanded change: +1
    (close) while the commanded value falls or is closed, -1 while it rises or is open, so a
    label never lags RoboTwin's 300-step gripper ramp the way a threshold on the value would.
    """
    check_arm(arm)
    start = EE_SLICES[arm].start
    poses = resampled.endposes[:, start : start + EE_POSE_DIM]
    grips = resampled.endposes[:, start + EE_POSE_DIM]
    tools = tcp_from_flange(poses)
    positions, rotations = tools[:, :3], quat_to_matrix(tools[:, 3:])

    moves = world_to_libero(np.diff(positions, axis=0)) / (settings.alpha_p * OSC_POSITION_SCALE_M)
    turns = np.einsum("tij,tkj->tik", rotations[1:], rotations[:-1])  # dR in the world
    turns = LIBERO_TO_WORLD.T @ turns @ LIBERO_TO_WORLD  # M.T dR M: the same delta in LIBERO
    spins = matrix_to_axis_angle(turns) / (settings.alpha_r * OSC_ROTATION_SCALE_RAD)

    clipped_moves, clipped_spins = np.clip(moves, -1.0, 1.0), np.clip(spins, -1.0, 1.0)
    clipped = float(
        np.mean(
            np.concatenate(
                [np.abs(moves) > 1.0, np.abs(spins) > 1.0],
                axis=1,
            )
        )
    )
    change = np.diff(grips)
    closing = (change < -GRIPPER_FLAT) | (
        (np.abs(change) <= GRIPPER_FLAT) & (grips[1:] < GRIPPER_CLOSED_BELOW)
    )
    actions = np.concatenate(
        [
            clipped_moves,
            matrix_to_rot6d(axis_angle_to_matrix(clipped_spins)),
            np.where(closing, 1.0, -1.0)[:, None],
        ],
        axis=1,
    )
    return actions, clipped


def build_prompt(demonstration: Demonstration, settings: Settings, arm: str) -> Prompt:
    """One demonstration resampled on its own times and converted into BPP's prompt.

    The rate is `stretch * 20` samples per demonstration second: a stretch above 1 slows the
    demonstration down, so a RoboTwin expert that moves faster than one LIBERO step does not
    clip (plan 3.4, Prompt speed).
    """
    rate = RATE_HZ * settings.stretch
    if demonstration.frames[0].gripper_joints is None:
        raise PolicyError(
            "the BPP adapter needs measured gripper joints: LIBERO's `gripper_states` is where "
            "the fingers are, and RoboTwin's gripper value is only the command (#36)"
        )
    resampled = resample(demonstration, rate_hz=rate)
    if len(resampled) < 2:
        raise PolicyError(f"the demonstration is {len(resampled)} samples long at {rate} Hz")
    actions, clipped = prompt_actions(resampled, arm, settings)
    chunks = max(1, math.ceil(len(actions) / PROMPT_CHUNK_N_ACTIONS))
    if chunks > settings.max_prompt_chunks:
        raise PolicyError(
            f"the demonstration is {len(actions)} steps at {rate} Hz, {chunks} prompt chunks; "
            f"the budget is {settings.max_prompt_chunks}"
        )
    frames = tuple(int(i) for i in np.arange(chunks) * PROMPT_CHUNK_N_ACTIONS)
    return Prompt(
        arm=arm,
        rate_hz=rate,
        actions=actions,
        frames=frames,
        clipped=clipped,
        resampled=resampled,
    )


# ------------------------------------------------------------------------ execution (plan 3.4)


def decode_action(action: np.ndarray, settings: Settings) -> tuple[np.ndarray, np.ndarray, float]:
    """One BPP action as (world position delta, world rotation delta, gripper target).

    The inverse of the prompt's encoding, times the gains: the tool centre moves
    `alpha_p * 0.05 * a` metres and turns by `alpha_r * 0.5 * a` radians. The gripper target is
    RoboTwin's commanded value, 1 (open) when BPP's gripper action is negative, else 0.
    """
    row = np.asarray(action, dtype=np.float64)
    if row.shape != (ACTION_DIM,):
        raise PolicyError(f"expected a BPP action of shape ({ACTION_DIM},), got {row.shape}")
    move = LIBERO_TO_WORLD @ (row[:3] * settings.alpha_p * OSC_POSITION_SCALE_M)
    spin = (
        matrix_to_axis_angle(rot6d_to_matrix(row[3:9])) * settings.alpha_r * OSC_ROTATION_SCALE_RAD
    )
    turn = LIBERO_TO_WORLD @ axis_angle_to_matrix(spin) @ LIBERO_TO_WORLD.T
    return move, turn, 1.0 if row[9] < 0 else 0.0


def group_bounds(chunk: np.ndarray, mode: str) -> list[tuple[int, int]]:
    """How a chunk's actions are split into `take_action` calls (plan 3.4).

    `ee_step` and `qpos_ik` send one action per call. `ee_grouped` sends 4, 4, 3 and 1 of them,
    the trailing single delta keeping the last two observations one 20 Hz step apart, and splits
    again wherever the gripper command flips, so a group never hides a grasp or a release.
    """
    rows = np.asarray(chunk, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != ACTION_DIM:
        raise PolicyError(f"expected a chunk of shape (k, {ACTION_DIM}), got {rows.shape}")
    if mode != "ee_grouped":
        return [(i, i + 1) for i in range(len(rows))]
    grips = rows[:, 9] < 0
    bounds: list[tuple[int, int]] = []
    start = 0
    sizes = list(GROUPED_SPLITS)
    while start < len(rows):
        size = sizes.pop(0) if sizes else GROUPED_SPLITS[-1]
        stop = min(start + size, len(rows))
        flips = np.flatnonzero(grips[start + 1 : stop] != grips[start])
        if flips.size:
            stop = start + 1 + int(flips[0])
        bounds.append((start, stop))
        start = stop
    return bounds


def aggregate(rows: np.ndarray, settings: Settings) -> np.ndarray:
    """One action standing for several: their deltas composed, in the same representation.

    Positions add; rotations compose in the world frame, in the order they would be applied.
    Exact, because decoding is linear in position and the composed rotation is re-encoded
    through the same gain. The gripper is the group's, which `group_bounds` keeps constant.
    """
    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    if len(rows) == 1:
        return rows[0].copy()
    turn = np.eye(3)
    for row in rows:
        _, step, _ = decode_action(row, settings)
        turn = step @ turn
    spin = matrix_to_axis_angle(turn) / (settings.alpha_r * OSC_ROTATION_SCALE_RAD)
    return np.concatenate(
        [rows[:, :3].sum(axis=0), matrix_to_rot6d(axis_angle_to_matrix(spin)), rows[-1, 9:]]
    )


class GroupedExecutor(ChunkExecutor):
    """A `ChunkExecutor` that queues every action its `plan` returns, not just `n_action`.

    `ee_grouped` plans one chunk and then executes it as four or more calls, so the queue holds
    what the plan produced: the groups themselves, already aggregated. The history and the
    re-planning are the base class's.
    """

    def __init__(self, plan, n_obs: int) -> None:
        super().__init__(plan, n_obs=n_obs, n_action=1)

    def act(self, observation: Any) -> np.ndarray:
        if not self._history:
            self._history.extend([observation] * self.n_obs)
        else:
            self._history.append(observation)
        if not self._queue:
            chunk = np.asarray(self._plan(list(self._history)))
            if chunk.ndim != 2 or len(chunk) < 1:
                raise PolicyError(
                    f"the model returned a chunk of shape {chunk.shape}; expected at least one "
                    "action as the rows of a 2-D array"
                )
            self._queue.extend(chunk)
            self.plans += 1
        return self._queue.popleft()


class Execution:
    """Turns BPP's actions into RoboTwin's, for one arm, over one episode (plan 3.3, 3.4).

    A virtual tool-centre target integrates the deltas, so motion below CuRobo's 5 mm goal
    tolerance is not thrown away, and is re-anchored to the measured pose only when tracking
    breaks down; a stall detector reports when the tool was told to move and did not. The idle
    arm holds the fixed target taken from the episode's first observation.
    """

    def __init__(self, settings: Settings, arm: str, arm_model: Any | None = None) -> None:
        self.settings = settings
        self.arm = check_arm(arm)
        self.arm_model = arm_model
        if settings.mode == "qpos_ik" and arm_model is None:
            raise PolicyError("mode qpos_ik needs the aloha kinematics; give the config a urdf")
        self.target = VirtualTarget(
            max_position_error_m=settings.max_position_error_m,
            max_rotation_error_rad=settings.max_rotation_error_rad,
        )
        self.stall = StallDetector(
            window=settings.stall_window, min_motion_m=settings.stall_motion_m
        )
        self.reset()

    def reset(self) -> None:
        self.target.reset()
        self.stall.reset()
        self.idle: IdleArmHold | None = None
        self.calls = 0
        self.unreachable = 0

    def start(self, observation: Any) -> None:
        """Take the idle arm's fixed hold from the episode's first observation."""
        if self.idle is None:
            other = "right" if self.arm == "left" else "left"
            self.idle = IdleArmHold.from_observation(observation, other)

    def measured(self, observation: Any) -> np.ndarray:
        """(7,) the active arm's flange pose, from the observation's `endpose`."""
        endpose = observation.endpose or {}
        try:
            pose = np.asarray(endpose[f"{self.arm}_endpose"], dtype=np.float64)
        except KeyError as exc:
            raise PolicyError(
                f"the observation has no {self.arm}_endpose; it has {sorted(endpose)}"
            ) from exc
        if pose.shape != (EE_POSE_DIM,):
            raise PolicyError(f"{self.arm}_endpose has shape {pose.shape}")
        return pose

    def act(self, action: np.ndarray, observation: Any) -> np.ndarray:
        """(1, 16) or (1, 14): the RoboTwin action this BPP action becomes."""
        self.start(observation)
        settings = self.settings
        flange = self.measured(observation)
        tool = tcp_from_flange(flange)
        move, turn, gripper = decode_action(action, settings)
        position, rotation = self.target.step(tool[:3], quat_to_matrix(tool[3:]), move, turn)
        self.stall.update(tool[:3], float(np.linalg.norm(move)))
        self.calls += 1

        target = np.eye(4)
        target[:3, :3], target[:3, 3] = rotation, position
        flange_target = flange_from_tcp(matrix_to_pose(target))
        if settings.mode == "qpos_ik":
            return self._qpos(flange_target, gripper, observation)
        assert self.idle is not None
        row = np.concatenate([flange_target, [gripper]])
        ordered = (
            np.concatenate([row, np.zeros(EE_POSE_DIM + 1)])
            if self.arm == "left"
            else np.concatenate([np.zeros(EE_POSE_DIM + 1), row])
        )
        return self.idle.apply_ee(ordered)[None, :]

    def _qpos(self, flange_target: np.ndarray, gripper: float, observation: Any) -> np.ndarray:
        assert self.idle is not None
        qpos = np.asarray(observation.qpos, dtype=np.float64)
        seed = qpos[QPOS_SLICES[self.arm]][:6]
        result = self.arm_model.inverse(flange_target, seed)
        if not result.converged:
            self.unreachable += 1
        row = np.concatenate([result.q, [gripper]])
        ordered = np.zeros(14)
        ordered[QPOS_SLICES[self.arm]] = row
        return self.idle.apply_qpos(ordered)[None, :]

    def info(self) -> dict[str, Any]:
        """JSON-ready, for `episode_info()`."""
        return {
            "calls": self.calls,
            "unreachable_targets": self.unreachable,
            **self.target.info(),
            **self.stall.info(),
        }


def arm_choice(demonstration: Demonstration, settings: Settings) -> ArmChoice:
    """The arm this adapter drives, by the toolkit's rule and the config's thresholds."""
    return choose_arm(demonstration, tie_m=settings.arm_tie_m, move_m=settings.arm_move_m)
