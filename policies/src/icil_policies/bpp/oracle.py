"""`BPPConversionReplay`, the conversion oracle for the BPP adapter (plan A2, O3).

The prompt's own actions, replayed through the whole conversion and execution chain: the same
resampling, the same gains, the same gripper labels, the same virtual target, the same execution
mode and the same idle hold as `BPPPolicy`, from the same config file — only the network is
missing. It is what a perfect model behind this adapter would do, so its V1 fraction is the
ceiling for any model behind it, and the gap to `replay` (and to `replay_ee`) is the structural
loss of driving one aloha arm with LIBERO-frame 20 Hz OSC deltas.

numpy only: it runs in the simulator's environment, in process, with no torch and no BPP.

    robotwin-icil eval --policy icil_policies.bpp:BPPConversionReplay \\
        --policy-arg config=policies/configs/bpp_liberogen_combination.yaml \\
        --camera-profile far_side --suite v1 --episodes 90 --seed 1000 --run-dir runs/o3
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robotwin_icil.demo import Demonstration
from robotwin_icil.policy import ICILPolicy, Observation, PolicyError

from ..common.kinematics import AlohaArm
from . import ADAPTER_VERSION
from .conversion import (
    Execution,
    Prompt,
    arm_choice,
    build_prompt,
    group_bounds,
    out_of_range_fraction,
    prompt_state,
)
from .conversion import aggregate as aggregate_actions
from .settings import ACTION_DIM, EXEC_ACTION_HORIZON, Settings, load

# What a step past the end of the prompt commands: no motion, and the gripper the prompt's last
# action commanded (`hold_past_the_end`). The demonstration ended where the expert succeeded, so
# holding there is the right thing to do — including holding a release open, which a constant
# gripper entry of 0 would instead close (`decode_action` closes on any value >= 0).
HOLD = np.zeros(ACTION_DIM)
HOLD[3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # rot6d of the identity
HOLD[9] = -1.0  # open, for the demonstration that has no action to hold at all


class BPPConversionReplay(ICILPolicy):
    """Replays the demonstration's own converted actions through the BPP adapter's chain."""

    name = "bpp_conversion_replay"

    def __init__(self, config: str | None = None, **overrides: Any) -> None:
        super().__init__()
        self.settings: Settings = load(config, **overrides)
        # Per instance, not per class: one adapter drives `ee` or `qpos` by its mode (plan 3.4).
        self.action_type = self.settings.action_type
        self.config = None if config is None else str(Path(config).resolve())
        self._arm_model = _arm_model(self.settings)
        self._normalizer = _normalizer(self.settings)
        self._reset()

    def _reset(self) -> None:
        self._prompt: Prompt | None = None
        self._choice = None
        self._execution: Execution | None = None
        self._queue: list[np.ndarray] = []
        self._cursor = 0
        self._held = 0
        self._out_of_range: dict[str, float] = {}

    def _set_demonstration(self, demonstration: Demonstration) -> None:
        self._choice = arm_choice(demonstration, self.settings)
        self._prompt = build_prompt(demonstration, self.settings, self._choice.arm)
        self._execution = Execution(self.settings, self._choice.arm, self._arm_model)
        self._out_of_range = _proprio_out_of_range(self._prompt, self.settings, self._normalizer)

    def _act(self, observation: Observation) -> np.ndarray:
        assert self._prompt is not None and self._execution is not None
        if not self._queue:
            self._queue = _next_calls(self._prompt.actions, self._cursor, self.settings)
            self._cursor += EXEC_ACTION_HORIZON
        action = self._queue.pop(0)
        return self._execution.act(action, observation)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "adapter": "icil_policies.bpp",
            "adapter_version": ADAPTER_VERSION,
            "checkpoint": None,
            "training_tasks": [],
            "camera_profile_required": self.settings.camera_profile,
            "oracle": "conversion",
            "config": self.config,
            "settings": self.settings.describe(),
        }

    def episode_info(self) -> dict[str, Any]:
        if self._prompt is None or self._choice is None or self._execution is None:
            return {}
        return {
            **self._choice.info(),
            "prompt_chunks": self._prompt.chunks,
            "prompt_steps": int(len(self._prompt.actions)),
            "prompt_rate_hz": self._prompt.rate_hz,
            "clipped_action_fraction": round(self._prompt.clipped, 6),
            "proprio_out_of_range": self._out_of_range,
            "actions_held_past_the_end": self._held,
            **self._execution.info(),
        }


def _next_calls(actions: np.ndarray, cursor: int, settings: Settings) -> list[np.ndarray]:
    """The actions of one `exec_action_horizon` chunk, grouped as the mode executes them."""
    chunk = actions[cursor : cursor + EXEC_ACTION_HORIZON]
    if len(chunk) == 0:
        return [hold_past_the_end(actions)]
    return [
        aggregate_actions(chunk[start:stop], settings)
        for start, stop in group_bounds(chunk, settings.mode, settings)
    ]


def hold_past_the_end(actions: np.ndarray) -> np.ndarray:
    """The action a step past the prompt takes: no motion, the last commanded gripper held.

    The gripper matters: a task that ends in a release wants the fingers to stay open, and a
    task that ends holding wants them shut. Taking it from the prompt's own last action keeps
    the oracle's ceiling free of an artifact the conversion invented.
    """
    hold = HOLD.copy()
    if len(actions):
        hold[9] = float(np.asarray(actions)[-1, 9])
    return hold


def _arm_model(settings: Settings) -> AlohaArm | None:
    if settings.mode != "qpos_ik":
        return None
    if not settings.urdf_path:
        raise PolicyError("mode qpos_ik needs urdf_path, aloha-agilex's URDF, in the config")
    return AlohaArm(settings.urdf_path, "left")


def _normalizer(settings: Settings) -> dict[str, dict[str, list[float]]]:
    """The checkpoint's normalizer as `icil-bpp slim` wrote it, or nothing.

    Plain JSON beside the slimmed checkpoint, so this oracle reports the same out-of-range
    fraction as `BPPPolicy` without loading torch.
    """
    path = Path(settings.normalizer) if settings.normalizer else None
    if path is None and settings.checkpoint:
        path = Path(settings.checkpoint) / "normalizer.json"
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PolicyError(f"cannot read the normalizer {path}: {exc}") from exc


def _proprio_out_of_range(
    prompt: Prompt, settings: Settings, normalizer: dict[str, dict[str, list[float]]]
) -> dict[str, float]:
    """How much of the prompt's proprioception the checkpoint's normalizer puts outside [-1, 1]."""
    if not normalizer:
        return {}
    state = prompt_state(prompt, settings)
    return {
        key: round(out_of_range_fraction(state[key], normalizer[key]), 6)
        for key in ("ee_pos", "gripper_states")
        if key in normalizer and key in state
    }
