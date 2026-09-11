"""`UniSkillPolicy`: the frozen skill encoder and a skill-conditioned diffusion policy (plan A3).

Per episode: the demonstration becomes skill rows once (`SkillExtractor`, one per 20 Hz step),
then the policy acts one 14-dim absolute qpos target per `act()` from a queue of `Ta` = 8. It
re-plans when the queue is empty, from the last `To` = 2 observations (padded by repetition at
the start, no warm-up steps) and the skill row of the number of actions executed so far, held at
the last row past the end (`conversion.skill_row`). Each re-plan samples `Tp` = 16 actions by
DDIM from noise drawn from this policy's own generator, which `seed()` reseeds.

The released UniSkill policy checkpoint is gone, so constructing this policy needs a checkpoint
exported from a network trained on RoboTwin exports (docs/models/uniskill.md). This module
imports no torch at import time; the model loads when the policy is constructed, in the
`icil-uniskill` environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from icil_policies.common.chunking import ChunkExecutor
from robotwin_icil.demo import Demonstration, DemonstrationError
from robotwin_icil.policy import ICILPolicy, Observation, PolicyError

from .config import load_config, verify_sha256
from .conversion import ADAPTER_VERSION, CAMERA_PROFILE, SkillCursor, policy_image

__all__ = ["ADAPTER_VERSION", "NO_CHECKPOINT", "UniSkillPolicy"]

NO_CHECKPOINT = (
    "UniSkillPolicy needs an exported policy checkpoint: pass checkpoint=PATH or set "
    "`checkpoint` in its config. UniSkill's released policy checkpoint is gone (its link "
    "returns 404), so the policy is a DiffusionPolicyUNet trained on RoboTwin exports outside "
    "the benchmark and exported with its EMA weights baked in; see docs/models/uniskill.md"
)
# Model-card skill settings that must match the encoder the policy is conditioned on here.
SKILL_SETTINGS = ("camera", "crop", "rate_hz", "k", "isd_sha256")


class UniSkillPolicy(ICILPolicy):
    name = "uniskill"
    action_type = "qpos"

    def __init__(
        self,
        config: str | None = None,
        checkpoint: str | None = None,
        checkpoint_sha256: str | None = None,
        device: str | None = None,
        skill_extractor: Any = None,
    ) -> None:
        super().__init__()
        self.config = load_config(
            config, checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256, device=device
        )
        path = self.config.checkpoint_path
        if path is None:
            raise PolicyError(NO_CHECKPOINT)
        self.checkpoint = str(path)
        self.checkpoint_sha256 = verify_sha256(
            path, self.config.checkpoint_sha256, "the UniSkill policy checkpoint"
        )
        import torch

        from . import model

        self._torch = torch
        self._model = model
        self._policy = model.load_checkpoint(path, self.config.device)
        if skill_extractor is None:
            from .skills import SkillExtractor

            skill_extractor = SkillExtractor(
                self.config.skills, self.config.device, self.config.augmentation
            )
        self._extractor = skill_extractor
        self._check_compatible()
        n_obs, n_action, _ = self._policy.horizons
        self._executor = ChunkExecutor(self._plan, n_obs=n_obs, n_action=n_action)
        self._cursor = SkillCursor(self._executor)
        self._generator = torch.Generator(device=self._policy.network.device)
        self._generator.manual_seed(self.config.seed)
        self._skills = None

    def _check_compatible(self) -> None:
        policy = self._policy
        if policy.ac_dim != 14:
            raise PolicyError(f"{self.checkpoint}: {policy.ac_dim}-dim actions, not 14-dim qpos")
        unknown = set(policy.low_dim_keys) - {"qpos"}
        if unknown:
            raise PolicyError(f"{self.checkpoint}: reads {sorted(unknown)}; only qpos is observed")
        if policy.skill_dim != self._extractor.skill_dim:
            raise PolicyError(
                f"{self.checkpoint}: conditioned on {policy.skill_dim}-dim skills, the encoder "
                f"gives {self._extractor.skill_dim}"
            )
        trained = policy.metadata.get("skills") or {}
        here = self._extractor.describe()
        for key in SKILL_SETTINGS:
            if key in trained and key in here and trained[key] != here[key]:
                raise PolicyError(
                    f"{self.checkpoint} was trained on skills with {key} {trained[key]!r}; "
                    f"the encoder here uses {here[key]!r}"
                )

    # --- the protocol -------------------------------------------------------------------

    def seed(self, seed: int) -> None:
        """Reseed the diffusion noise; never torch's global RNG, which RoboTwin reseeds."""
        self._generator.manual_seed(int(seed))

    def _reset(self) -> None:
        self._cursor.reset()
        self._skills = None

    def _set_demonstration(self, demonstration: Demonstration) -> None:
        try:
            skills = np.asarray(self._extractor.extract(demonstration), dtype=np.float32)
        except DemonstrationError as exc:
            raise PolicyError(f"{self.name}: {exc}") from exc
        if skills.ndim != 2 or len(skills) == 0 or skills.shape[1] != self._policy.skill_dim:
            raise PolicyError(f"{self.name}: skill rows of shape {skills.shape}")
        self._skills = self._torch.from_numpy(skills).to(self._policy.network.device)
        self._cursor.set_rows(len(skills))

    def _act(self, observation: Observation) -> np.ndarray:
        return self._cursor.act(observation)

    def _plan(self, history: list[Observation]) -> np.ndarray:
        row = self._cursor.row()
        skill = self._skills[row][None]
        actions = self._model.sample(self._policy, self._inputs(history), skill, self._generator)
        return actions[0].detach().cpu().numpy().astype(np.float64)

    def _inputs(self, history: list[Observation]) -> dict[str, Any]:
        """(1, To, ...) tensors per input key, processed as robomimic processes its dataset."""
        import robomimic.utils.obs_utils as ObsUtils

        torch = self._torch
        device = self._policy.network.device
        inputs = {}
        for key in self._policy.rgb_keys:
            _, height, width = self._policy.shapes[key]
            frames = []
            for observation in history:
                if key not in observation.images:
                    raise PolicyError(
                        f"{self.name}: the observation has no {key!r}; the policy reads "
                        f"{list(self._policy.rgb_keys)} under the {CAMERA_PROFILE!r} profile"
                    )
                frames.append(policy_image(observation.images[key], height, width))
            stacked = torch.from_numpy(np.stack(frames))
            inputs[key] = ObsUtils.process_obs(stacked, obs_key=key)[None].to(device)
        for key in self._policy.low_dim_keys:
            qpos = torch.as_tensor(np.stack([o.qpos for o in history]), dtype=torch.float32)
            inputs[key] = ObsUtils.process_obs(qpos, obs_key=key)[None].to(device)
        return inputs

    # --- what the run records -------------------------------------------------------------

    def parameter_checksum(self) -> str:
        """sha256 of every weight the policy runs: the policy network, the ISD, the depth model."""
        return self._model.parameter_checksum(self._policy.network.nets, *self._extractor.modules())

    def describe(self) -> dict[str, Any]:
        metadata = self._policy.metadata
        n_obs, n_action, n_pred = self._policy.horizons
        skills = self._extractor.describe()
        return {
            **super().describe(),
            "adapter": "uniskill",
            "adapter_version": ADAPTER_VERSION,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_tasks": metadata.get("training_tasks", "unknown"),
            "training_regime": metadata.get("training_regime", "unknown"),
            "camera_profile_required": CAMERA_PROFILE,
            "parameter_checksum": self.parameter_checksum(),
            "k": skills["k"],
            "augmentation": metadata.get("augmentation", "unknown"),
            "skill_encoder": skills,
            "horizons": {"observation": n_obs, "action": n_action, "prediction": n_pred},
            "inference_steps": self._policy.inference_steps,
            "ema": "baked into the checkpoint at export",
            "config": self.config.as_dict(),
        }

    def episode_info(self) -> dict[str, Any]:
        """The skill row at each re-plan against the demonstration's progress through its rows."""
        cursor = self._cursor
        rows = cursor.n_rows
        last = cursor.replans[-1][1] if cursor.replans else None
        return {
            "skill_rows": rows,
            "executed_steps": cursor.executed,
            "replan_steps": [step for step, _ in cursor.replans],
            "replan_rows": [row for _, row in cursor.replans],
            "skill_progress": None if last is None or rows < 2 else last / (rows - 1),
            "held_past_end": rows > 0 and cursor.executed > rows - 1,
        }
