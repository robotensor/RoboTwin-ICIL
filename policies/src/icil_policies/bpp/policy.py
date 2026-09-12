"""`BPPPolicy`: the released BPP checkpoint, prompted with one RoboTwin demonstration.

Runs in the `icil-bpp` environment, reached from the simulator's process through the core's
built-in `remote` (#39), which never imports model code:

    robotwin-icil eval --policy remote \\
        --policy-arg policy=icil_policies.bpp:BPPPolicy \\
        --policy-arg python=$ICIL_HOME/envs/icil-bpp/bin/python \\
        --policy-arg config=policies/configs/bpp_liberogen_combination.yaml \\
        --camera-profile far_side --suite v1 --episodes 90 --seed 42 --run-dir runs/bpp

One episode, in order: `seed()` seeds this process's torch RNG; `reset()` clears the model's
prompt and the action queue; `set_demonstration()` converts the one demonstration and calls the
model's `prompt()` exactly once; each `act()` adds an observation to a history of two, and takes
the next action from a queue of `exec_action_horizon` 12 out of the 16 `predict_action` returns.
There is no warm-up: BPP's own runner opens the gripper for ten steps first, which here would
come out of the task's step budget, and the history is padded by repeating the first observation
exactly as that runner's wrapper pads it.

**Sampling.** BPP's `DiffusionUnetPolicy.predict_action` takes no generator: it calls
`conditional_sample`, which draws `torch.randn` from the process's global RNG
(`policy/diffusion_unet_policy.py:237`). `seed()` therefore seeds that global RNG per episode.
That is safe only out of process, which is how this adapter runs: RoboTwin reseeds torch's
global RNG with the *scene* seed whenever it builds a scene, so an in-process adapter drawing
from the global RNG would sample as a function of privileged state. In the model server nothing
else touches torch's RNG, and the seed comes from the policy's own stream. Nothing enforced
that, so `__init__` now refuses outright where SAPIEN is importable (`refuse_in_process`).

The policy is frozen: the model is loaded in eval mode with gradients off, nothing here writes a
parameter, and `describe()` carries a `parameter_checksum` the runner audits at both ends of a
run.
"""

from __future__ import annotations

import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robotwin_icil.demo import Demonstration
from robotwin_icil.policy import ICILPolicy, Observation, PolicyError

from ..common.chunking import ChunkExecutor
from ..common.kinematics import AlohaArm
from . import ADAPTER_VERSION
from .conversion import (
    Execution,
    GroupedExecutor,
    Prompt,
    aggregate,
    arm_choice,
    build_prompt,
    group_bounds,
    observation_state,
    out_of_range_fraction,
    prompt_state,
)
from .model import load_model, parameter_checksum
from .settings import (
    ACTION_DIM,
    EXEC_ACTION_HORIZON,
    OBS_HORIZON,
    PROMPT_CHUNK_N_ACTIONS,
    Settings,
    load,
)

# The proprioception keys whose normalized range is reported per episode (plan 3.6): `ee_ori` is
# a rot6d the normalizer leaves alone, so only these two can fall outside [-1, 1].
PROPRIO_RANGE_KEYS = ("ee_pos", "gripper_states")


class BPPPolicy(ICILPolicy):
    """The BPP LIBERO-Gen Combination checkpoint behind the RoboTwin conversion of `conversion`."""

    name = "bpp"

    def __init__(
        self,
        config: str | None = None,
        device: str = "cuda",
        allow_in_process: bool = False,
        **overrides: Any,
    ) -> None:
        super().__init__()
        if not allow_in_process:
            refuse_in_process()
        self.settings: Settings = load(config, **overrides)
        # Per instance, not per class: the execution mode decides whether this adapter drives
        # `take_action('ee')` or `take_action('qpos')` (plan 3.4). `remote` carries it over.
        self.action_type = self.settings.action_type
        self.config = None if config is None else str(Path(config).resolve())
        self.device = device
        if not self.settings.checkpoint:
            raise PolicyError("bpp config: checkpoint must name `icil-bpp slim`'s output")
        self.model = load_model(
            self.settings.checkpoint, device, self.settings.checkpoint_sha256 or None
        )
        self._checksum = parameter_checksum(self.model)
        self._normalizer = self._normalizer_params()
        self._chunker = _chunker(self.model)
        self._arm_model = (
            AlohaArm(self.settings.urdf_path, "left")
            if self.settings.mode == "qpos_ik" and self.settings.urdf_path
            else None
        )
        self._reset()

    # ----------------------------------------------------------------- the episode lifecycle

    def seed(self, seed: int) -> None:
        """Seed this process's torch RNG, which is where BPP's diffusion noise comes from."""
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        self._seed = int(seed)

    def _reset(self) -> None:
        self._prompt: Prompt | None = None
        self._choice = None
        self._execution: Execution | None = None
        self._executor: ChunkExecutor | None = None
        self._out_of_range: dict[str, float] = {}
        self._observed: list[dict[str, np.ndarray]] = []
        if getattr(self, "model", None) is not None:
            self.model.reset(action_exec_horizon=EXEC_ACTION_HORIZON)

    def _set_demonstration(self, demonstration: Demonstration) -> None:
        import torch

        self._choice = arm_choice(demonstration, self.settings)
        self._prompt = build_prompt(demonstration, self.settings, self._choice.arm)
        state = prompt_state(self._prompt, self.settings)
        self._out_of_range = self._proprio_out_of_range(state)
        self._execution = Execution(self.settings, self._choice.arm, self._arm_model)
        self._executor = _executor(self._plan, self.settings)
        with torch.inference_mode():
            self.model.prompt(self._prompt_dict(state, self._prompt.actions))

    def _act(self, observation: Observation) -> np.ndarray:
        if self._executor is None or self._execution is None:
            raise PolicyError(f"{self.name}: act() before set_demonstration()")
        action = self._executor.act(observation)
        return self._execution.act(action, observation)

    # ------------------------------------------------------------------------ the model side

    def _plan(self, history) -> np.ndarray:
        """One `predict_action` from the last two observations, as rows this adapter executes."""
        import torch

        assert self._choice is not None
        states = [observation_state(obs, self.settings, self._choice.arm) for obs in history]
        # The proprioception the model actually conditioned on, kept (without the images) for
        # the rollout's own out-of-range fraction: it is the arm leaving LIBERO's training range
        # mid-episode that the prompt's fraction cannot see (plan 3.6).
        self._observed.append({key: states[-1][key] for key in PROPRIO_RANGE_KEYS})
        batch = {
            key: torch.from_numpy(
                np.stack([state[key] for state in states]).astype(np.float32)[None]
            ).to(self.device)
            for key in states[0]
        }
        # `predict_action` normalizes the dict it is given, whose keys are the observation keys
        # themselves (`utils/prompt_util.py:239-253`); BPP's own runner passes it flat too
        # (`env_runner/libero_image_runner.py:703`). A `{"obs": ...}` wrapper would be looked up
        # in the normalizer as a key of its own.
        with torch.inference_mode():
            predicted = self.model.predict_action(batch)["action"]
        chunk = predicted.detach().to("cpu").numpy()[0][:EXEC_ACTION_HORIZON]
        if chunk.shape != (EXEC_ACTION_HORIZON, ACTION_DIM):
            raise PolicyError(f"BPP returned a chunk of shape {chunk.shape}")
        if self.settings.mode != "ee_grouped":
            return chunk
        return np.stack(
            [
                aggregate(chunk[start:stop], self.settings)
                for start, stop in group_bounds(chunk, self.settings.mode, self.settings)
            ]
        )

    def _prompt_dict(self, state: dict[str, np.ndarray], actions: np.ndarray) -> dict:
        """The prompt BPP's `prompt()` takes, built by BPP's own chunker and collate.

        `prompt_state` already reads one observation per chunk, which is what BPP's sampler does
        with `downsample_obs_by_chunk=True` (`common/sampler.py:936`), so the chunker runs with
        `obs_predownsampled=True` and only reshapes the actions into (L, 20, 10), zero-padding
        the last partial chunk as `pad_end_prompt_actions: zeros` asks.
        """
        import torch
        from behavior_prompting.common.np_util import add_batch_dim, remove_batch_dim
        from behavior_prompting.train_network.utils.prompt_util import collate_prompts

        sample = {
            "obs": {key: value.astype(np.float32) for key, value in state.items()},
            "action": actions.astype(np.float32),
            "metadata": {},
        }
        chunked = remove_batch_dim(
            self._chunker.chunk_prompt(add_batch_dim(sample), obs_predownsampled=True)
        )
        chunked["obs"] = {key: torch.from_numpy(value) for key, value in chunked["obs"].items()}
        chunked["action"] = torch.from_numpy(chunked["action"])
        batch = collate_prompts([{"obs": {"prompt": chunked}}])
        prompt = batch["obs"]["prompt"]
        return {
            "obs": {key: value.to(self.device) for key, value in prompt["obs"].items()},
            "action": prompt["action"].to(self.device),
            "metadata": {"mask": prompt["metadata"]["mask"].to(self.device)},
        }

    def _normalizer_params(self) -> dict[str, dict[str, np.ndarray]]:
        """The checkpoint's scale and offset per proprioception key, as numpy.

        Read once, in `__init__`: `close()` drops the model, and an episode's record is written
        after it in a server that is shutting down.
        """
        params = self.model.normalizer.params_dict
        return {
            key: {
                "scale": params[key]["scale"].detach().cpu().numpy(),
                "offset": params[key]["offset"].detach().cpu().numpy(),
            }
            for key in PROPRIO_RANGE_KEYS
            if key in params
        }

    def _proprio_out_of_range(self, state: Mapping[str, np.ndarray]) -> dict[str, float]:
        """How much of this proprioception the checkpoint's normalizer puts outside [-1, 1]."""
        return {
            key: round(out_of_range_fraction(state[key], numbers), 6)
            for key, numbers in self._normalizer.items()
            if key in state
        }

    def _rollout_out_of_range(self) -> dict[str, float]:
        """The same fraction over the observations the model was actually conditioned on."""
        if not self._observed:
            return {}
        stacked = {
            key: np.stack([state[key] for state in self._observed]) for key in self._observed[0]
        }
        return self._proprio_out_of_range(stacked)

    # ---------------------------------------------------------------------------- the record

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "adapter": "icil_policies.bpp",
            "adapter_version": ADAPTER_VERSION,
            "checkpoint": self.settings.source_checkpoint,
            "checkpoint_sha256": self.settings.source_sha256,
            # No RoboTwin task was ever trained on: this is a transfer row. What it did see is
            # LIBERO-Gen spatial combinations, which `training_domains` says and no V1 task is.
            "training_tasks": [],
            "training_domains": ["LIBERO-Gen spatial combinations (LIBERO_Tabletop_Manipulation)"],
            "camera_profile_required": self.settings.camera_profile,
            # Recomputed, never the cached start value: the runner's frozen-policy audit calls
            # `describe()` again at the end of a run and compares the two, and a cached constant
            # would compare equal however the weights had changed. `_checksum` stays only as
            # what the model loaded with, so a run that has already closed still describes it.
            "parameter_checksum": (
                self._checksum if self.model is None else parameter_checksum(self.model)
            ),
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
            "proprio_out_of_range_rollout": self._rollout_out_of_range(),
            "plans": self._executor.plans if self._executor is not None else 0,
            **self._execution.info(),
        }

    def environment(self) -> dict[str, str]:
        import torch

        environment = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": str(torch.version.cuda),
            "device": self.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
        try:
            import behavior_prompting

            environment["behavior_prompting"] = str(
                Path(behavior_prompting.__file__).resolve().parent.parent
            )
        except ImportError:  # pragma: no cover
            pass
        return environment

    def close(self) -> None:
        import torch

        self.model = None
        self._executor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def refuse_in_process() -> None:
    """Refuse to construct this policy in a process that can also run the simulator.

    `seed()` seeds the process's *global* torch RNG, which is where BPP's diffusion noise comes
    from, and RoboTwin calls `torch.manual_seed(scene_seed)` on every build: in one process the
    noise would become a function of the scene seed, the privileged state CLAUDE.md keeps out of
    a policy. The supported path is `remote`/`serve`, where the model has a process of its own
    and no simulator in it. `allow_in_process=True` is for a caller that has checked (BPP's own
    LIBERO runner, say, which is not RoboTwin).
    """
    import importlib.util
    import sys

    present = "sapien" in sys.modules
    if not present:
        try:
            present = importlib.util.find_spec("sapien") is not None
        except (ImportError, ValueError):  # pragma: no cover - a broken sapien install
            present = True
    if present:
        raise PolicyError(
            "bpp: SAPIEN is importable here, so this may be the simulator's process, where "
            "seeding torch's global RNG would make the diffusion noise a function of the scene "
            "seed. Run the model behind `--policy remote --policy-arg python=.../icil-bpp/bin/"
            "python`, or pass allow_in_process=True if there is no simulator in this process"
        )


def _chunker(model: Any) -> Any:
    """BPP's own `PromptActionChunker`, over the shape_meta the model was built from."""
    from behavior_prompting.train_network.utils.prompt_util import PromptActionChunker

    from .model import compose_config

    shape_meta = compose_config().shape_meta
    if int(shape_meta["prompt_chunk_n_actions"]) != PROMPT_CHUNK_N_ACTIONS:
        raise PolicyError(
            f"this composition chunks {shape_meta['prompt_chunk_n_actions']} actions, "
            f"not {PROMPT_CHUNK_N_ACTIONS}"
        )
    return PromptActionChunker(shape_meta)


def _executor(plan, settings: Settings) -> ChunkExecutor:
    """One action per `act()` for `ee_step` and `qpos_ik`; one group per `act()` when grouped."""
    if settings.mode == "ee_grouped":
        return GroupedExecutor(plan, n_obs=OBS_HORIZON)
    return ChunkExecutor(plan, n_obs=OBS_HORIZON, n_action=EXEC_ACTION_HORIZON)
