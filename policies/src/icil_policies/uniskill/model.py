"""The skill-conditioned policy: the robomimic fork's `DiffusionPolicyUNet` for RoboTwin.

- `robotwin_config()` is the fork's `configs/uniskill_policy.json` for RoboTwin (plan B3,
  `uniskill_robotwin`): `far_side_camera`, `left_camera` and `right_camera` at 128x128 with the
  fork's 116-pixel crop, `qpos` as the only low-dimensional input, 14-dim absolute qpos actions
  with min-max normalisation, To 2, Ta 8, Tp 16, DDIM with 10 of 100 steps, EMA power 0.75.
- An exported checkpoint (`save_checkpoint`) holds that config, the shapes, the action
  normaliser, a model card (`metadata`: training tasks, regime, skill settings) and the network
  weights with the EMA average already baked in (`bake_ema`, at export time). The fork's own
  rollout sampled with the live weights (`EMAModel.model_cls` is the live network), a deviation
  recorded in docs/models/uniskill.md; loading baked weights once, before the evaluator's start
  checksum, means the evaluator never calls `EMAModel.copy_to`, a parameter write.
- `load_checkpoint` builds the network without an optimizer or an EMA model, loads the weights
  strictly and switches to eval mode; `sample` runs DDIM from noise drawn from the policy's own
  `torch.Generator`, never from torch's global RNG, which RoboTwin reseeds with every scene.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robotwin_icil.policy import PolicyError

from .conversion import ADAPTER_VERSION

__all__ = ["ADAPTER_VERSION", "FORMAT", "LoadedPolicy", "load_checkpoint", "robotwin_config"]

FORMAT = "icil-uniskill-policy/1"
ALGO_NAME = "diffusion_policy"
ACTION_DIM = 14
RGB_KEYS = ("far_side_camera", "left_camera", "right_camera")
LOW_DIM_KEYS = ("qpos",)
IMAGE_SIZE = 128  # the fork's LIBERO renders
IMAGE_CROP = 116  # the fork's CropRandomizer; a centre crop at evaluation

# configs/uniskill_policy.json at 2803ad6, with RoboTwin's cameras, qpos and 14-dim actions.
ROBOTWIN_CONFIG: dict[str, Any] = {
    "algo_name": ALGO_NAME,
    "train": {
        "seq_length": 15,
        "pad_seq_length": True,
        "frame_stack": 2,
        "pad_frame_stack": True,
        "goal_mode": "skill",
        "skill_aug": True,
        "aug_num": 5,
        "hdf5_normalize_obs": False,
        "dataset_keys": ["actions"],
        "action_keys": ["actions"],
        "action_config": {"actions": {"normalization": "min_max"}},
    },
    "algo": {
        "horizon": {"observation_horizon": 2, "action_horizon": 8, "prediction_horizon": 16},
        "skill": {"enabled": True, "skill_dim": 64, "dropout": False},
        "lang": {"enabled": False},
        "subgoal": {"enabled": False},
        "unet": {
            "enabled": True,
            "diffusion_step_embed_dim": 256,
            "down_dims": [256, 512, 1024],
            "kernel_size": 5,
            "n_groups": 8,
        },
        "ema": {"enabled": True, "power": 0.75},
        "ddpm": {"enabled": False},
        "ddim": {
            "enabled": True,
            "num_train_timesteps": 100,
            "num_inference_timesteps": 10,
            "beta_schedule": "squaredcos_cap_v2",
            "clip_sample": True,
            "set_alpha_to_one": True,
            "steps_offset": 0,
            "prediction_type": "epsilon",
        },
    },
    "observation": {
        "modalities": {
            "obs": {"low_dim": list(LOW_DIM_KEYS), "rgb": list(RGB_KEYS), "depth": [], "scan": []},
            "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
        },
        "encoder": {
            "rgb": {
                "core_class": "VisualCore",
                "core_kwargs": {
                    "feature_dimension": 64,
                    "backbone_class": "ResNet18Conv",
                    "backbone_kwargs": {"pretrained": False, "input_coord_conv": False},
                    "pool_class": "SpatialSoftmax",
                    "pool_kwargs": {
                        "num_kp": 32,
                        "learnable_temperature": False,
                        "temperature": 1.0,
                        "noise_std": 0.0,
                    },
                },
                "obs_randomizer_class": "CropRandomizer",
                "obs_randomizer_kwargs": {
                    "crop_height": IMAGE_CROP,
                    "crop_width": IMAGE_CROP,
                    "num_crops": 1,
                    "pos_enc": False,
                },
            }
        },
    },
}


def _update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def robotwin_config(overrides: Mapping[str, Any] | None = None):
    """The fork's config object for RoboTwin, with `overrides` (a nested dict) applied."""
    from robomimic.config import config_factory

    config = config_factory(ALGO_NAME)
    with config.values_unlocked():
        config.update(_update(ROBOTWIN_CONFIG, overrides or {}))
    return config


def config_from_json(text: str):
    """A fork config from its JSON dump, as the fork's `evaluate.py` rebuilds one."""
    from robomimic.config import config_factory

    data = json.loads(text)
    config = config_factory(data["algo_name"])
    with config.values_unlocked():
        config.update(data)
    return config


def default_shapes(image_size: int = IMAGE_SIZE) -> dict[str, list[int]]:
    """Input shapes, channels first, as robomimic's `shape_metadata['all_shapes']` has them."""
    shapes = {key: [3, image_size, image_size] for key in RGB_KEYS}
    shapes.update({key: [ACTION_DIM] for key in LOW_DIM_KEYS})
    return shapes


def build_network(config, shapes: Mapping[str, list[int]], ac_dim: int, device):
    """The fork's `DiffusionPolicyUNet` for `config`, built without an optimizer.

    robomimic's `Algo` creates optimizers in its constructor; the benchmark's evaluator must
    hold none, so a subclass skips that step. Weight initialisation runs off the global RNG.
    """
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.algo.diffusion_policy import DiffusionPolicyUNet

    class FrozenDiffusionPolicyUNet(DiffusionPolicyUNet):
        def _create_optimizers(self) -> None:
            self.optimizers = {}
            self.lr_schedulers = {}

    ObsUtils.initialize_obs_utils_with_config(config)
    with torch.random.fork_rng(devices=[]):
        return FrozenDiffusionPolicyUNet(
            algo_config=config.algo,
            obs_config=config.observation,
            global_config=config,
            obs_key_shapes={key: list(value) for key, value in shapes.items()},
            ac_dim=ac_dim,
            device=torch.device(device),
        )


def bake_ema(network, ema_state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """The network's state dict with its parameters replaced by the EMA average.

    Export time only, on a copy of the trained network: diffusers' `EMAModel.state_dict()`
    keeps `shadow_params` in the order of `nets.parameters()`; buffers are the network's own.
    """
    shadow = list(ema_state["shadow_params"])
    parameters = list(network.nets.parameters())
    if len(shadow) != len(parameters):
        raise ValueError(f"{len(shadow)} EMA shadow parameters for {len(parameters)} parameters")
    by_identity = {id(p): s for p, s in zip(parameters, shadow, strict=True)}
    state = {}
    for name, tensor in network.nets.state_dict(keep_vars=True).items():
        value = by_identity.get(id(tensor), tensor)
        if value.shape != tensor.shape:
            raise ValueError(
                f"EMA {name} has shape {tuple(value.shape)}, not {tuple(tensor.shape)}"
            )
        state[name] = value.detach().clone().cpu()
    return state


def save_checkpoint(
    path: str | Path,
    config,
    shapes: Mapping[str, list[int]],
    ac_dim: int,
    nets_state: Mapping[str, torch.Tensor],
    action_scale: np.ndarray,
    action_offset: np.ndarray,
    metadata: Mapping[str, Any],
) -> str:
    """Write an exported checkpoint; returns its sha256. `nets_state` has the EMA baked in.

    The action normaliser follows robomimic's: normalised = (action - offset) / scale.
    """
    payload = {
        "format": FORMAT,
        "algo_name": ALGO_NAME,
        "config": config.dump(),
        "shape_metadata": {"all_shapes": {k: list(v) for k, v in shapes.items()}, "ac_dim": ac_dim},
        "nets": {name: tensor.detach().cpu() for name, tensor in nets_state.items()},
        "action_normalization_stats": {
            "actions": {
                "scale": np.asarray(action_scale, dtype=np.float64).reshape(1, -1).tolist(),
                "offset": np.asarray(action_offset, dtype=np.float64).reshape(1, -1).tolist(),
            }
        },
        "metadata": json.loads(json.dumps(dict(metadata))),
    }
    torch.save(payload, path)
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class LoadedPolicy:
    """An exported checkpoint, built and loaded, ready to sample."""

    network: Any  # the fork's DiffusionPolicyUNet, eval mode, no optimizer, no EMA
    config: Any
    shapes: dict[str, list[int]]
    ac_dim: int
    action_scale: torch.Tensor  # (ac_dim,)
    action_offset: torch.Tensor
    metadata: dict[str, Any]

    @property
    def horizons(self) -> tuple[int, int, int]:
        h = self.config.algo.horizon
        return h.observation_horizon, h.action_horizon, h.prediction_horizon

    @property
    def rgb_keys(self) -> tuple[str, ...]:
        return tuple(self.config.observation.modalities.obs.rgb)

    @property
    def low_dim_keys(self) -> tuple[str, ...]:
        return tuple(self.config.observation.modalities.obs.low_dim)

    @property
    def skill_dim(self) -> int:
        return int(self.config.algo.skill.skill_dim)

    @property
    def inference_steps(self) -> int:
        algo = self.config.algo
        return int((algo.ddim if algo.ddim.enabled else algo.ddpm).num_inference_timesteps)


def load_checkpoint(path: str | Path, device) -> LoadedPolicy:
    """Build the network an exported checkpoint describes and load its weights, once."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        found = payload.get("format") if isinstance(payload, dict) else type(payload).__name__
        raise PolicyError(
            f"{path} is not an exported UniSkill policy checkpoint ({FORMAT}), it is {found!r}: "
            "a training checkpoint of the fork must first be exported with its EMA weights baked in"
        )
    config = config_from_json(payload["config"])
    algo = config.algo
    if not (algo.skill.enabled and algo.unet.enabled):
        raise PolicyError(f"{path}: needs a skill-conditioned UNet diffusion policy")
    if algo.skill.dropout:
        # The fork trains through a dropout head on the skill but samples without it.
        raise PolicyError(f"{path}: skill.dropout differs between the fork's training and sampling")
    if algo.lang.enabled or algo.subgoal.enabled:
        raise PolicyError(f"{path}: language and subgoal conditioning are not supported")
    with config.values_unlocked():
        config.algo.ema.enabled = False  # baked in: no EMA model at evaluation
    shapes = {k: list(v) for k, v in payload["shape_metadata"]["all_shapes"].items()}
    ac_dim = int(payload["shape_metadata"]["ac_dim"])
    network = build_network(config, shapes, ac_dim, device)
    network.nets.load_state_dict(payload["nets"], strict=True)
    network.set_eval()
    network.nets.requires_grad_(False)
    stats = payload["action_normalization_stats"]["actions"]
    return LoadedPolicy(
        network=network,
        config=config,
        shapes=shapes,
        ac_dim=ac_dim,
        action_scale=torch.tensor(stats["scale"], dtype=torch.float32).reshape(-1),
        action_offset=torch.tensor(stats["offset"], dtype=torch.float32).reshape(-1),
        metadata=dict(payload.get("metadata") or {}),
    )


@torch.inference_mode()
def sample(
    policy: LoadedPolicy,
    obs: Mapping[str, torch.Tensor],
    skill: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """(B, Tp - To + 1, ac_dim) actions, unnormalised, from observations (B, To, ...) and skills
    (B, skill_dim): the fork's `_get_action_trajectory`, with noise from `generator`."""
    import robomimic.utils.tensor_utils as TensorUtils

    network = policy.network
    nets = network.nets
    n_obs, _, n_pred = policy.horizons
    features = TensorUtils.time_distributed(
        {"obs": dict(obs), "goal": None}, nets["policy"]["obs_encoder"], inputs_as_kwargs=True
    )
    condition = torch.cat([features.flatten(start_dim=1), skill], dim=-1)
    batch = condition.shape[0]
    noisy = torch.randn(
        (batch, n_pred, policy.ac_dim), generator=generator, device=generator.device
    ).to(network.device)
    scheduler = network.noise_scheduler
    scheduler.set_timesteps(policy.inference_steps)
    for t in scheduler.timesteps:
        noise = nets["policy"]["noise_pred_net"](sample=noisy, timestep=t, global_cond=condition)
        noisy = scheduler.step(
            model_output=noise, timestep=t, sample=noisy, generator=generator
        ).prev_sample
    actions = noisy[:, n_obs - 1 :]
    scale = policy.action_scale.to(actions.device)
    offset = policy.action_offset.to(actions.device)
    return actions * scale + offset


def parameter_checksum(*modules: torch.nn.Module) -> str:
    """sha256 of every parameter and buffer of `modules`, by name, dtype, shape and bytes."""
    digest = hashlib.sha256()
    for index, module in enumerate(modules):
        for name, tensor in sorted(module.state_dict().items()):
            data = tensor.detach().cpu().contiguous()
            digest.update(f"{index}:{name}:{data.dtype}:{tuple(data.shape)}".encode())
            digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
