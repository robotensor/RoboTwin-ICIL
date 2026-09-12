"""`SkillExtractor`: UniSkill's frozen skill encoder on a demonstration (plan A3).

The ISD (`dynamics/idm.py`, weights `idm.pth`) reads a pair of frames and their depth from
Depth-Anything-V2-Small and returns a 64-dim skill. On a demonstration it returns one row per
20 Hz step, pairing step i with step min(i + k, N - 1) (`conversion.skill_steps`).

Preprocessing follows the ISD's own training code (`diffusion/dataset/base_dataset.py` and
`diffusion/train_uniskill.py`), not `extract_skill.py`:

- each view is resized to 224x224 as a PIL image with bilinear interpolation and scaled to
  [0, 1] (torchvision's `Resize` then `ToTensor`), the ISD's visual input;
- Depth-Anything's slow `DPTImageProcessor` takes that [0, 1] image with `do_rescale=False`
  (518x518, ImageNet normalisation); each predicted depth map is scaled to [0, 1] by its own
  minimum and maximum, then bilinearly resized to 224x224.

`extract_skill.py` hands the processor raw 0-255 frames with `do_rescale=False`, which the ISD
never saw in training. Depth runs once per step and is reused for both frames of every pair.

Both models are frozen: loaded once, `requires_grad_(False)`, `eval()`, run under
`torch.inference_mode()`, and their weights verified against the config's sha256 first.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

from robotwin_icil.demo import Demonstration

from .config import SkillEncoder, load_config, verify_sha256
from .conversion import (
    ADAPTER_VERSION,
    Augmentation,
    SkillSteps,
    augment_views,
    draw_augmentation,
    skill_steps,
)

__all__ = ["ADAPTER_VERSION", "SkillExtractor", "load_depth", "load_isd"]

DEPTH_WEIGHTS = "model.safetensors"


@contextlib.contextmanager
def _without_imagenet_download() -> Iterator[None]:
    """Build torchvision ResNet-18s without their ImageNet weights.

    `dynamics/idm.py` builds its visual encoder with `resnet18(pretrained=True)`, which
    downloads ImageNet weights; `idm.pth` overwrites every one of them (loaded strictly), so the
    download is skipped and a server starts offline.
    """
    original = torchvision.models.resnet18

    def resnet18(*args: Any, pretrained: bool = False, weights: Any = None, **kwargs: Any):
        return original(*args, weights=None, **kwargs)

    torchvision.models.resnet18 = resnet18
    try:
        yield
    finally:
        torchvision.models.resnet18 = original


def load_isd(settings: SkillEncoder, device: torch.device | str) -> tuple[torch.nn.Module, str]:
    """The ISD with `idm.pth` loaded strictly, frozen, and the file's sha256."""
    from dynamics.idm import IDM  # the ISD checkout, on the env's path

    digest = verify_sha256(settings.isd_path, settings.isd_sha256, "UniSkill's skill encoder")
    # Its constructor initialises weights and runs a probe input: keep that off the global RNG.
    with torch.random.fork_rng(devices=[]), _without_imagenet_download():
        idm = IDM(
            num_layers=settings.num_layers,
            num_heads=settings.num_heads,
            hidden_dim=settings.hidden_dim,
            skill_dim=settings.skill_dim,
            out_dim=settings.out_dim,
            idm_resolution=settings.resolution,
        )
    # A plain state dict: loads with weights_only=True, unlike the fork's own checkpoints.
    state = torch.load(settings.isd_path, map_location="cpu", weights_only=True)
    idm.load_state_dict(state, strict=True)
    return idm.requires_grad_(False).eval().to(device), digest


def load_depth(settings: SkillEncoder, device: torch.device | str) -> tuple[Any, Any, str]:
    """Depth-Anything-V2-Small from its local directory: processor, frozen model, sha256."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    digest = verify_sha256(
        settings.depth_dir / DEPTH_WEIGHTS, settings.depth_sha256, "Depth-Anything-V2-Small"
    )
    # The slow processor, which the ISD was trained with (transformers 4.48's only one for DPT).
    processor = AutoImageProcessor.from_pretrained(
        settings.depth_dir, use_fast=False, local_files_only=True
    )
    model = AutoModelForDepthEstimation.from_pretrained(settings.depth_dir, local_files_only=True)
    return processor, model.requires_grad_(False).eval().to(device), digest


class SkillExtractor:
    """Skill rows of a demonstration, from the frozen ISD and depth model."""

    def __init__(
        self,
        settings: SkillEncoder | None = None,
        device: torch.device | str = "cuda",
        augmentation: Augmentation | None = None,
    ) -> None:
        config = load_config()
        self.settings = config.skills if settings is None else settings
        self.augmentation = config.augmentation if augmentation is None else augmentation
        self.device = torch.device(device)
        self.idm, self.isd_sha256 = load_isd(self.settings, self.device)
        self.depth_processor, self.depth_model, self.depth_sha256 = load_depth(
            self.settings, self.device
        )

    @property
    def skill_dim(self) -> int:
        return self.settings.skill_dim

    def modules(self) -> tuple[torch.nn.Module, ...]:
        """The frozen networks, for a parameter checksum."""
        return (self.idm, self.depth_model)

    def describe(self) -> dict[str, Any]:
        s = self.settings
        return {
            "isd_sha256": self.isd_sha256,
            "depth_sha256": self.depth_sha256,
            "camera": s.camera,
            "crop": s.crop,
            "rate_hz": s.rate_hz,
            "k": s.k,
            "resolution": s.resolution,
            "skill_dim": s.skill_dim,
        }

    def steps(self, demonstration: Demonstration) -> SkillSteps:
        s = self.settings
        return skill_steps(demonstration, s.camera, s.rate_hz, s.k, s.crop)

    def extract(
        self, demonstration: Demonstration, variant: int | None = None, key: int = 0
    ) -> np.ndarray:
        """(N, skill_dim) float32: row i is the skill from step i to step min(i + k, N - 1).

        `variant` None is the base skills (`base.npy`), what the policy is conditioned on at
        evaluation; 0, 1, ... are the augmented variants `aug_<variant>` of demonstration `key`.
        """
        steps = self.steps(demonstration)
        views = steps.views
        if variant is not None:
            views = augment_views(views, draw_augmentation(self.augmentation, variant, key))
        return self.encode(views, steps.future)

    def extract_variants(self, demonstration: Demonstration, key: int = 0) -> dict[str, np.ndarray]:
        """`base` and every `aug_<i>`, keyed as the fork's skill directory names its files."""
        variants = {"base": self.extract(demonstration)}
        for variant in range(self.augmentation.variants):
            variants[f"aug_{variant}"] = self.extract(demonstration, variant, key)
        return variants

    @torch.inference_mode()
    def encode(self, views: np.ndarray, future: np.ndarray) -> np.ndarray:
        """(N, skill_dim) skills of views (N, h, w, 3) uint8 paired with `future` (N,)."""
        future = np.asarray(future)
        if len(views) != len(future) or len(views) == 0:
            raise ValueError(f"{len(views)} views and {len(future)} pairs")
        images = self._images(views)
        depth = self._depth(images)
        size = self.settings.batch_size
        skills = []
        for start in range(0, len(views), size):
            now = torch.arange(start, min(start + size, len(views)))
            later = torch.from_numpy(future[now.numpy()])
            visual = torch.stack([images[now], images[later]], dim=1).to(self.device)
            depth_pair = torch.stack([depth[now], depth[later]], dim=1)
            # (B, 1, skill_dim): the ISD keeps its last frame's time axis.
            skill = self.idm(depth_pair, visual, return_skill=True)
            skills.append(skill.reshape(len(now), -1).float().cpu())
        return torch.cat(skills).numpy().astype(np.float32, copy=False)

    def _images(self, views: np.ndarray) -> torch.Tensor:
        """(N, 3, R, R) float in [0, 1] on the CPU: PIL bilinear resize, then to a tensor."""
        size = [self.settings.resolution] * 2
        bilinear = TF.InterpolationMode.BILINEAR
        return torch.stack(
            [TF.to_tensor(TF.resize(Image.fromarray(v), size, bilinear)) for v in views]
        )

    def _depth(self, images: torch.Tensor) -> torch.Tensor:
        """(N, R, R) depth on the device, each map scaled to [0, 1] by its own extremes."""
        maps = []
        for batch in images.split(self.settings.batch_size):
            inputs = self.depth_processor(images=list(batch), do_rescale=False, return_tensors="pt")
            depth = self.depth_model(pixel_values=inputs["pixel_values"].to(self.device))
            depth = depth.predicted_depth
            low = depth.flatten(1).min(-1).values[:, None, None]
            high = depth.flatten(1).max(-1).values[:, None, None]
            depth = (depth - low) / (high - low)
            size = (self.settings.resolution,) * 2
            maps.append(F.interpolate(depth[:, None], size=size, mode="bilinear")[:, 0])
        return torch.cat(maps)
