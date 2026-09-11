"""The UniSkill adapter's configuration: `uniskill.yml`, model paths and their sha256 checks.

`load_config()` reads the packaged `uniskill.yml`, then a user's file over it (only the keys it
names), then keyword overrides. Unknown keys are refused rather than ignored, so a misspelt
setting cannot silently fall back to its default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robotwin_icil.policy import PolicyError

from .conversion import CAMERA, ISD_CROP, ISD_RESOLUTION, RATE_HZ, SKILL_INTERVAL, Augmentation

DEFAULT_CONFIG = Path(__file__).with_name("uniskill.yml")


def icil_home() -> Path:
    """Where models and model environments live: $ICIL_HOME, default ~/.cache/robotwin-icil."""
    return Path(os.environ.get("ICIL_HOME") or "~/.cache/robotwin-icil").expanduser()


def model_path(value: str | os.PathLike) -> Path:
    """An absolute path: `~` expanded, and a relative one taken under $ICIL_HOME/models."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else icil_home() / "models" / path


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: str | os.PathLike, expected: str | None, what: str) -> str:
    """The file's sha256, after checking it exists and, when `expected` is given, matches it."""
    path = Path(path)
    if not path.is_file():
        raise PolicyError(f"{what} not found at {path}")
    actual = sha256_file(path)
    if expected is not None and actual != expected.lower():
        raise PolicyError(f"{what} at {path} has sha256 {actual}, expected {expected}")
    return actual


@dataclass(frozen=True)
class SkillEncoder:
    """The frozen ISD and depth model, and how a demonstration becomes their input."""

    isd: str
    isd_sha256: str | None
    depth: str
    depth_sha256: str | None
    camera: str = CAMERA
    crop: int = ISD_CROP
    rate_hz: float = RATE_HZ
    k: int = SKILL_INTERVAL
    resolution: int = ISD_RESOLUTION
    batch_size: int = 32
    num_layers: int = 8
    num_heads: int = 4
    hidden_dim: int = 256
    skill_dim: int = 64
    out_dim: int = 768

    @property
    def isd_path(self) -> Path:
        return model_path(self.isd)

    @property
    def depth_dir(self) -> Path:
        return model_path(self.depth)


@dataclass(frozen=True)
class UniSkillConfig:
    checkpoint: str | None
    checkpoint_sha256: str | None
    device: str
    seed: int
    skills: SkillEncoder
    augmentation: Augmentation

    @property
    def checkpoint_path(self) -> Path | None:
        return None if self.checkpoint is None else model_path(self.checkpoint)

    def as_dict(self) -> dict[str, Any]:
        """JSON values, for `describe()`."""
        data = dataclasses.asdict(self)
        data["augmentation"]["crop_scale"] = list(self.augmentation.crop_scale)
        return data


def _merge(base: dict[str, Any], update: dict[str, Any], where: str) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if key not in base:
            raise PolicyError(f"{where}: unknown setting {key!r}; known are {sorted(base)}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise PolicyError(f"{where}: {key!r} must be a mapping, got {value!r}")
            merged[key] = _merge(base[key], value, f"{where}: {key}")
        else:
            merged[key] = value
    return merged


def _read(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise PolicyError(f"cannot read the UniSkill config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PolicyError(f"the UniSkill config {path} is not a mapping")
    return data


def load_config(path: str | os.PathLike | None = None, **overrides: Any) -> UniSkillConfig:
    """The packaged defaults, then the file at `path` over them, then `overrides` (top level)."""
    data = _read(DEFAULT_CONFIG)
    if path is not None:
        data = _merge(data, _read(Path(path).expanduser()), str(path))
    data = _merge(data, {k: v for k, v in overrides.items() if v is not None}, "overrides")
    try:
        augmentation = dict(data["augmentation"])
        augmentation["crop_scale"] = tuple(augmentation["crop_scale"])
        return UniSkillConfig(
            checkpoint=None if data["checkpoint"] is None else str(data["checkpoint"]),
            checkpoint_sha256=data["checkpoint_sha256"],
            device=str(data["device"]),
            seed=int(data["seed"]),
            skills=SkillEncoder(**data["skills"]),
            augmentation=Augmentation(**augmentation),
        )
    except (TypeError, ValueError) as exc:
        raise PolicyError(f"invalid UniSkill config: {exc}") from exc
