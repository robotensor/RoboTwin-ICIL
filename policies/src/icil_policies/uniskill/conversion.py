"""UniSkill's conversion math, in numpy: skill rows, their indexing, and the images each model sees.

Constants and their sources (docs/models/uniskill.md lists them all):

- `RATE_HZ` 20: the ISD's LIBERO frames and the policy's control rate; the demonstration is
  resampled on its frame times (`icil_policies.common.resample`), never on frame index.
- `SKILL_INTERVAL` k = 20 steps, 1.0 s at 20 Hz: the paper's setting for LIBERO.
- One skill row per resampled step, the future frame clamped at the end: row i pairs step i with
  step min(i + k, N - 1). That is the released `skills.zip` convention (as many rows as the
  demonstration has steps), not `extract_skill.py`'s N - k rows.
- `ISD_CROP` 180: the ISD sees `far_side_camera` (320x180, 45° vertical) through a centred
  180x180 square, which is LIBERO's 45° field; the same crop at training and evaluation. Frames
  are upright, as RoboTwin renders them: recomputed released skills matched upright LIBERO
  frames (cosine 0.964) far better than flipped ones (0.84).
- `ISD_RESOLUTION` 224: `dynamics/idm.py`'s `idm_resolution`.

At evaluation the policy re-plans every `Ta` actions with the skill row of the number of
actions executed so far (`skill_row`), held at the last row past the end: the step count is the
index, as in the fork's `evaluate.py`. One qpos action per resampled step keeps that count aligned
with the rows (`icil_policies.common.oracles.resampled_qpos_actions`).

Skill augmentation (`aug_0..4`) makes variants of a demonstration's skills for training. Neither
the paper nor the code says how the released variants were made. ASSUMPTION: colour jitter
(brightness, contrast, saturation) and a small square crop, drawn once per demonstration and
variant from a seeded generator and applied to every frame, so a variant is one consistent
video; the parameters are `Augmentation`'s, set in the adapter's config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from icil_policies.common.images import area_weights, bilinear_weights, crop_start
from icil_policies.common.resample import Resampled, resample
from robotwin_icil.demo import Demonstration, DemonstrationError

ADAPTER_VERSION = "1"

RATE_HZ = 20.0
SKILL_INTERVAL = 20
CAMERA = "far_side_camera"
CAMERA_PROFILE = "far_side"
ISD_CROP = 180
ISD_RESOLUTION = 224


def future_rows(n_rows: int, k: int = SKILL_INTERVAL) -> np.ndarray:
    """(n_rows,) the step each skill row looks ahead to: min(i + k, n_rows - 1)."""
    if n_rows < 1:
        raise ValueError(f"a demonstration has at least one step, got {n_rows}")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return np.minimum(np.arange(n_rows) + k, n_rows - 1)


def skill_row(executed: int, n_rows: int) -> int:
    """The row a re-plan after `executed` actions uses: that count, held at the last row."""
    if executed < 0:
        raise ValueError(f"executed must be non-negative, got {executed}")
    if n_rows < 1:
        raise ValueError(f"there are no skill rows to index, got {n_rows}")
    return min(int(executed), n_rows - 1)


class SkillCursor:
    """The executed-action count that picks each re-plan's skill row, and the order of the two.

    One cursor per episode, wrapping a `ChunkExecutor`. `act()` returns one action and only
    then advances the count, so the row a re-plan reads is the number of actions executed
    *before* it; `row()` is what the plan callback asks for, and records `(executed, row)` so
    the episode can report where each re-plan sat in the demonstration. Keeping both here, and
    not in the policy, keeps the ordering under the numpy tests that run in CI.
    """

    def __init__(self, executor: Any) -> None:
        self._executor = executor
        self.n_rows = 0
        self.executed = 0
        self.replans: list[tuple[int, int]] = []

    def reset(self) -> None:
        """Forget the episode: the chunk executor's state, the count and the re-plan record."""
        self._executor.reset()
        self.n_rows = 0
        self.executed = 0
        self.replans = []

    def set_rows(self, n_rows: int) -> None:
        """Record how many skill rows this episode's demonstration gave."""
        if n_rows < 1:
            raise ValueError(f"there are no skill rows to index, got {n_rows}")
        self.n_rows = int(n_rows)

    def row(self) -> int:
        """The row a re-plan conditions on now: `skill_row` of the actions executed so far."""
        row = skill_row(self.executed, self.n_rows)
        self.replans.append((self.executed, row))
        return row

    def act(self, observation: Any) -> np.ndarray:
        """One action from the executor, then the count advances."""
        action = self._executor.act(observation)
        self.executed += 1
        return action


def isd_view(image: np.ndarray, crop: int = ISD_CROP) -> np.ndarray:
    """The centred `crop` x `crop` square of a uint8 frame the ISD sees; a view of `image`."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (h, w, 3) uint8 frame, got {image.shape} {image.dtype}")
    height, width = image.shape[:2]
    top = crop_start(height / 2, crop, height)
    left = crop_start(width / 2, crop, width)
    return image[top : top + crop, left : left + crop]


@dataclass(frozen=True)
class SkillSteps:
    """A demonstration at the skill rate: what the ISD pairs, one row per step."""

    resampled: Resampled
    views: np.ndarray  # (N, crop, crop, 3) uint8, the ISD's view of each step's nearest frame
    future: np.ndarray  # (N,) the step each row pairs with

    def __len__(self) -> int:
        return len(self.future)


def skill_steps(
    demonstration: Demonstration,
    camera: str = CAMERA,
    rate_hz: float = RATE_HZ,
    k: int = SKILL_INTERVAL,
    crop: int = ISD_CROP,
) -> SkillSteps:
    """The demonstration resampled to `rate_hz` on its times, with each step's view and pair."""
    if camera not in demonstration.cameras:
        raise DemonstrationError(
            f"UniSkill's skill encoder reads {camera!r}, which the camera profile "
            f"{CAMERA_PROFILE!r} renders; this demonstration has {list(demonstration.cameras)}"
        )
    resampled = resample(demonstration, rate_hz)
    views = np.stack([isd_view(image, crop) for image in resampled.images(camera)])
    return SkillSteps(resampled=resampled, views=views, future=future_rows(len(resampled), k))


@dataclass(frozen=True)
class Augmentation:
    """How skill variants `aug_0..` are drawn (ASSUMPTION, see the module docstring).

    Each strength s draws a factor from [1 - s, 1 + s]; `crop_scale` bounds the side of the
    square crop as a fraction of the ISD's view.
    """

    variants: int = 5
    seed: int = 0
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    crop_scale: tuple[float, float] = (0.9, 1.0)

    def __post_init__(self) -> None:
        if isinstance(self.variants, bool) or not isinstance(self.variants, int):
            raise ValueError(f"variants must be an integer, got {self.variants!r}")
        if self.variants < 0:
            raise ValueError(f"variants must be non-negative, got {self.variants}")
        for name in ("brightness", "contrast", "saturation"):
            strength = getattr(self, name)
            if not 0 <= strength < 1:
                raise ValueError(f"{name} must be in [0, 1), got {strength}")
        low, high = self.crop_scale
        if not 0 < low <= high <= 1:
            raise ValueError(f"crop_scale must satisfy 0 < low <= high <= 1, got {self.crop_scale}")
        object.__setattr__(self, "crop_scale", (float(low), float(high)))


@dataclass(frozen=True)
class AugmentationDraw:
    """One variant's parameters, applied alike to every frame of one demonstration."""

    brightness: float
    contrast: float
    saturation: float
    scale: float
    top: float  # where the crop sits in the free margin, 0 (top) to 1 (bottom)
    left: float
    variant: int = field(default=0)


def draw_augmentation(augmentation: Augmentation, variant: int, key: int = 0) -> AugmentationDraw:
    """Variant `variant` of the demonstration `key` (for example its export index), seeded."""
    if not 0 <= variant < augmentation.variants:
        raise ValueError(f"variant must be in [0, {augmentation.variants}), got {variant}")
    rng = np.random.default_rng([augmentation.seed, key, variant])

    def factor(strength: float) -> float:
        return float(rng.uniform(1 - strength, 1 + strength))

    return AugmentationDraw(
        brightness=factor(augmentation.brightness),
        contrast=factor(augmentation.contrast),
        saturation=factor(augmentation.saturation),
        scale=float(rng.uniform(*augmentation.crop_scale)),
        top=float(rng.uniform()),
        left=float(rng.uniform()),
        variant=variant,
    )


# ITU-R 601 luma, as torchvision's rgb_to_grayscale weighs the channels.
_LUMA = np.array([0.299, 0.587, 0.114])


def augment_views(views: np.ndarray, draw: AugmentationDraw) -> np.ndarray:
    """(N, s', s', 3) uint8: colour jitter then a square crop of side round(scale * s).

    Brightness scales every channel, contrast blends each frame with its mean grey level and
    saturation each pixel with its grey, each clamped to [0, 255] as torchvision's
    `adjust_brightness`, `adjust_contrast` and `adjust_saturation` do; in that fixed order.
    """
    views = np.asarray(views)
    if views.dtype != np.uint8 or views.ndim != 4 or views.shape[-1] != 3:
        raise ValueError(f"expected (N, h, w, 3) uint8 views, got {views.shape} {views.dtype}")
    x = np.clip(views.astype(np.float64) * draw.brightness, 0, 255)
    mean = (x @ _LUMA).mean(axis=(1, 2))[:, None, None, None]
    x = np.clip(draw.contrast * x + (1 - draw.contrast) * mean, 0, 255)
    grey = (x @ _LUMA)[..., None]
    x = np.clip(draw.saturation * x + (1 - draw.saturation) * grey, 0, 255)
    height, width = views.shape[1:3]
    side = max(1, round(draw.scale * min(height, width)))
    top = round(draw.top * (height - side))
    left = round(draw.left * (width - side))
    x = x[:, top : top + side, left : left + side]
    return np.rint(x).astype(np.uint8)


def policy_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """(height, width, 3) uint8: a camera frame as the policy's encoder receives it.

    Each axis is area-averaged when it shrinks and bilinearly interpolated (`align_corners=False`)
    when it grows, then rounded to whole levels, as a stored training frame is. Training exports
    and evaluation both go through this function, so the two see the same pixels.
    """
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (h, w, 3) uint8 frame, got {image.shape} {image.dtype}")

    def weights(n_in: int, n_out: int) -> np.ndarray:
        return area_weights(n_in, n_out) if n_out <= n_in else bilinear_weights(n_in, n_out)

    rows = weights(image.shape[0], height)
    columns = weights(image.shape[1], width)
    out = np.einsum("ip,pqc->iqc", rows, image.astype(np.float64))
    out = np.einsum("jq,iqc->ijc", columns, out)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
