"""The image path for a model trained on LIBERO's 128-pixel renders (plan 3.1), in numpy.

aloha's arms sit 0.30 m either side of the centre line, so a centred square crop puts the acting
arm near the image edge, where LIBERO's Panda sits centred. The path:

1. `crop_square` cuts a square around a column, the projection of the acting arm's workspace
   centre: a constant of the camera profile, never scene information. From `far_side_camera`'s
   320x180 that is 180x180, and clamping keeps the column within 90-230.
2. `area_resize` to 128: each output pixel is the mean of the input area it covers, as a
   128-pixel render would integrate it (OpenCV's `INTER_AREA` when shrinking).
3. `bilinear_resize` to 224 with `align_corners=False` and no antialiasing:
   `torch.nn.functional.interpolate(mode="bilinear", align_corners=False)`, which is how BPP
   brings its 128-pixel LIBERO frames to 224 (`train_network/dataset/
   libero_replay_image_dataset.py`, `_receding_rgb_numpy_thwc_to_float_chw`, and
   `train_network/env_runner/libero_image_runner.py`).

`arm_centred_view` runs the three, so the model sees a 128-pixel frame brought to 224, as in
training, rather than a sharper 224 image.

Images are (h, w) or (h, w, channels) arrays; resizes return float64, `arm_centred_view` float32.
Both resizes are separable: a weight matrix per axis, `rows @ image @ columns.T`.
"""

from __future__ import annotations

import math

import numpy as np

CROP = 180
MIDDLE = 128
OUTPUT = 224


def crop_start(centre: float, size: int, extent: int) -> int:
    """The first index of a `size` window centred on `centre`, clamped inside `extent`."""
    if size > extent:
        raise ValueError(f"cannot crop {size} pixels from {extent}")
    return min(max(math.floor(centre - size / 2 + 0.5), 0), extent - size)


def crop_square(
    image: np.ndarray, size: int, centre_col: float, centre_row: float | None = None
) -> np.ndarray:
    """A `size` x `size` view of the image centred on (row, col), shifted to stay inside it.

    The row defaults to the image's middle. A view: it shares memory with `image`.
    """
    height, width = image.shape[:2]
    top = crop_start(height / 2 if centre_row is None else centre_row, size, height)
    left = crop_start(centre_col, size, width)
    return image[top : top + size, left : left + size]


def area_weights(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in): output pixel i averages input interval [i s, (i + 1) s), s = n_in / n_out."""
    if n_out > n_in:
        raise ValueError(f"area resizing only shrinks; asked for {n_in} -> {n_out}")
    scale = n_in / n_out
    starts = np.arange(n_out)[:, None] * scale
    pixels = np.arange(n_in)[None, :]
    overlap = np.minimum(pixels + 1, starts + scale) - np.maximum(pixels, starts)
    return np.clip(overlap, 0, None) / scale


def bilinear_weights(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in): half-pixel centres (`align_corners=False`), sources clamped at the edges."""
    scale = n_in / n_out
    source = np.maximum((np.arange(n_out) + 0.5) * scale - 0.5, 0.0)
    low = np.minimum(np.floor(source).astype(int), n_in - 1)
    high = np.minimum(low + 1, n_in - 1)
    fraction = source - low
    weights = np.zeros((n_out, n_in))
    rows = np.arange(n_out)
    np.add.at(weights, (rows, low), 1 - fraction)
    np.add.at(weights, (rows, high), fraction)
    return weights


def area_resize(image: np.ndarray, height: int, width: int | None = None) -> np.ndarray:
    """Shrink to (height, width) by pixel-area averaging; width defaults to height."""
    width = height if width is None else width
    return _separable(
        image, area_weights(image.shape[0], height), area_weights(image.shape[1], width)
    )


def bilinear_resize(image: np.ndarray, height: int, width: int | None = None) -> np.ndarray:
    """Resize to (height, width) bilinearly, `align_corners=False`; width defaults to height."""
    width = height if width is None else width
    return _separable(
        image, bilinear_weights(image.shape[0], height), bilinear_weights(image.shape[1], width)
    )


def arm_centred_view(
    image: np.ndarray,
    centre_col: float,
    crop: int = CROP,
    middle: int = MIDDLE,
    output: int = OUTPUT,
    quantize: bool = True,
) -> np.ndarray:
    """(output, output, 3) float32 in [0, 1] from a uint8 frame: crop, area to 128, bilinear to 224.

    With `quantize`, the 128-pixel stage is rounded to whole 0-255 levels, as a stored LIBERO
    frame is, before it is scaled to [0, 1] and resized, which is the order BPP's loader follows.
    Only uint8 frames are accepted: a float frame already in [0, 1] would come out near black.
    """
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise ValueError(f"expected a uint8 frame with 0-255 levels, got {image.dtype}")
    small = area_resize(crop_square(image, crop, centre_col), middle)
    if quantize:
        small = np.clip(np.rint(small), 0, 255)
    return bilinear_resize(small / 255.0, output).astype(np.float32)


def _separable(image: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim not in (2, 3):
        raise ValueError(f"expected an (h, w) or (h, w, channels) image, got {image.shape}")
    flat = image if image.ndim == 3 else image[..., None]
    out = np.einsum("ip,pqc->iqc", rows, flat)
    out = np.einsum("jq,iqc->ijc", columns, out)
    return out if image.ndim == 3 else out[..., 0]
