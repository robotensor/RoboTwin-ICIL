"""PNG files from uint8 arrays, with the standard library alone.

`robotwin-icil cameras` writes one image per camera to be checked by eye. That is all this is for,
so it writes 8-bit grey, rgb or rgba, unfiltered, and reads nothing.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Channels -> PNG colour type: grey, truecolour, truecolour with alpha.
_COLOR_TYPES = {1: 0, 3: 2, 4: 6}


def encode(image: np.ndarray) -> bytes:
    """An `(h, w)`, `(h, w, 3)` or `(h, w, 4)` uint8 image as the bytes of a PNG file."""
    pixels = np.asarray(image)
    if pixels.dtype != np.uint8:
        raise ValueError(f"a PNG here is 8-bit: expected uint8, got {pixels.dtype}")
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    if pixels.ndim != 3 or pixels.shape[2] not in _COLOR_TYPES or 0 in pixels.shape[:2]:
        raise ValueError(f"expected an (h, w), (h, w, 3) or (h, w, 4) image, got {pixels.shape}")
    height, width, channels = pixels.shape
    rows = np.ascontiguousarray(pixels).reshape(height, width * channels)
    # Every scanline starts with its filter type; 0 is none.
    scanlines = np.concatenate([np.zeros((height, 1), dtype=np.uint8), rows], axis=1)
    header = struct.pack(">IIBBBBB", width, height, 8, _COLOR_TYPES[channels], 0, 0, 0)
    return b"".join(
        [
            _SIGNATURE,
            _chunk(b"IHDR", header),
            _chunk(b"IDAT", zlib.compress(scanlines.tobytes(), 6)),
            _chunk(b"IEND", b""),
        ]
    )


def write_png(path: Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(image))
    return path


def _chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)
