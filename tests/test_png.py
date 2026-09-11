import struct
import zlib

import numpy as np
import pytest

from robotwin_icil.png import encode, write_png


def decode(data: bytes) -> np.ndarray:
    """Just enough of a PNG reader to check the writer: unfiltered 8-bit images."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset, chunks = 8, []
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        kind, body = data[offset + 4 : offset + 8], data[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])
        assert crc == zlib.crc32(kind + body) & 0xFFFFFFFF, kind
        chunks.append((kind, body))
        offset += 12 + length
    assert [kind for kind, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    width, height, depth, color, *_ = struct.unpack(">IIBBBBB", chunks[0][1])
    assert depth == 8
    channels = {0: 1, 2: 3, 6: 4}[color]
    rows = np.frombuffer(zlib.decompress(chunks[1][1]), dtype=np.uint8)
    rows = rows.reshape(height, 1 + width * channels)
    assert not rows[:, 0].any()
    return rows[:, 1:].reshape(height, width, channels)


@pytest.mark.parametrize("channels", [3, 4])
def test_an_image_round_trips(channels):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(180, 320, channels), dtype=np.uint8)
    assert np.array_equal(decode(encode(image)), image)


def test_grey_and_non_contiguous_images_are_written_as_they_look(tmp_path):
    grey = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert np.array_equal(decode(encode(grey))[:, :, 0], grey)
    flipped = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)[:, ::-1]
    path = write_png(tmp_path / "nested" / "camera.png", flipped)
    assert np.array_equal(decode(path.read_bytes()), flipped)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 4, 3), dtype=np.float32),
        np.zeros((4, 4, 2), dtype=np.uint8),
        np.zeros((0, 4, 3), dtype=np.uint8),
        np.zeros(4, dtype=np.uint8),
    ],
)
def test_images_a_png_cannot_hold_are_refused(image):
    with pytest.raises(ValueError):
        encode(image)
