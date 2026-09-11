import numpy as np
import pytest

from icil_policies.common import images

RNG = np.random.default_rng(0)


def test_a_centred_crop_of_the_far_side_frame():
    frame = np.arange(180 * 320).reshape(180, 320)
    crop = images.crop_square(frame, 180, centre_col=160)
    assert crop.shape == (180, 180)
    assert crop[0, 0] == frame[0, 70] and crop[-1, -1] == frame[-1, 249]
    assert np.shares_memory(crop, frame)


@pytest.mark.parametrize(
    ("centre", "start"), [(0, 0), (90, 0), (91, 1), (229, 139), (230, 140), (400, 140)]
)
def test_the_crop_column_is_clamped_to_90_230(centre, start):
    assert images.crop_start(centre, 180, 320) == start


def test_a_crop_can_move_vertically_and_refuses_to_outgrow_the_frame():
    frame = np.arange(240 * 320).reshape(240, 320)
    crop = images.crop_square(frame, 100, centre_col=50, centre_row=200)
    assert crop[0, 0] == frame[140, 0]
    with pytest.raises(ValueError, match="cannot crop 250"):
        images.crop_square(frame, 250, centre_col=160)


def test_area_resizing_averages_covered_pixels():
    image = RNG.uniform(0, 255, size=(8, 6, 3))
    halved = images.area_resize(image, 4, 3)
    blocks = image.reshape(4, 2, 3, 2, 3).mean(axis=(1, 3))
    np.testing.assert_allclose(halved, blocks)
    np.testing.assert_allclose(images.area_resize(np.full((180, 180, 3), 77.0), 128), 77.0)
    weights = images.area_weights(180, 128)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(weights.sum(axis=0), 128 / 180)  # every input pixel counts once
    with pytest.raises(ValueError, match="only shrinks"):
        images.area_resize(image, 16)


def test_bilinear_resizing_uses_half_pixel_centres():
    row = np.array([[0.0, 1.0]])
    np.testing.assert_allclose(images.bilinear_resize(row, 1, 4), [[0, 0.25, 0.75, 1]])
    image = RNG.uniform(size=(5, 7))
    np.testing.assert_allclose(images.bilinear_resize(image, 5, 7), image)
    np.testing.assert_allclose(images.bilinear_weights(128, 224).sum(axis=1), 1.0)


def test_the_arm_centred_view():
    frame = RNG.integers(0, 256, size=(180, 320, 3), dtype=np.uint8)
    view = images.arm_centred_view(frame, centre_col=120)
    assert view.shape == (224, 224, 3) and view.dtype == np.float32
    assert 0.0 <= view.min() and view.max() <= 1.0
    flat = images.arm_centred_view(np.full((180, 320, 3), 51, dtype=np.uint8), centre_col=200)
    np.testing.assert_allclose(flat, 51 / 255, atol=1e-7)
    sharp = images.arm_centred_view(frame, centre_col=120, quantize=False)
    assert np.abs(sharp - view).max() <= 0.5 / 255 + 1e-6


def test_images_must_be_two_or_three_dimensional():
    with pytest.raises(ValueError, match=r"\(h, w\)"):
        images.bilinear_resize(np.zeros((2, 2, 2, 2)), 4)


def test_area_resizing_matches_opencv_when_importable():
    cv2 = pytest.importorskip("cv2")
    # In float64, so float32 rounding cannot hide a convention error.
    image = RNG.uniform(size=(180, 180, 3))
    expected = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    np.testing.assert_allclose(images.area_resize(image, 128), expected, atol=1e-12)
    wide = RNG.uniform(size=(240, 320, 3))
    expected = cv2.resize(wide, (128, 100), interpolation=cv2.INTER_AREA)
    np.testing.assert_allclose(images.area_resize(wide, 100, 128), expected, atol=1e-12)


def test_bilinear_resizing_matches_torch_when_importable():
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")
    for shape, size in [
        ((128, 128), (224, 224)),
        ((128, 96), (224, 150)),
        ((240, 240), (224, 224)),
    ]:
        # In float64: torch's float32 source indices differ from exact ones by about 2e-5.
        image = RNG.uniform(size=(*shape, 3))
        tensor = torch.from_numpy(np.moveaxis(image, -1, 0)[None].copy())
        expected = functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
        expected = np.moveaxis(expected[0].numpy(), 0, -1)
        np.testing.assert_allclose(images.bilinear_resize(image, *size), expected, atol=1e-12)


def test_the_whole_path_matches_opencv_and_torch_when_importable():
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")
    frame = RNG.integers(0, 256, size=(180, 320, 3), dtype=np.uint8)
    crop = frame[:, 40:220].astype(np.float64)  # centre column 130
    small = np.clip(np.rint(cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)), 0, 255)
    tensor = torch.from_numpy(np.moveaxis(small / 255.0, -1, 0)[None].copy())
    expected = functional.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    expected = np.moveaxis(expected[0].numpy(), 0, -1)
    # The view is float32, so it matches to float32 precision.
    np.testing.assert_allclose(images.arm_centred_view(frame, centre_col=130), expected, atol=1e-6)
