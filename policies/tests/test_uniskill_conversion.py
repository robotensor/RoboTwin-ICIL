import numpy as np
import pytest

from icil_policies.common.chunking import ChunkExecutor
from icil_policies.common.oracles import resampled_qpos_actions
from icil_policies.common.resample import resample
from icil_policies.testing import assert_conversion_pinned, conversion_digest
from icil_policies.uniskill import conversion
from icil_policies.uniskill.conversion import (
    Augmentation,
    SkillCursor,
    augment_views,
    draw_augmentation,
    future_rows,
    isd_view,
    policy_image,
    skill_row,
    skill_steps,
)
from robotwin_icil.demo import DemonstrationError
from synthetic import camera_demonstration

# One entry per ADAPTER_VERSION ever released; added, never edited.
PINS = {"1": "c08dd7aac688ceffb05b40e85dc1bc7c45d229080935bacdad2614e3bc1b0450"}
FAR_SIDE = (180, 320)
STEP = 1 / 250


def _far_side(times, **kwargs):
    return camera_demonstration(times, camera="far_side_camera", shape=FAR_SIDE, **kwargs)


def test_each_row_looks_k_steps_ahead_clamped_at_the_last_step():
    np.testing.assert_array_equal(future_rows(5, 2), [2, 3, 4, 4, 4])
    np.testing.assert_array_equal(future_rows(1, 20), [0])
    rows = future_rows(50, 20)
    assert len(rows) == 50  # one row per step, not extract_skill.py's 50 - 20
    np.testing.assert_array_equal(rows[:30], np.arange(20, 50))
    assert set(rows[30:]) == {49}


@pytest.mark.parametrize(("n", "k"), [(0, 20), (5, 0), (5, -1), (5, 1.5), (5, True)])
def test_rows_need_a_step_and_a_positive_integer_k(n, k):
    with pytest.raises(ValueError):
        future_rows(n, k)


def test_the_skill_row_is_the_executed_count_held_past_the_end():
    assert [skill_row(executed, 3) for executed in range(6)] == [0, 1, 2, 2, 2, 2]
    for bad in ((-1, 3), (0, 0)):
        with pytest.raises(ValueError):
            skill_row(*bad)


def test_replans_every_ta_actions_take_the_row_of_the_actions_executed():
    """Through the cursor `UniSkillPolicy._act` and `_plan` use, so CI covers their order."""
    planned = []

    def plan(history):
        planned.append(cursor.row())
        return np.zeros((16, 14))

    cursor = SkillCursor(ChunkExecutor(plan, n_obs=2, n_action=8))
    cursor.set_rows(20)
    for step in range(40):
        cursor.act(step)
    # An action leaves before the count advances, so each re-plan reads the count before it.
    assert planned == [0, 8, 16, 19, 19]
    assert [step for step, _ in cursor.replans] == [0, 8, 16, 24, 32]
    assert [row for _, row in cursor.replans] == planned
    assert cursor.executed == 40


def test_the_cursor_forgets_the_episode_and_needs_rows_to_index():
    cursor = SkillCursor(ChunkExecutor(lambda history: np.zeros((16, 14)), n_obs=2, n_action=8))
    cursor.set_rows(3)
    cursor.act(0)
    cursor.reset()
    assert (cursor.executed, cursor.n_rows, cursor.replans) == (0, 0, [])
    with pytest.raises(ValueError):
        cursor.set_rows(0)


def test_skill_rows_align_with_resampled_steps_and_actions():
    # RoboTwin's pattern: frames after steps 0, 1, 16, 31, the primitive's last and a tie.
    steps = np.array([0, 1, 16, 31, 40, 40, 41, 56, 71, 86, 101, 116, 131, 146, 150, 400])
    demo = _far_side(steps * STEP)
    sampled = skill_steps(demo)
    resampled = resample(demo, 20.0)
    assert len(sampled) == len(resampled) == len(resampled_qpos_actions(demo)) == 33
    np.testing.assert_array_equal(sampled.resampled.source, resampled.source)
    np.testing.assert_array_equal(sampled.future, future_rows(33, 20))
    # Each step's view is the centred 180x180 square of its nearest frame.
    for step, frame in enumerate(resampled.source):
        np.testing.assert_array_equal(
            sampled.views[step], demo.frames[frame].images["far_side_camera"][:, 70:250]
        )


def test_skill_steps_need_the_far_side_camera():
    demo = camera_demonstration(np.arange(3) * 0.05, camera="head_camera", shape=(240, 320))
    with pytest.raises(DemonstrationError, match="far_side"):
        skill_steps(demo)


def test_the_isd_view_is_a_centred_square():
    image = np.arange(180 * 320 * 3, dtype=np.int64).reshape(180, 320, 3).astype(np.uint8)
    view = isd_view(image)
    assert view.shape == (180, 180, 3) and np.shares_memory(view, image)
    np.testing.assert_array_equal(view, image[:, 70:250])
    with pytest.raises(ValueError):
        isd_view(image.astype(np.float32))
    with pytest.raises(ValueError):
        isd_view(image[:100])


def _views(count=3, side=40):
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (count, side, side, 3), dtype=np.uint8)


def test_augmentation_is_deterministic_per_seed_key_and_variant():
    augmentation = Augmentation()
    views = _views()
    first = augment_views(views, draw_augmentation(augmentation, 2, key=11))
    again = augment_views(views, draw_augmentation(Augmentation(), 2, key=11))
    np.testing.assert_array_equal(first, again)
    others = [
        draw_augmentation(augmentation, 3, key=11),
        draw_augmentation(augmentation, 2, key=12),
        draw_augmentation(Augmentation(seed=1), 2, key=11),
    ]
    for draw in others:
        out = augment_views(views, draw)
        assert out.shape != first.shape or not np.array_equal(out, first)


def test_a_variant_is_one_crop_and_one_jitter_for_every_frame():
    augmentation = Augmentation(crop_scale=(0.5, 0.5))
    views = np.repeat(_views(1), 4, axis=0)
    out = augment_views(views, draw_augmentation(augmentation, 0))
    assert out.shape == (4, 20, 20, 3) and out.dtype == np.uint8
    for frame in out[1:]:
        np.testing.assert_array_equal(frame, out[0])


def test_augmentation_with_no_strength_and_no_crop_is_the_identity():
    augmentation = Augmentation(brightness=0, contrast=0, saturation=0, crop_scale=(1, 1))
    views = _views()
    np.testing.assert_array_equal(augment_views(views, draw_augmentation(augmentation, 4)), views)


def test_augmentation_parameters_are_checked():
    for bad in (
        {"variants": -1},
        {"brightness": 1.0},
        {"contrast": -0.1},
        {"crop_scale": (0.0, 1.0)},
        {"crop_scale": (0.9, 0.8)},
    ):
        with pytest.raises(ValueError):
            Augmentation(**bad)
    with pytest.raises(ValueError):
        draw_augmentation(Augmentation(variants=5), 5)


def test_the_policy_image_averages_when_shrinking_and_interpolates_when_growing():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:2, :2] = 200
    small = policy_image(image, 2, 2)
    np.testing.assert_array_equal(small[..., 0], [[200, 0], [0, 0]])
    grown = policy_image(image, 8, 2)
    assert grown.shape == (8, 2, 3) and grown.dtype == np.uint8
    np.testing.assert_array_equal(grown[0, :, 0], [200, 0])
    with pytest.raises(ValueError):
        policy_image(image.astype(np.float64), 2, 2)


def test_the_conversion_is_pinned_to_its_version():
    steps = np.array([0, 1, 16, 31, 40, 40, 41, 56, 71, 300])
    demo = _far_side(steps * STEP)
    sampled = skill_steps(demo, k=3)
    draw = draw_augmentation(Augmentation(), 1, key=5)
    digest = conversion_digest(
        sampled.future,
        sampled.resampled.source,
        sampled.views[:, ::17, ::17],
        augment_views(sampled.views[:2], draw)[:, ::13, ::13],
        policy_image(demo.frames[3].images["far_side_camera"], 32, 48),
    )
    assert_conversion_pinned(conversion, digest, PINS)
