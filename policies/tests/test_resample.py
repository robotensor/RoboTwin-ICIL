import math

import numpy as np
import pytest

from icil_policies.common.resample import resample, sample_times
from icil_policies.common.rotations import axis_angle_to_quat, quat_to_axis_angle
from synthetic import CAMERA, demonstration, endpose_row

STEP = 1 / 250  # one physics step


def _moving(times, speed=0.1, **kwargs):
    """A demonstration whose left flange moves along x at `speed` m/s."""
    rows = [endpose_row((speed * t, 0, 0)) for t in times]
    return demonstration(rows, times=times, **kwargs)


def test_evenly_timed_linear_motion_is_resampled_exactly():
    times = np.arange(51) / 50  # 1 s at 50 Hz
    out = resample(_moving(times), rate_hz=20)
    assert len(out) == 21
    np.testing.assert_allclose(out.times, np.arange(21) / 20)
    np.testing.assert_allclose(out.endposes[:, 0], 0.1 * out.times, atol=1e-15)


def test_robotwin_spacing_with_ties_is_resampled_on_time_not_index():
    # A primitive's pattern: a frame before its first step, after steps 1, 16, 31 and its last
    # (40), then the next primitive's first frame at the same instant.
    steps = np.array([0, 1, 16, 31, 40, 40, 41, 56, 71])
    times = steps * STEP
    out = resample(_moving(times), rate_hz=20)
    np.testing.assert_allclose(out.times, np.arange(len(out)) / 20)
    inside = out.times <= times[-1]
    np.testing.assert_allclose(out.endposes[inside, 0], 0.1 * out.times[inside], atol=1e-15)
    # The grid overshoots the final frame (0.284 s) by one sample, which holds that frame.
    assert out.times[-1] == pytest.approx(0.3) and not inside[-1]
    assert out.endposes[-1, 0] == pytest.approx(0.1 * times[-1])


def test_a_sample_at_a_tied_time_takes_the_last_frame_there():
    times = [0.0, 0.05, 0.05, 0.1]
    rows = [endpose_row(left_gripper=g) for g in (1.0, 1.0, 0.0, 0.0)]
    qpos = np.zeros((4, 14))
    qpos[2:, 6] = 0.3
    out = resample(demonstration(rows, times=times, qpos=qpos), rate_hz=20)
    np.testing.assert_allclose(out.times, [0, 0.05, 0.1])
    np.testing.assert_array_equal(out.endposes[:, 7], [1, 0, 0])
    np.testing.assert_array_equal(out.qpos[:, 6], [0, 0.3, 0.3])
    np.testing.assert_array_equal(out.held, [0, 2, 3])


def test_float32_step_times_still_tie_on_the_grid():
    # The core's clock: steps times SAPIEN's float32 timestep, so step 25 is 5 ns past 0.1 s.
    steps = np.array([0, 1, 16, 24, 25, 25, 26, 40])
    times = steps * float(np.float32(STEP))
    assert times[4] > 0.1
    rows = [
        endpose_row((0.1 * t, 0, 0), left_gripper=1.0 if i < 5 else 0.0)
        for i, t in enumerate(times)
    ]
    qpos = np.zeros((len(times), 14))
    qpos[5:, 6] = 0.3
    out = resample(demonstration(rows, times=times, qpos=qpos), rate_hz=20)
    # The gripper command issued at step 25 shows at the 0.1 s sample, not 50 ms later.
    np.testing.assert_array_equal(out.held, [0, 1, 5, 6, 7])
    np.testing.assert_array_equal(out.endposes[:, 7], [1, 1, 0, 0, 0])
    np.testing.assert_array_equal(out.qpos[:, 6], [0, 0, 0.3, 0.3, 0.3])
    np.testing.assert_array_equal(out.source, [0, 2, 5, 7, 7])  # nearest; 5 is at 0.1 s
    inside = out.times <= times[-1]
    np.testing.assert_allclose(out.endposes[inside, 0], 0.1 * out.times[inside], atol=1e-9)
    # A demonstration ending on a grid point gets no extra sample past it.
    ends = np.array([0, 250]) * float(np.float32(STEP))
    np.testing.assert_allclose(sample_times(ends, 20), np.arange(21) / 20)


def test_grippers_are_held_while_positions_interpolate():
    times = [0.0, 0.1]
    rows = [endpose_row((0, 0, 0), left_gripper=1.0), endpose_row((0.1, 0, 0), left_gripper=0.0)]
    qpos = np.zeros((2, 14))
    qpos[1, :6], qpos[1, 6], qpos[1, 13] = 1.0, 0.0, 0.5
    qpos[0, 6] = 1.0
    joints = {"left": [[0.04, 0.04], [0.0, 0.0]], "right": [[0.01], [0.02]]}
    out = resample(demonstration(rows, times=times, qpos=qpos, gripper_joints=joints), rate_hz=20)
    np.testing.assert_allclose(out.endposes[:, 0], [0, 0.05, 0.1])
    np.testing.assert_array_equal(out.endposes[:, 7], [1, 1, 0])
    np.testing.assert_allclose(out.qpos[:, 0], [0, 0.5, 1])
    np.testing.assert_array_equal(out.qpos[:, 6], [1, 1, 0])
    np.testing.assert_array_equal(out.qpos[:, 13], [0, 0, 0.5])
    np.testing.assert_array_equal(out.gripper_joints["left"], [[0.04, 0.04], [0.04, 0.04], [0, 0]])
    np.testing.assert_array_equal(out.gripper_joints["right"], [[0.01], [0.01], [0.02]])


def test_orientations_are_slerped():
    times = [0.0, 0.1]
    rows = [
        endpose_row(right_quat=axis_angle_to_quat([0, 0, 0.0])),
        endpose_row(right_quat=axis_angle_to_quat([0, 0, 1.0])),
    ]
    out = resample(demonstration(rows, times=times), rate_hz=20)
    np.testing.assert_allclose(
        quat_to_axis_angle(out.endposes[:, 11:15])[:, 2], [0, 0.5, 1.0], atol=1e-12
    )
    np.testing.assert_allclose(out.endposes[:, 3:7], np.tile([1.0, 0, 0, 0], (3, 1)))


def test_images_come_from_the_nearest_frame_the_earlier_on_a_tie():
    times = [0.0, 0.03, 0.1, 0.16]
    out = resample(_moving(times), rate_hz=20)
    # samples 0, 0.05, 0.1, 0.15 and 0.2, past the last frame: nearest frames 0, 1 (0.02 away
    # against 0.05), 2, 3 and 3
    np.testing.assert_array_equal(out.source, [0, 1, 2, 3, 3])
    images = out.images(CAMERA)
    assert images.shape == (5, 2, 3, 3)
    np.testing.assert_array_equal(images[:, 0, 0, 0], [0, 1, 2, 3, 3])
    tied = resample(_moving([0.0, 0.1]), rate_hz=20)
    np.testing.assert_array_equal(tied.source, [0, 0, 1])
    with pytest.raises(ValueError, match="no camera"):
        out.images("head_camera")


def test_the_final_frame_is_always_sampled_unless_asked_not_to():
    times = [0.0, 0.12]
    with_end = resample(_moving(times), rate_hz=20)
    np.testing.assert_allclose(with_end.times, [0, 0.05, 0.1, 0.15])
    np.testing.assert_allclose(with_end.endposes[-1, 0], 0.1 * 0.12)
    without = resample(_moving(times), rate_hz=20, include_end=False)
    np.testing.assert_allclose(without.times, [0, 0.05, 0.1])
    on_grid = resample(_moving([0.0, 0.1]), rate_hz=20)
    np.testing.assert_allclose(on_grid.times, [0, 0.05, 0.1])


def test_the_grid_starts_at_the_first_frame_and_tolerates_float_error():
    np.testing.assert_allclose(sample_times([2.0, 2.3], 10), [2.0, 2.1, 2.2, 2.3])
    assert len(sample_times([0.0, 94 * 0.05], 20)) == 95
    for rate in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="rate_hz"):
            sample_times([0.0, 1.0], rate)


def test_untimed_demonstrations_use_the_estimated_times():
    rows = [endpose_row((0.01 * i, 0, 0)) for i in range(5)]
    demo = demonstration(rows, frequency=10)
    out = resample(demo, rate_hz=20)
    np.testing.assert_allclose(out.times, np.arange(9) / 20)
    np.testing.assert_allclose(out.endposes[:, 0], 0.1 * out.times, atol=1e-15)
    assert out.gripper_joints is None


def test_a_faster_grid_is_a_time_stretch():
    out = resample(_moving(np.arange(11) / 10), rate_hz=40)
    assert len(out) == 41
