import json

import numpy as np
import pytest

from icil_policies.common.rotations import axis_angle_to_matrix
from icil_policies.common.virtual_target import StallDetector, VirtualTarget

EYE = np.eye(3)
MM = 0.001


def test_the_first_step_anchors_on_the_measured_pose():
    target = VirtualTarget()
    position, rotation = target.step([0.1, 0.2, 0.3], EYE, [MM, 0, 0])
    np.testing.assert_allclose(position, [0.101, 0.2, 0.3])
    np.testing.assert_array_equal(rotation, EYE)
    assert target.reanchors == {"tracking": 0, "plan_failed": 0}


def test_residuals_below_the_bound_add_up():
    # The arm does not move at all (every plan counts it as there); the target keeps going.
    target = VirtualTarget(max_position_error_m=0.01)
    for _ in range(9):
        position, _ = target.step([0, 0, 0], EYE, [MM, 0, 0])
    assert position[0] == pytest.approx(9 * MM)
    assert target.reanchors["tracking"] == 0


def test_a_tracking_error_past_the_bound_re_anchors_on_the_measured_pose():
    target = VirtualTarget(max_position_error_m=0.01)
    target.step([0, 0, 0], EYE, [0.008, 0, 0])
    position, _ = target.step([0, 0, 0], EYE, [0.004, 0, 0])  # error 8 mm: kept
    assert position[0] == pytest.approx(0.012)
    position, _ = target.step([0.0005, 0, 0], EYE, [MM, 0, 0])  # error 11.5 mm: re-anchored
    assert position[0] == pytest.approx(0.0015)
    assert target.reanchors == {"tracking": 1, "plan_failed": 0}


def test_a_rotation_error_past_its_bound_re_anchors_too():
    target = VirtualTarget(max_rotation_error_rad=0.1)
    target.step([0, 0, 0], EYE, [0, 0, 0], axis_angle_to_matrix([0, 0, 0.15]))
    assert target.tracking_error([0, 0, 0], EYE)[1] == pytest.approx(0.15)
    _, rotation = target.step([0, 0, 0], EYE, [0, 0, 0])
    np.testing.assert_allclose(rotation, EYE)
    assert target.reanchors["tracking"] == 1


def test_a_failed_plan_re_anchors_whatever_the_error():
    target = VirtualTarget()
    target.step([0, 0, 0], EYE, [MM, 0, 0])
    position, _ = target.step([0.0002, 0, 0], EYE, [MM, 0, 0], plan_failed=True)
    assert position[0] == pytest.approx(0.0012)
    assert target.reanchors == {"tracking": 0, "plan_failed": 1}


def test_rotation_deltas_multiply_on_the_left_in_the_world_frame():
    start = axis_angle_to_matrix([0.3, 0, 0])
    turn = axis_angle_to_matrix([0, 0, 0.05])
    target = VirtualTarget()
    _, rotation = target.step([0, 0, 0], start, [0, 0, 0], turn)
    np.testing.assert_allclose(rotation, turn @ start)
    assert not np.allclose(rotation, start @ turn)


def test_reset_forgets_the_target_and_counts():
    target = VirtualTarget()
    target.step([0, 0, 0], EYE, [0.02, 0, 0])
    target.step([0, 0, 0], EYE, [0, 0, 0])
    assert json.loads(json.dumps(target.info())) == {"reanchors": {"tracking": 1, "plan_failed": 0}}
    target.reset()
    assert target.position is None and target.reanchors["tracking"] == 0
    with pytest.raises(ValueError, match="no target yet"):
        target.tracking_error([0, 0, 0], EYE)
    with pytest.raises(ValueError, match="positive"):
        VirtualTarget(max_position_error_m=0)


def test_a_tool_told_to_move_that_does_not_is_one_stall_event():
    detector = StallDetector(window=3, min_motion_m=0.002)
    flags = [detector.update([0, 0, 0], commanded_m=MM) for _ in range(6)]
    assert flags == [False, False, False, True, True, True]
    assert detector.events == 1
    # It moves again, then stalls again: a second event.
    for x in (0.003, 0.006, 0.009):
        assert not detector.update([x, 0, 0], commanded_m=0.003)
    for _ in range(3):
        detector.update([0.009, 0, 0], commanded_m=MM)
    assert detector.stalled and detector.events == 2
    assert json.loads(json.dumps(detector.info())) == {"stall_events": 2}


def test_a_tool_told_to_stay_still_is_not_stalled():
    detector = StallDetector(window=3, min_motion_m=0.002)
    assert not any(detector.update([0, 0, 0], commanded_m=0.0) for _ in range(10))
    detector.reset()
    assert detector.events == 0 and not detector.stalled


@pytest.mark.parametrize(("window", "motion"), [(0, 0.002), (2.5, 0.002), (3, 0.0)])
def test_stall_settings_are_checked(window, motion):
    with pytest.raises(ValueError):
        StallDetector(window=window, min_motion_m=motion)
