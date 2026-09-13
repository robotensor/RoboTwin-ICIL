from dataclasses import replace

import numpy as np
import pytest

from robotwin_icil.demo import (
    BIMANUAL_EE_DIM,
    BIMANUAL_QPOS_DIM,
    Demonstration,
    DemonstrationError,
    Frame,
)


def frame(index: int, cameras=("head_camera",), qpos_dim=BIMANUAL_QPOS_DIM) -> Frame:
    return Frame(
        index=index,
        images={name: np.full((4, 4, 3), index, dtype=np.uint8) for name in cameras},
        qpos=np.full(qpos_dim, float(index)),
        endpose={"left_gripper": 1.0, "right_gripper": 1.0},
    )


def demo(n=5, **kwargs) -> Demonstration:
    return Demonstration(frames=tuple(frame(i) for i in range(n)), frequency=15, **kwargs)


def test_shapes_and_derived_views():
    d = demo(n=5)
    assert len(d) == 5
    assert d.cameras == ("head_camera",)
    assert d.qpos().shape == (5, BIMANUAL_QPOS_DIM)
    assert d.images("head_camera").shape == (5, 4, 4, 3)
    assert d.duration_s == pytest.approx(4 / 15)


def test_actions_are_the_next_state():
    # The expert is position-controlled, so replaying `actions()` from the same initial state
    # reproduces the trajectory. Off-by-one here would silently shift every replay rollout.
    d = demo(n=4)
    actions = d.actions()
    assert actions.shape == (3, BIMANUAL_QPOS_DIM)
    np.testing.assert_array_equal(actions, d.qpos()[1:])


def test_a_single_frame_is_not_a_demonstration():
    with pytest.raises(DemonstrationError):
        Demonstration(frames=(frame(0),), frequency=15)


def test_frequency_must_be_positive():
    with pytest.raises(DemonstrationError):
        Demonstration(frames=(frame(0), frame(1)), frequency=0)


def test_frame_rejects_the_wrong_qpos_width():
    with pytest.raises(DemonstrationError):
        frame(0, qpos_dim=7)


def test_frame_rejects_a_non_rgb_image():
    with pytest.raises(DemonstrationError):
        Frame(
            index=0,
            images={"head_camera": np.zeros((4, 4))},
            qpos=np.zeros(BIMANUAL_QPOS_DIM),
            endpose={},
        )


def test_cameras_must_not_change_mid_demonstration():
    frames = (frame(0), Frame(index=1, images={}, qpos=np.zeros(BIMANUAL_QPOS_DIM), endpose={}))
    with pytest.raises(DemonstrationError):
        Demonstration(frames=frames, frequency=15)


def test_frame_indices_must_increase():
    with pytest.raises(DemonstrationError):
        Demonstration(frames=(frame(1), frame(0)), frequency=15)


def test_unknown_camera_is_rejected():
    with pytest.raises(DemonstrationError):
        demo().images("wrist_camera")


def timed(times, cameras=("head_camera",)) -> tuple[Frame, ...]:
    return tuple(
        Frame(
            index=i,
            images={name: np.full((4, 4, 3), i, dtype=np.uint8) for name in cameras},
            qpos=np.full(BIMANUAL_QPOS_DIM, float(i)),
            endpose={},
            time_s=t,
        )
        for i, t in enumerate(times)
    )


def test_recorded_times_are_returned_as_they_are():
    # Ties are a frame taken with no physics step since the last: a primitive boundary.
    d = Demonstration(frames=timed([0.0, 0.004, 0.064, 0.064, 0.1]), frequency=50 / 3)
    np.testing.assert_array_equal(d.times(), [0.0, 0.004, 0.064, 0.064, 0.1])


def test_duration_is_the_span_of_the_times():
    # Five timed frames over 0.1 s would read as 5 / (50 / 3) = 0.3 s by frame count.
    d = Demonstration(frames=timed([0.0, 0.004, 0.064, 0.064, 0.1]), frequency=50 / 3)
    assert d.duration_s == pytest.approx(0.1)
    # Untimed, the duplicate at a boundary adds no time.
    frames = (frame(0), frame(1), frame(2), replace(frame(2), index=3))
    assert Demonstration(frames=frames, frequency=15).duration_s == pytest.approx(2 / 15)


def test_times_must_not_run_backwards():
    with pytest.raises(DemonstrationError, match="frame 2 is timed at 0.01 s, before frame 1"):
        Demonstration(frames=timed([0.0, 0.02, 0.01]), frequency=15)


def test_times_are_all_or_nothing():
    with pytest.raises(DemonstrationError, match="some frames have a time_s"):
        Demonstration(frames=timed([0.0, None, 0.1]), frequency=15)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_frame_time_must_be_finite(bad):
    # NaN compares false both ways, so it would slip past the ordering check.
    with pytest.raises(DemonstrationError, match="time_s is"):
        timed([bad])


def test_untimed_frames_are_spaced_at_the_nominal_rate():
    np.testing.assert_allclose(demo(n=4).times(), [0.0, 1 / 15, 2 / 15, 3 / 15])


def test_untimed_duplicates_share_a_time():
    # Demonstrations recorded before the clock: the exact duplicate at a primitive boundary is
    # collapsed, the rest keep 1 / frequency.
    frames = (frame(0), frame(1), replace(frame(1), index=2), frame(3))
    d = Demonstration(frames=frames, frequency=10)
    np.testing.assert_allclose(d.times(), [0.0, 0.1, 0.1, 0.2])


@pytest.mark.parametrize(
    "change",
    [
        {"qpos": np.full(BIMANUAL_QPOS_DIM, 9.0)},
        {"endpose": {"left_gripper": 0.5, "right_gripper": 1.0}},
        {"images": {"head_camera": np.full((4, 4, 3), 9, dtype=np.uint8)}},
        {"gripper_joints": {"left": np.array([0.01, 0.01]), "right": np.array([0.0, 0.0])}},
    ],
)
def test_a_frame_that_differs_in_anything_is_not_a_duplicate(change):
    joints = {"left": np.array([0.02, 0.02]), "right": np.array([0.0, 0.0])}
    base = [replace(frame(i), gripper_joints=joints) for i in range(2)]
    frames = (*base, replace(base[1], index=2, **change))
    d = Demonstration(frames=frames, frequency=10)
    np.testing.assert_allclose(d.times(), [0.0, 0.1, 0.2])


def posed(left_xyz, right_xyz, left_gripper=1.0, right_gripper=1.0) -> dict:
    """An endpose as RoboTwin's `get_obs()` reports it: lists of xyz + wxyz, and floats."""
    return {
        "left_endpose": [*left_xyz, 1.0, 0.0, 0.0, 0.0],
        "left_gripper": left_gripper,
        "right_endpose": [*right_xyz, 0.0, 1.0, 0.0, 0.0],
        "right_gripper": right_gripper,
    }


def ee_demo(endposes) -> Demonstration:
    return Demonstration(
        frames=tuple(replace(frame(i), endpose=pose) for i, pose in enumerate(endposes)),
        frequency=15,
    )


def test_endposes_are_in_take_action_ee_layout():
    d = ee_demo([posed((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), 0.25, 0.75)] * 2)
    endposes = d.endposes()
    assert endposes.shape == (2, BIMANUAL_EE_DIM) == (2, 16)
    # [left xyz, left wxyz, left gripper, right xyz, right wxyz, right gripper]
    np.testing.assert_array_equal(
        endposes[0],
        [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.25, 0.4, 0.5, 0.6, 0.0, 1.0, 0.0, 0.0, 0.75],
    )


def test_ee_actions_are_the_next_endpose():
    d = ee_demo([posed((0.0, 0.0, z), (0.0, 0.0, -z)) for z in (0.0, 0.1, 0.2, 0.3)])
    actions = d.ee_actions()
    assert actions.shape == (3, BIMANUAL_EE_DIM)
    np.testing.assert_array_equal(actions, d.endposes()[1:])
    assert actions[0, 2] == pytest.approx(0.1) and actions[-1, 10] == pytest.approx(-0.3)


def test_frames_without_an_endpose_have_no_ee_view():
    with pytest.raises(DemonstrationError, match="endpose has no 'left_endpose'"):
        demo().endposes()


def test_an_endpose_of_the_wrong_width_is_rejected():
    pose = posed((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    pose["right_endpose"] = pose["right_endpose"][:3]
    with pytest.raises(DemonstrationError, match="right_endpose has shape"):
        ee_demo([pose, pose]).endposes()


def test_arms_moved_are_those_whose_flange_travels_past_the_threshold():
    still = (0.0, 0.0, 0.0)
    left_moves = ee_demo([posed((0.0, 0.0, z), still) for z in (0.0, 0.01, 0.03)])
    assert left_moves.arms_moved() == ("left",)
    assert left_moves.arms_moved(threshold_m=0.05) == ()
    both = ee_demo([posed(still, still), posed((0.03, 0.0, 0.0), (0.0, 0.03, 0.0))])
    assert both.arms_moved() == ("left", "right")


def test_arms_moved_counts_path_length_not_displacement():
    # Out 1.5 cm and back: no net displacement, a 3 cm path.
    still = (0.0, 0.0, 0.0)
    d = ee_demo([posed(still, still), posed(still, (0.015, 0.0, 0.0)), posed(still, still)])
    assert d.arms_moved() == ("right",)


GRIPPER_JOINTS = {"left": np.array([0.045, 0.045]), "right": np.array([0.012, 0.012])}


def test_frames_carry_measured_gripper_joints():
    d = Demonstration(
        frames=tuple(replace(frame(i), gripper_joints=GRIPPER_JOINTS) for i in range(2)),
        frequency=15,
    )
    np.testing.assert_array_equal(d.frames[1].gripper_joints["right"], [0.012, 0.012])
    assert frame(0).gripper_joints is None  # frames recorded before they were read


@pytest.mark.parametrize(
    "bad",
    [
        {"left": np.array([0.0, 0.0])},
        {"left": np.array([0.0]), "right": np.array([0.0]), "head": np.array([0.0])},
        {"left": np.zeros((2, 2)), "right": np.array([0.0])},
        {"left": np.array([np.nan]), "right": np.array([0.0])},
    ],
)
def test_malformed_gripper_joints_are_rejected(bad):
    with pytest.raises(DemonstrationError, match="gripper"):
        replace(frame(0), gripper_joints=bad)


def test_gripper_joints_are_all_or_nothing():
    frames = (replace(frame(0), gripper_joints=GRIPPER_JOINTS), frame(1))
    with pytest.raises(DemonstrationError, match="some frames have gripper_joints"):
        Demonstration(frames=frames, frequency=15)
