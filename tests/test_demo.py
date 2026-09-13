from types import SimpleNamespace

import numpy as np
import pytest

from robotwin_icil.demo import (
    MOVED_THRESHOLD,
    Demonstration,
    DemonstrationError,
    Frame,
    arms_moved,
)

# aloha-agilex's joint vector; a dual Franka's is 16. The container takes its width from its frames.
QPOS_DIM = 14


def frame(index: int, cameras=("head_camera",), qpos_dim=QPOS_DIM) -> Frame:
    return Frame(
        index=index,
        images={name: np.full((4, 4, 3), index, dtype=np.uint8) for name in cameras},
        qpos=np.full(qpos_dim, float(index)),
        endpose={"left_gripper": 1.0, "right_gripper": 1.0},
    )


def demo(n=5, qpos_dim=QPOS_DIM, **kwargs) -> Demonstration:
    return Demonstration(
        frames=tuple(frame(i, qpos_dim=qpos_dim) for i in range(n)), frequency=15, **kwargs
    )


@pytest.mark.parametrize("qpos_dim", [14, 16])
def test_shapes_and_derived_views(qpos_dim):
    d = demo(n=5, qpos_dim=qpos_dim)
    assert len(d) == 5
    assert d.cameras == ("head_camera",)
    assert d.qpos_dim == qpos_dim
    assert d.qpos().shape == (5, qpos_dim)
    assert d.images("head_camera").shape == (5, 4, 4, 3)
    assert d.duration_s == pytest.approx(5 / 15)


def test_actions_are_the_next_state():
    # The expert is position-controlled, so replaying `actions()` from the same initial state
    # reproduces the trajectory. Off-by-one here would silently shift every replay rollout.
    d = demo(n=4)
    actions = d.actions()
    assert actions.shape == (3, QPOS_DIM)
    np.testing.assert_array_equal(actions, d.qpos()[1:])


def test_a_single_frame_is_not_a_demonstration():
    with pytest.raises(DemonstrationError):
        Demonstration(frames=(frame(0),), frequency=15)


def test_frequency_must_be_positive():
    with pytest.raises(DemonstrationError):
        Demonstration(frames=(frame(0), frame(1)), frequency=0)


def test_frame_rejects_a_qpos_that_is_not_a_flat_vector():
    with pytest.raises(DemonstrationError):
        Frame(index=0, images={}, qpos=np.zeros((2, 7)), endpose={})
    with pytest.raises(DemonstrationError):
        Frame(index=0, images={}, qpos=np.zeros(0), endpose={})


def test_qpos_width_must_not_change_mid_demonstration():
    # A width is a robot; frames of two widths cannot be one robot's trajectory.
    with pytest.raises(DemonstrationError, match="16-wide qpos, expected 14"):
        Demonstration(frames=(frame(0), frame(1, qpos_dim=16)), frequency=15)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_frame_rejects_a_non_finite_qpos(value):
    # A NaN never exceeds a threshold, so arms_moved would report the arm as still; refuse it.
    qpos = np.zeros(QPOS_DIM)
    qpos[3] = value
    with pytest.raises(DemonstrationError, match="non-finite"):
        Frame(index=0, images={}, qpos=qpos, endpose={})


def test_frame_rejects_a_non_rgb_image():
    with pytest.raises(DemonstrationError):
        Frame(
            index=0,
            images={"head_camera": np.zeros((4, 4))},
            qpos=np.zeros(QPOS_DIM),
            endpose={},
        )


def test_cameras_must_not_change_mid_demonstration():
    frames = (frame(0), Frame(index=1, images={}, qpos=np.zeros(QPOS_DIM), endpose={}))
    with pytest.raises(DemonstrationError):
        Demonstration(frames=frames, frequency=15)


def test_frame_indices_must_increase():
    with pytest.raises(DemonstrationError):
        Demonstration(frames=(frame(1), frame(0)), frequency=15)


def test_unknown_camera_is_rejected():
    with pytest.raises(DemonstrationError):
        demo().images("wrist_camera")


def trajectory(*rows) -> Demonstration:
    frames = tuple(
        Frame(index=i, images={}, qpos=np.asarray(row, dtype=float), endpose={})
        for i, row in enumerate(rows)
    )
    return Demonstration(frames=frames, frequency=15)


def moved(*joints: tuple[int, float]) -> np.ndarray:
    row = np.zeros(QPOS_DIM)
    for joint, value in joints:
        row[joint] = value
    return row


REST = np.zeros(QPOS_DIM)


def test_arms_moved_names_the_half_of_the_row_that_left_its_first_frame():
    # Left arm is joints 0-5 plus gripper 6; right arm is 7-12 plus gripper 13.
    assert arms_moved(trajectory(REST, moved((9, 0.3)), REST)) == ("right",)
    assert arms_moved(trajectory(REST, moved((2, -0.3)))) == ("left",)
    assert arms_moved(trajectory(REST, moved((2, 0.3), (9, 0.3)))) == ("left", "right")


def test_arms_moved_counts_a_gripper_that_opens():
    assert arms_moved(trajectory(REST, moved((6, 1.0)))) == ("left",)
    assert arms_moved(trajectory(REST, moved((13, 1.0)))) == ("right",)


def test_arms_moved_ignores_motion_within_the_threshold():
    # The same number gates a joint (radians) and a gripper (fraction of full opening); reaching
    # it exactly is not moving, exceeding it is.
    at = MOVED_THRESHOLD
    assert arms_moved(trajectory(REST, moved((3, at), (10, -at)))) == ()
    assert arms_moved(trajectory(REST, moved((6, at), (13, -at)))) == ()
    assert arms_moved(trajectory(REST, moved((3, at + 0.01)))) == ("left",)
    assert arms_moved(trajectory(REST, moved((13, at + 0.01)))) == ("right",)
    assert arms_moved(trajectory(REST, moved((3, 0.3))), threshold=0.5) == ()


def test_arms_moved_measures_from_the_first_frame_not_from_rest():
    # An arm that starts away from zero and stays there has not moved.
    start = moved((0, 1.2), (7, -0.9))
    assert arms_moved(trajectory(start, start, start)) == ()
    assert arms_moved(trajectory(start, start + moved((7, 0.3)))) == ("right",)


def test_arms_moved_refuses_a_row_that_does_not_split_into_two_arms():
    odd = SimpleNamespace(qpos=lambda: np.zeros((3, 15)))
    with pytest.raises(DemonstrationError, match="two equal arms"):
        arms_moved(odd)
