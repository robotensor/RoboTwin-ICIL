import numpy as np
import pytest

from robotwin_icil.demo import BIMANUAL_QPOS_DIM, Demonstration, DemonstrationError, Frame


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
    assert d.duration_s == pytest.approx(5 / 15)


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
