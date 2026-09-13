import numpy as np
import pytest

from robotwin_icil.demo import Demonstration, DemonstrationError, Frame

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
