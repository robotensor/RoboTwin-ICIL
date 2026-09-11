import json

import numpy as np
import pytest

from icil_policies.common.arms import IdleArmHold, choose_arm
from icil_policies.common.rotations import axis_angle_to_quat
from synthetic import demonstration, endpose_row, observation

FRAMES = 11
TIMES = np.arange(FRAMES) * 0.1


def _demo(left, right, times=TIMES, **rows):
    """Left and right flange positions per frame, (T, 3) each."""
    return demonstration(
        [endpose_row(lp, rp, **rows) for lp, rp in zip(left, right, strict=True)], times=times
    )


def _line(start, end, count=FRAMES):
    return np.linspace(start, end, count)


def test_the_arm_travelling_further_is_chosen():
    still = np.zeros((11, 3))
    choice = choose_arm(_demo(_line([0, 0, 0], [0.2, 0, 0]), still))
    assert choice.arm == "left" and choice.rule == "path"
    assert choice.idle == "right"
    assert choice.path_m["left"] == pytest.approx(0.2)
    assert choose_arm(_demo(still, _line([0, 0, 0], [0, 0.3, 0]))).arm == "right"


def test_travel_is_measured_at_the_tool_centre_not_the_flange():
    # The left flange stays put but turns a quarter turn about z: its tool centre, 0.12 m out,
    # sweeps 0.19 m. The right flange slides 0.1 m.
    quats = [axis_angle_to_quat([0, 0, a]) for a in np.linspace(0, np.pi / 2, 11)]
    rows = [endpose_row((0, 0, 0), (0.01 * i, 0, 0), left_quat=q) for i, q in enumerate(quats)]
    choice = choose_arm(demonstration(rows, times=TIMES))
    assert choice.arm == "left"
    assert choice.path_m["left"] == pytest.approx(0.12 * np.pi / 2, rel=1e-2)


def test_a_tie_goes_to_the_arm_that_moves_first():
    left = np.vstack([np.zeros((3, 3)), _line([0, 0, 0], [0.2, 0, 0], 8)])
    right = np.vstack([_line([0, 0, 0], [0.2, 0, 0], 8), np.full((3, 3), [0.2, 0, 0])])
    choice = choose_arm(_demo(left, right))
    assert choice.arm == "right" and choice.rule == "first_move"
    assert choice.first_move_s["right"] < choice.first_move_s["left"]


def test_an_arm_that_never_moves_loses_a_tie_to_one_that_does():
    left = np.vstack([np.zeros((10, 3)), [[0.006, 0, 0]]])
    right = np.zeros((11, 3))
    choice = choose_arm(_demo(left, right))
    assert choice.arm == "left" and choice.rule == "first_move"
    assert choice.first_move_s["right"] is None


def test_a_full_tie_goes_to_the_left_arm():
    still = np.zeros((11, 3))
    assert (choice := choose_arm(_demo(still, still))).arm == "left" and choice.rule == "default"
    both = _line([0, 0, 0], [0.2, 0, 0])
    assert choose_arm(_demo(both, both + [1.0, 0, 0])).rule == "default"


def test_the_choice_reports_the_demonstrations_arm_set_as_json():
    demo = _demo(_line([0, 0, 0], [0.2, 0, 0]), _line([0, 0, 0], [0, 0.05, 0]))
    choice = choose_arm(demo)
    assert choice.moved == demo.arms_moved() == ("left", "right")
    info = json.loads(json.dumps(choice.info()))
    assert info["active_arm"] == "left" and info["demonstration_arms"] == ["left", "right"]


def _first_observation():
    row = endpose_row((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), left_gripper=1.0, right_gripper=0.8)
    qpos = np.arange(14, dtype=np.float64)
    return observation(row, qpos=qpos)


def test_the_idle_arm_holds_its_first_observation_in_qpos_actions():
    hold = IdleArmHold.from_observation(_first_observation(), "right")
    np.testing.assert_array_equal(hold.qpos, np.arange(7, 14))
    actions = np.full((3, 14), -1.0)
    held = hold.apply_qpos(actions)
    np.testing.assert_array_equal(held[:, :7], -1.0)
    np.testing.assert_array_equal(held[:, 7:], np.tile(np.arange(7, 14), (3, 1)))
    np.testing.assert_array_equal(actions, -1.0)  # a copy


def test_the_idle_arm_holds_its_first_observation_in_ee_actions():
    hold = IdleArmHold.from_observation(_first_observation(), "left")
    np.testing.assert_array_equal(hold.ee, [0.1, 0.2, 0.3, 1, 0, 0, 0, 1.0])
    held = hold.apply_ee(np.zeros(16))
    np.testing.assert_array_equal(held[:8], hold.ee)
    np.testing.assert_array_equal(held[8:], 0.0)
    with pytest.raises(ValueError, match=r"\(\.\.\., 16\)"):
        hold.apply_ee(np.zeros(14))


def test_an_observation_without_an_endpose_holds_qpos_only():
    first = _first_observation()
    bare = type(first)(step=0, images=first.images, qpos=first.qpos)
    hold = IdleArmHold.from_observation(bare, "left")
    assert hold.ee is None
    hold.apply_qpos(np.zeros(14))
    with pytest.raises(ValueError, match="no endpose"):
        hold.apply_ee(np.zeros(16))
    with pytest.raises(ValueError, match="arm must be one of"):
        IdleArmHold.from_observation(first, "both")
