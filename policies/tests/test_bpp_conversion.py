"""The BPP adapter's conversion: frames, proprioception, images, prompts and execution.

Pure numpy, no torch and no simulator. The digest at the end pins the whole chain to
`ADAPTER_VERSION`, so conversion math that changes without a bump fails here.
"""

import numpy as np
import pytest

from icil_policies.bpp import ADAPTER_VERSION, conversion
from icil_policies.bpp import settings as bpp_settings
from icil_policies.bpp.synthetic import AGENTVIEW, WRISTS, demonstration, flange_of, observation
from icil_policies.common.frames import TCP_OFFSET_M, arm_base, flange_from_tcp, tcp_from_flange
from icil_policies.common.rotations import axis_angle_to_matrix, matrix_to_rot6d, rot6d_to_matrix
from icil_policies.testing import assert_conversion_pinned, conversion_digest
from robotwin_icil.policy import PolicyError

SETTINGS = bpp_settings.load()
# One digest per released ADAPTER_VERSION; entries are added, never edited.
PINS = {"1": "c0eb76c9b60bd25294a06f896e62b6bc701f4533a9d3b8b4945b72990a0e472f"}


# ------------------------------------------------------------------------------ frames


def test_the_tool_centre_is_where_the_flange_points():
    tool = np.array([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])
    flange = flange_from_tcp(tool)
    assert flange[0] == pytest.approx(tool[0] - TCP_OFFSET_M)
    np.testing.assert_allclose(tcp_from_flange(flange), tool, atol=1e-12)


def test_proprioception_sits_in_liberos_frame_at_the_tool_centre():
    arm = "left"
    tool = arm_base(arm) + np.array([0.0, 0.25, -0.04])
    state = conversion.proprioception(flange_of(tool), 0.045, SETTINGS, arm)
    # p_L = M.T (p_world - b_arm) + b_Panda: the world's +y (the robot's forward) is LIBERO's +x.
    np.testing.assert_allclose(
        state["ee_pos"], np.array([0.25, 0.0, -0.04]) + SETTINGS.libero_base_m, atol=1e-12
    )
    assert state["ee_ori"].shape == (6,)
    # The correction takes robosuite's +z approach onto aloha's flange +x, so an identity flange
    # comes out as a rotation whose third row is the world axis the tool points along.
    rotation = rot6d_to_matrix(state["ee_ori"])
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_the_two_arms_map_to_the_same_libero_position():
    offset = np.array([0.0, 0.25, -0.04])
    left = conversion.proprioception(flange_of(arm_base("left") + offset), 0.0, SETTINGS, "left")
    right = conversion.proprioception(flange_of(arm_base("right") + offset), 0.0, SETTINGS, "right")
    np.testing.assert_allclose(left["ee_pos"], right["ee_pos"], atol=1e-12)


def test_the_gripper_maps_alohas_finger_range_onto_liberos():
    np.testing.assert_allclose(conversion.gripper_states(0.045, SETTINGS), [0.04, -0.04])
    np.testing.assert_allclose(conversion.gripper_states(-0.01, SETTINGS), [0.0, -0.0])
    half = conversion.gripper_states(0.0175, SETTINGS)
    assert half[0] == pytest.approx(0.02) and half[1] == pytest.approx(-0.02)


def test_a_demonstration_without_measured_fingers_is_refused():
    with pytest.raises(PolicyError, match="measured gripper joints"):
        conversion.finger_of(None, "left", "frame 0")
    with pytest.raises(PolicyError, match="measured gripper joints"):
        conversion.finger_of({}, "left", "frame 0")


# ------------------------------------------------------------------------------ cameras


def test_the_crop_follows_the_arm_and_clamps_inside_the_frame():
    left, right = (
        conversion.crop_column(SETTINGS, "left"),
        conversion.crop_column(SETTINGS, "right"),
    )
    # far_side_camera looks back at the robot with `left = (1, 0, 0)`, so image left is the
    # robot's right: the left arm sits in the right half of the frame.
    assert right < 160 < left
    from icil_policies.common.images import crop_start

    # The columns follow the profile's own projection: far_side stands where a third-person
    # training view does, so a change of pose moves them.
    assert crop_start(left, SETTINGS.crop, 320) == 121
    assert crop_start(right, SETTINGS.crop, 320) == 18


def test_a_pinned_crop_column_wins_over_the_projection():
    pinned = bpp_settings.load(crop_column={"left": 100.0, "right": 200.0})
    assert conversion.crop_column(pinned, "left") == 100.0
    with pytest.raises(PolicyError, match="no 'right' entry"):
        conversion.crop_column(bpp_settings.load(crop_column={"left": 1.0}), "right")


def test_the_views_are_what_bpps_model_sees():
    demo = demonstration(steps=4)
    views = conversion.views(demo.frames[2].images, SETTINGS, "left")
    assert set(views) == {"agentview_rgb", "eye_in_hand_rgb"}
    for view in views.values():
        assert view.shape == (3, 224, 224) and view.dtype == np.float32
        assert 0.0 <= view.min() and view.max() <= 1.0
    # A frame of one level comes out at that level, whatever the crop and the two resizes.
    np.testing.assert_allclose(views["agentview_rgb"], 2 / 255, atol=1e-6)
    np.testing.assert_allclose(views["eye_in_hand_rgb"], 9 / 255, atol=1e-6)


def test_the_wrist_roll_turns_the_wrist_image_only():
    images = demonstration(steps=2).frames[0].images
    images = {
        **images,
        WRISTS[0]: np.tile(np.arange(320, dtype=np.uint8), (240, 1))[..., None].repeat(3, 2),
    }
    upright = conversion.views(images, SETTINGS, "left")["eye_in_hand_rgb"]
    rolled = conversion.views(images, bpp_settings.load(wrist_roll_quarter_turns=1), "left")[
        "eye_in_hand_rgb"
    ]
    assert not np.allclose(upright, rolled)


def test_a_missing_camera_names_the_profile():
    demo = demonstration(steps=2)
    images = {name: image for name, image in demo.frames[0].images.items() if name != AGENTVIEW}
    with pytest.raises(PolicyError, match="far_side"):
        conversion.views(images, SETTINGS, "left")


# ------------------------------------------------------------------------------- prompt


def test_a_prompt_action_is_the_move_divided_by_the_gain():
    step = 0.002
    demo = demonstration(steps=21, step_m=step)
    prompt = conversion.build_prompt(demo, SETTINGS, "left")
    assert prompt.actions.shape == (20, 10)
    # The tool slides along the world's +y, which is LIBERO's +x, one unit being alpha_p * 0.05 m.
    unit = SETTINGS.alpha_p * bpp_settings.OSC_POSITION_SCALE_M
    np.testing.assert_allclose(prompt.actions[:, 0], step / unit, atol=1e-9)
    np.testing.assert_allclose(prompt.actions[:, 1:3], 0.0, atol=1e-9)
    np.testing.assert_allclose(
        prompt.actions[:, 3:9], np.tile(matrix_to_rot6d(np.eye(3)), (20, 1)), atol=1e-9
    )
    assert prompt.clipped == 0.0 and prompt.chunks == 1


def test_a_fast_demonstration_clips_and_says_how_much():
    prompt = conversion.build_prompt(demonstration(steps=6, step_m=0.05), SETTINGS, "left")
    assert prompt.actions.max() == pytest.approx(1.0)
    assert prompt.clipped > 0.0


def test_a_rotation_is_encoded_as_the_rot6d_of_the_scaled_axis_angle():
    turn = 0.02
    demo = demonstration(steps=3, turn_rad=turn * 2, step_m=0.0)
    prompt = conversion.build_prompt(demo, SETTINGS, "left")
    unit = SETTINGS.alpha_r * bpp_settings.OSC_ROTATION_SCALE_RAD
    expected = matrix_to_rot6d(axis_angle_to_matrix([0.0, 0.0, turn / unit]))
    np.testing.assert_allclose(prompt.actions[0, 3:9], expected, atol=1e-9)


def test_the_gripper_label_follows_the_onset_of_the_command():
    # Open, then a slow close over several frames, then closed: +1 while falling or closed.
    values = np.array([1.0, 1.0, 0.6, 0.2, 0.0, 0.0, 0.4, 1.0, 1.0])
    prompt = conversion.build_prompt(demonstration(steps=9, grippers=values), SETTINGS, "left")
    np.testing.assert_allclose(prompt.actions[:, 9], [-1, 1, 1, 1, 1, -1, -1, -1])


def test_the_stretch_resamples_the_demonstration_more_densely():
    demo = demonstration(steps=21)
    slow = conversion.build_prompt(demo, bpp_settings.load(stretch=2.0), "left")
    assert slow.rate_hz == 40.0 and len(slow.actions) == 2 * 20
    assert slow.chunks == 2


def test_the_chunk_budget_refuses_an_endless_demonstration():
    tight = bpp_settings.load(max_prompt_chunks=1)
    with pytest.raises(PolicyError, match="prompt chunks"):
        conversion.build_prompt(demonstration(steps=60), tight, "left")


def test_the_prompt_reads_one_observation_per_chunk():
    demo = demonstration(steps=45)
    prompt = conversion.build_prompt(demo, SETTINGS, "left")
    assert prompt.frames == (0, 20, 40)
    state = conversion.prompt_state(prompt, SETTINGS)
    assert state["agentview_rgb"].shape == (3, 3, 224, 224)
    assert state["ee_pos"].shape == (3, 3) and state["gripper_states"].shape == (3, 2)


# ---------------------------------------------------------------------------- execution


def test_an_action_decodes_into_the_world_frame():
    move, turn, gripper = conversion.decode_action(
        np.array([1.0, 0, 0, 1, 0, 0, 0, 1, 0, -1.0]), SETTINGS
    )
    # LIBERO's +x is the world's +y, and a negative gripper action opens.
    np.testing.assert_allclose(
        move, [0.0, SETTINGS.alpha_p * bpp_settings.OSC_POSITION_SCALE_M, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(turn, np.eye(3), atol=1e-12)
    assert gripper == 1.0
    assert conversion.decode_action(np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5]), SETTINGS)[2] == 0.0


def test_grouping_splits_a_chunk_four_four_three_one():
    chunk = np.zeros((12, 10))
    chunk[:, 9] = -1.0
    assert conversion.group_bounds(chunk, "ee_step") == [(i, i + 1) for i in range(12)]
    assert conversion.group_bounds(chunk, "ee_grouped") == [(0, 4), (4, 8), (8, 11), (11, 12)]


def test_grouping_splits_again_where_the_gripper_flips():
    chunk = np.zeros((12, 10))
    chunk[:, 9] = -1.0
    chunk[2:, 9] = 1.0  # the gripper closes two steps into the first group
    assert conversion.group_bounds(chunk, "ee_grouped")[:2] == [(0, 2), (2, 6)]
    for start, stop in conversion.group_bounds(chunk, "ee_grouped"):
        assert len(set(chunk[start:stop, 9])) == 1


def test_an_aggregated_group_moves_exactly_as_its_steps_would():
    rows = np.zeros((3, 10))
    rows[:, :3] = [[0.2, 0.0, 0.0], [0.1, 0.3, 0.0], [0.0, 0.0, 0.4]]
    rows[:, 3:9] = matrix_to_rot6d(axis_angle_to_matrix([0.0, 0.0, 0.05]))
    rows[:, 9] = -1.0
    total = conversion.aggregate(rows, SETTINGS)
    moves = [conversion.decode_action(row, SETTINGS) for row in rows]
    move, turn, gripper = conversion.decode_action(total, SETTINGS)
    np.testing.assert_allclose(move, sum(m for m, _, _ in moves), atol=1e-12)
    composed = np.eye(3)
    for _, step, _ in moves:
        composed = step @ composed
    np.testing.assert_allclose(turn, composed, atol=1e-9)
    assert gripper == 1.0


def _composed(rows):
    composed = np.eye(3)
    for row in rows:
        _, step, _ = conversion.decode_action(row, SETTINGS)
        composed = step @ composed
    return composed


def test_a_group_is_cut_before_its_rotation_outgrows_one_action():
    # Four full-scale rotations about one axis compose past what a single action encodes; the
    # aggregated group would otherwise wrap and turn the other way.
    chunk = np.zeros((12, 10))
    chunk[:, 3:9] = matrix_to_rot6d(axis_angle_to_matrix([0.0, 0.0, 1.0]))
    chunk[:, 9] = -1.0
    step_rad = SETTINGS.alpha_r * bpp_settings.OSC_ROTATION_SCALE_RAD
    assert 4 * step_rad > conversion.rotation_encoding_limit(SETTINGS)
    bounds = conversion.group_bounds(chunk, "ee_grouped", SETTINGS)
    assert bounds[0] == (0, 3)
    for start, stop in bounds:
        total = conversion.aggregate(chunk[start:stop], SETTINGS)
        np.testing.assert_allclose(
            conversion.decode_action(total, SETTINGS)[1], _composed(chunk[start:stop]), atol=1e-9
        )


def test_aggregating_a_rotation_too_large_to_encode_is_refused():
    rows = np.zeros((4, 10))
    rows[:, 3:9] = matrix_to_rot6d(axis_angle_to_matrix([0.0, 0.0, 1.0]))
    rows[:, 9] = -1.0
    with pytest.raises(PolicyError, match="composes"):
        conversion.aggregate(rows, SETTINGS)


def _drive(settings, actions, moving=False):
    """Run actions through `Execution`. The observation stands still unless `moving`, so the
    virtual target is what advances, not the robot."""
    demo = demonstration(steps=8)
    execution = conversion.Execution(settings, "left")
    out = []
    for i, action in enumerate(actions):
        index = min(i, len(demo) - 1) if moving else 0
        out.append(execution.act(action, observation(demo, index=index, step=i)))
    return np.concatenate(out), execution


def test_an_ee_action_drives_one_arm_and_holds_the_other():
    action = np.zeros(10)
    action[0] = 1.0
    action[3:9] = matrix_to_rot6d(np.eye(3))
    action[9] = -1.0
    rows, execution = _drive(SETTINGS, [action, action])
    assert rows.shape == (2, 16)
    idle = np.asarray(demonstration(steps=8).frames[0].endpose["right_endpose"])
    np.testing.assert_allclose(rows[:, 8:15], np.tile(idle, (2, 1)), atol=1e-12)
    np.testing.assert_allclose(rows[:, 15], 1.0)  # the idle gripper stays open
    assert rows[0, 7] == 1.0  # the active gripper opens on a negative gripper action
    # Each call moves the target one gain-scaled unit along the world's +y.
    unit = SETTINGS.alpha_p * bpp_settings.OSC_POSITION_SCALE_M
    assert rows[1, 1] - rows[0, 1] == pytest.approx(unit, abs=1e-9)
    assert execution.info()["calls"] == 2


def test_a_robot_that_lags_its_target_is_re_anchored_not_run_away_with():
    # The observation advances 2 mm per call while the model commands 12 mm: once the tracking
    # error passes `max_position_error_m` the target is re-anchored to where the arm really is,
    # so the adapter never chases a target the planner cannot reach.
    action = np.zeros(10)
    action[0] = 1.0
    action[3:9] = matrix_to_rot6d(np.eye(3))
    action[9] = -1.0
    rows, execution = _drive(SETTINGS, [action] * 6, moving=True)
    unit = SETTINGS.alpha_p * bpp_settings.OSC_POSITION_SCALE_M
    assert execution.info()["reanchors"]["tracking"] > 0
    # Re-anchored, the target sits one step ahead of the arm rather than five steps ahead.
    assert rows[-1, 1] - rows[0, 1] < 5 * unit


def test_the_virtual_target_keeps_motion_the_planner_would_lose():
    # The observation never moves, as a planner inside its goal tolerance would leave it: the
    # target still advances one unit per call instead of being re-anchored to a standstill.
    small = bpp_settings.load(alpha_p=0.02)
    action = np.zeros(10)
    action[0] = 1.0
    action[3:9] = matrix_to_rot6d(np.eye(3))
    demo = demonstration(steps=2)
    execution = conversion.Execution(small, "left")
    first = execution.act(action, observation(demo, 0, 0))
    third = None
    for step in range(1, 4):
        third = execution.act(action, observation(demo, 0, step))
    unit = small.alpha_p * bpp_settings.OSC_POSITION_SCALE_M
    assert third[0, 1] - first[0, 1] == pytest.approx(3 * unit, abs=1e-9)
    assert execution.info()["reanchors"] == {"tracking": 0, "plan_failed": 0}


def test_qpos_ik_returns_joint_targets_for_one_arm():
    urdf = str(bpp_settings.Path(__file__).resolve().parent / "data" / "aloha_like.urdf")
    settings = bpp_settings.load(mode="qpos_ik", urdf_path=urdf)
    assert settings.action_type == "qpos"
    from icil_policies.common.kinematics import AlohaArm

    execution = conversion.Execution(settings, "left", AlohaArm(urdf, "left"))
    action = np.zeros(10)
    action[3:9] = matrix_to_rot6d(np.eye(3))
    action[9] = -1.0
    demo = demonstration(steps=2)
    rows = execution.act(action, observation(demo, 0, 0))
    assert rows.shape == (1, 14)
    # The idle arm holds the joints and the gripper of the episode's first observation.
    np.testing.assert_allclose(rows[0, 7:14], 0.2, atol=1e-12)
    assert rows[0, 6] == 1.0  # the active gripper opens


def test_qpos_ik_without_a_urdf_is_refused():
    with pytest.raises(PolicyError, match="qpos_ik"):
        conversion.Execution(bpp_settings.load(mode="qpos_ik"), "left")


# ------------------------------------------------------------------------------- pinned


def test_the_conversion_is_pinned_to_the_adapter_version():
    demo = demonstration(steps=45, grippers=np.r_[np.ones(20), np.zeros(25)], turn_rad=0.3)
    prompt = conversion.build_prompt(demo, SETTINGS, "left")
    state = conversion.prompt_state(prompt, SETTINGS)
    rows, _ = _drive(SETTINGS, [prompt.actions[i] for i in range(4)])
    digest = conversion_digest(
        prompt.actions,
        state["ee_pos"],
        state["ee_ori"],
        state["gripper_states"],
        state["agentview_rgb"][:, :, ::32, ::32],
        state["eye_in_hand_rgb"][:, :, ::32, ::32],
        rows,
    )
    import icil_policies.bpp as adapter

    assert adapter.ADAPTER_VERSION == ADAPTER_VERSION
    assert_conversion_pinned(adapter, digest, PINS)
