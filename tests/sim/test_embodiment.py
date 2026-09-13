"""A run can choose its robot: dual Franka builds 16 wide, and its expert's demonstration is too."""

import pytest

from robotwin_icil import generate, robotwin

pytestmark = pytest.mark.sim

TASK = "click_bell"


def _build(task_env, config, seeds=range(5)):
    """Set the task up on the first stable seed; unstable placements are rejections, not evidence.

    Closes the env on any failure to build, as `generate.attempt` does: a half-built env left
    open holds the GPU and can hang the next test's camera read.
    """
    for seed in seeds:
        try:
            task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **config.resolve(TASK))
        except robotwin.unstable_error():
            robotwin.close(task_env)
            continue
        except Exception:
            robotwin.close(task_env)
            raise
        return seed
    pytest.fail("every seed was unstable")


@pytest.mark.parametrize(
    ("embodiment", "qpos_dim", "dual_arm"),
    [("aloha-agilex", 14, True), ("franka-panda", 16, False)],
)
def test_the_scene_is_built_with_the_chosen_robot(embodiment, qpos_dim, dual_arm):
    # aloha-agilex is one URDF with two six-joint arms; franka-panda is two seven-joint arms, one
    # URDF each. Both report joints then a gripper per arm, so the widths are 14 and 16.
    task_env = robotwin.load_task(TASK)
    config = robotwin.SceneConfig(embodiment=embodiment)
    _build(task_env, config)
    try:
        assert robotwin.action_dims(task_env) == {"qpos": qpos_dim, "ee": 16}
        assert robotwin.observation(task_env)["qpos"].shape == (qpos_dim,)
        fingerprint = robotwin.fingerprint(task_env)
        assert fingerprint.robot_qpos.shape == (qpos_dim,)
        assert (embodiment in fingerprint.extras["embodiment"]) and (
            ("|" not in fingerprint.extras["embodiment"]) is dual_arm
        )
        assert task_env.robot.is_dual_arm is dual_arm
    finally:
        task_env.close_env()


def test_two_franka_arms_stand_the_documented_distance_apart():
    task_env = robotwin.load_task(TASK)
    _build(task_env, robotwin.SceneConfig(embodiment="franka-panda"))
    try:
        left = task_env.robot.left_entity.get_root_pose().p
        right = task_env.robot.right_entity.get_root_pose().p
        assert abs(float(right[0] - left[0])) == pytest.approx(robotwin.FRANKA_ARM_DISTANCE_M)
    finally:
        task_env.close_env()


def test_the_franka_expert_produces_a_sixteen_wide_demonstration():
    # The whole point of choosing the robot: the demonstration a policy is handed on Franka has
    # 16-wide frames, captured through the same path as aloha's 14-wide ones.
    task_env = robotwin.load_task(TASK)
    config = robotwin.SceneConfig(embodiment="franka-panda")
    outcomes = []
    for seed in range(8):
        outcome, demonstration, initial = generate.attempt(
            task_env, seed, config.resolve(TASK), config.save_freq, 0
        )
        if outcome.rejection is None:
            assert demonstration.qpos_dim == 16
            assert demonstration.qpos().shape[1] == 16 and demonstration.actions().shape[1] == 16
            assert initial.robot_qpos.shape == (16,)
            return
        outcomes.append(f"seed {seed}: {outcome.rejection.value} {outcome.detail}".rstrip())
    pytest.fail("the Franka expert solved none of eight click_bell seeds:\n" + "\n".join(outcomes))
