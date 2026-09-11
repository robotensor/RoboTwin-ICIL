"""aloha-agilex's numpy kinematics against RoboTwin's own endposes, in the simulator.

`qpos_ik` trusts `AlohaArm` to know where a joint configuration puts the flange. RoboTwin
computes the endpose from SAPIEN's link pose (`Robot._trans_endpose`); this drives both arms to
a few joint configurations with `take_action('qpos')`, compares `AlohaArm.endpose` of the
measured joints with the endpose `get_obs()` reports for that same state, and solves each
reported endpose back to joints. Errors are printed (`pytest -s`).

Run from the main checkout, in the simulator env: `PYTHONPATH=src:policies/src $RT -m pytest
policies/tests -m sim`.
"""

import numpy as np
import pytest

from icil_policies.common.kinematics import AlohaArm
from icil_policies.common.rotations import quat_to_matrix, relative_angle
from robotwin_icil import robotwin

pytestmark = pytest.mark.sim

TASK = "click_bell"  # the fastest expert; any task's scene holds the same robot
SEEDS = range(8)
URDF = (
    robotwin.ROBOTWIN_ROOT
    / "assets"
    / "embodiments"
    / "aloha-agilex"
    / "urdf"
    / "arx5_description_isaac.urdf"
)
# Joint targets for both arms, home first; moderate, so the arms stay clear of the table.
CONFIGURATIONS = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.3, 0.4, 0.3, 0.2, -0.3, 0.5),
    (-0.3, 0.8, 0.6, -0.4, 0.4, -0.6),
    (0.5, 0.2, 0.1, 0.6, 0.2, 1.0),
)
# The model and RoboTwin share the URDF and agree to float32 precision: 4e-7 m and 1e-6 rad,
# measured, with `robot_pose`'s rounded 0.707 quaternion costing nothing measurable. A model
# error of a fraction of a millimetre, such as a base offset or an anchor mix-up, must fail.
POSITION_TOLERANCE_M = 1e-5
ROTATION_TOLERANCE_RAD = 1e-4


def _build(task_env, seed):
    """Build one seed's scene; None, with the env closed, when it does not settle."""
    try:
        task_env.setup_demo(
            now_ep_num=0, seed=seed, is_test=True, **robotwin.SceneConfig().resolve(TASK)
        )
    except robotwin.unstable_error():
        robotwin.close(task_env)
        return None
    return task_env


def _stable_seed(task_env):
    for seed in SEEDS:
        if _build(task_env, seed) is not None:
            robotwin.close(task_env)
            return seed
    pytest.fail(f"{TASK}: every seed of {list(SEEDS)} was unstable")


def _measured_joints(task_env, side: str) -> np.ndarray:
    """An arm's six measured joint positions, gripper left out."""
    robot = task_env.robot
    state = (
        robot.get_left_arm_real_jointState()
        if side == "left"
        else robot.get_right_arm_real_jointState()
    )
    return np.asarray(state[:-1], dtype=np.float64)


def test_forward_kinematics_matches_robotwins_endposes():
    arms = {side: AlohaArm(URDF, side) for side in ("left", "right")}
    task_env = robotwin.load_task(TASK)
    seed = _stable_seed(task_env)
    assert _build(task_env, seed) is not None
    try:
        print(f"\n{TASK} seed {seed}: AlohaArm against RoboTwin's endpose")
        for config in CONFIGURATIONS:
            action = robotwin.observation(task_env)["qpos"].copy()
            action[0:6] = config
            action[7:13] = config
            task_env.take_action(action, action_type="qpos")
            reported = robotwin.observation(task_env)["endpose"]
            for side, arm in arms.items():
                q = _measured_joints(task_env, side)
                expected = np.asarray(reported[f"{side}_endpose"], dtype=np.float64)
                computed = arm.endpose(q)
                position = float(np.linalg.norm(computed[:3] - expected[:3]))
                rotation = float(
                    relative_angle(quat_to_matrix(computed[3:]), quat_to_matrix(expected[3:]))
                )
                solved = arm.inverse(expected, q + 0.05)
                print(
                    f"  {side} q={np.round(q, 3).tolist()}: {position * 1000:.4f} mm, "
                    f"{rotation:.2e} rad; IK {'converged' if solved.converged else 'FAILED'} "
                    f"in {solved.iterations} iterations"
                )
                assert position < POSITION_TOLERANCE_M, (side, config, position)
                assert rotation < ROTATION_TOLERANCE_RAD, (side, config, rotation)
                assert solved.converged, (side, config, solved)
                np.testing.assert_allclose(arm.endpose(solved.q)[:3], expected[:3], atol=1e-5)
    finally:
        robotwin.close(task_env)
