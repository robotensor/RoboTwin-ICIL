"""The BPP adapter's execution path in the real simulator: the oracle drives one arm, live.

Pure tests check the conversion arithmetic; only RoboTwin can say whether the actions it
produces plan, move the arm they name, and leave the other one where it was. This runs
`BPPConversionReplay` — the conversion oracle, the same chain `BPPPolicy` executes with a model
in front of it — on one scene of the fastest V1 task, under the `far_side` camera profile the
adapter requires.

Run from the main checkout, in the simulator env:

    PYTHONPATH=src:policies/src $RT -m pytest policies/tests -m sim
"""

import numpy as np
import pytest

from icil_policies.bpp.oracle import BPPConversionReplay
from icil_policies.bpp.settings import load
from robotwin_icil import robotwin
from robotwin_icil.generate import generate, scene_seeds
from robotwin_icil.policy import Observation

pytestmark = pytest.mark.sim

TASK = "click_bell"  # the fastest expert
PROFILE = "far_side"
CALLS = 12
SAVE_FREQ = 15
# What the adapter drives per call: a full-scale action is about 12 mm of tool motion, so a
# dozen calls must move the active arm well past a millimetre and leave the idle one still.
ACTIVE_MOTION_M = 0.005
IDLE_MOTION_M = 1e-3


@pytest.fixture(scope="module")
def settings():
    return load()


def _scene(config):
    """One scene of TASK with a successful expert demonstration, or a skip."""
    task_env = robotwin.load_task(TASK)
    seeds = scene_seeds(0, 0, 8)
    generated = generate(task_env, seeds, lambda: config.resolve(TASK), SAVE_FREQ, 0)
    if not generated.ok:
        robotwin.close(task_env)
        pytest.skip(f"{TASK}: no successful expert demonstration in 8 seeds")
    robotwin.close(task_env)
    return generated


def _observation(task_env, step):
    raw = robotwin.observation(task_env)
    return Observation(
        step=step,
        images=raw["images"],
        qpos=raw["qpos"],
        endpose=raw["endpose"],
        gripper_joints=raw["gripper_joints"],
    )


def _tool(observation, arm):
    from icil_policies.common.frames import tcp_from_flange

    return tcp_from_flange(np.asarray(observation.endpose[f"{arm}_endpose"]))[:3]


def test_the_conversion_oracle_drives_one_arm_in_the_simulator(settings):
    config = robotwin.SceneConfig(camera_profile=PROFILE)
    generated = _scene(config)
    policy = BPPConversionReplay()
    policy.seed(1)
    policy.reset()
    policy.set_demonstration(generated.demonstration)

    task_env = robotwin.load_task(TASK)
    task_env.setup_demo(now_ep_num=0, seed=generated.seed, is_test=True, **config.resolve(TASK))
    try:
        first = _observation(task_env, 0)
        arm = policy.episode_info()["active_arm"]
        idle = "right" if arm == "left" else "left"
        start_active, start_idle = _tool(first, arm), _tool(first, idle)
        for step in range(CALLS):
            observation = _observation(task_env, step)
            for action in policy.act(observation):
                task_env.take_action(action, action_type=policy.action_type)
            if robotwin.episode_over(task_env):
                break
        last = _observation(task_env, CALLS)
        moved_active = float(np.linalg.norm(_tool(last, arm) - start_active))
        moved_idle = float(np.linalg.norm(_tool(last, idle) - start_idle))
        print(
            f"\n{TASK}: active {arm} moved {moved_active * 1000:.1f} mm, idle {idle} "
            f"{moved_idle * 1000:.2f} mm over {CALLS} calls; {policy.episode_info()}"
        )
        assert moved_active > ACTIVE_MOTION_M
        assert moved_idle < IDLE_MOTION_M
        info = policy.episode_info()
        assert info["prompt_chunks"] >= 1 and info["calls"] == CALLS
    finally:
        robotwin.close(task_env)


def test_the_far_side_profile_renders_every_camera_the_adapter_reads(settings):
    config = robotwin.SceneConfig(camera_profile=PROFILE)
    task_env = robotwin.load_task(TASK)
    task_env.setup_demo(now_ep_num=0, seed=0, is_test=True, **config.resolve(TASK))
    try:
        images = robotwin.observation(task_env)["images"]
        for name in (settings.agentview_camera, *settings.wrist_cameras):
            assert name in images, f"{name} missing from {sorted(images)}"
        width, height, _ = settings.agentview_geometry()
        assert images[settings.agentview_camera].shape == (height, width, 3)
        from icil_policies.bpp.conversion import views

        converted = views(images, settings, "left")
        assert converted["agentview_rgb"].shape == (3, 224, 224)
        assert 0.0 <= converted["agentview_rgb"].min() <= converted["agentview_rgb"].max() <= 1.0
    finally:
        robotwin.close(task_env)
