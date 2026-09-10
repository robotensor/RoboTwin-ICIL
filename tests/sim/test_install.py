"""The simulator install works end to end: a task builds and RoboTwin's expert solves it."""

import pytest

from robotwin_icil import robotwin

pytestmark = pytest.mark.sim


def test_a_task_sets_up_and_its_expert_succeeds():
    task_env = robotwin.load_task("place_object_basket")
    for seed in range(5):
        try:
            task_env.setup_demo(
                now_ep_num=0, seed=seed, is_test=True, **robotwin.SceneConfig().resolve()
            )
        except robotwin.unstable_error():
            continue
        try:
            task_env.play_once()
            solved = task_env.plan_success and task_env.check_success()
        finally:
            task_env.close_env()
        if solved:
            return
    pytest.fail("RoboTwin's expert solved none of five seeds; the install is broken")
