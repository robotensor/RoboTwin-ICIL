"""The simulator install works end to end: a task builds and RoboTwin's expert solves it."""

import pytest

from robotwin_icil import robotwin

pytestmark = pytest.mark.sim

TASK = "place_object_basket"


def test_a_task_sets_up_and_its_expert_succeeds():
    # RoboTwin's expert fails some seeds outright — an infeasible grasp raises from `play_once()`,
    # a plan comes up short. Upstream's collector counts those as failed seeds and moves on, and
    # so does this check: only an expert that solves none of eight seeds means a broken install.
    task_env = robotwin.load_task(TASK)
    outcomes = []
    for seed in range(8):
        try:
            task_env.setup_demo(
                now_ep_num=0, seed=seed, is_test=True, **robotwin.SceneConfig().resolve(TASK)
            )
        except robotwin.unstable_error():
            outcomes.append(f"seed {seed}: unstable")
            continue
        try:
            task_env.play_once()
            if task_env.plan_success and task_env.check_success():
                return
            outcomes.append(f"seed {seed}: the expert did not succeed")
        except Exception as exc:
            outcomes.append(f"seed {seed}: {type(exc).__name__}: {exc}")
        finally:
            task_env.close_env()
    pytest.fail("RoboTwin's expert solved none of eight seeds:\n" + "\n".join(outcomes))
