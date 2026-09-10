"""RoboTwin gets the task name it needs, and so the task's own step limit."""

import pytest
import yaml

from robotwin_icil import robotwin

pytestmark = pytest.mark.sim


def test_a_built_scene_gets_its_tasks_step_limit():
    # Without `task_name`, RoboTwin prints "None not in step limit file" and allows 1000 steps.
    limits_file = robotwin.ROBOTWIN_ROOT / "env_cfg" / "task_config" / "_eval_step_limit.yml"
    limits = yaml.safe_load(limits_file.read_text(encoding="utf-8"))
    args = robotwin.SceneConfig().resolve("click_bell")
    assert args["task_name"] == "click_bell"

    task_env = robotwin.load_task("click_bell")
    for seed in range(5):
        try:
            task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        except robotwin.unstable_error():
            continue
        try:
            assert task_env.step_lim == limits["click_bell"] != 1000
        finally:
            task_env.close_env()
        return
    pytest.fail("every seed was unstable")
