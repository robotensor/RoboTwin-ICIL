"""Same Scene holds in the real simulator: one seed, one scene, however many times it is built."""

import pytest

from robotwin_icil import robotwin, scene

pytestmark = pytest.mark.sim

# One task per V1 category, so a category-specific source of drift cannot hide.
TASKS = ["place_object_basket", "stack_blocks_two", "click_bell"]
SEEDS = [0, 1, 2]


def _build(task_env, task, seed):
    task_env.setup_demo(
        now_ep_num=0, seed=seed, is_test=True, **robotwin.SceneConfig().resolve(task)
    )
    try:
        return robotwin.fingerprint(task_env)
    finally:
        task_env.close_env()


@pytest.mark.parametrize("task", TASKS)
def test_same_seed_rebuilds_the_same_scene(task):
    task_env = robotwin.load_task(task)
    checked = 0
    for seed in SEEDS:
        try:
            first = _build(task_env, task, seed)
        except robotwin.unstable_error():
            continue  # a generation rejection, not a determinism question
        second = _build(task_env, task, seed)
        mismatches = scene.compare(first, second)
        assert mismatches == [], f"{task} seed {seed}: " + "; ".join(map(str, mismatches))
        checked += 1
    assert checked, f"{task}: every seed was unstable, nothing was checked"


def test_a_different_seed_is_a_different_scene():
    # Guards the test above: a fingerprint too coarse to tell scenes apart would pass it vacuously.
    task = "place_object_basket"
    task_env = robotwin.load_task(task)
    fingerprints = []
    for seed in range(6):
        try:
            fingerprints.append(_build(task_env, task, seed))
        except robotwin.unstable_error():
            continue
        if len(fingerprints) == 2:
            break
    assert len(fingerprints) == 2
    assert scene.compare(fingerprints[0], fingerprints[1]) != []
