"""Camera profiles keep every seed's scene in the real simulator.

Every static camera draws from numpy's global RNG after seeding and before `load_actors`, and
`camera_profiles` refuses any profile that changes how many draw. This checks the claim itself:
`stock` and `far_side` build the same actors, articulations and robot for three seeds of one task
per V1 category. Only the cameras may differ.
"""

import dataclasses

import numpy as np
import pytest

from robotwin_icil import camera_profiles, robotwin, scene

pytestmark = pytest.mark.sim

# One task per V1 category: Press / Push, Pick and Place, Stacking.
TASKS = ["click_bell", "place_a2b_left", "stack_blocks_two"]
SEEDS = [0, 1, 2]


def _build(task_env, task, seed, profile):
    """The scene's fingerprint and first images, or None when the seed does not settle."""
    args = robotwin.SceneConfig(camera_profile=profile).resolve(task)
    try:
        task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    except robotwin.unstable_error():
        robotwin.close(task_env)
        return None
    try:
        return robotwin.fingerprint(task_env), robotwin.observation(task_env)["images"]
    finally:
        robotwin.close(task_env)


def test_far_side_resolves_on_robotwins_own_configs():
    args = robotwin.SceneConfig(camera_profile="far_side").resolve("click_bell")
    cameras = camera_profiles.static_cameras(args)
    assert [(c["name"], c["type"]) for c in cameras] == [
        ("head_camera", "D435"),
        ("far_side_camera", "L515"),
    ]


@pytest.mark.parametrize("task", TASKS)
def test_far_side_builds_the_same_scene_as_stock(task):
    task_env = robotwin.load_task(task)
    checked = 0
    for seed in SEEDS:
        stock = _build(task_env, task, seed, "stock")
        far_side = _build(task_env, task, seed, "far_side")
        if stock is None or far_side is None:
            # An unstable seed is skipped, but the two profiles must agree that it is unstable.
            assert stock is far_side is None, f"{task} seed {seed}: only one profile settled"
            continue
        (stock_scene, stock_images), (far_scene, far_images) = stock, far_side

        mismatches = scene.compare(
            dataclasses.replace(stock_scene, cameras={}),
            dataclasses.replace(far_scene, cameras={}),
        )
        assert mismatches == [], f"{task} seed {seed}: " + "; ".join(map(str, mismatches))

        assert set(stock_scene.cameras) ^ set(far_scene.cameras) == {
            "front_camera",
            "far_side_camera",
        }
        assert set(stock_images) - {"front_camera"} == set(far_images) - {"far_side_camera"}
        image = far_images["far_side_camera"]
        assert image.shape == (180, 320, 3) and image.dtype == np.uint8
        assert np.ptp(image) > 0, "far_side_camera rendered a single colour"
        checked += 1
    assert checked, f"{task}: every seed was unstable, nothing was checked"
