"""The survey's capture without images runs the same expert on the real simulator.

`robot_state` stands in for `get_obs` on the claim that no expert reads what rendering produces.
On dual Franka, the robot the one-arm survey runs, each seed's attempt with and without images
must end the same way — success or the same rejection — with as many frames, the same joints
and the same arms moved. Run it alone on the GPU, from a checkout with RoboTwin's assets.
"""

import numpy as np
import pytest

from robotwin_icil import generate, robotwin
from robotwin_icil.demo import arms_moved

pytestmark = pytest.mark.sim

CONFIG = robotwin.SceneConfig(embodiment="franka-panda")


def _attempt(task_env, task: str, seed: int, images: bool) -> dict:
    outcome, demonstration, _ = generate.attempt(
        task_env, seed, CONFIG.resolve(task), CONFIG.save_freq, 0, images=images
    )
    if demonstration is None:
        return {"outcome": outcome.rejection.value, "frames": None, "arms": None, "qpos": None}
    assert (demonstration.cameras == ()) is not images
    return {
        "outcome": "ok",
        "frames": len(demonstration),
        "arms": arms_moved(demonstration),
        "qpos": demonstration.qpos(),
    }


def _differences(rendered: dict, plain: dict) -> list[str]:
    found = [
        f"{key}: {rendered[key]} with images, {plain[key]} without"
        for key in ("outcome", "frames", "arms")
        if rendered[key] != plain[key]
    ]
    if (
        not found
        and rendered["qpos"] is not None
        and not np.allclose(rendered["qpos"], plain["qpos"])
    ):
        largest = float(np.abs(rendered["qpos"] - plain["qpos"]).max())
        found.append(f"qpos: rows differ by up to {largest:.3g}")
    return found


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("task", ["click_bell", "place_empty_cup"])
def test_an_attempt_without_images_ends_as_the_rendered_one_does(task, seed):
    task_env = robotwin.load_task(task)
    rendered = _attempt(task_env, task, seed, images=True)
    plain = _attempt(task_env, task, seed, images=False)
    differences = _differences(rendered, plain)
    if differences:
        # The Franka expert has produced 44 and 45 frames from one click_bell scene on two rendered
        # runs. Only a second rendered run that agrees with the first makes a difference evidence
        # against leaving the cameras out.
        again = _differences(rendered, _attempt(task_env, task, seed, images=True))
        if again:
            pytest.xfail(
                f"{task} seed {seed}: two rendered runs already differ ({'; '.join(again)}), "
                f"so this seed cannot show what rendering changes ({'; '.join(differences)})"
            )
    assert not differences, f"{task} seed {seed}: " + "; ".join(differences)
