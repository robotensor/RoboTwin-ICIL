"""The survey's capture without images runs the same expert on the real simulator.

`robot_state` stands in for `get_obs` on the claim that no expert reads what rendering produces.
On dual Franka, the robot the one-arm survey runs, each seed's attempt with and without images
must end the same way — success or the same rejection — with as many frames, the same joints
and the same arms moved. Run it alone on the GPU, from a checkout with RoboTwin's assets.

The verdict on a pair of runs is plain Python and is tested here without the simulator.
"""

import numpy as np
import pytest

from robotwin_icil import generate, robotwin
from robotwin_icil.demo import arms_moved

CONFIG = robotwin.SceneConfig(embodiment="franka-panda")

SAME = "same"
UNREPEATABLE = "unrepeatable"
DIFFERENT = "different"


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


def _within(rendered: list[dict], plain: dict) -> bool:
    """Whether the run without images ends as a rendered run does, frames between theirs."""
    alike = [
        run
        for run in rendered
        if (run["outcome"], run["arms"]) == (plain["outcome"], plain["arms"])
    ]
    if not alike:
        return False
    if plain["frames"] is None:
        return True
    counts = [run["frames"] for run in alike]
    return min(counts) <= plain["frames"] <= max(counts)


def _verdict(rendered: list[dict], plain: dict) -> tuple[str, str]:
    """`SAME`, `UNREPEATABLE` or `DIFFERENT`, and the differences that decided it.

    Without images, a run is the same when it matches any rendered run. It is unrepeatable, and the
    seed shows nothing, only when two rendered runs already differ from each other and it stays
    inside what they span: the outcome and arms of one of them, and a frame count between theirs.
    Anything else is a difference images made.
    """
    against = [_differences(run, plain) for run in rendered]
    if not all(against):
        return SAME, ""
    reason = "; ".join(
        f"rendered run {number}: {', '.join(found)}" for number, found in enumerate(against, 1)
    )
    spread = _differences(rendered[0], rendered[1]) if len(rendered) > 1 else []
    if not spread:
        return DIFFERENT, f"every rendered run agrees, and without images differs ({reason})"
    if not _within(rendered, plain):
        return DIFFERENT, f"without images leaves what the rendered runs span ({reason})"
    return UNREPEATABLE, (
        f"two rendered runs already differ ({'; '.join(spread)}), and without images stays "
        f"between them ({reason})"
    )


@pytest.mark.sim
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("task", ["click_bell", "place_empty_cup"])
def test_an_attempt_without_images_ends_as_the_rendered_one_does(task, seed):
    task_env = robotwin.load_task(task)
    rendered = [_attempt(task_env, task, seed, images=True)]
    plain = _attempt(task_env, task, seed, images=False)
    if _differences(rendered[0], plain):
        # The Franka expert does not always repeat itself: click_bell at eval seed 42 (scene seed
        # 191664963) gave a 45-frame demonstration on one rendered run and 44 on two more
        # (CHANGELOG, #81). A second rendered run tells that apart from a change images make.
        rendered.append(_attempt(task_env, task, seed, images=True))
    verdict, reason = _verdict(rendered, plain)
    if verdict == UNREPEATABLE:
        pytest.xfail(f"{task} seed {seed}: {reason}")
    assert verdict == SAME, f"{task} seed {seed}: {reason}"


def _run(outcome: str = "ok", frames: int | None = 44, arms=("right",), shift: float = 0.0) -> dict:
    if outcome != "ok":
        return {"outcome": outcome, "frames": None, "arms": None, "qpos": None}
    return {
        "outcome": outcome,
        "frames": frames,
        "arms": arms,
        "qpos": np.full((frames, 16), shift),
    }


def test_a_run_without_images_that_matches_either_rendered_run_is_the_same():
    assert _verdict([_run(frames=44)], _run(frames=44)) == (SAME, "")
    assert _verdict([_run(frames=44), _run(frames=45)], _run(frames=45)) == (SAME, "")
    failed = _run("plan_failed")
    assert _verdict([failed], _run("plan_failed")) == (SAME, "")


def test_a_run_without_images_that_differs_from_agreeing_rendered_runs_is_different():
    verdict, reason = _verdict([_run()], _run(shift=0.1))
    assert verdict == DIFFERENT and "qpos" in reason
    verdict, reason = _verdict([_run(), _run()], _run(shift=0.1))
    assert verdict == DIFFERENT and "every rendered run agrees" in reason


def test_a_failure_without_images_between_two_rendered_successes_is_different():
    verdict, reason = _verdict([_run(frames=44), _run(frames=45)], _run("expert_failed"))
    assert verdict == DIFFERENT
    assert "ok with images, expert_failed without" in reason


def test_a_run_without_images_that_moves_other_arms_than_both_rendered_runs_is_different():
    rendered = [_run(frames=44), _run(frames=45)]
    assert _verdict(rendered, _run(frames=45, arms=("left",)))[0] == DIFFERENT
    assert _verdict(rendered, _run(frames=45, arms=("left", "right")))[0] == DIFFERENT


def test_a_frame_count_outside_the_rendered_runs_is_different():
    rendered = [_run(frames=44), _run(frames=45)]
    assert _verdict(rendered, _run(frames=46))[0] == DIFFERENT
    # One rendered run failed: the span is the one success's frame count.
    assert _verdict([_run(frames=44), _run("plan_failed")], _run(frames=45))[0] == DIFFERENT


def test_a_run_without_images_between_two_differing_rendered_runs_is_unrepeatable():
    rendered = [_run(frames=44), _run(frames=45)]
    verdict, reason = _verdict(rendered, _run(frames=45, shift=0.1))
    assert verdict == UNREPEATABLE
    assert "frames: 44 with images, 45 without" in reason
    rendered = [_run(frames=44), _run("plan_failed")]
    assert _verdict(rendered, _run("plan_failed", shift=0.1))[0] == SAME
    assert _verdict(rendered, _run(frames=44, shift=0.1))[0] == UNREPEATABLE
