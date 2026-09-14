"""The survey's capture without images runs the same expert on the real simulator.

`robot_state` stands in for `get_obs` on the claim that no expert reads what rendering produces.
On dual Franka, the robot the one-arm survey runs, each seed's attempt with and without images
must end the same way — success or the same rejection — with as many frames, the same joints,
the same endposes and the same arms moved. `robot_state` reads the endpose itself rather than
taking `get_obs`'s, so only the real simulator can show the two agree. Run it alone on the GPU,
from a checkout with RoboTwin's assets.

The verdict on a pair of runs is plain Python and is tested here without the simulator.
"""

import numpy as np
import pytest

from robotwin_icil import generate, robotwin
from robotwin_icil.demo import Demonstration, Frame, arms_moved

CONFIG = robotwin.SceneConfig(embodiment="franka-panda")

SAME = "same"
UNREPEATABLE = "unrepeatable"
DIFFERENT = "different"


def _attempt(task_env, task: str, seed: int, images: bool) -> dict:
    outcome, demonstration, _ = generate.attempt(
        task_env, seed, CONFIG.resolve(task), CONFIG.save_freq, 0, images=images
    )
    if demonstration is None:
        return {
            "outcome": outcome.rejection.value,
            "frames": None,
            "arms": None,
            "qpos": None,
            "endpose": None,
        }
    assert (demonstration.cameras == ()) is not images
    endpose = _endposes(demonstration)
    assert endpose, "the config records no endpose, so there is none to compare"
    return {
        "outcome": "ok",
        "frames": len(demonstration),
        "arms": arms_moved(demonstration),
        "qpos": demonstration.qpos(),
        "endpose": endpose,
    }


def _endposes(demonstration) -> dict[str, np.ndarray]:
    """Each endpose entry over the frames, in the frame's key order: (T, 7) poses, (T,) grippers."""
    return {
        key: np.asarray([frame.endpose[key] for frame in demonstration.frames], dtype=np.float64)
        for key in demonstration.frames[0].endpose
    }


def _differences(rendered: dict, plain: dict) -> list[str]:
    found = [
        f"{key}: {rendered[key]} with images, {plain[key]} without"
        for key in ("outcome", "frames", "arms")
        if rendered[key] != plain[key]
    ]
    if found or rendered["qpos"] is None:
        return found
    if not np.allclose(rendered["qpos"], plain["qpos"]):
        largest = float(np.abs(rendered["qpos"] - plain["qpos"]).max())
        found.append(f"qpos: rows differ by up to {largest:.3g}")
    keys, plain_keys = list(rendered["endpose"]), list(plain["endpose"])
    if keys != plain_keys:
        found.append(f"endpose: keys {keys} with images, {plain_keys} without")
        return found
    for key in keys:
        if not np.allclose(rendered["endpose"][key], plain["endpose"][key]):
            largest = float(np.abs(rendered["endpose"][key] - plain["endpose"][key]).max())
            found.append(f"{key}: differs by up to {largest:.3g}")
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


ENDPOSE_KEYS = ("left_endpose", "left_gripper", "right_endpose", "right_gripper")


def _run(
    outcome: str = "ok",
    frames: int | None = 44,
    arms=("right",),
    shift: float = 0.0,
    endpose_shift: float = 0.0,
    endpose_keys: tuple[str, ...] = ENDPOSE_KEYS,
) -> dict:
    if outcome != "ok":
        return {"outcome": outcome, "frames": None, "arms": None, "qpos": None, "endpose": None}
    return {
        "outcome": outcome,
        "frames": frames,
        "arms": arms,
        "qpos": np.full((frames, 16), shift),
        "endpose": {
            key: np.full((frames, 7) if key.endswith("endpose") else (frames,), endpose_shift)
            for key in endpose_keys
        },
    }


def test_endposes_stack_each_entry_over_the_frames_in_the_frames_key_order():
    frames = tuple(
        Frame(
            index=index,
            images={},
            qpos=np.zeros(16),
            endpose={"right_endpose": [float(index)] * 7, "right_gripper": 0.5},
        )
        for index in range(3)
    )
    endpose = _endposes(Demonstration(frames=frames, frequency=250 / 15))
    assert list(endpose) == ["right_endpose", "right_gripper"]
    assert endpose["right_endpose"].shape == (3, 7) and endpose["right_endpose"][2, 6] == 2.0
    assert endpose["right_gripper"].shape == (3,)


def test_an_endpose_that_moves_without_images_is_a_difference_though_the_joints_agree():
    verdict, reason = _verdict([_run()], _run(endpose_shift=0.1))
    assert verdict == DIFFERENT
    assert "left_endpose: differs by up to 0.1" in reason and "qpos" not in reason
    verdict, reason = _verdict([_run()], _run(endpose_keys=ENDPOSE_KEYS[::-1]))
    assert verdict == DIFFERENT and "endpose: keys" in reason
    assert _verdict([_run()], _run(endpose_keys=()))[0] == DIFFERENT


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
