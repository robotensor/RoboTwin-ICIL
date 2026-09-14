"""The survey's capture without images runs the same expert on the real simulator.

`robot_state` stands in for `get_obs` on the claim that no expert reads what rendering produces,
and only the real simulator can test that. It cannot test it seed by seed with an exact match,
because RoboTwin's expert does not repeat itself between runs of one scene: two rendered runs of
one seed have differed by up to 0.13 rad in their qpos rows; identical runs recorded 77 and 78
frames on aloha-agilex (#81) and 44 and 45 on two Frankas; and the one run of an exact comparison
here failed click_bell seed 0 on 47 frames rendered against 48 without. None of that tells
rendering apart from the expert's own variation.

So on dual Franka, the robot the one-arm survey runs, each seed runs with images, without, then
with images again, and the run without images must stay within what the two rendered runs span:

- the same outcome, success or the same rejection, unless the rendered runs already disagree;
- a frame count within max(the rendered runs' frame spread, `FRAME_SLACK`) of a rendered success;
- the same arms moved, and the same endpose keys, as every rendered success;
- when all three have as many frames, qpos and each endpose entry row by row, no further from the
  nearer rendered run than the rendered runs are from each other, plus `MARGIN`.

`robot_state` reads the endpose itself rather than taking `get_obs`'s, which is why endposes are
compared as well as joints. Run it alone on the GPU, from a checkout with RoboTwin's assets: a
second simulator process can run CuRobo out of memory and fail every seed.

The verdict on three runs is plain Python and is tested here without the simulator.
"""

import numpy as np
import pytest

from robotwin_icil import generate, robotwin
from robotwin_icil.demo import MOVED_THRESHOLD, Demonstration, Frame, arms_moved

CONFIG = robotwin.SceneConfig(embodiment="franka-panda")

# Runs of one seed have ended a frame apart (77 and 78, 44 and 45, 47 and 48), so rendered runs
# that agree on their frame count still leave that much room.
FRAME_SLACK = 2
# On top of the rendered runs' own difference: the arms reader's threshold, radians for a joint and
# a fraction of travel for a gripper (metres and quaternion components for an endpose), so a run
# without images may drift by less than it takes to count an arm as moving.
MARGIN = MOVED_THRESHOLD


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


def _largest(a: np.ndarray, b: np.ndarray) -> float:
    """The largest difference between two runs' rows, frame for frame."""
    return float(np.abs(a - b).max()) if a.size else 0.0


def _differences(rendered: list[dict], plain: dict) -> list[str]:
    """What puts the run without images outside what two rendered runs of its seed span.

    Empty when nothing does. A rendered run that failed has no frames, arms or rows, so only the
    rendered successes bound those; the rows are compared only when both rendered runs and the run
    without images have as many frames, since only then do the rendered runs say how far apart
    two runs' rows may be.
    """
    first, second = rendered
    found = []
    if first["outcome"] == second["outcome"] != plain["outcome"]:
        found.append(
            f"outcome: {first['outcome']} on both rendered runs, {plain['outcome']} without images"
        )
    succeeded = [
        (number, run) for number, run in enumerate(rendered, 1) if run["frames"] is not None
    ]
    if plain["frames"] is None or not succeeded:
        return found

    counts = [run["frames"] for _, run in succeeded]
    slack = max(max(counts) - min(counts), FRAME_SLACK)
    if min(abs(plain["frames"] - count) for count in counts) > slack:
        rendered_counts = " and ".join(str(count) for count in counts)
        found.append(
            f"frames: {plain['frames']} without images, {rendered_counts} rendered, "
            f"more than {slack} from the nearer"
        )
    for number, run in succeeded:
        if run["arms"] != plain["arms"]:
            found.append(f"arms: {run['arms']} on rendered run {number}, {plain['arms']} without")
    keys = list(plain["endpose"])
    keys_agree = True
    for number, run in succeeded:
        if list(run["endpose"]) != keys:
            keys_agree = False
            found.append(
                f"endpose: keys {list(run['endpose'])} on rendered run {number}, {keys} without"
            )
    frame_counts = {first["frames"], second["frames"], plain["frames"]}
    if len(succeeded) < 2 or not keys_agree or len(frame_counts) > 1:
        return found

    for key in ["qpos", *keys]:
        rows = [run["qpos"] if key == "qpos" else run["endpose"][key] for run in (*rendered, plain)]
        between = _largest(rows[0], rows[1])
        nearer = min(_largest(rows[2], rows[0]), _largest(rows[2], rows[1]))
        if nearer > between + MARGIN:
            found.append(
                f"{key}: rows differ by up to {nearer:.3g} from the nearer rendered run, "
                f"and the rendered runs by {between:.3g}"
            )
    return found


@pytest.mark.sim
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("task", ["click_bell", "place_empty_cup"])
def test_an_attempt_without_images_stays_within_two_rendered_runs(task, seed):
    task_env = robotwin.load_task(task)
    first = _attempt(task_env, task, seed, images=True)
    plain = _attempt(task_env, task, seed, images=False)
    second = _attempt(task_env, task, seed, images=True)
    found = _differences([first, second], plain)
    assert not found, f"{task} seed {seed}: {'; '.join(found)}"


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


def test_a_run_without_images_like_both_rendered_runs_has_no_difference():
    assert _differences([_run(), _run()], _run()) == []
    failed = _run("plan_failed")
    assert _differences([failed, failed], _run("plan_failed")) == []


def test_an_outcome_both_rendered_runs_share_must_be_the_outcome_without_images():
    found = _differences([_run(), _run()], _run("expert_failed"))
    assert found == ["outcome: ok on both rendered runs, expert_failed without images"]
    failed = _run("plan_failed")
    assert _differences([failed, failed], _run())[0].startswith("outcome: plan_failed")


def test_rendered_runs_that_disagree_on_the_outcome_leave_it_open():
    rendered = [_run(frames=47), _run("plan_failed")]
    assert _differences(rendered, _run("expert_failed")) == []
    assert _differences(rendered, _run(frames=48)) == []
    assert _differences([_run("plan_failed"), _run("expert_error")], _run()) == []


def test_frames_may_differ_by_the_rendered_spread_or_the_slack_whichever_is_larger():
    # The one run of the exact comparison: 47 frames rendered, 48 without.
    assert _differences([_run(frames=47), _run(frames=47)], _run(frames=48)) == []
    assert _differences([_run(frames=47), _run(frames=47)], _run(frames=45)) == []
    found = _differences([_run(frames=47), _run(frames=47)], _run(frames=50))
    assert found == ["frames: 50 without images, 47 and 47 rendered, more than 2 from the nearer"]
    assert _differences([_run(frames=44), _run(frames=48)], _run(frames=52)) == []
    assert _differences([_run(frames=44), _run(frames=48)], _run(frames=53))[0].startswith("frames")
    # One rendered run failed: the slack is measured from the one success.
    assert _differences([_run(frames=44), _run("plan_failed")], _run(frames=47))[0].startswith(
        "frames: 47 without images, 44 rendered"
    )


def test_rows_are_held_to_the_rendered_runs_difference_plus_the_margin():
    assert _differences([_run(), _run()], _run(shift=0.04)) == []
    found = _differences([_run(), _run()], _run(shift=0.06))
    assert len(found) == 1 and found[0].startswith("qpos: rows differ by up to 0.06")
    assert _differences([_run(), _run(shift=0.1)], _run(shift=0.12)) == []
    assert _differences([_run(), _run(shift=0.1)], _run(shift=0.3))[0].startswith("qpos")


def test_rows_are_not_compared_unless_all_three_runs_have_as_many_frames():
    assert _differences([_run(frames=44), _run(frames=45)], _run(frames=45, shift=1.0)) == []
    assert _differences([_run(frames=45), _run(frames=45)], _run(frames=44, shift=1.0)) == []
    assert _differences([_run(frames=44), _run(frames=45)], _run(frames=44, shift=1.0)) == []
    assert _differences([_run(frames=45), _run("plan_failed")], _run(frames=45, shift=1.0)) == []


def test_a_run_without_images_must_move_the_arms_every_rendered_success_moved():
    rendered = [_run(frames=44), _run(frames=45)]
    assert _differences(rendered, _run(frames=45, arms=("left",)))[:1] == [
        "arms: ('right',) on rendered run 1, ('left',) without"
    ]
    assert len(_differences(rendered, _run(arms=("left", "right")))) == 2
    assert _differences([_run(), _run("plan_failed")], _run(arms=("left",))) == [
        "arms: ('right',) on rendered run 1, ('left',) without"
    ]


def test_an_endpose_that_moves_without_images_is_a_difference_though_the_joints_agree():
    found = _differences([_run(), _run()], _run(endpose_shift=0.1))
    assert "left_endpose: rows differ by up to 0.1" in found[0]
    assert not any(line.startswith("qpos") for line in found)
    found = _differences([_run(), _run()], _run(endpose_keys=ENDPOSE_KEYS[::-1]))
    assert len(found) == 2 and all(line.startswith("endpose: keys") for line in found)
    assert _differences([_run(), _run()], _run(endpose_keys=()))[0].startswith("endpose: keys")
