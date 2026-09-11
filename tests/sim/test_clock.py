"""The physics clock in the real simulator: when RoboTwin's expert takes its frames.

`take_dense_action` records a frame before its first physics step, after every `save_freq`-th
step from the first, and after its last; `together_move_to_pose` does the same in a loop of its
own. A primitive of n steps is therefore framed at its steps 0, 1, 1 + save_freq, ..., n, and the
next primitive's first frame duplicates this one's last. The clock must see exactly that, and
change nothing it watches.
"""

import contextlib

import numpy as np
import pytest

from robotwin_icil import generate, robotwin, scene

pytestmark = pytest.mark.sim

TASK = "click_bell"  # the fastest expert: three moves of several primitives each
SEEDS = range(8)


def _build(task_env, seed):
    """Build one seed's scene; None, with the env closed, when it does not settle."""
    try:
        task_env.setup_demo(
            now_ep_num=0, seed=seed, is_test=True, **robotwin.SceneConfig().resolve(TASK)
        )
    except robotwin.unstable_error():
        robotwin.close(task_env)
        return None
    return task_env


def test_the_experts_frames_follow_the_capture_pattern():
    config = robotwin.SceneConfig()
    task_env = robotwin.load_task(TASK)
    for seed in SEEDS:
        _, demonstration, initial = generate.attempt(
            task_env, seed, config.resolve(TASK), config.save_freq, 0
        )
        if demonstration is not None:
            break
    else:
        pytest.fail(f"{TASK}: the expert solved none of seeds {list(SEEDS)}")

    timestep = 1.0 / (demonstration.frequency * config.save_freq)
    steps = demonstration.times() / timestep
    np.testing.assert_allclose(steps, np.rint(steps), atol=1e-6)  # frames fall between steps
    steps = np.rint(steps).astype(int)
    gaps = np.diff(steps)
    assert steps[0] == 0, "the scene's settle is not part of the demonstration's time"
    assert gaps.min() == 0 and gaps.max() == config.save_freq
    # Per primitive: one frame a single step in, then save_freq apart, a remainder, and the next
    # primitive's first frame on top of its last.
    assert np.count_nonzero(gaps == config.save_freq) > len(gaps) / 2
    assert np.count_nonzero(gaps == 1) >= 2
    assert np.count_nonzero(gaps == 0) >= 2
    for i in np.flatnonzero(gaps == 0):
        # A zero gap is a frame taken with no physics step since the last: the same state.
        np.testing.assert_array_equal(
            demonstration.frames[i].qpos, demonstration.frames[i + 1].qpos
        )

    # The clock is gone with the expert's scene: rebuilding the seed is the scene it started in.
    assert _build(task_env, seed) is not None
    try:
        mismatches = scene.compare(initial, robotwin.fingerprint(task_env))
    finally:
        robotwin.close(task_env)
    assert mismatches == [], "; ".join(map(str, mismatches))


def _acted(task_env, seed, clocked):
    """One seed's fingerprints before and after three fixed actions, and the clock if any."""
    if _build(task_env, seed) is None:
        return None
    try:
        before = robotwin.fingerprint(task_env)
        target = robotwin.observation(task_env)["qpos"].copy()
        target[[0, 7]] += 0.05  # the first joint of each arm
        with contextlib.ExitStack() as stack:
            ticks = stack.enter_context(robotwin.clock(task_env)) if clocked else None
            for _ in range(3):
                task_env.take_action(target, action_type="qpos")
        assert "step" not in vars(task_env.scene), "the clock outlived its block"
        return before, robotwin.fingerprint(task_env), ticks
    finally:
        robotwin.close(task_env)


def test_the_clock_changes_no_scene():
    task_env = robotwin.load_task(TASK)
    for seed in SEEDS:
        plain = _acted(task_env, seed, clocked=False)
        clocked = _acted(task_env, seed, clocked=True)
        if plain is None or clocked is None:
            assert plain is clocked is None, f"seed {seed}: only one build settled"
            continue
        assert scene.compare(plain[0], plain[1]) != [], "the actions moved nothing"
        for without, with_clock in zip(plain[:2], clocked[:2], strict=True):
            mismatches = scene.compare(without, with_clock)
            assert mismatches == [], f"seed {seed}: " + "; ".join(map(str, mismatches))
        assert clocked[2].steps > 0
        return
    pytest.fail(f"{TASK}: every seed was unstable, nothing was checked")
