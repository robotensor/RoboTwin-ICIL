"""The `ee` action path in the real simulator: how an idle arm holds, and what one call costs.

`take_action(action, action_type='ee')` plans both arms with CuRobo from their measured joints
and runs max(left_n, right_n) physics steps: at least 31 when a plan succeeds, 50 with that arm
not commanded when it fails. An adapter driving one arm must hold the other, and must size its
targets against CuRobo's goal tolerance. The numbers are printed (`pytest -s`) for those
decisions; only the fixed hold and the replay's record are asserted.
"""

import time

import numpy as np
import pytest

from robotwin_icil import robotwin, tasks
from robotwin_icil.episode import EpisodeSpec, run_episode
from robotwin_icil.policy import ReplayEEPolicy
from robotwin_icil.records import Status

pytestmark = pytest.mark.sim

TASK = "click_bell"  # the fastest expert
SEEDS = range(8)
HOLD_CALLS = 10
HOLD_TOLERANCE_RAD = 1e-3
# Tool-centre displacements probed, straight up. A pure translation moves the tool centre, 0.12 m
# along the flange's own +x axis on aloha-agilex, and the flange by the same vector.
PROBE_MM = (0.0, 1.0, 5.0, 10.0, 20.0)


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


def _stable_seed(task_env):
    for seed in SEEDS:
        if _build(task_env, seed) is not None:
            robotwin.close(task_env)
            return seed
    pytest.fail(f"{TASK}: every seed of {list(SEEDS)} was unstable")


def _ee_action(endpose) -> np.ndarray:
    """The 16 numbers `take_action('ee')` reads, from what `get_obs()` reports."""
    return np.concatenate(
        [
            endpose["left_endpose"],
            [endpose["left_gripper"]],
            endpose["right_endpose"],
            [endpose["right_gripper"]],
        ]
    ).astype(np.float64)


def _arm_joints(task_env) -> np.ndarray:
    """Both arms' measured joint positions, grippers left out: what drift is measured on."""
    robot = task_env.robot
    left = robot.get_left_arm_real_jointState()[:-1]
    right = robot.get_right_arm_real_jointState()[:-1]
    return np.asarray(left + right, dtype=np.float64)


def _hold(task_env, seed, echo):
    """The largest joint drift after each of HOLD_CALLS calls holding both arms still.

    The fixed hold sends the first observation's endposes every call; the echoed hold sends the
    endposes observed just before each call, so every plan re-bases on where the arm now is.
    """
    assert _build(task_env, seed) is not None
    try:
        start = _arm_joints(task_env)
        fixed = _ee_action(robotwin.observation(task_env)["endpose"])
        drift = []
        for _ in range(HOLD_CALLS):
            if echo:
                action = _ee_action(robotwin.observation(task_env)["endpose"])
            else:
                action = fixed
            task_env.take_action(action, action_type="ee")
            drift.append(float(np.abs(_arm_joints(task_env) - start).max()))
        return drift
    finally:
        robotwin.close(task_env)


def test_a_fixed_hold_keeps_both_arms_still():
    task_env = robotwin.load_task(TASK)
    seed = _stable_seed(task_env)
    fixed = _hold(task_env, seed, echo=False)
    echoed = _hold(task_env, seed, echo=True)
    print(f"\n{TASK} seed {seed}: max joint drift (rad) after each of {HOLD_CALLS} ee calls")
    print("  fixed hold:  " + " ".join(f"{d:.2e}" for d in fixed))
    print("  echoed hold: " + " ".join(f"{d:.2e}" for d in echoed))
    assert max(fixed) < HOLD_TOLERANCE_RAD, f"the fixed hold drifted {max(fixed):.2e} rad"


def test_probe_what_one_ee_call_costs():
    task_env = robotwin.load_task(TASK)
    seed = _stable_seed(task_env)
    assert _build(task_env, seed) is not None
    try:
        started = time.perf_counter()
        obs = robotwin.observation(task_env)
        observed_s = time.perf_counter() - started
        print(
            f"\n{TASK} seed {seed}: one get_obs() with cameras {sorted(obs['images'])} "
            f"took {observed_s * 1000:.1f} ms"
        )

        # Planning alone, every target from the first state, left arm.
        flange = np.asarray(obs["endpose"]["left_endpose"], dtype=np.float64)
        for mm in PROBE_MM:
            target = flange.copy()
            target[2] += mm / 1000
            started = time.perf_counter()
            result = task_env.robot.left_plan_path(target.tolist())
            planned_s = time.perf_counter() - started
            length = len(result["position"]) if result["status"] == "Success" else None
            print(
                f"  plan +{mm:4.1f} mm: CuRobo {result['status']}, {length} waypoints, "
                f"{planned_s * 1000:.1f} ms"
            )

        # Executed: one call per target from wherever the last left the arm, the right arm
        # held at its first-observation endpose.
        hold = _ee_action(obs["endpose"])
        for mm in PROBE_MM:
            action = hold.copy()
            action[:7] = robotwin.observation(task_env)["endpose"]["left_endpose"]
            action[2] += mm / 1000
            started = time.perf_counter()
            with robotwin.clock(task_env) as ticks:
                task_env.take_action(action, action_type="ee")
            acted_s = time.perf_counter() - started
            print(
                f"  take_action +{mm:4.1f} mm: {ticks.steps} physics steps, {acted_s * 1000:.1f} ms"
            )
    finally:
        robotwin.close(task_env)


class _Kept(ReplayEEPolicy):
    """replay_ee, keeping the demonstration it was handed for the printout."""

    def _set_demonstration(self, demonstration):
        super()._set_demonstration(demonstration)
        self.demonstration = demonstration


def test_replay_ee_runs_an_episode_to_a_record():
    policy = _Kept()
    spec = EpisodeSpec(episode=0, task=tasks.table()[TASK], global_seed=42, max_expert_attempts=5)
    record = run_episode(spec, policy, robotwin.SceneConfig())
    print(
        f"\nreplay_ee on {TASK}: {record.status.value}, success {record.success}, "
        f"{record.steps}/{record.step_limit} calls, {record.physics_steps} physics steps, "
        f"{record.demonstration_frames} demonstration frames, "
        f"arms moved {policy.demonstration.arms_moved()}, {record.duration_s} s; {record.detail}"
    )
    assert record.status is Status.SCORED, record.detail
    assert record.steps > 0 and record.physics_steps > 0
