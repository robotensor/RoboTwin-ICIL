"""The Same Scene protocol, end to end, against a fake RoboTwin env."""

from types import SimpleNamespace

import numpy as np
import pytest

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import robotwin, tasks
from robotwin_icil.demo import Demonstration, Frame
from robotwin_icil.episode import EpisodeSpec, rollout, run_episode
from robotwin_icil.generate import scene_seeds
from robotwin_icil.policy import (
    DummyPolicy,
    EpisodeInfo,
    ICILPolicy,
    PolicyError,
    PolicyUnreachable,
    ReplayPolicy,
)
from robotwin_icil.records import SAME_SCENE, Status


@pytest.fixture(autouse=True)
def _fake_unstable(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)


def spec(episode=0, attempts=5, global_seed=0):
    return EpisodeSpec(
        episode=episode,
        task=tasks.table()["place_object_basket"],
        global_seed=global_seed,
        max_expert_attempts=attempts,
    )


@pytest.mark.parametrize("qpos_dim", [14, 16])
def test_replay_succeeds_from_the_same_scene(qpos_dim):
    # 14 is aloha-agilex, 16 a dual Franka: the loop takes its widths from the robot it is given.
    env = FakeTaskEnv(qpos_dim=qpos_dim)
    record = run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=env)
    assert record.status is Status.SCORED and record.success
    assert record.evaluation_setting == SAME_SCENE and record.skill_category == "pick_and_place"
    assert record.steps == env.expert_steps and record.demonstration_frames == env.expert_steps + 1
    assert record.scene_seed == scene_seeds(0, 0, 5)[0]
    assert record.embodiment == "fake-arms"
    assert robotwin.action_dims(env) == {"qpos": qpos_dim, "ee": 16}


def test_the_rollout_starts_from_a_rebuilt_scene_not_the_experts_final_state():
    env = FakeTaskEnv()
    record = run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=env)
    assert env.setups == [record.scene_seed, record.scene_seed]
    assert env.closed >= 2


def test_a_policy_that_ignores_the_demonstration_fails():
    record = run_episode(spec(), DummyPolicy(), FakeConfig(), task_env=FakeTaskEnv(step_lim=20))
    assert record.status is Status.SCORED and record.success is False
    assert record.steps == 20 and record.step_limit == 20


def test_expert_failures_are_rejections_not_model_failures():
    seeds = scene_seeds(0, 0, 5)
    env = FakeTaskEnv(
        unstable_seeds={seeds[0]}, plan_fails_on={seeds[1]}, expert_misses_on={seeds[2]}
    )
    record = run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=env)
    assert record.status is Status.SCORED and record.success
    assert record.scene_seed == seeds[3]
    assert record.expert_generation_attempts == 4
    assert record.rejections == {"expert_failed": 1, "plan_failed": 1, "unstable": 1}


def test_an_exhausted_budget_is_a_rejected_episode():
    env = FakeTaskEnv(unstable_seeds=set(scene_seeds(0, 0, 2)))
    record = run_episode(spec(attempts=2), ReplayPolicy(), FakeConfig(), task_env=env)
    assert record.status is Status.REJECTED and record.success is None and record.scene_seed is None
    assert record.rejections == {"unstable": 2}


def test_scene_drift_makes_the_episode_invalid_before_the_policy_acts():
    policy = ReplayPolicy()
    record = run_episode(spec(), policy, FakeConfig(), task_env=FakeTaskEnv(drift=True))
    assert record.status is Status.INVALID and record.success is None
    assert "cube" in record.detail
    assert record.scene_max_error == pytest.approx(0.01 * np.sqrt(3))
    assert (
        policy._demonstration is None
    )  # never handed a demonstration for a scene it would not see


class _Spy(ReplayPolicy):
    name = "spy"

    def __init__(self):
        super().__init__()
        self.calls = []
        self.episodes = []

    def reset(self, episode=None):
        self.calls.append("reset")
        self.episodes.append(episode)
        super().reset(episode)

    def set_demonstration(self, demonstration):
        self.calls.append("demonstration")
        super().set_demonstration(demonstration)


def test_each_episode_resets_the_policy_then_gives_it_one_demonstration():
    policy = _Spy()
    run_episode(spec(0), policy, FakeConfig(), task_env=FakeTaskEnv())
    run_episode(spec(1), policy, FakeConfig(), task_env=FakeTaskEnv(qpos_dim=16))
    assert policy.calls == ["reset", "demonstration", "reset", "demonstration"]
    # Told the robot and its live widths; a benchmark run draws no seed for the policy.
    assert policy.episodes == [
        EpisodeInfo(embodiment="fake-arms", action_dims={"qpos": 14, "ee": 16}, seed=None),
        EpisodeInfo(embodiment="fake-arms", action_dims={"qpos": 16, "ee": 16}, seed=None),
    ]


class _GoneAway(ReplayPolicy):
    name = "gone-away"

    def _act(self, observation):
        raise PolicyUnreachable("the policy is unreachable: act: the policy went away")


def test_a_policy_that_cannot_be_reached_is_never_a_failed_rollout():
    # Nobody's result: a benchmark run stops on it rather than record the episode as failed.
    env = FakeTaskEnv()
    with pytest.raises(PolicyUnreachable, match="went away"):
        run_episode(spec(), _GoneAway(), FakeConfig(), task_env=env)
    assert env.closed == 2


class _Broken(ICILPolicy):
    name = "broken"

    def _act(self, observation):
        return np.zeros(3)


def test_a_policy_breaking_the_protocol_stops_the_run():
    with pytest.raises(PolicyError):
        run_episode(spec(), _Broken(), FakeConfig(), task_env=FakeTaskEnv())


class _CannotLoad(ReplayPolicy):
    name = "cannot-load"

    def _set_demonstration(self, demonstration):
        raise ValueError("adapter could not load the demo")


def test_an_adapter_that_cannot_take_the_demonstration_stops_the_run():
    # Only run-unit scores a policy at fault as a failure; a benchmark run surfaces the bug.
    env = FakeTaskEnv()
    with pytest.raises(ValueError, match="could not load"):
        run_episode(spec(), _CannotLoad(), FakeConfig(), task_env=env)
    assert env.closed == 2


def test_a_simulator_error_mid_rollout_is_a_failed_rollout():
    record = run_episode(
        spec(), ReplayPolicy(), FakeConfig(), task_env=FakeTaskEnv(rollout_raises_at=2)
    )
    assert record.status is Status.SCORED and record.success is False
    assert record.detail.startswith("rollout error")


def test_a_robot_whose_widths_cannot_be_read_is_a_harness_bug_not_a_failed_rollout():
    # The widths are read off the arms before the policy acts; a seam that cannot read them would
    # otherwise score every episode as a "rollout error" failure, a stream of zeros to average.
    env = FakeTaskEnv()
    env.setup_demo(seed=scene_seeds(0, 0, 5)[0])
    env.robot = SimpleNamespace()  # no get_left_arm_jointState / get_right_arm_jointState
    policy = ReplayPolicy()
    policy.reset()
    frames = tuple(Frame(index, {}, np.zeros(14), {}) for index in range(2))
    policy.set_demonstration(Demonstration(frames=frames, frequency=1.0))
    with pytest.raises(AttributeError):
        rollout(env, policy)


def test_both_scenes_are_built_under_the_tasks_name():
    # RoboTwin looks the task's step limit up by name; an unnamed scene silently gets 1000 steps.
    env = FakeTaskEnv()
    run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=env)
    assert env.task_names == ["place_object_basket", "place_object_basket"]


def test_a_record_says_why_seeds_were_rejected():
    seeds = scene_seeds(0, 0, 5)
    env = FakeTaskEnv(unstable_seeds={seeds[0]})
    record = run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=env)
    assert record.rejection_details == {"unstable": f"objects unstable in seed {seeds[0]}"}


def test_a_scene_that_fails_to_build_stops_the_run():
    # RoboTwin can leave the env half-built after such a failure; recording it as a rejected
    # seed is how 320 seeds became "expert_error" in the first V1 run.
    env = FakeTaskEnv(setup_raises_on={scene_seeds(0, 0, 5)[0]})
    with pytest.raises(robotwin.RoboTwinError, match="planner failed to construct"):
        run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=env)


def test_running_out_of_gpu_memory_stops_the_run():
    env = FakeTaskEnv(oom_on_play={scene_seeds(0, 0, 5)[0]})
    with pytest.raises(robotwin.RoboTwinError, match="out of memory"):
        run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=env)


def test_losing_the_gpu_in_the_expert_stops_the_run():
    # Recorded as a rejection, a device loss once turned one run into 40 bogus expert_errors.
    env = FakeTaskEnv(device_lost_on_play={scene_seeds(0, 0, 5)[0]})
    with pytest.raises(robotwin.RoboTwinError, match="lost the GPU"):
        run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=env)


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        ("CUDA out of memory. Tried to allocate 20.00 MiB.", "ran out of memory"),
        ("vk::Device::waitForFences: ErrorDeviceLost", "lost the GPU"),
    ],
)
def test_a_gpu_failure_mid_rollout_stops_the_run_rather_than_failing_the_policy(error, reason):
    # The simulator failed, not the policy: a failure recorded against it would be a wrong score.
    env = FakeTaskEnv(rollout_raises_at=2, rollout_error=error)
    with pytest.raises(robotwin.RoboTwinError, match=f"{reason} during the rollout"):
        run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=env)
    assert env.closed == 2


def test_an_expert_that_raises_is_still_a_rejected_seed():
    env = FakeTaskEnv(expert_raises_on={scene_seeds(0, 0, 5)[0]})
    record = run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=env)
    assert record.status is Status.SCORED and record.rejections == {"expert_error": 1}
    assert "target_pose cannot be None" in record.rejection_details["expert_error"]
