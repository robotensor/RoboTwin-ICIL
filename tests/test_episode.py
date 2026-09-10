"""The Same Scene protocol, end to end, against a fake RoboTwin env."""

import numpy as np
import pytest

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import robotwin, tasks
from robotwin_icil.episode import EpisodeSpec, run_episode
from robotwin_icil.generate import scene_seeds
from robotwin_icil.policy import DummyPolicy, ICILPolicy, PolicyError, ReplayPolicy
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


def test_replay_succeeds_from_the_same_scene():
    env = FakeTaskEnv()
    record = run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=env)
    assert record.status is Status.SCORED and record.success
    assert record.evaluation_setting == SAME_SCENE and record.skill_category == "pick_and_place"
    assert record.steps == env.expert_steps and record.demonstration_frames == env.expert_steps + 1
    assert record.scene_seed == scene_seeds(0, 0, 5)[0]


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

    def reset(self):
        self.calls.append("reset")
        super().reset()

    def set_demonstration(self, demonstration):
        self.calls.append("demonstration")
        super().set_demonstration(demonstration)


def test_each_episode_resets_the_policy_then_gives_it_one_demonstration():
    policy = _Spy()
    run_episode(spec(0), policy, FakeConfig(), task_env=FakeTaskEnv())
    run_episode(spec(1), policy, FakeConfig(), task_env=FakeTaskEnv())
    assert policy.calls == ["reset", "demonstration", "reset", "demonstration"]


class _Broken(ICILPolicy):
    name = "broken"

    def _act(self, observation):
        return np.zeros(3)


def test_a_policy_breaking_the_protocol_stops_the_run():
    with pytest.raises(PolicyError):
        run_episode(spec(), _Broken(), FakeConfig(), task_env=FakeTaskEnv())


def test_a_simulator_error_mid_rollout_is_a_failed_rollout():
    record = run_episode(
        spec(), ReplayPolicy(), FakeConfig(), task_env=FakeTaskEnv(rollout_raises_at=2)
    )
    assert record.status is Status.SCORED and record.success is False
    assert record.detail.startswith("rollout error")
