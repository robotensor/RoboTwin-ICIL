"""The Same Scene protocol, end to end, against a fake RoboTwin env."""

import json

import numpy as np
import pytest

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import robotwin, tasks
from robotwin_icil.episode import EpisodeSpec, run_episode
from robotwin_icil.generate import policy_seed, scene_seeds
from robotwin_icil.policy import (
    DummyPolicy,
    ICILPolicy,
    PolicyError,
    ReplayEEPolicy,
    ReplayPolicy,
)
from robotwin_icil.records import SAME_SCENE, EpisodeRecord, Status


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


def test_replay_ee_succeeds_from_the_same_scene_through_the_ee_path():
    env = FakeTaskEnv()
    record = run_episode(spec(), ReplayEEPolicy(), FakeConfig(), task_env=env)
    assert record.status is Status.SCORED and record.success
    assert record.steps == env.expert_steps
    assert set(env.action_types) == {"ee"}


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

    def seed(self, seed):
        self.calls.append(("seed", seed))

    def reset(self):
        self.calls.append("reset")
        super().reset()

    def set_demonstration(self, demonstration):
        self.calls.append("demonstration")
        super().set_demonstration(demonstration)


def test_each_episode_seeds_and_resets_the_policy_then_gives_it_one_demonstration():
    policy = _Spy()
    first = run_episode(spec(0), policy, FakeConfig(), task_env=FakeTaskEnv())
    second = run_episode(spec(1), policy, FakeConfig(), task_env=FakeTaskEnv())
    seeds = [policy_seed(0, 0), policy_seed(0, 1)]
    assert policy.calls == [
        ("seed", seeds[0]),
        "reset",
        "demonstration",
        ("seed", seeds[1]),
        "reset",
        "demonstration",
    ]
    # Never the scene seed: that one rebuilds the scene, target and all.
    assert seeds[0] != first.scene_seed and seeds[1] != second.scene_seed


def test_a_policy_is_not_seeded_for_a_scene_it_never_sees():
    policy = _Spy()
    run_episode(spec(), policy, FakeConfig(), task_env=FakeTaskEnv(drift=True))
    assert policy.calls == []


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


def test_an_expert_that_raises_is_still_a_rejected_seed():
    env = FakeTaskEnv(expert_raises_on={scene_seeds(0, 0, 5)[0]})
    record = run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=env)
    assert record.status is Status.SCORED and record.rejections == {"expert_error": 1}
    assert "target_pose cannot be None" in record.rejection_details["expert_error"]


class _Watcher(ReplayPolicy):
    name = "watcher"

    def _reset(self):
        super()._reset()
        self.observations = []

    def _act(self, observation):
        self.observations.append(observation)
        return super()._act(observation)


def test_the_rollout_is_timed_in_physics_steps():
    # Each take_action runs four physics steps here; the evaluation scene's settle is not counted.
    env = FakeTaskEnv(physics_per_action=4)
    policy = _Watcher()
    record = run_episode(spec(), policy, FakeConfig(), task_env=env)
    assert record.status is Status.SCORED and record.success
    assert record.physics_steps == 4 * record.steps
    assert [o.time_s for o in policy.observations] == pytest.approx(
        [4 * k / 250 for k in range(record.steps)]
    )
    assert env.closed_while_clocked == 0


def test_episodes_without_a_rollout_ran_no_physics_steps():
    drifted = run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=FakeTaskEnv(drift=True))
    rejected = run_episode(
        spec(attempts=1),
        ReplayPolicy(),
        FakeConfig(),
        task_env=FakeTaskEnv(unstable_seeds=set(scene_seeds(0, 0, 1))),
    )
    assert drifted.status is Status.INVALID and rejected.status is Status.REJECTED
    assert drifted.physics_steps == rejected.physics_steps == 0


def test_observations_carry_the_measured_gripper_joints():
    policy = _Watcher()
    run_episode(spec(), policy, FakeConfig(), task_env=FakeTaskEnv())
    assert policy.observations
    for observation in policy.observations:
        assert set(observation.gripper_joints) == {"left", "right"}
        assert observation.gripper_joints["left"].shape == (2,)


class _Reporting(ReplayPolicy):
    """Reports what `info` holds after each rollout."""

    name = "reporting"

    def __init__(self, info):
        super().__init__()
        self.info = info
        self.asked = 0

    def episode_info(self):
        self.asked += 1
        return self.info


def test_what_a_policy_reports_is_recorded_with_its_episode():
    policy = _Reporting({"active_arm": "left", "chunks": 3, "clipped": (1, 2), "note": None})
    record = run_episode(spec(), policy, FakeConfig(), task_env=FakeTaskEnv())
    assert record.status is Status.SCORED and policy.asked == 1
    # Stored as JSON reads it back: the record round-trips unchanged.
    expected = {"active_arm": "left", "chunks": 3, "clipped": [1, 2], "note": None}
    assert record.policy_info == expected
    assert EpisodeRecord.from_json(json.loads(json.dumps(record.to_json()))) == record


def test_a_policy_that_reports_nothing_records_an_empty_mapping():
    record = run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=FakeTaskEnv())
    assert record.policy_info == {}


@pytest.mark.parametrize(
    "info, key",
    [({"arm": np.int64(1)}, "arm"), ({"ok": 1, "fraction": float("nan")}, "fraction")],
)
def test_a_report_that_is_not_json_stops_the_run_naming_the_key(info, key):
    with pytest.raises(PolicyError, match=f"episode_info\\(\\): '{key}' is not JSON"):
        run_episode(spec(), _Reporting(info), FakeConfig(), task_env=FakeTaskEnv())


def test_a_policy_is_asked_for_a_report_only_after_a_rollout():
    policy = _Reporting({"chunks": 3})
    record = run_episode(spec(), policy, FakeConfig(), task_env=FakeTaskEnv(drift=True))
    assert record.status is Status.INVALID and record.policy_info == {} and policy.asked == 0


def test_a_record_names_its_action_path_and_the_arms_the_demonstration_moved():
    env = FakeTaskEnv()
    qpos = run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=env)
    ee = run_episode(spec(), ReplayEEPolicy(), FakeConfig(), task_env=FakeTaskEnv())
    assert (qpos.action_type, ee.action_type) == ("qpos", "ee")
    # The fake expert drives every joint of both arms towards a random target.
    assert qpos.demonstration_arms == ee.demonstration_arms == ("left", "right")
    assert EpisodeRecord.from_json(json.loads(json.dumps(qpos.to_json()))) == qpos


def test_the_arms_are_those_arms_moved_reports_and_none_without_a_demonstration():
    recorded = []

    class _Keeper(ReplayPolicy):
        def _set_demonstration(self, demonstration):
            super()._set_demonstration(demonstration)
            recorded.append(demonstration.arms_moved())

    scored = run_episode(spec(), _Keeper(), FakeConfig(), task_env=FakeTaskEnv())
    assert scored.demonstration_arms == recorded[0]
    drifted = run_episode(spec(), ReplayPolicy(), FakeConfig(), task_env=FakeTaskEnv(drift=True))
    assert drifted.status is Status.INVALID and drifted.demonstration_arms == ("left", "right")
    rejected = run_episode(
        spec(attempts=1),
        ReplayEEPolicy(),
        FakeConfig(),
        task_env=FakeTaskEnv(unstable_seeds=set(scene_seeds(0, 0, 1))),
    )
    assert rejected.status is Status.REJECTED
    assert rejected.demonstration_arms is None and rejected.action_type == "ee"
