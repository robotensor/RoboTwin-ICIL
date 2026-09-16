"""A demonstration is built once and saved, and an episode is evaluated from the saved file."""

import dataclasses
import json
import os

import numpy as np
import pytest

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import prompt, robotwin, scene, unit
from robotwin_icil.demo import Demonstration
from robotwin_icil.policy import (
    EpisodeInfo,
    ICILPolicy,
    PolicyUnreachable,
    ReplayPolicy,
    Unscorable,
)
from robotwin_icil.records import git_commit

pytest.importorskip("imageio_ffmpeg")

TASK = "click_bell"
SEED = 11


@pytest.fixture(autouse=True)
def _fake_sim(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "SceneConfig", FakeConfig)


def assert_read_result_shape(result):
    """The fields the orchestrator's `read_result` promises, for either command's result.json."""
    assert {"success", "void", "steps", "error"} <= set(result)
    assert isinstance(result["void"], bool)
    assert (result["success"] is None) == result["void"]
    if result["void"]:
        assert result["steps"] is None and isinstance(result["error"], str) and result["error"]
        assert result["void_cause"] in ("harness", "policy")
    else:
        assert isinstance(result["success"], bool) and isinstance(result["steps"], int)
        assert result["void_cause"] is None


def materialized(tmp_path, env=None, **config):
    out = tmp_path / "prompt"
    done = unit.materialize(TASK, SEED, FakeConfig(**config), out, task_env=env or FakeTaskEnv())
    return done, out


def test_materialize_writes_the_prompt_the_clip_and_the_result(tmp_path):
    env = FakeTaskEnv()
    done, out = materialized(tmp_path, env, save_freq=5)
    assert done.ok and [a.rejection for a in done.generated.attempts] == [None]
    assert {p.name for p in out.iterdir()} == {"prompt.npz", "demonstration.mp4", "result.json"}
    assert env.setups == [SEED] and env.closed == 1

    result = unit.read_result(out)
    assert result == done.result
    assert result["ok"] is True and result["task"] == TASK and result["scene_seed"] == SEED
    assert result["frames"] == env.expert_steps + 1 and result["cameras"] == ["head_camera"]
    assert result["prompt_sha256"] == prompt.sha256_of(out / "prompt.npz")
    assert result["video"] == "demonstration.mp4" and result["embodiment"] == "fake-arms"
    assert_read_result_shape(result)
    assert result["success"] is True and result["void"] is False and result["error"] is None
    assert result["steps"] == env.expert_steps  # the demonstration's actions

    demonstration, meta = prompt.read_prompt(out / "prompt.npz")
    assert len(demonstration) == env.expert_steps + 1 and demonstration.timed
    assert meta["task"] == TASK and meta["scene_seed"] == SEED
    assert meta["embodiment"] == {
        "name": "fake-arms",
        "robotwin": ["fake-arms"],
        "choice": "fake-arms",
    }
    assert meta["task_config"] == "fake" and meta["save_freq"] == 5
    assert meta["head_camera"] is None and meta["overrides"] == {}
    assert meta["frames"] == len(demonstration) and meta["cameras"] == ["head_camera"]
    assert meta["expert"] == {
        "scene_seeds": [SEED],
        "attempts": [{"seed": SEED, "rejection": None, "detail": ""}],
        "rejections": {},
    }
    assert result["scene_seeds"] == [SEED] and result["attempts"] == meta["expert"]["attempts"]
    assert result["rejections"] == {} and result["void_cause"] is None
    assert meta["benchmark_commit"] and meta["robotwin_commit"]
    recorded = scene.SceneFingerprint.from_json(meta["scene"]["fingerprint"])
    assert scene.digest(recorded) == meta["scene"]["sha256"] == result["scene_sha256"]
    assert scene.compare(recorded, done.initial) == []


def test_a_rejected_seed_writes_its_rejection_and_no_prompt(tmp_path):
    out = tmp_path / "prompt"
    out.mkdir()
    (out / "prompt.npz").write_bytes(b"stale")  # from an earlier command in the same directory
    done = unit.materialize(
        TASK, SEED, FakeConfig(), out, task_env=FakeTaskEnv(unstable_seeds={SEED})
    )
    assert not done.ok and done.demonstration is None
    assert {p.name for p in out.iterdir()} == {"result.json"}
    result = unit.read_result(out)
    assert result["ok"] is False and result["rejection"] == "unstable"
    assert result["detail"] == f"objects unstable in seed {SEED}"
    assert result["task"] == TASK and result["scene_seeds"] == [SEED]
    assert result["attempts"] == [
        {"seed": SEED, "rejection": "unstable", "detail": f"objects unstable in seed {SEED}"}
    ]
    # No seed was chosen: the attempts say which one was tried.
    assert result["scene_seed"] is None and result["rejections"] == {"unstable": 1}
    # Read as the orchestrator reads any result: void, with the rejection as the reason, and the
    # harness's - the expert's - to answer for, never a policy's.
    assert_read_result_shape(result)
    assert result["void"] is True and result["success"] is None and result["steps"] is None
    assert result["void_cause"] == "harness"
    assert result["error"] == f"expert rejected the seed: unstable: objects unstable in seed {SEED}"


def test_an_expert_that_fails_is_a_rejection_too(tmp_path):
    done, out = materialized(tmp_path, FakeTaskEnv(expert_misses_on={SEED}))
    assert not done.ok and unit.read_result(out)["rejection"] == "expert_failed"


def test_a_broken_simulator_is_an_error_not_a_rejection(tmp_path):
    with pytest.raises(robotwin.RoboTwinError, match="planner failed to construct"):
        materialized(tmp_path, FakeTaskEnv(broken_setup=True))


def test_materialize_tries_candidates_in_order_and_keeps_the_first_the_expert_solves(tmp_path):
    env = FakeTaskEnv(unstable_seeds={3}, expert_misses_on={5})
    out = tmp_path / "prompt"
    done = unit.materialize(TASK, [3, 5, 8, 13], FakeConfig(), out, task_env=env)
    assert done.ok and done.generated.seed == 8
    # 13 is never built: the first success ends the search.
    assert env.setups == [3, 5, 8] and env.closed == 3
    assert {p.name for p in out.iterdir()} == {"prompt.npz", "demonstration.mp4", "result.json"}

    result = unit.read_result(out)
    assert_read_result_shape(result)
    assert result["ok"] and result["success"] and result["scene_seed"] == 8
    assert result["scene_seeds"] == [3, 5, 8, 13]
    assert result["attempts"] == [
        {"seed": 3, "rejection": "unstable", "detail": "objects unstable in seed 3"},
        {"seed": 5, "rejection": "expert_failed", "detail": ""},
        {"seed": 8, "rejection": None, "detail": ""},
    ]
    assert result["rejections"] == {"expert_failed": 1, "unstable": 1}

    # The prompt is the chosen seed's, and meta records how it was found.
    _, meta = prompt.read_raw(out / "prompt.npz")
    assert meta["scene_seed"] == 8
    assert meta["expert"] == {
        "scene_seeds": [3, 5, 8, 13],
        "attempts": result["attempts"],
        "rejections": result["rejections"],
    }
    run = unit.run_unit(
        out / "prompt.npz", ReplayPolicy(), tmp_path / "run", task_env=FakeTaskEnv()
    )
    assert run["success"] is True and run["scene_seed"] == 8


def test_every_candidate_rejected_is_a_void_harness_result_naming_each(tmp_path):
    env = FakeTaskEnv(unstable_seeds={3}, plan_fails_on={5})
    out = tmp_path / "prompt"
    done = unit.materialize(TASK, [3, 5], FakeConfig(), out, task_env=env)
    assert not done.ok and done.demonstration is None and env.setups == [3, 5]
    assert {p.name for p in out.iterdir()} == {"result.json"}

    result = unit.read_result(out)
    assert_read_result_shape(result)
    assert result["ok"] is False and result["void"] is True and result["success"] is None
    assert result["void_cause"] == "harness" and result["scene_seed"] is None
    assert result["rejections"] == {"plan_failed": 1, "unstable": 1}
    assert [a["seed"] for a in result["attempts"]] == [3, 5]
    assert result["error"] == (
        "expert rejected all 2 candidate seeds: "
        "seed 3: unstable: objects unstable in seed 3; seed 5: plan_failed"
    )
    # The last candidate's rejection, where a single seed's result has always kept it.
    assert result["rejection"] == "plan_failed" and result["detail"] == ""


@pytest.mark.parametrize(
    ("seeds", "reason"),
    [
        ([], "at least one candidate"),
        ([3, 5, 3], "scene seed 3 is given more than once"),
        ([-1], "not an integer in [0, 2**32)"),
        ([2**32], "not an integer in [0, 2**32)"),
        ([True], "not an integer"),
    ],
)
def test_a_candidate_list_it_must_not_try_is_refused_before_any_scene(tmp_path, seeds, reason):
    env = FakeTaskEnv()
    with pytest.raises(unit.UnitError) as refused:
        unit.materialize(TASK, seeds, FakeConfig(), tmp_path / "prompt", task_env=env)
    assert reason in str(refused.value) and env.setups == []


def test_run_unit_succeeds_with_replay_from_the_file(tmp_path):
    _, out = materialized(tmp_path)
    env = FakeTaskEnv()
    result = unit.run_unit(out / "prompt.npz", ReplayPolicy(), tmp_path / "run", task_env=env)
    assert result["success"] is True and result["void"] is False and result["error"] is None
    assert result["steps"] == env.expert_steps and result["step_limit"] == env.step_lim
    assert result["scene_max_error"] == 0.0 and result["model"] == "replay"
    assert result["policy"] == "replay" and result["served_policy"] is None
    assert result["embodiment"] == "fake-arms" and result["task"] == TASK
    assert result["scene_seed"] == SEED and result["evaluation_setting"] == "same_scene"
    assert result["prompt_sha256"] == prompt.sha256_of(out / "prompt.npz")
    assert result["video"] == "evaluation.mp4"
    assert (tmp_path / "run" / "evaluation.mp4").is_file()
    assert unit.read_result(tmp_path / "run") == result
    assert_read_result_shape(result)
    # The scene was rebuilt from meta, under the task's name, and closed afterwards.
    assert env.setups == [SEED] and env.task_names == [TASK] and env.closed == 1
    # What evaluated it, as every episode records: the checkpoint and both commits.
    assert result["checkpoint"] is None
    assert result["benchmark_commit"] == git_commit(robotwin.REPO_ROOT)
    assert result["robotwin_commit"] == git_commit(robotwin.ROBOTWIN_ROOT)
    assert result["benchmark_commit"] and result["robotwin_commit"]


def test_run_unit_records_the_policys_checkpoint_even_when_void(tmp_path):
    class Checkpointed(ReplayPolicy):
        name = "checkpointed"

        def describe(self):
            return {**super().describe(), "model": "icrt", "checkpoint": "/ckpt/icrt.pt"}

    result = unit.run_unit(tmp_path / "missing.npz", Checkpointed(), tmp_path / "run")
    assert result["void"] and result["model"] == "icrt"
    assert result["checkpoint"] == "/ckpt/icrt.pt" and result["benchmark_commit"]


def test_a_policy_that_fails_is_a_failure_not_a_void(tmp_path):
    _, out = materialized(tmp_path)

    class Still(ICILPolicy):
        name = "still"

        def _act(self, observation):
            return observation.qpos

    result = unit.run_unit(
        out / "prompt.npz", Still(), tmp_path / "run", task_env=FakeTaskEnv(step_lim=8)
    )
    assert result["success"] is False and result["void"] is False and result["steps"] == 8


def test_two_runs_from_one_prompt_rebuild_the_same_scene(tmp_path):
    _, out = materialized(tmp_path)
    _, meta = prompt.read_raw(out / "prompt.npz")
    results = [
        unit.run_unit(
            out / "prompt.npz", ReplayPolicy(), tmp_path / f"run{i}", task_env=FakeTaskEnv()
        )
        for i in range(2)
    ]
    assert all(r["success"] and not r["void"] and r["scene_max_error"] == 0.0 for r in results)
    # What each run saw is on its result: the rebuilt scene's digest, not only the verdict.
    recorded = meta["scene"]["sha256"]
    assert [r["scene_sha256"] for r in results] == [recorded, recorded]
    assert [r["live_scene_sha256"] for r in results] == [recorded, recorded]


def test_a_rebuild_within_tolerance_reports_how_far_it_was(tmp_path):
    # Not bit-identical but inside the tolerance: scored, and the result says by how much rather
    # than a hard-coded 0.0, with the rebuilt scene's own digest.
    _, out = materialized(tmp_path)

    def nudge(meta):
        meta["scene"]["fingerprint"]["actors"]["cube"][0] += 3e-6
        recorded = scene.SceneFingerprint.from_json(meta["scene"]["fingerprint"])
        meta["scene"]["sha256"] = scene.digest(recorded)

    _retag(out / "prompt.npz", nudge)
    result = unit.run_unit(
        out / "prompt.npz", ReplayPolicy(), tmp_path / "run", task_env=FakeTaskEnv()
    )
    assert result["void"] is False and result["success"] is True
    assert result["scene_max_error"] == pytest.approx(3e-6, rel=1e-3)
    assert result["live_scene_sha256"] != result["scene_sha256"]


def _retag(path, edit):
    """Rewrite a prompt with its meta edited, as tampering would."""
    arrays, meta = prompt.read_raw(path)
    edit(meta)
    np.savez_compressed(path, **arrays, meta=json.dumps(meta, sort_keys=True))


def test_run_unit_voids_on_a_tampered_digest(tmp_path):
    _, out = materialized(tmp_path)
    original = json.loads(json.dumps(prompt.read_raw(out / "prompt.npz")[1]))

    def forge(meta):
        meta["scene"]["sha256"] = "0" * 64

    _retag(out / "prompt.npz", forge)
    env, policy = FakeTaskEnv(), ReplayPolicy()
    result = unit.run_unit(out / "prompt.npz", policy, tmp_path / "run", task_env=env)
    assert result["void"] is True and result["success"] is None and result["steps"] is None
    assert "inconsistent" in result["error"] and original["scene"]["sha256"][:12] in result["error"]
    assert env.setups == [] and policy._demonstration is None  # nothing built, nothing handed over
    assert unit.read_result(tmp_path / "run")["void"] is True


def test_run_unit_voids_when_the_recorded_scene_was_edited(tmp_path):
    # Moving a recorded pose without recomputing the digest is caught the same way; recomputing
    # it is caught by the live fingerprint, below.
    _, out = materialized(tmp_path)

    def move(meta):
        meta["scene"]["fingerprint"]["actors"]["cube"][0] += 0.05

    _retag(out / "prompt.npz", move)
    result = unit.run_unit(
        out / "prompt.npz", ReplayPolicy(), tmp_path / "run", task_env=FakeTaskEnv()
    )
    assert result["void"] and "inconsistent" in result["error"]


def test_run_unit_voids_on_scene_drift(tmp_path):
    env = FakeTaskEnv(drift=True)  # the second build of a seed moves the cube
    _, out = materialized(tmp_path, env)
    policy = ReplayPolicy()
    result = unit.run_unit(out / "prompt.npz", policy, tmp_path / "run", task_env=env)
    assert result["void"] is True and result["success"] is None and result["steps"] is None
    assert result["error"].startswith("scene drift") and "cube" in result["error"]
    assert result["scene_max_error"] == pytest.approx(0.01 * np.sqrt(3))
    assert result["live_scene_sha256"] not in (None, result["scene_sha256"])
    assert policy._demonstration is None
    assert result["video"] == "evaluation.mp4"  # the drifted first frame, as evidence


def test_a_prompt_from_another_scene_is_caught_by_the_live_fingerprint(tmp_path):
    # A consistent meta (fingerprint and digest agree) for a scene that is not the one the seed
    # builds: only the rebuilt scene can catch it.
    _, out = materialized(tmp_path)

    def move(meta):
        meta["scene"]["fingerprint"]["actors"]["cube"][0] += 0.05
        recorded = scene.SceneFingerprint.from_json(meta["scene"]["fingerprint"])
        meta["scene"]["sha256"] = scene.digest(recorded)

    _retag(out / "prompt.npz", move)
    result = unit.run_unit(
        out / "prompt.npz", ReplayPolicy(), tmp_path / "run", task_env=FakeTaskEnv()
    )
    assert result["void"] and result["error"].startswith("scene drift")
    assert result["scene_max_error"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (lambda meta: meta.pop("scene"), "no 'scene'"),
        (lambda meta: meta.__setitem__("embodiment", "aloha-agilex"), "embodiment has no name"),
        (lambda meta: meta["embodiment"].pop("choice"), "embodiment has no choice"),
        (lambda meta: meta["embodiment"].__setitem__("choice", 3), "embodiment choice is 3"),
        (lambda meta: meta.__setitem__("scene_seed", "11"), "not an integer"),
        (lambda meta: meta.__setitem__("save_freq", "fifteen"), "malformed"),
        (lambda meta: meta["scene"].__setitem__("fingerprint", {"actors": {}}), "malformed"),
        (lambda meta: meta.__setitem__("task", "nope"), "task 'nope' is not one"),
        (lambda meta: meta.__setitem__("task", 5), "task 5 is not one"),
    ],
)
def test_a_malformed_meta_voids_the_unit_before_any_scene(tmp_path, edit, reason):
    _, out = materialized(tmp_path)
    _retag(out / "prompt.npz", edit)
    env = FakeTaskEnv()
    result = unit.run_unit(out / "prompt.npz", ReplayPolicy(), tmp_path / "run", task_env=env)
    assert result["void"] and result["success"] is None and reason in result["error"]
    assert env.setups == []


def test_a_config_the_simulator_refuses_voids_the_unit(tmp_path, monkeypatch):
    # A meta whose config RoboTwin will not take (an unknown robot, an override it refuses) is
    # the prompt's fault, not a harness error that ends the command.
    _, out = materialized(tmp_path)

    def refuse(**fields):
        raise robotwin.RoboTwinError("unknown embodiment 'ur5-wsg'")

    monkeypatch.setattr(robotwin, "SceneConfig", refuse)
    env = FakeTaskEnv()
    result = unit.run_unit(out / "prompt.npz", ReplayPolicy(), tmp_path / "run", task_env=env)
    assert result["void"] and "unknown embodiment 'ur5-wsg'" in result["error"]
    assert env.setups == []


def test_an_unreadable_prompt_voids_the_unit(tmp_path):
    result = unit.run_unit(tmp_path / "missing.npz", ReplayPolicy(), tmp_path / "run")
    assert result["void"] and result["error"].startswith("unreadable prompt")
    assert result["task"] is None and (tmp_path / "run" / "result.json").is_file()


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda path: path.write_bytes(b""), id="empty"),
        pytest.param(
            lambda path: path.write_bytes(path.read_bytes()[: path.stat().st_size // 2]),
            id="truncated",
        ),
        pytest.param(
            lambda path: _rewrite(path, lambda a: {**a, "frequency": np.asarray([1.0, 2.0])}),
            id="non-scalar-frequency",
        ),
    ],
)
def test_a_corrupt_prompt_voids_the_unit(tmp_path, corrupt):
    _, out = materialized(tmp_path)
    corrupt(out / "prompt.npz")
    env = FakeTaskEnv()
    result = unit.run_unit(out / "prompt.npz", ReplayPolicy(), tmp_path / "run", task_env=env)
    assert result["void"] and result["error"].startswith("unreadable prompt")
    assert unit.read_result(tmp_path / "run") == result and env.setups == []


def _rewrite(path, edit):
    """Rewrite a prompt with its arrays edited and its meta untouched."""
    arrays, meta = prompt.read_raw(path)
    np.savez_compressed(path, **edit(arrays), meta=json.dumps(meta, sort_keys=True))


def test_a_gpu_failure_mid_rollout_voids_the_unit(tmp_path, monkeypatch):
    # Who held the GPU goes on the result: a served policy sharing the device could have filled
    # it, which only whoever started the policy can tell from these process ids.
    held = [{"pid": os.getpid(), "used_mib": 5120}, {"pid": 4242, "used_mib": 23000}]
    monkeypatch.setattr(robotwin, "gpu_processes", lambda: held)
    _, out = materialized(tmp_path)
    env = FakeTaskEnv(
        rollout_raises_at=2, rollout_error="vk::Device::waitForFences: ErrorDeviceLost"
    )
    result = unit.run_unit(out / "prompt.npz", ReplayPolicy(), tmp_path / "run", task_env=env)
    assert_read_result_shape(result)
    assert result["void"] is True and "lost the GPU during the rollout" in result["error"]
    assert result["void_cause"] == "harness"
    assert result["gpu_processes"] == held and result["run_unit_pid"] == os.getpid()
    assert unit.read_result(tmp_path / "run") == result and env.closed == 1


def test_a_unit_that_did_not_fail_on_the_gpu_records_no_gpu_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(robotwin, "gpu_processes", lambda: pytest.fail("not a GPU failure"))
    _, out = materialized(tmp_path)
    result = unit.run_unit(
        out / "prompt.npz", ReplayPolicy(), tmp_path / "run", task_env=FakeTaskEnv()
    )
    assert result["success"] is True and "gpu_processes" not in result


def test_a_harness_fault_while_evaluating_voids_the_unit_with_its_reason(tmp_path, monkeypatch):
    # Not a RoboTwinError and not the policy's doing: the harness failed to fingerprint the rebuilt
    # scene. The unit is void and result.json says why, rather than the command dying without one.
    _, out = materialized(tmp_path)

    def broken(env):
        raise AttributeError("'NoneType' object has no attribute 'get_pose'")

    monkeypatch.setattr(robotwin, "fingerprint", broken)
    env, policy = FakeTaskEnv(), _Recorder()
    result = unit.run_unit(out / "prompt.npz", policy, tmp_path / "run", task_env=env)
    assert_read_result_shape(result)
    assert result["void"] is True and result["error"] == (
        "harness error while evaluating: AttributeError: "
        "'NoneType' object has no attribute 'get_pose'"
    )
    assert result["scene_sha256"] and result["live_scene_sha256"] is None
    assert unit.read_result(tmp_path / "run") == result
    assert env.setups == [SEED] and env.closed == 1 and policy.demonstrations == []


class _Faulty(ReplayPolicy):
    """A replay policy that breaks in one named place."""

    name = "faulty"

    def __init__(self, where):
        super().__init__()
        self.where = where

    def _reset(self):
        super()._reset()
        if self.where == "reset":
            raise RuntimeError("weights missing")

    def _set_demonstration(self, demonstration):
        if self.where == "set_demonstration":
            raise ValueError("adapter could not load the demo")
        super()._set_demonstration(demonstration)

    def _act(self, observation):
        action = super()._act(observation)
        if self.where == "act-width" and self._cursor == 3:
            return action[:3]
        if self.where == "act-nan" and self._cursor == 3:
            return np.full_like(action, np.nan)
        return action


@pytest.mark.parametrize(
    ("where", "steps", "reason"),
    [
        ("reset", 0, "policy failed before acting: RuntimeError: weights missing"),
        ("set_demonstration", 0, "policy failed before acting: ValueError: adapter could not"),
        ("act-width", 2, "policy broke the protocol: faulty: act() returned shape (1, 3)"),
        ("act-nan", 2, "policy broke the protocol: faulty: act() returned a non-finite action"),
    ],
)
def test_a_policy_at_fault_fails_the_unit_rather_than_voiding_it(tmp_path, where, steps, reason):
    # A void unit is thrown out of the score; a policy that could void a unit by raising or by
    # returning junk could throw out the units it is losing. What the policy did is its result.
    _, out = materialized(tmp_path)
    env = FakeTaskEnv()
    result = unit.run_unit(out / "prompt.npz", _Faulty(where), tmp_path / "run", task_env=env)
    assert_read_result_shape(result)
    assert result["success"] is False and result["void"] is False and result["error"] is None
    assert result["steps"] == steps and result["detail"].startswith(reason)
    assert unit.read_result(tmp_path / "run") == result and env.closed == 1


class _Unreachable(ReplayPolicy):
    """A replay policy whose connection fails in one named place, with `error`."""

    name = "unreachable"

    def __init__(self, where, error=PolicyUnreachable):
        super().__init__()
        self.where, self.error = where, error

    def _reset(self):
        super()._reset()
        if self.where == "reset":
            raise self.error("the policy is unreachable: connect: nothing listened")

    def _set_demonstration(self, demonstration):
        if self.where == "set_demonstration":
            raise self.error("the policy is unreachable: prompt: no answer within 1s")
        super()._set_demonstration(demonstration)

    def _act(self, observation):
        action = super()._act(observation)
        if self.where == "act" and self._cursor == 3:
            raise self.error("the policy is unreachable: act: the policy went away")
        return action


@pytest.mark.parametrize("where", ["reset", "set_demonstration", "act"])
def test_a_policy_that_cannot_be_reached_voids_the_unit_on_the_policy(tmp_path, where):
    # Wherever the connection is lost, nothing the policy did was scored and the harness did
    # nothing wrong: void, charged to the policy, with the reason.
    _, out = materialized(tmp_path)
    env = FakeTaskEnv()
    result = unit.run_unit(out / "prompt.npz", _Unreachable(where), tmp_path / "run", task_env=env)
    assert_read_result_shape(result)
    assert result["void"] is True and result["void_cause"] == "policy"
    assert result["success"] is None and result["steps"] is None
    assert result["error"].startswith("the policy is unreachable: ")
    assert unit.read_result(tmp_path / "run") == result and env.closed == 1


def test_an_unscorable_harness_fault_in_a_policy_call_voids_on_the_harness(tmp_path):
    _, out = materialized(tmp_path)
    policy = _Unreachable("act", error=Unscorable)
    result = unit.run_unit(out / "prompt.npz", policy, tmp_path / "run", task_env=FakeTaskEnv())
    assert_read_result_shape(result)
    assert result["void"] is True and result["void_cause"] == "harness"


class _Told(ReplayPolicy):
    """Keeps the episode info each reset handed it."""

    name = "told"

    def __init__(self):
        super().__init__()
        self.told = []

    def _reset(self):
        super()._reset()
        self.told.append(self.episode)


def test_the_policy_is_reset_with_public_facts_and_a_seed_drawn_from_the_prompt(tmp_path):
    _, out = materialized(tmp_path)
    policy = _Told()
    result = unit.run_unit(out / "prompt.npz", policy, tmp_path / "run", task_env=FakeTaskEnv())
    assert result["success"] is True
    [told] = policy.told
    seed = unit.episode_seed(prompt.sha256_of(out / "prompt.npz"))
    assert told == EpisodeInfo(
        embodiment="fake-arms", action_dims={"qpos": 14, "ee": 16}, seed=seed
    )
    assert {f.name for f in dataclasses.fields(told)} == {"embodiment", "action_dims", "seed"}
    # The same prompt gives every policy the same seed; it is not the scene seed, nor bits of the
    # digest a duel publishes.
    assert unit.episode_seed(result["prompt_sha256"]) == seed and 0 <= seed < 2**31
    assert seed != SEED and f"{seed:08x}" not in result["prompt_sha256"]
    assert unit.episode_seed("0" * 64) != unit.episode_seed("1" * 64)


class _Recorder(ReplayPolicy):
    """Keeps everything it was handed, so a test can look for privileged data in it."""

    name = "recorder"

    def __init__(self):
        super().__init__()
        self.demonstrations = []
        self.observations = []

    def _set_demonstration(self, demonstration):
        self.demonstrations.append(demonstration)
        super()._set_demonstration(demonstration)

    def _act(self, observation):
        self.observations.append(observation)
        return super()._act(observation)


def test_nothing_privileged_reaches_the_policy(tmp_path):
    _, out = materialized(tmp_path)
    policy = _Recorder()
    result = unit.run_unit(out / "prompt.npz", policy, tmp_path / "run", task_env=FakeTaskEnv())
    assert result["success"] is True
    [demonstration] = policy.demonstrations
    assert isinstance(demonstration, Demonstration)
    assert {f.name for f in dataclasses.fields(demonstration)} == {"frames", "frequency", "cameras"}
    assert {f.name for f in dataclasses.fields(demonstration.frames[0])} == {
        "index",
        "images",
        "qpos",
        "endpose",
        "time_s",
    }
    assert {f.name for f in dataclasses.fields(policy.observations[0])} == {
        "step",
        "images",
        "qpos",
        "endpose",
        "instruction",
    }
    _, meta = prompt.read_raw(out / "prompt.npz")
    handed = json.dumps(
        [
            {k: v for k, v in dataclasses.asdict(frame).items() if k != "images"}
            for frame in demonstration.frames
        ]
        + [
            {k: v for k, v in dataclasses.asdict(o).items() if k != "images"}
            for o in policy.observations
        ],
        default=lambda value: value.tolist(),
    )
    for secret in (TASK, "scene_seed", meta["scene"]["sha256"], "fingerprint", "meta", "task"):
        assert secret not in handed
    assert policy.observations[0].instruction == "Follow the demonstrated behavior."


def test_scene_config_is_rebuilt_from_meta(monkeypatch):
    monkeypatch.undo()  # the real SceneConfig: a plain dataclass, no simulator to construct it
    meta = {
        "task_config": "demo_randomized",
        "save_freq": 5,
        "head_camera": "L515",
        "overrides": {"render_freq": 0},
        "embodiment": {
            "name": "franka-panda",
            "robotwin": ["franka-panda", "franka-panda", 0.8],
            "choice": "franka-panda",
        },
    }
    assert unit.scene_config_from(meta) == robotwin.SceneConfig(
        task_config="demo_randomized",
        save_freq=5,
        head_camera="L515",
        overrides={"render_freq": 0},
        embodiment="franka-panda",
    )
    assert unit.scene_config_from({**meta, "overrides": {}, "head_camera": None}) == (
        robotwin.SceneConfig(task_config="demo_randomized", save_freq=5, embodiment="franka-panda")
    )
    # A robot the task config chose is rebuilt by the task config again, whatever it is named:
    # its name need not be one --embodiment takes, nor its arms 0.8 m apart.
    by_config = {
        **meta,
        "embodiment": {
            "name": "franka-panda",
            "robotwin": ["franka-panda", "franka-panda", 0.6],
            "choice": None,
        },
    }
    assert unit.scene_config_from(by_config).embodiment is None


def test_run_unit_will_not_write_into_its_prompts_directory(tmp_path):
    _, out = materialized(tmp_path)
    before = sorted(p.name for p in out.iterdir())
    with pytest.raises(unit.UnitError, match="holds the prompt"):
        unit.run_unit(out / "prompt.npz", ReplayPolicy(), out, task_env=FakeTaskEnv())
    assert sorted(p.name for p in out.iterdir()) == before and unit.read_result(out)["ok"]
