"""A demonstration is built once and saved, and an episode is evaluated from the saved file."""

import dataclasses
import json

import numpy as np
import pytest

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import prompt, robotwin, scene, unit
from robotwin_icil.demo import Demonstration
from robotwin_icil.policy import ICILPolicy, ReplayPolicy

pytest.importorskip("imageio_ffmpeg")

TASK = "click_bell"
SEED = 11


@pytest.fixture(autouse=True)
def _fake_sim(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "SceneConfig", FakeConfig)


def materialized(tmp_path, env=None, **config):
    out = tmp_path / "prompt"
    done = unit.materialize(TASK, SEED, FakeConfig(**config), out, task_env=env or FakeTaskEnv())
    return done, out


def test_materialize_writes_the_prompt_the_clip_and_the_result(tmp_path):
    env = FakeTaskEnv()
    done, out = materialized(tmp_path, env, save_freq=5)
    assert done.ok and done.attempt.rejection is None
    assert {p.name for p in out.iterdir()} == {"prompt.npz", "demonstration.mp4", "result.json"}
    assert env.setups == [SEED] and env.closed == 1

    result = unit.read_result(out)
    assert result == done.result
    assert result["ok"] is True and result["task"] == TASK and result["scene_seed"] == SEED
    assert result["frames"] == env.expert_steps + 1 and result["cameras"] == ["head_camera"]
    assert result["prompt_sha256"] == prompt.sha256_of(out / "prompt.npz")
    assert result["video"] == "demonstration.mp4" and result["embodiment"] == "fake-arms"

    demonstration, meta = prompt.read_prompt(out / "prompt.npz")
    assert len(demonstration) == env.expert_steps + 1 and demonstration.timed
    assert meta["task"] == TASK and meta["scene_seed"] == SEED
    assert meta["embodiment"] == {"name": "fake-arms", "robotwin": ["fake-arms"]}
    assert meta["task_config"] == "fake" and meta["save_freq"] == 5
    assert meta["head_camera"] is None and meta["overrides"] == {}
    assert meta["frames"] == len(demonstration) and meta["cameras"] == ["head_camera"]
    assert meta["expert"] == {"attempts": 1, "rejections": {}, "rejection_details": {}}
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
    assert result["task"] == TASK and result["scene_seed"] == SEED and result["attempts"] == 1


def test_an_expert_that_fails_is_a_rejection_too(tmp_path):
    done, out = materialized(tmp_path, FakeTaskEnv(expert_misses_on={SEED}))
    assert not done.ok and unit.read_result(out)["rejection"] == "expert_failed"


def test_a_broken_simulator_is_an_error_not_a_rejection(tmp_path):
    with pytest.raises(robotwin.RoboTwinError, match="planner failed to construct"):
        materialized(tmp_path, FakeTaskEnv(broken_setup=True))


def test_run_unit_succeeds_with_replay_from_the_file(tmp_path):
    _, out = materialized(tmp_path)
    env = FakeTaskEnv()
    result = unit.run_unit(out / "prompt.npz", ReplayPolicy(), tmp_path / "run", task_env=env)
    assert result["success"] is True and result["void"] is False and result["error"] is None
    assert result["steps"] == env.expert_steps and result["step_limit"] == env.step_lim
    assert result["scene_max_error"] == 0.0 and result["model"] == "replay"
    assert result["embodiment"] == "fake-arms" and result["task"] == TASK
    assert result["scene_seed"] == SEED and result["evaluation_setting"] == "same_scene"
    assert result["prompt_sha256"] == prompt.sha256_of(out / "prompt.npz")
    assert result["video"] == "evaluation.mp4"
    assert (tmp_path / "run" / "evaluation.mp4").is_file()
    assert unit.read_result(tmp_path / "run") == result
    # The scene was rebuilt from meta, under the task's name, and closed afterwards.
    assert env.setups == [SEED] and env.task_names == [TASK] and env.closed == 1


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
    results = [
        unit.run_unit(
            out / "prompt.npz", ReplayPolicy(), tmp_path / f"run{i}", task_env=FakeTaskEnv()
        )
        for i in range(2)
    ]
    assert all(r["success"] and not r["void"] and r["scene_max_error"] == 0.0 for r in results)


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
        (lambda meta: meta.__setitem__("scene_seed", "11"), "not an integer"),
        (lambda meta: meta.__setitem__("save_freq", "fifteen"), "malformed"),
        (lambda meta: meta["scene"].__setitem__("fingerprint", {"actors": {}}), "malformed"),
    ],
)
def test_a_malformed_meta_voids_the_unit_before_any_scene(tmp_path, edit, reason):
    _, out = materialized(tmp_path)
    _retag(out / "prompt.npz", edit)
    env = FakeTaskEnv()
    result = unit.run_unit(out / "prompt.npz", ReplayPolicy(), tmp_path / "run", task_env=env)
    assert result["void"] and result["success"] is None and reason in result["error"]
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


def test_a_policy_breaking_the_protocol_voids_the_unit(tmp_path):
    _, out = materialized(tmp_path)

    class Broken(ICILPolicy):
        name = "broken"

        def _act(self, observation):
            return np.zeros(3)

    result = unit.run_unit(out / "prompt.npz", Broken(), tmp_path / "run", task_env=FakeTaskEnv())
    assert result["void"] and result["error"].startswith("policy broke the protocol")
    assert result["success"] is None and result["steps"] is None


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
        "embodiment": {"name": "franka-panda", "robotwin": ["franka-panda", "franka-panda", 0.8]},
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
