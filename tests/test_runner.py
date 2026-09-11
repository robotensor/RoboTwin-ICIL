import copy
import json

import pytest

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import camera_profiles, robotwin, runner, tasks
from robotwin_icil.generate import policy_seed
from robotwin_icil.policy import PolicyError, ReplayPolicy
from robotwin_icil.records import RecordError, RunDir


@pytest.fixture
def fake_sim(monkeypatch):
    """Route the runner's simulator calls to fake envs; returns the envs it creates, in order."""
    created: list[tuple[str, FakeTaskEnv]] = []

    def load_task(name):
        created.append((name, FakeTaskEnv()))
        return created[-1][1]

    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "load_task", load_task)
    monkeypatch.setattr(robotwin, "clear_render_cache", lambda: None)
    return created


def spec(tmp_path, **overrides):
    base = dict(
        run_dir=tmp_path / "run",
        tasks=tasks.table().suite("v1")[:2],
        suite="v1",
        episodes=4,
        global_seed=3,
    )
    base.update(overrides)
    return runner.RunSpec(**base)


def quiet(_line):
    pass


def test_assign_is_balanced_ordered_and_pure():
    suite = tasks.table().suite("v1")[:3]
    plan = runner.assign(suite, 7)
    assert [task.name for task in plan] == [suite[i % 3].name for i in range(7)]
    with pytest.raises(ValueError):
        runner.assign((), 3)


def test_a_run_records_every_episode_with_one_env_per_task(tmp_path, fake_sim):
    records = runner.run(spec(tmp_path), ReplayPolicy(), FakeConfig(), log=quiet)
    assert [r.episode for r in records] == [0, 1, 2, 3]
    assert all(r.scored and r.success for r in records)
    assert [name for name, _ in fake_sim] == [t.name for t in tasks.table().suite("v1")[:2]]


def test_the_manifest_records_what_was_run(tmp_path, fake_sim):
    runner.run(spec(tmp_path), ReplayPolicy(), FakeConfig(), log=quiet)
    manifest = RunDir(tmp_path / "run").manifest()
    assert manifest.global_seed == 3 and manifest.suite == "v1" and manifest.episodes == 4
    assert manifest.policy["policy"] == "replay"
    assert manifest.benchmark_commit  # this checkout is a git repository


def test_a_resumed_run_only_runs_what_is_missing(tmp_path, fake_sim):
    runner.run(spec(tmp_path), ReplayPolicy(), FakeConfig(), log=quiet)
    episodes = (tmp_path / "run" / "episodes.jsonl").read_text().splitlines()
    (tmp_path / "run" / "episodes.jsonl").write_text("\n".join(episodes[:2]) + "\n")
    fake_sim.clear()

    records = runner.run(spec(tmp_path), ReplayPolicy(), FakeConfig(), log=quiet)
    assert [r.episode for r in records] == [0, 1, 2, 3]
    assert sum(len(env.setups) for _, env in fake_sim) == 2 * 2  # two episodes, two builds each
    assert (tmp_path / "run" / "episodes.jsonl").read_text().splitlines()[:2] == episodes[:2]


def test_a_different_run_cannot_reuse_the_directory(tmp_path, fake_sim):
    runner.run(spec(tmp_path), ReplayPolicy(), FakeConfig(), log=quiet)
    with pytest.raises(RecordError):
        runner.run(spec(tmp_path, global_seed=4), ReplayPolicy(), FakeConfig(), log=quiet)


def test_one_env_is_alive_at_a_time(tmp_path, monkeypatch, fake_sim):
    # Each env holds CuRobo planners on the GPU; holding all of a suite's at once ran it out.
    events = []
    load = robotwin.load_task
    monkeypatch.setattr(
        robotwin, "load_task", lambda name: events.append(f"load {name}") or load(name)
    )
    monkeypatch.setattr(robotwin, "free_gpu", lambda: events.append("free"))
    runner.run(spec(tmp_path), ReplayPolicy(), FakeConfig(), log=quiet)
    first, second = (task.name for task in tasks.table().suite("v1")[:2])
    assert events == [f"load {first}", "free", f"load {second}", "free"]


def test_episodes_run_task_by_task_but_keep_their_assignment(tmp_path, fake_sim):
    records = runner.run(spec(tmp_path), ReplayPolicy(), FakeConfig(), log=quiet)
    suite = tasks.table().suite("v1")[:2]
    assert [r.task for r in records] == [suite[i % 2].name for i in range(4)]
    lines = (tmp_path / "run" / "episodes.jsonl").read_text().splitlines()
    assert [json.loads(line)["episode"] for line in lines] == [0, 2, 1, 3]


def test_a_broken_simulator_stops_the_run_and_keeps_what_was_recorded(
    tmp_path, monkeypatch, fake_sim
):
    second = tasks.table().suite("v1")[1].name
    monkeypatch.setattr(
        robotwin, "load_task", lambda name: FakeTaskEnv(broken_setup=(name == second))
    )
    with pytest.raises(robotwin.RoboTwinError):
        runner.run(spec(tmp_path), ReplayPolicy(), FakeConfig(), log=quiet)
    # The first task's episodes are on disk, so rerunning the same command resumes from there.
    assert [r.episode for r in RunDir(tmp_path / "run").records()] == [0, 2]


class FarSideConfig(FakeConfig):
    """A config whose args carry an embodiment's static cameras, under the far_side profile."""

    camera_profile = "far_side"

    def resolve(self, task_name=None):
        from test_cameras import STATIC_CAMERA_LIST

        args = {
            **super().resolve(task_name),
            "camera": {"head_camera_type": "D435", "collect_head_camera": True},
            "left_embodiment_config": {"static_camera_list": copy.deepcopy(STATIC_CAMERA_LIST)},
        }
        return camera_profiles.apply(args, self.camera_profile)


def test_the_manifest_records_the_camera_profile(tmp_path, fake_sim):
    runner.run(spec(tmp_path), ReplayPolicy(), FakeConfig(), log=quiet)
    stock = RunDir(tmp_path / "run").manifest()
    assert stock.benchmark_config["camera_profile"] == camera_profiles.get("stock").identity()

    runner.run(spec(tmp_path, run_dir=tmp_path / "far"), ReplayPolicy(), FarSideConfig(), log=quiet)
    far_side = RunDir(tmp_path / "far").manifest()
    assert far_side.benchmark_config["camera_profile"]["name"] == "far_side"
    cameras = far_side.robotwin_config["static_cameras"]
    assert [(c["name"], c["type"]) for c in cameras] == [
        ("head_camera", "D435"),
        ("far_side_camera", "L515"),
    ]


def test_clips_film_the_camera_the_run_names(tmp_path, monkeypatch, fake_sim):
    cameras = []

    class Recording(runner.EpisodeVideo):
        def __init__(self, directory, camera):
            cameras.append(camera)
            super().__init__(directory, camera=camera)

        def demonstration(self, demonstration):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(runner, "EpisodeVideo", Recording)
    run = spec(tmp_path, episodes=2, video=True, video_camera="far_side_camera")
    runner.run(run, ReplayPolicy(), FakeConfig(), log=quiet)
    assert cameras == ["far_side_camera", "far_side_camera"]


class Counting(ReplayPolicy):
    """Counts `describe()` calls: an adapter's description may hash every parameter it has."""

    name = "counting"

    def __init__(self):
        super().__init__()
        self.described = 0

    def describe(self):
        self.described += 1
        return super().describe()


def test_the_policy_is_described_at_the_start_and_end_of_a_run_not_per_episode(tmp_path, fake_sim):
    policy = Counting()
    records = runner.run(spec(tmp_path), policy, FakeConfig(), log=quiet)
    assert policy.described == 2
    assert {r.model for r in records} == {"counting"}


class Seeded(ReplayPolicy):
    """Remembers the seeds it is handed, in the order it is handed them."""

    name = "seeded"

    def __init__(self):
        super().__init__()
        self.seeds = []

    def seed(self, seed):
        self.seeds.append(seed)


def test_a_policy_seed_does_not_depend_on_resume_order(tmp_path, fake_sim):
    policy = Seeded()
    records = runner.run(spec(tmp_path), policy, FakeConfig(), log=quiet)
    # Episodes run task by task: 0 and 2, then 1 and 3.
    assert policy.seeds == [policy_seed(3, episode) for episode in (0, 2, 1, 3)]
    assert not set(policy.seeds) & {r.scene_seed for r in records}

    episodes = (tmp_path / "run" / "episodes.jsonl").read_text().splitlines()
    (tmp_path / "run" / "episodes.jsonl").write_text("\n".join(episodes[:2]) + "\n")
    resumed = Seeded()
    runner.run(spec(tmp_path), resumed, FakeConfig(), log=quiet)
    assert resumed.seeds == policy.seeds[2:]


class Closing(Counting):
    """Counts `close()` calls: a model server or a GPU model must be released exactly once."""

    name = "closing"

    def __init__(self):
        super().__init__()
        self.closed = 0

    def close(self):
        self.closed += 1


def test_the_run_closes_the_policy_once(tmp_path, fake_sim):
    policy = Closing()
    runner.run(spec(tmp_path), policy, FakeConfig(), log=quiet)
    assert policy.closed == 1


def test_the_policy_is_closed_once_when_an_episode_raises(tmp_path, monkeypatch, fake_sim):
    monkeypatch.setattr(robotwin, "load_task", lambda name: FakeTaskEnv(broken_setup=True))
    policy = Closing()
    with pytest.raises(robotwin.RoboTwinError):
        runner.run(spec(tmp_path), policy, FakeConfig(), log=quiet)
    assert policy.closed == 1
    # No checksum, and the run did not finish: nothing to audit, so no second description.
    assert policy.described == 1


def test_the_policy_is_closed_when_the_run_is_refused(tmp_path, fake_sim):
    runner.run(spec(tmp_path), ReplayPolicy(), FakeConfig(), log=quiet)
    policy = Closing()
    with pytest.raises(RecordError):
        runner.run(spec(tmp_path, global_seed=4), policy, FakeConfig(), log=quiet)
    assert policy.closed == 1


class Provenanced(ReplayPolicy):
    name = "provenanced"

    def __init__(self, reported):
        super().__init__()
        self.reported = reported

    def environment(self):
        return self.reported


def test_the_manifest_records_the_policys_own_environment(tmp_path, fake_sim):
    reported = {"torch": "2.7.0+cu128", "gpu": "NVIDIA GeForce RTX 5090", "bpp": "0123abc"}
    runner.run(spec(tmp_path), Provenanced(reported), FakeConfig(), log=quiet)
    manifest = RunDir(tmp_path / "run").manifest()
    assert manifest.policy_environment == reported
    assert "policy_environment" not in manifest.identity()
    # Another machine, another torch: the run still resumes.
    runner.run(spec(tmp_path), Provenanced({"torch": "2.8.0"}), FakeConfig(), log=quiet)


def test_a_policy_environment_holds_strings(tmp_path, fake_sim):
    with pytest.raises(PolicyError, match="environment\\(\\): 'cuda' must be a string, not float"):
        runner.run(spec(tmp_path), Provenanced({"cuda": 12.8}), FakeConfig(), log=quiet)
    assert not (tmp_path / "run" / "manifest.json").exists()


class Described(ReplayPolicy):
    """Describes itself with whatever convention keys a test gives it."""

    name = "described"

    def __init__(self, **fields):
        super().__init__()
        self.fields = fields
        self.closed = 0

    def describe(self):
        return {**super().describe(), **self.fields}

    def close(self):
        self.closed += 1


def test_a_policy_needing_another_camera_profile_is_refused_up_front(tmp_path, fake_sim):
    policy = Described(camera_profile_required="far_side")
    with pytest.raises(PolicyError, match="requires camera profile 'far_side' but the run uses"):
        runner.run(spec(tmp_path), policy, FakeConfig(), log=quiet)
    assert fake_sim == []  # not one env built, not one seed drawn
    assert not (tmp_path / "run" / "manifest.json").exists()
    assert policy.closed == 1  # refused before the manifest, closed all the same

    runner.run(spec(tmp_path), Described(camera_profile_required="stock"), FakeConfig(), log=quiet)
    assert RunDir(tmp_path / "run").manifest().policy["camera_profile_required"] == "stock"


def test_a_description_off_the_convention_refuses_the_run(tmp_path, fake_sim):
    unhashed = Described(checkpoint_sha256="abc")
    with pytest.raises(PolicyError, match="'checkpoint_sha256' must be 64 hex digits"):
        runner.run(spec(tmp_path), unhashed, FakeConfig(), log=quiet)
    misspelt = Described(camera_profile_required="farside")
    with pytest.raises(PolicyError, match="'camera_profile_required' must be a camera profile's"):
        runner.run(spec(tmp_path), misspelt, FakeConfig(), log=quiet)
    assert fake_sim == []
    assert (unhashed.closed, misspelt.closed) == (1, 1)


def test_a_description_resumes_as_json_reads_it_back(tmp_path, fake_sim):
    policy = Described(training_tasks=("click_bell", "stack_blocks_two"))
    runner.run(spec(tmp_path), policy, FakeConfig(), log=quiet)
    manifest = RunDir(tmp_path / "run").manifest()
    assert manifest.policy["training_tasks"] == ["click_bell", "stack_blocks_two"]
    runner.run(spec(tmp_path), policy, FakeConfig(), log=quiet)  # a tuple is not a new run


class Learning(ReplayPolicy):
    """A policy that writes its parameters as it acts: what the frozen-policy audit catches."""

    name = "learning"

    def __init__(self, checksum="0" * 8):
        super().__init__()
        self.checksum = checksum
        self.updates = 0
        self.closed = 0

    def _act(self, observation):
        self.updates += 1
        return super()._act(observation)

    def describe(self):
        checksum = None if self.checksum is None else f"{self.checksum}-{self.updates}"
        return {**super().describe(), "parameter_checksum": checksum}

    def close(self):
        self.closed += 1


class Frozen(Learning):
    name = "frozen"

    def describe(self):
        return {**ReplayPolicy.describe(self), "parameter_checksum": self.checksum}


def test_a_policy_whose_parameters_change_fails_the_audit(tmp_path, fake_sim):
    policy = Learning()
    with pytest.raises(PolicyError, match="parameters changed during the run: the policy must be"):
        runner.run(spec(tmp_path), policy, FakeConfig(), log=quiet)
    assert policy.closed == 1
    # What ran stays on disk, and so does the failure, which outlives the error.
    assert len(RunDir(tmp_path / "run").records()) == 4
    audit = json.loads((tmp_path / "run" / "audit.json").read_text())
    assert audit["frozen"] is False and audit["policy"] == "learning"
    assert audit["parameter_checksum"]["start"] == "00000000-0"
    assert audit["parameter_checksum"]["end"] != "00000000-0"


def test_a_run_that_failed_the_audit_does_not_resume(tmp_path, fake_sim):
    with pytest.raises(PolicyError, match="parameters changed"):
        runner.run(spec(tmp_path), Learning(), FakeConfig(), log=quiet)
    # A fresh copy starts from the checksum the manifest holds, so its identity matches: only the
    # recorded failure stops it.
    with pytest.raises(RecordError, match="failed the frozen-policy audit"):
        runner.run(spec(tmp_path), Learning(), FakeConfig(), log=quiet)


class Drifting(Learning):
    """Changes its checksum each time it is described, whether or not it acted."""

    name = "drifting"

    def describe(self):
        self.updates += 1
        return super().describe()


def test_a_refused_run_does_not_mark_the_directory_it_was_refused(tmp_path, fake_sim):
    runner.run(spec(tmp_path), ReplayPolicy(), FakeConfig(), log=quiet)
    with pytest.raises(PolicyError, match="parameters changed") as raised:
        runner.run(spec(tmp_path, global_seed=4), Drifting(), FakeConfig(), log=quiet)
    assert isinstance(raised.value.__context__, RecordError)
    assert not (tmp_path / "run" / "audit.json").exists()
    runner.run(spec(tmp_path), ReplayPolicy(), FakeConfig(), log=quiet)  # still resumes


def test_a_frozen_policy_passes_the_audit(tmp_path, fake_sim):
    records = runner.run(spec(tmp_path), Frozen(), FakeConfig(), log=quiet)
    assert len(records) == 4


def test_the_audit_runs_when_the_run_raises_too(tmp_path, monkeypatch, fake_sim):
    second = tasks.table().suite("v1")[1].name
    monkeypatch.setattr(
        robotwin, "load_task", lambda name: FakeTaskEnv(broken_setup=(name == second))
    )
    policy = Learning()
    with pytest.raises(PolicyError, match="parameters changed") as raised:
        runner.run(spec(tmp_path), policy, FakeConfig(), log=quiet)
    assert isinstance(raised.value.__context__, robotwin.RoboTwinError)
    assert policy.closed == 1
    assert (tmp_path / "run" / "audit.json").exists()


def test_a_policy_without_a_checksum_is_not_audited(tmp_path, fake_sim):
    records = runner.run(spec(tmp_path), Learning(checksum=None), FakeConfig(), log=quiet)
    assert len(records) == 4


def test_the_manifest_records_the_policy_config_and_a_resume_must_share_it(tmp_path, fake_sim):
    config = {"config": tmp_path / "bpp.yml", "temperature": 0.5, "deterministic": True}
    runner.run(spec(tmp_path, policy_config=config), ReplayPolicy(), FakeConfig(), log=quiet)
    recorded = RunDir(tmp_path / "run").manifest().policy_config
    assert recorded == {
        "config": str(tmp_path / "bpp.yml"),
        "temperature": 0.5,
        "deterministic": True,
    }

    runner.run(spec(tmp_path, policy_config=config), ReplayPolicy(), FakeConfig(), log=quiet)
    other = {**config, "temperature": 0.7}
    with pytest.raises(RecordError, match="already holds a different run"):
        runner.run(spec(tmp_path, policy_config=other), ReplayPolicy(), FakeConfig(), log=quiet)
    with pytest.raises(RecordError, match="already holds a different run"):
        runner.run(spec(tmp_path), ReplayPolicy(), FakeConfig(), log=quiet)


def test_a_resume_under_another_adapter_version_is_refused(tmp_path, fake_sim):
    runner.run(spec(tmp_path), Described(adapter_version="0.1.0"), FakeConfig(), log=quiet)
    with pytest.raises(RecordError, match="already holds a different run"):
        runner.run(spec(tmp_path), Described(adapter_version="0.2.0"), FakeConfig(), log=quiet)
