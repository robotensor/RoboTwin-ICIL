import copy
import json

import pytest

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import camera_profiles, robotwin, runner, tasks
from robotwin_icil.policy import ReplayPolicy
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


def test_the_policy_is_described_once_per_run_not_per_episode(tmp_path, fake_sim):
    policy = Counting()
    records = runner.run(spec(tmp_path), policy, FakeConfig(), log=quiet)
    assert policy.described == 1
    assert {r.model for r in records} == {"counting"}
