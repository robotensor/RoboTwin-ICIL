import pytest

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import robotwin, runner, tasks
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
