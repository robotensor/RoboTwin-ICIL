import json
import subprocess
from pathlib import Path

import pytest

from robotwin_icil.records import (
    SAME_SCENE,
    EpisodeRecord,
    RecordError,
    RunDir,
    RunManifest,
    Status,
    git_commit,
    write_json,
)


def record(episode=0, status=Status.SCORED, success=True, **overrides) -> EpisodeRecord:
    base = dict(
        episode=episode,
        evaluation_setting=SAME_SCENE,
        skill_category="pick_and_place",
        task="place_object_basket",
        scene_seed=123,
        status=status,
        success=success,
        steps=40,
        step_limit=400,
        demonstration_frames=30,
        expert_generation_attempts=1,
        rejections={},
        scene_max_error=0.0,
        model="replay",
        embodiment="aloha-agilex",
    )
    base.update(overrides)
    return EpisodeRecord(**base)


def manifest(**overrides) -> RunManifest:
    base = dict(
        global_seed=42,
        evaluation_setting=SAME_SCENE,
        tasks=("place_object_basket",),
        episodes=10,
        max_expert_attempts=20,
        policy={"policy": "replay"},
        benchmark_commit="abc",
        robotwin_commit="def",
        benchmark_config={"embodiment": "aloha-agilex"},
        robotwin_config={
            "task_config": "demo_clean",
            "embodiment": ["aloha-agilex"],
            "embodiment_name": "aloha-agilex",
        },
    )
    base.update(overrides)
    return RunManifest(**base)


def test_only_scored_episodes_carry_a_success():
    with pytest.raises(RecordError):
        record(status=Status.SCORED, success=None)
    with pytest.raises(RecordError):
        record(status=Status.REJECTED, success=False)
    assert not record(status=Status.REJECTED, success=None).scored


def test_records_round_trip_through_json():
    original = record(rejections={"unstable": 2}, expert_generation_attempts=3)
    restored = EpisodeRecord.from_json(json.loads(json.dumps(original.to_json())))
    assert restored == original
    assert restored.status is Status.SCORED


def test_run_dir_appends_and_reads_back(tmp_path):
    run = RunDir(tmp_path / "run")
    run.start(manifest())
    run.append(record(0))
    run.append(record(1, success=False))
    assert [r.episode for r in run.records()] == [0, 1]
    assert run.completed() == {0, 1}
    assert run.manifest() == manifest()


def test_resuming_the_same_run_is_allowed_but_not_a_different_one(tmp_path):
    run = RunDir(tmp_path)
    run.start(manifest())
    run.start(
        manifest(environment={"python": "3.10.99"})
    )  # the machine may differ, the run may not
    with pytest.raises(RecordError):
        run.start(manifest(global_seed=7))
    with pytest.raises(RecordError):
        run.start(manifest(arms="1"))  # one-arm and two-arm runs are different runs


def test_a_manifest_records_only_an_arms_value_a_run_can_ask_for():
    assert manifest(arms="1").arms == "1"
    with pytest.raises(RecordError, match="arms"):
        manifest(arms="switching")
    with pytest.raises(RecordError, match="arms"):
        RunManifest.from_json({**manifest().to_json(), "arms": 3})


def test_a_manifest_written_before_arms_existed_still_loads(tmp_path):
    data = manifest().to_json()
    del data["arms"]
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    loaded = RunDir(tmp_path).manifest()
    assert loaded.arms == "2"
    assert loaded == manifest()


@pytest.mark.parametrize("suite", ["legacy", None])
def test_legacy_suite_metadata_is_readable_and_does_not_change_run_identity(tmp_path, suite):
    data = {**manifest().to_json(), "suite": suite}
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    run = RunDir(tmp_path)
    loaded = run.manifest()
    assert loaded == manifest()
    assert RunManifest.from_json(data) == loaded
    assert "suite" not in loaded.to_json()
    run.start(manifest())
    # Dropping a label must not allow a different task selection to resume this run.
    with pytest.raises(RecordError, match="different run"):
        run.start(manifest(tasks=("click_bell",)))
    assert data["suite"] == suite  # reading does not mutate the caller's data


def test_a_run_on_another_robot_cannot_continue_this_one(tmp_path):
    # Same seed, task selection and policy, but 16-wide Franka episodes would be averaged with 14-wide
    # aloha ones: the embodiment is part of what a resumed run must share.
    run = RunDir(tmp_path)
    run.start(manifest())
    with pytest.raises(RecordError, match="different run"):
        run.start(manifest(benchmark_config={"embodiment": "franka-panda"}))
    with pytest.raises(RecordError, match="different run"):
        run.start(
            manifest(
                robotwin_config={
                    "task_config": "demo_clean",
                    "embodiment": ["franka-panda", "franka-panda", 0.8],
                    "embodiment_name": "franka-panda",
                }
            )
        )


def test_a_torn_final_line_is_ignored_but_earlier_corruption_is_not(tmp_path):
    run = RunDir(tmp_path)
    run.start(manifest())
    run.append(record(0))
    with run.episodes_path.open("a", encoding="utf-8") as handle:
        handle.write('{"episode": 1, "trunc')  # what a killed run leaves behind
    assert [r.episode for r in run.records()] == [0]

    run.episodes_path.write_text(
        '{"broken\n' + json.dumps(record(0).to_json()) + "\n", encoding="utf-8"
    )
    with pytest.raises(RecordError):
        run.records()


def test_an_episode_recorded_twice_is_an_error(tmp_path):
    run = RunDir(tmp_path)
    run.start(manifest())
    run.append(record(0))
    run.append(record(0))
    with pytest.raises(RecordError):
        run.records()


def test_git_commit_of_a_non_repository_is_none(tmp_path):
    assert git_commit(tmp_path) is None


def test_git_commit_is_a_checkouts_own_never_the_repository_around_it(tmp_path):
    # A wheel's package sits wherever its environment does, which may be inside another repository.
    repo = tmp_path / "unrelated"
    site = repo / ".venv" / "lib" / "python3.10"
    site.mkdir(parents=True)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "README").write_text("another project\n")
    subprocess.run([*git, "add", "README"], check=True)
    subprocess.run([*git, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "c"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert git_commit(repo) == head
    assert git_commit(site) is None


def test_records_written_before_rejection_details_still_load():
    data = record().to_json()
    data.pop("rejection_details")
    assert EpisodeRecord.from_json(data).rejection_details == {}


def test_records_written_before_a_run_could_choose_its_robot_are_aloha():
    data = record(embodiment="franka-panda").to_json()
    assert EpisodeRecord.from_json(data).embodiment == "franka-panda"
    data.pop("embodiment")
    assert EpisodeRecord.from_json(data).embodiment == "aloha-agilex"


def test_write_json_replaces_the_file_whole_and_keeps_key_order(tmp_path):
    path = tmp_path / "out.json"
    path.write_text("old", encoding="utf-8")
    write_json(path, [{"task": "click_bell", "seeds": 2, "arms": ["left"]}])
    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == [{"task": "click_bell", "seeds": 2, "arms": ["left"]}]
    assert text.index('"task"') < text.index('"seeds"') < text.index('"arms"')
    assert text.endswith("\n") and not path.with_suffix(".json.tmp").exists()


def test_write_json_torn_mid_write_leaves_the_previous_file(tmp_path, monkeypatch):
    # A survey runs for hours and rewrites its json after every task; a kill that lands during
    # the write must leave the last complete payload, not a truncated one.
    path = tmp_path / "out.json"
    path.write_text('{"complete": true}\n', encoding="utf-8")
    original = Path.write_text

    def torn(self, text, *args, **kwargs):
        original(self, text[: len(text) // 2], *args, **kwargs)
        raise OSError("killed mid-write")

    monkeypatch.setattr(Path, "write_text", torn)
    with pytest.raises(OSError):
        write_json(path, {"complete": False, "more": list(range(50))})
    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": True}
