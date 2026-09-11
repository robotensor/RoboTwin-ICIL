import json
from dataclasses import replace

import pytest

from robotwin_icil import camera_profiles
from robotwin_icil.records import (
    SAME_SCENE,
    EpisodeRecord,
    RecordError,
    RunDir,
    RunManifest,
    Status,
    git_commit,
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
    )
    base.update(overrides)
    return EpisodeRecord(**base)


def manifest(**overrides) -> RunManifest:
    base = dict(
        global_seed=42,
        evaluation_setting=SAME_SCENE,
        suite="v1",
        tasks=("place_object_basket",),
        episodes=10,
        max_expert_attempts=20,
        policy={"policy": "replay"},
        benchmark_commit="abc",
        robotwin_commit="def",
        benchmark_config={},
        robotwin_config={"task_config": "demo_clean"},
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
    run.start(manifest(policy_environment={"torch": "2.7.0", "gpu": "RTX 5090"}))
    with pytest.raises(RecordError):
        run.start(manifest(global_seed=7))


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


def test_records_written_before_rejection_details_still_load():
    data = record().to_json()
    data.pop("rejection_details")
    assert EpisodeRecord.from_json(data).rejection_details == {}


def profiled(profile, static_cameras, **overrides) -> RunManifest:
    """A manifest written since camera profiles, which records the profile and static cameras."""
    return manifest(
        benchmark_config={
            "task_config": "demo_clean",
            "camera_profile": camera_profiles.get(profile).identity(),
        },
        robotwin_config={"task_config": "demo_clean", "static_cameras": static_cameras},
        **overrides,
    )


STOCK_CAMERAS = [{"name": "head_camera"}, {"name": "front_camera"}]
FAR_SIDE_CAMERAS = [{"name": "head_camera"}, {"name": "far_side_camera"}]


def test_a_manifest_from_before_camera_profiles_is_a_stock_run(tmp_path):
    legacy = manifest(benchmark_config={"task_config": "demo_clean"})
    assert legacy.identity() == profiled("stock", STOCK_CAMERAS).identity()
    assert legacy.identity() != profiled("far_side", FAR_SIDE_CAMERAS).identity()

    run = RunDir(tmp_path)
    run.start(legacy)
    run.start(profiled("stock", STOCK_CAMERAS))  # a V1 run directory still resumes
    with pytest.raises(RecordError):
        run.start(profiled("far_side", FAR_SIDE_CAMERAS))


def test_a_camera_profile_is_part_of_a_runs_identity():
    far_side = profiled("far_side", FAR_SIDE_CAMERAS)
    assert far_side.identity() == profiled("far_side", FAR_SIDE_CAMERAS).identity()
    assert far_side.identity() != profiled("far_side", STOCK_CAMERAS).identity()
    edited = dict(far_side.benchmark_config, camera_profile={"name": "far_side", "sha256": "0"})
    assert far_side.identity() != replace(far_side, benchmark_config=edited).identity()


def test_a_manifest_from_before_camera_profiles_still_loads(tmp_path):
    data = manifest(benchmark_config={"task_config": "demo_clean"}).to_json()
    (tmp_path / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    assert "camera_profile" not in RunDir(tmp_path).manifest().benchmark_config


def test_records_written_before_physics_steps_still_load(tmp_path):
    data = record(physics_steps=812).to_json()
    assert EpisodeRecord.from_json(data).physics_steps == 812
    data.pop("physics_steps")
    assert EpisodeRecord.from_json(data).physics_steps == 0

    run = RunDir(tmp_path)
    run.start(manifest())
    run.episodes_path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    assert [r.physics_steps for r in run.records()] == [0]


def test_records_written_before_policy_info_still_load(tmp_path):
    data = record(policy_info={"active_arm": "right"}).to_json()
    assert EpisodeRecord.from_json(data).policy_info == {"active_arm": "right"}
    data.pop("policy_info")
    assert EpisodeRecord.from_json(data).policy_info == {}

    run = RunDir(tmp_path)
    run.start(manifest())
    run.episodes_path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    assert [r.policy_info for r in run.records()] == [{}]


def test_a_manifest_from_before_policy_environments_still_loads(tmp_path):
    data = manifest().to_json()
    data.pop("policy_environment")
    (tmp_path / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    assert RunDir(tmp_path).manifest().policy_environment == {}


def test_a_manifest_from_before_policy_configs_is_a_run_without_policy_arguments(tmp_path):
    data = manifest().to_json()
    data.pop("policy_config")
    (tmp_path / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    run = RunDir(tmp_path)
    legacy = run.manifest()
    assert legacy.policy_config == {}
    assert legacy.identity() == manifest(policy_config={}).identity()
    assert legacy.identity() != manifest(policy_config={"temperature": 0.5}).identity()
    run.start(manifest())  # an older run directory still resumes with no --policy-arg
    with pytest.raises(RecordError):
        run.start(manifest(policy_config={"temperature": 0.5}))


def test_a_run_that_failed_the_audit_is_not_resumed(tmp_path):
    run = RunDir(tmp_path)
    run.start(manifest())
    run.check_audit()  # nothing recorded, nothing refused
    run.fail_audit({"policy": "learning", "parameter_checksum": {"start": "a", "end": "b"}})
    assert json.loads((tmp_path / "audit.json").read_text()) == {
        "frozen": False,
        "policy": "learning",
        "parameter_checksum": {"start": "a", "end": "b"},
    }
    with pytest.raises(RecordError, match="failed the frozen-policy audit"):
        run.check_audit()
    with pytest.raises(RecordError, match="failed the frozen-policy audit"):
        run.start(manifest())
