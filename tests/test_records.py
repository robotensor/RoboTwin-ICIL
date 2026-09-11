import json
import subprocess
import sys
import types
from dataclasses import replace
from importlib import metadata

import pytest

from robotwin_icil import camera_profiles, records
from robotwin_icil.records import (
    SAME_SCENE,
    EpisodeRecord,
    RecordError,
    RunDir,
    RunManifest,
    Status,
    environment,
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


def test_records_written_before_action_types_and_demonstration_arms_still_load(tmp_path):
    data = record(action_type="ee", demonstration_arms=("left",)).to_json()
    assert json.loads(json.dumps(data))["demonstration_arms"] == ["left"]
    restored = EpisodeRecord.from_json(json.loads(json.dumps(data)))
    assert restored == record(action_type="ee", demonstration_arms=("left",))
    assert (restored.action_type, restored.demonstration_arms) == ("ee", ("left",))
    for key in ("action_type", "demonstration_arms"):
        data.pop(key)
    legacy = EpisodeRecord.from_json(data)
    assert (legacy.action_type, legacy.demonstration_arms) == (None, None)

    run = RunDir(tmp_path)
    run.start(manifest())
    run.append(record(0, action_type="qpos", demonstration_arms=("left", "right")))
    with run.episodes_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**data, "episode": 1}) + "\n")
    loaded = run.records()
    assert [(r.action_type, r.demonstration_arms) for r in loaded] == [
        ("qpos", ("left", "right")),
        (None, None),
    ]


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


def _installed(monkeypatch, versions, nvidia_smi=None, torch=None, curobo=None):
    """Fake what `environment()` probes: distributions, nvidia-smi's output, torch and curobo."""
    calls = []
    real_run = subprocess.run

    def version(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    def run(command, **kwargs):
        if command[0] != "nvidia-smi":  # platform.platform() asks `uname -p` the same way
            return real_run(command, **kwargs)
        calls.append((command, kwargs))
        if nvidia_smi is None:
            raise FileNotFoundError(2, "No such file or directory", command[0])
        if isinstance(nvidia_smi, BaseException):
            raise nvidia_smi
        return subprocess.CompletedProcess(command, 0, stdout=nvidia_smi, stderr="")

    monkeypatch.setattr(records.metadata, "version", version)
    monkeypatch.setattr(records.subprocess, "run", run)
    # None in sys.modules makes the import raise ImportError, as a missing package does.
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "curobo", curobo)
    return calls


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def test_environment_with_nothing_installed_is_python_and_platform(monkeypatch):
    calls = _installed(monkeypatch, {})
    assert set(environment()) == {"python", "platform"}
    # nvidia-smi was asked once, under a timeout, so a hung driver cannot stall a run.
    [(command, kwargs)] = calls
    assert command[0] == "nvidia-smi" and 0 < kwargs["timeout"] <= 10


def test_environment_names_the_gpu_driver_torch_cuda_curobo_and_sapien(monkeypatch):
    torch = _module("torch", __version__="2.8.0+cu128", version=types.SimpleNamespace(cuda="12.8"))
    _installed(
        monkeypatch,
        {"nvidia_curobo": "0.7.8", "sapien": "3.0.0b1"},
        nvidia_smi="NVIDIA GeForce RTX 5090, 575.57.08\nNVIDIA GeForce RTX 5090, 575.57.08\n",
        torch=torch,
    )
    found = environment()
    assert {k: v for k, v in found.items() if k not in ("python", "platform")} == {
        "gpu": "NVIDIA GeForce RTX 5090; NVIDIA GeForce RTX 5090",
        "gpu_driver": "575.57.08",
        "torch": "2.8.0+cu128",
        "torch_cuda": "12.8",
        "curobo": "0.7.8",
        "sapien": "3.0.0b1",
    }
    assert all(isinstance(value, str) for value in found.values())
    assert "environment" not in manifest(environment=found).identity()


def test_environment_falls_back_to_curobos_own_version(monkeypatch):
    cpu_torch = _module("torch", __version__="2.4.1", version=types.SimpleNamespace(cuda=None))
    _installed(
        monkeypatch, {}, torch=cpu_torch, curobo=_module("curobo", __version__="0.7.7.post1")
    )
    found = environment()
    assert (found["torch"], found["curobo"]) == ("2.4.1", "0.7.7.post1")
    assert "torch_cuda" not in found and "sapien" not in found


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired("nvidia-smi", 5.0),
        subprocess.CalledProcessError(9, "nvidia-smi"),
        PermissionError(13, "Permission denied"),
    ],
)
def test_environment_never_raises(monkeypatch, failure):
    _installed(monkeypatch, {"sapien": "3.0.0b1"}, nvidia_smi=failure)

    def broken_import(name, package=None):
        raise OSError(f"lib{name}.so: cannot open shared object file")

    monkeypatch.setattr(records.importlib, "import_module", broken_import)
    assert set(environment()) == {"python", "platform", "sapien"}


def test_environment_skips_what_nvidia_smi_does_not_list(monkeypatch):
    _installed(monkeypatch, {}, nvidia_smi="No devices were found\n")
    assert "gpu" not in environment()
