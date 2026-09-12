"""The half of the plugin that must work with no simulator.

That is not a convenience: the validator host reads a catalogue, CI checks a unit list and a third
party verifies a published prompt, and none of them can build a RoboTwin scene. Every test here
runs under `pytest -m "not sim"`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from icil_benchmark_robotwin import BENCHMARK, BENCHMARK_API_VERSION
from icil_benchmark_robotwin.plugin import PROMPT_NAME, VIEWS
from icil_benchmark_robotwin.prompt import PROMPT_SCHEMA, channel_of, prompt_sha256
from icil_benchmark_robotwin.units import derive_units


def test_the_plugin_names_itself_and_its_abi():
    info = BENCHMARK.info()
    assert info["id"] == BENCHMARK.id == "robotwin"
    assert info["api_version"] == BENCHMARK_API_VERSION
    assert info["protocol"] == "same_scene_1demo"


def test_it_offers_only_the_view_that_is_meaningful_here():
    """Same Scene makes the sensorimotor view degenerate - replaying the demonstration's own
    actions solves the episode - so this benchmark does not offer it."""
    assert VIEWS == ("video_only",)
    assert "sensorimotor" not in BENCHMARK.info()["views"]


def test_the_catalogue_is_the_menu_not_the_meal():
    cat = BENCHMARK.catalogue()
    assert "v1" in cat["suites"] and cat["suites"]["v1"]
    assert set(cat["suites"]["v1"]) <= set(cat["tasks"])
    for task in cat["tasks"].values():
        assert task["category"] in cat["categories"]
    # No demonstrations: a catalogue says what exists, not what a duel will run on.
    assert "demos" not in cat and "demonstrations" not in cat


def test_units_are_a_pure_function_of_their_arguments():
    first = BENCHMARK.derive_units(seed_material="duel-1", count=9, suite="v1")
    again = BENCHMARK.derive_units(seed_material="duel-1", count=9, suite="v1")
    assert first == again
    other = BENCHMARK.derive_units(seed_material="duel-2", count=9, suite="v1")
    assert [u["scene_seed"] for u in other] != [u["scene_seed"] for u in first]


def test_units_are_derived_by_hash_not_by_a_library_rng():
    """A stream that depends on a numpy version is not reproducible by someone holding the
    published record, so the derivation is pinned to a value here."""
    units = derive_units(seed_material="golden", count=3, suite="v1")
    assert [u.scene_seed for u in units] == [850551942, 798494248, 64716227]


def test_units_cover_every_task_before_repeating_one():
    suite = BENCHMARK.catalogue()["suites"]["v1"]
    units = BENCHMARK.derive_units(seed_material="x", count=len(suite), suite="v1")
    assert [u["task"] for u in units] == list(suite)
    assert all(u["skill_category"] for u in units)


def test_a_unit_round_trips_through_its_published_form():
    from icil_benchmark_robotwin.units import BenchmarkUnit

    unit = derive_units(seed_material="x", count=1)[0]
    assert BenchmarkUnit.from_dict(unit.as_dict()) == unit
    # Extra keys the orchestrator adds do not break the round trip.
    assert BenchmarkUnit.from_dict({**unit.as_dict(), "unit_id": "rp-000"}) == unit


def test_an_empty_or_unknown_suite_is_refused():
    with pytest.raises(ValueError):
        derive_units(seed_material="x", count=0)
    from robotwin_icil.tasks import TaskTableError

    with pytest.raises(TaskTableError, match="unknown suite"):
        derive_units(seed_material="x", count=1, suite="no-such-suite")


# ------------------------------------------------------------------ prompts


def _write_prompt(path: Path, *, task="click_bell", seed=7, frames=4, cameras=("head_camera",)):
    arrays = {
        "times": np.linspace(0, 1, frames, dtype=np.float64),
        "qpos": np.zeros((frames, 14), dtype=np.float32),
        "endpose": np.zeros((frames, 16), dtype=np.float32),
        "actions": np.zeros((frames - 1, 14), dtype=np.float32),
    }
    for camera in cameras:
        arrays[f"frames_{camera}"] = np.zeros((frames, 4, 4, 3), dtype=np.uint8)
    meta = {
        "schema": PROMPT_SCHEMA,
        "task": task,
        "scene_seed": seed,
        "setting": "same_scene",
        "cameras": list(cameras),
        "frequency": 16.67,
        "steps": frames - 1,
        "fingerprint_sha256": "a" * 64,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)
    return path


def test_a_prompt_verifies_against_the_unit_that_asked_for_it(tmp_path):
    _write_prompt(tmp_path / PROMPT_NAME)
    verdict = BENCHMARK.verify_prompt(
        path=str(tmp_path), unit={"task": "click_bell", "scene_seed": 7}
    )
    assert verdict["ok"], verdict["problems"]
    assert verdict["sha256"] == prompt_sha256(tmp_path / PROMPT_NAME)
    assert verdict["frames"] == 4 and verdict["cameras"] == ["head_camera"]


def test_a_prompt_for_another_task_or_scene_is_refused(tmp_path):
    _write_prompt(tmp_path / PROMPT_NAME, task="click_bell", seed=7)
    wrong_task = BENCHMARK.verify_prompt(path=str(tmp_path), unit={"task": "press_stapler"})
    assert not wrong_task["ok"] and any("task" in p for p in wrong_task["problems"])
    wrong_seed = BENCHMARK.verify_prompt(path=str(tmp_path), unit={"scene_seed": 8})
    assert not wrong_seed["ok"] and any("scene_seed" in p for p in wrong_seed["problems"])


def test_a_missing_or_corrupt_prompt_is_reported_not_raised(tmp_path):
    """A duel needs a verdict, and a corrupt prompt is a materializing fault to substitute for."""
    absent = BENCHMARK.verify_prompt(path=str(tmp_path), unit={})
    assert not absent["ok"] and "missing" in absent["problems"][0]

    (tmp_path / PROMPT_NAME).write_bytes(b"not an npz")
    corrupt = BENCHMARK.verify_prompt(path=str(tmp_path), unit={})
    assert not corrupt["ok"] and "unreadable" in corrupt["problems"][0]


def test_a_demonstration_of_one_frame_is_refused(tmp_path):
    _write_prompt(tmp_path / PROMPT_NAME, frames=1)
    verdict = BENCHMARK.verify_prompt(path=str(tmp_path), unit={})
    assert not verdict["ok"] and any("frames" in p for p in verdict["problems"])


def test_the_digest_is_the_bytes_so_a_third_party_can_recompute_it(tmp_path):
    a = _write_prompt(tmp_path / "a.npz")
    b = _write_prompt(tmp_path / "b.npz")
    assert prompt_sha256(a) == prompt_sha256(b)
    _write_prompt(tmp_path / "c.npz", seed=8)
    assert prompt_sha256(a) != prompt_sha256(tmp_path / "c.npz")


def test_only_meta_belongs_to_no_channel():
    """The orchestrator's view allows or drops whole channels, so an array in none of them would
    never reach a policy - which is correct for `meta`, and an oversight for anything else."""
    assert channel_of("frames_head_camera") == "video"
    assert channel_of("qpos") == "proprio" and channel_of("endpose") == "proprio"
    assert channel_of("actions") == "actions"
    assert channel_of("times") == "metadata"
    assert channel_of("meta") is None


# ------------------------------------------------------------------ results


def test_a_result_is_read_back_as_the_orchestrator_expects(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"success": True, "void": False, "steps": 41}))
    result = BENCHMARK.read_result(out_dir=str(tmp_path))
    assert result["success"] is True and result["void"] is False and result["steps"] == 41


def test_a_void_unit_has_no_success_either_way(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"void": True, "error": "scene drift"}))
    result = BENCHMARK.read_result(out_dir=str(tmp_path))
    assert result["void"] is True and result["success"] is None


def test_a_missing_result_is_void_rather_than_a_loss(tmp_path):
    """The subprocess died; that is the harness's fault, not the model's."""
    result = BENCHMARK.read_result(out_dir=str(tmp_path))
    assert result["void"] is True and result["success"] is None and result["error"]


def test_the_plugin_says_which_array_carries_which_channel():
    """The orchestrator's demonstration view allows or drops whole channels, and only the
    benchmark knows what its arrays mean. What a policy may *see* stays the orchestrator's
    decision - this is just the map it needs to apply one."""
    channels = BENCHMARK.info()["demo_channels"]
    assert channels["actions"] == ["actions"]
    assert set(channels["proprio"]) == {"qpos", "endpose"}
    assert channels["video"] == ["frames_*"]
    # Every channel the prompt writer knows about is published; a new one cannot be forgotten.
    from icil_benchmark_robotwin.prompt import CHANNELS

    assert set(channels) == set(CHANNELS)


def test_the_channel_map_is_spelled_the_way_the_orchestrator_reads_it():
    """Both spellings here are the orchestrator's, and getting either wrong fails silently: its
    view is an allow-list, so an entry that matches nothing hands the policy nothing.

    - a prefix ends in `*`, because the orchestrator cannot enumerate this benchmark's cameras;
    - `times` sits in `metadata`, the channel every view keeps, because no field's modality list
      claims frame timestamps and a view that dropped them would leave the demonstration untimed.
    """
    channels = BENCHMARK.info()["demo_channels"]
    assert channels["video"] == ["frames_*"]
    assert channels["metadata"] == ["times"]
    # The prefix matches what `dump` writes, whatever cameras a scene config turns on.
    prefix = channels["video"][0].removesuffix("*")
    for camera in ("head_camera", "left_camera", "right_camera", "front_camera"):
        assert f"frames_{camera}".startswith(prefix)
