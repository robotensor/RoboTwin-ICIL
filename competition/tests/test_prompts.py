"""verify_prompt reads the file; read_result reads either command's result."""

import hashlib
import json

import numpy as np
import pytest

import robotwin_icil
from icil_benchmark_robotwin import BENCHMARK
from robotwin_icil import prompt


def test_the_prompt_a_unit_asked_for_is_verified(make_prompt, franka_unit):
    path = make_prompt()
    verdict = BENCHMARK.verify_prompt(path=str(path), unit=franka_unit)
    assert verdict["ok"] is True and verdict["problems"] == []
    assert verdict["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (verdict["task"], verdict["scene_seed"], verdict["embodiment"]) == (
        "click_bell",
        11,
        "franka-panda",
    )
    # The unit as a duel hands it over, reserved keys and all.
    duel = {**franka_unit, "unit_id": "fu-000", "skill": "franka_press_push", "seed": 9}
    assert BENCHMARK.verify_prompt(path=str(path), unit=duel)["ok"]


@pytest.mark.parametrize(
    ("made", "reason"),
    [
        ({"task": "click_alarmclock"}, "the prompt is of task 'click_alarmclock'"),
        ({"scene_seed": 12}, "scene seed 12 is not one of the unit's candidates"),
        ({"embodiment": "aloha-agilex"}, "built on 'aloha-agilex'"),
        ({"qpos_dim": 14}, "qpos is 14 wide; franka-panda is 16"),
    ],
)
def test_a_prompt_that_is_not_the_units_is_refused(make_prompt, franka_unit, made, reason):
    verdict = BENCHMARK.verify_prompt(path=str(make_prompt(**made)), unit=franka_unit)
    assert verdict["ok"] is False and any(reason in p for p in verdict["problems"])
    assert len(verdict["sha256"]) == 64


def _rewrite(path, arrays=None, meta=None):
    old_arrays, old_meta = prompt.read_raw(path)
    arrays = old_arrays if arrays is None else arrays(old_arrays)
    meta = old_meta if meta is None else meta(old_meta)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays, meta=json.dumps(meta))


def test_a_pickled_npz_is_refused_without_being_unpickled(make_prompt, franka_unit, tmp_path):
    path = make_prompt()
    arrays, meta = prompt.read_raw(path)
    pickled = tmp_path / "pickled.npz"
    with open(pickled, "wb") as handle:
        np.savez(handle, **arrays, meta=json.dumps(meta), extra=np.array([{"a": 1}], dtype=object))
    verdict = BENCHMARK.verify_prompt(path=str(pickled), unit=franka_unit)
    assert verdict["ok"] is False and "allow_pickle=False" in " ".join(verdict["problems"])


@pytest.mark.parametrize(
    ("arrays", "meta", "reason"),
    [
        (lambda a: {**a, "notes": np.zeros(3)}, None, "arrays in no published channel: notes"),
        (lambda a: {k: v for k, v in a.items() if k != "actions"}, None, "no 'actions' array"),
        (lambda a: {**a, "qpos": a["qpos"].astype(np.float32)}, None, "qpos is float32"),
        (None, lambda m: {**m, "frames": 99}, "meta says 99 frames"),
        (None, lambda m: {**m, "scene": {**m["scene"], "sha256": "0" * 64}}, "was edited"),
        (None, lambda m: {k: v for k, v in m.items() if k != "scene"}, "no readable scene"),
        (
            None,
            lambda m: {**m, "embodiment": {**m["embodiment"], "choice": None}},
            "chosen as None",
        ),
    ],
)
def test_a_prompt_whose_arrays_or_meta_do_not_hold_up_is_refused(
    make_prompt, franka_unit, arrays, meta, reason
):
    path = make_prompt()
    _rewrite(path, arrays, meta)
    verdict = BENCHMARK.verify_prompt(path=str(path), unit=franka_unit)
    assert verdict["ok"] is False and any(reason in p for p in verdict["problems"]), verdict


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("task_config", "demo_randomized"),
        ("save_freq", 5),
        ("save_freq", 15.0),
        ("overrides", {"step_lim": 100000}),
        ("head_camera", "D435"),
        ("evaluation_setting", "different_object_pose"),
        ("task_config", None),
    ],
)
def test_a_prompt_built_under_another_scene_config_is_refused(make_prompt, franka_unit, key, value):
    # run-unit rebuilds and scores the scene from these: its task, seed and robot can all be the
    # unit's and it still be another unit.
    path = make_prompt()
    _rewrite(path, meta=lambda m: {**m, key: value})
    verdict = BENCHMARK.verify_prompt(path=str(path), unit=franka_unit)
    assert verdict["ok"] is False
    assert any(f"the prompt's {key} is {value!r}" in p for p in verdict["problems"]), verdict


def _expert(**changes):
    """A meta edit replacing fields of the expert record `make_prompt` wrote."""
    return lambda m: {**m, "expert": {**m["expert"], **changes}}


KEPT_11 = {"seed": 11, "rejection": None, "detail": ""}


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (_expert(scene_seeds=[11]), "was given candidates [11]; the unit's are [5, 11, 17, 23]"),
        (
            # A later candidate cherry-picked: the first was never tried.
            _expert(attempts=[KEPT_11]),
            "tried seeds [11], not the unit's candidates in order",
        ),
        (
            _expert(attempts=[{"seed": 5, "rejection": None, "detail": ""}, KEPT_11]),
            "keeps no rejection for candidate(s) [5]",
        ),
        (
            _expert(attempts=[{"seed": 5, "rejection": "unstable", "detail": ""}]),
            "last tried seed 5, not its scene seed 11",
        ),
        (
            _expert(
                attempts=[
                    {"seed": 5, "rejection": "unstable", "detail": ""},
                    {**KEPT_11, "rejection": "unstable"},
                ]
            ),
            "rejects the seed the prompt was built on",
        ),
        (lambda m: {k: v for k, v in m.items() if k != "expert"}, "was given candidates None"),
    ],
)
def test_a_prompt_whose_expert_record_is_not_the_units_is_refused(
    make_prompt, franka_unit, edit, reason
):
    # The scene seed alone being a candidate says nothing of how it was chosen.
    path = make_prompt()
    _rewrite(path, meta=edit)
    verdict = BENCHMARK.verify_prompt(path=str(path), unit=franka_unit)
    assert verdict["ok"] is False and any(reason in p for p in verdict["problems"]), verdict


def test_a_prompt_holding_a_camera_no_unit_observes_is_refused(make_prompt, franka_unit):
    # run-unit hands the policy every camera a prompt holds; meta edited to match hides nothing.
    path = make_prompt()

    def extra_camera(arrays):
        return {**arrays, "frames_extra": np.zeros_like(arrays["frames_head_camera"])}

    _rewrite(path, extra_camera, lambda m: {**m, "cameras": ["extra", "head_camera"]})
    verdict = BENCHMARK.verify_prompt(path=str(path), unit=franka_unit)
    assert verdict["ok"] is False
    assert any("camera(s) extra, which no unit observes" in p for p in verdict["problems"])
    assert len(verdict["problems"]) == 1, verdict  # the arrays and meta agree with each other


def test_a_prompt_whose_meta_leaves_out_its_scene_config_is_refused(make_prompt, franka_unit):
    path = make_prompt()
    _rewrite(path, meta=lambda m: {k: v for k, v in m.items() if k != "overrides"})
    verdict = BENCHMARK.verify_prompt(path=str(path), unit=franka_unit)
    assert verdict["ok"] is False and any("overrides is None" in p for p in verdict["problems"])


def test_an_unreadable_prompt_is_refused(tmp_path, franka_unit):
    missing = BENCHMARK.verify_prompt(path=str(tmp_path / "missing.npz"), unit=franka_unit)
    assert missing["ok"] is False and missing["sha256"] is None
    (tmp_path / "junk.npz").write_bytes(b"not a zip")
    junk = BENCHMARK.verify_prompt(path=str(tmp_path / "junk.npz"), unit=franka_unit)
    assert junk["ok"] is False and "cannot read prompt" in junk["problems"][0]


#: The benchmark source this plugin runs, which every command it builds records in its result.
MINE = robotwin_icil.source_sha256()


def _result(tmp_path, doc):
    """`doc` written as result.json — an object stamped with this plugin's source unless it names
    one — and read back."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    if isinstance(doc, dict):
        doc = {"source_sha256": MINE, **doc}
    (tmp_path / "result.json").write_text(json.dumps(doc) if not isinstance(doc, str) else doc)
    return BENCHMARK.read_result(out_dir=str(tmp_path))


def test_a_scored_success_and_a_failure_are_read_as_written(tmp_path):
    written = {"success": True, "void": False, "void_cause": None, "steps": 36, "error": None}
    result = _result(tmp_path / "ok", {**written, "live_scene_sha256": "ab" * 32})
    assert result == {**written, "live_scene_sha256": "ab" * 32, "source_sha256": MINE}
    failed = _result(tmp_path / "failed", {**written, "success": False, "detail": "boom"})
    assert failed["success"] is False and failed["void"] is False and failed["void_cause"] is None


@pytest.mark.parametrize("cause", ["harness", "policy"])
def test_a_void_result_keeps_its_cause_and_its_reason(tmp_path, cause):
    doc = {"success": None, "void": True, "void_cause": cause, "steps": None, "error": "why"}
    result = _result(tmp_path, doc)
    assert result == {**doc, "source_sha256": MINE}


@pytest.mark.parametrize("source", ["0" * 64, None])
def test_a_result_other_benchmark_source_wrote_is_void_on_the_harness(tmp_path, source):
    # A scored success, as far as its fields go, from code this plugin did not build the argv for.
    doc = {"success": True, "void": False, "void_cause": None, "steps": 36, "error": None}
    if source is not None:
        doc["source_sha256"] = source
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "result.json").write_text(json.dumps(doc))
    result = BENCHMARK.read_result(out_dir=str(tmp_path))
    assert result["void"] is True and result["success"] is None and result["steps"] is None
    assert result["void_cause"] == "harness"
    assert f"written by benchmark source {source!r}, not the {MINE}" in result["error"]


def test_every_candidate_rejected_is_a_void_harness_result(tmp_path):
    doc = {
        "ok": False,
        "success": None,
        "void": True,
        "void_cause": "harness",
        "steps": None,
        "error": "expert rejected all 4 candidate seeds: ...",
        "attempts": [{"seed": 5, "rejection": "unstable", "detail": ""}],
    }
    result = _result(tmp_path, doc)
    assert result["void"] is True and result["void_cause"] == "harness"
    assert result["attempts"] == doc["attempts"]


@pytest.mark.parametrize(
    ("doc", "error"),
    [
        ("{not json", "unreadable result.json"),
        ("[1, 2]", "holds list, not an object"),
        ({"success": None, "void": False, "steps": 3}, "without a reason"),
        ({"success": True}, "without a reason"),
        ({"success": None, "void": True, "void_cause": "martians"}, "without a reason"),
    ],
)
def test_a_result_that_does_not_hold_up_is_void_on_the_harness(tmp_path, doc, error):
    result = _result(tmp_path, doc)
    assert result["void"] is True and result["success"] is None and result["steps"] is None
    assert result["void_cause"] == "harness" and error in result["error"]


def test_a_missing_result_is_void_on_the_harness(tmp_path):
    result = BENCHMARK.read_result(out_dir=str(tmp_path))
    assert result["void"] is True and result["void_cause"] == "harness"
    assert "unreadable result.json" in result["error"]
