"""A demonstration built once and saved is evaluated from the file, in the real simulator."""

import pytest

from robotwin_icil import prompt, robotwin, unit
from robotwin_icil.policy import ReplayPolicy

pytestmark = pytest.mark.sim

TASK = "click_bell"
QPOS_DIM = {"aloha-agilex": 14, "franka-panda": 16}


def _materialize(tmp_path, embodiment, seeds=range(8)):
    """Materialize the first candidate the expert solves; rejected seeds are outcomes, not
    evidence, and materialize itself tries the next one."""
    out = tmp_path / "prompt"
    done = unit.materialize(TASK, list(seeds), robotwin.SceneConfig(embodiment=embodiment), out)
    if not done.ok:
        pytest.fail(f"the {embodiment} expert solved no {TASK} candidate: {done.result['error']}")
    assert done.result["attempts"][-1] == {
        "seed": done.result["scene_seed"],
        "rejection": None,
        "detail": "",
    }
    return done.result["scene_seed"], out, done


@pytest.mark.parametrize("embodiment", sorted(QPOS_DIM))
def test_replay_succeeds_from_the_saved_prompt(tmp_path, embodiment):
    # The whole loop through the files: the expert's demonstration written by materialize, read
    # back by run-unit, its scene rebuilt and verified, and the replay oracle solving it.
    seed, out, done = _materialize(tmp_path, embodiment)
    assert {p.name for p in out.iterdir()} == {"prompt.npz", "demonstration.mp4", "result.json"}
    assert unit.read_result(out)["ok"] is True

    demonstration, meta = prompt.read_prompt(out / "prompt.npz")
    assert meta["task"] == TASK and meta["scene_seed"] == seed
    assert meta["embodiment"]["name"] == embodiment
    assert demonstration.qpos_dim == QPOS_DIM[embodiment] and demonstration.timed
    assert len(demonstration) == meta["frames"] == done.result["frames"]

    result = unit.run_unit(out / "prompt.npz", ReplayPolicy(), tmp_path / "run")
    assert result["void"] is False, result["error"]
    assert result["success"] is True and result["embodiment"] == embodiment
    assert result["steps"] > 0 and result["live_scene_sha256"]
    assert (tmp_path / "run" / "evaluation.mp4").is_file()
    assert unit.read_result(tmp_path / "run") == result


def test_two_runs_from_one_prompt_rebuild_identical_scenes(tmp_path):
    # Observed through run-unit itself: each run records the digest of the scene it rebuilt from
    # meta, fingerprinted before anyone acted. Identical fingerprints are identical digests.
    _, out, done = _materialize(tmp_path, "aloha-agilex")
    results = [
        unit.run_unit(out / "prompt.npz", ReplayPolicy(), tmp_path / f"run{i}") for i in range(2)
    ]
    for result in results:
        assert result["void"] is False, result["error"]
        assert result["scene_sha256"] == done.result["scene_sha256"]
    assert results[0]["live_scene_sha256"] == results[1]["live_scene_sha256"]
    assert results[0]["scene_max_error"] == results[1]["scene_max_error"]
