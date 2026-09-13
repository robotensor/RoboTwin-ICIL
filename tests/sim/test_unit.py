"""A demonstration built once and saved is evaluated from the file, in the real simulator."""

import pytest

from robotwin_icil import prompt, robotwin, scene, unit
from robotwin_icil.policy import ReplayPolicy

pytestmark = pytest.mark.sim

TASK = "click_bell"
QPOS_DIM = {"aloha-agilex": 14, "franka-panda": 16}


def _materialize(tmp_path, embodiment, seeds=range(8)):
    """Materialize the first seed the expert solves; rejected seeds are outcomes, not evidence."""
    config = robotwin.SceneConfig(embodiment=embodiment)
    outcomes = []
    for seed in seeds:
        done = unit.materialize(TASK, seed, config, tmp_path / f"seed{seed}")
        if done.ok:
            return seed, tmp_path / f"seed{seed}", done
        outcomes.append(f"seed {seed}: {done.result['rejection']} {done.result['detail']}".rstrip())
    pytest.fail(
        f"the {embodiment} expert solved none of {len(outcomes)} {TASK} seeds:\n"
        + "\n".join(outcomes)
    )


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
    assert result["scene_max_error"] == 0.0 and result["steps"] > 0
    assert (tmp_path / "run" / "evaluation.mp4").is_file()
    assert unit.read_result(tmp_path / "run") == result


def test_two_runs_from_one_prompt_rebuild_identical_scenes(tmp_path):
    seed, out, _ = _materialize(tmp_path, "aloha-agilex")
    _, meta = prompt.read_raw(out / "prompt.npz")
    recorded = scene.SceneFingerprint.from_json(meta["scene"]["fingerprint"])
    config = unit.scene_config_from(meta)

    # What run-unit sees on each run: the scene rebuilt from meta, fingerprinted before anyone acts.
    live = []
    task_env = robotwin.load_task(TASK)
    for _ in range(2):
        task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **config.resolve(TASK))
        try:
            live.append(robotwin.fingerprint(task_env))
        finally:
            task_env.close_env()
    for fingerprint in live:
        assert scene.compare(recorded, fingerprint) == []
    assert scene.compare(live[0], live[1]) == []
    assert scene.digest(live[0]) == scene.digest(live[1])

    for i in range(2):
        result = unit.run_unit(out / "prompt.npz", ReplayPolicy(), tmp_path / f"run{i}")
        assert result["void"] is False and result["scene_max_error"] == 0.0, result["error"]
