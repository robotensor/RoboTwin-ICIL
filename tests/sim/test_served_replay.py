"""A policy served at an address plays a saved prompt in the real simulator.

Needs the `icil-policy` distribution installed in the simulator environment, and its replay
example (see `tests/conftest.py`); skips without them.
"""

import json

import pytest

from robotwin_icil import cli, robotwin, unit

pytestmark = pytest.mark.sim

TASK = "click_bell"


@pytest.mark.parametrize("embodiment", ["aloha-agilex", "franka-panda"])
def test_the_served_replay_example_succeeds_from_the_saved_prompt(
    tmp_path, embodiment, serve_policy, replay_manifest
):
    # Rejected candidates are the expert's outcomes, not evidence; materialize tries the next.
    config = robotwin.SceneConfig(embodiment=embodiment)
    done = unit.materialize(TASK, list(range(8)), config, tmp_path / "prompt")
    if not done.ok:
        pytest.fail(f"the {embodiment} expert solved no {TASK} candidate: {done.result['error']}")

    served = serve_policy(replay_manifest)
    argv = ["run-unit", "--prompt", str(tmp_path / "prompt" / "prompt.npz")]
    argv += ["--policy-address", served.address, "--authkey-env", served.authkey_env]
    argv += ["--policy-log", str(served.log_file), "--out", str(tmp_path / "run")]
    assert cli.main(argv) == 0
    result = json.loads((tmp_path / "run" / "result.json").read_text())
    assert result["void"] is False, result["error"]
    assert result["success"] is True and result["void_cause"] is None
    assert result["policy"] == "remote" and result["served_policy"] == "replay.policy:ReplayPolicy"
    assert result["embodiment"] == embodiment and result["scene_seed"] == done.result["scene_seed"]
    assert result["steps"] > 0 and (tmp_path / "run" / "evaluation.mp4").is_file()
    # run-unit closed the client, and the server exited with it.
    assert served.process.wait(timeout=30) == 0
