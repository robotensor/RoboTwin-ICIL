"""A served policy records what it records in process, in the real simulator.

Milestone M1's check of the policy server: replay behind `python -m robotwin_icil.serve` (here
under this same interpreter) against replay in process, on the same task, seed and config.
"""

import pytest

from robotwin_icil import robotwin, tasks
from robotwin_icil.episode import EpisodeSpec, run_episode
from robotwin_icil.policy import ReplayPolicy
from robotwin_icil.records import Status
from robotwin_icil.remote import RemotePolicy

pytestmark = pytest.mark.sim

TASK = "click_bell"


def test_served_replay_records_what_in_process_replay_records(tmp_path):
    # Built first, as `eval` builds it: before the RoboTwin seam moves the working directory.
    served = RemotePolicy(policy="replay", log=tmp_path / "serve.log")
    spec = EpisodeSpec(episode=0, task=tasks.table()[TASK], global_seed=42, max_expert_attempts=5)
    try:
        local = run_episode(spec, ReplayPolicy(), robotwin.SceneConfig())
        remote = run_episode(spec, served, robotwin.SceneConfig())
    finally:
        served.close()
    assert local.status is Status.SCORED and local.success
    untimed = [{**record.to_json(), "duration_s": 0.0} for record in (local, remote)]
    assert untimed[1] == untimed[0]
    assert served._server.process.returncode == 0
