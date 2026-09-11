"""The resampled qpos replay in the real simulator: the ceiling of a 20 Hz qpos model.

The native replay plays every recorded frame and scores 18/18 on V1; the resampled replay plays
one target per 20 Hz step of the same demonstration. It must come out close to the native one;
V1's full measurement is a run of `--policy icil_policies.common.oracles:ResampledQposReplay`.
"""

import pytest

from icil_policies.common.oracles import ResampledQposReplay
from robotwin_icil import robotwin, tasks
from robotwin_icil.episode import EpisodeSpec, run_episode
from robotwin_icil.records import Status

pytestmark = pytest.mark.sim

TASK = "click_bell"  # the fastest expert


def test_the_resampled_replay_solves_click_bell():
    policy = ResampledQposReplay()
    spec = EpisodeSpec(episode=0, task=tasks.table()[TASK], global_seed=42, max_expert_attempts=5)
    record = run_episode(spec, policy, robotwin.SceneConfig())
    assert record.status is Status.SCORED, record.detail
    print(
        f"\nresampled_qpos_replay on {TASK}: success {record.success}, "
        f"{record.steps}/{record.step_limit} calls for {record.demonstration_frames} frames, "
        f"{policy.episode_info()}; {record.detail}"
    )
    assert record.success
