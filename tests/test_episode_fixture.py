"""The records `run_episode` writes are pinned, so splitting the episode in two cannot change them.

`tests/fixtures/eval_records.json` was written by the code before the episode was split into a
generation half and an evaluation half (#84); timings aside, the same inputs must still give the
same records.
"""

import json
from pathlib import Path

import pytest

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import robotwin, tasks
from robotwin_icil.episode import EpisodeSpec, run_episode
from robotwin_icil.generate import scene_seeds
from robotwin_icil.policy import DummyPolicy, ReplayPolicy

FIXTURE = Path(__file__).parent / "fixtures" / "eval_records.json"


@pytest.fixture(autouse=True)
def _fake_unstable(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)


def _spec(attempts=5):
    return EpisodeSpec(
        episode=0,
        task=tasks.table()["place_object_basket"],
        global_seed=0,
        max_expert_attempts=attempts,
    )


def _cases():
    seeds = scene_seeds(0, 0, 5)
    return {
        "replay_scored": lambda: run_episode(
            _spec(), ReplayPolicy(), FakeConfig(), task_env=FakeTaskEnv()
        ),
        "dummy_fails": lambda: run_episode(
            _spec(), DummyPolicy(), FakeConfig(), task_env=FakeTaskEnv(step_lim=20)
        ),
        "rejections_then_success": lambda: run_episode(
            _spec(),
            ReplayPolicy(),
            FakeConfig(),
            task_env=FakeTaskEnv(
                unstable_seeds={seeds[0]}, plan_fails_on={seeds[1]}, expert_misses_on={seeds[2]}
            ),
        ),
        "budget_exhausted": lambda: run_episode(
            _spec(attempts=2),
            ReplayPolicy(),
            FakeConfig(),
            task_env=FakeTaskEnv(unstable_seeds=set(scene_seeds(0, 0, 2))),
        ),
        "scene_drift": lambda: run_episode(
            _spec(), ReplayPolicy(), FakeConfig(), task_env=FakeTaskEnv(drift=True)
        ),
        "rollout_error": lambda: run_episode(
            _spec(), ReplayPolicy(), FakeConfig(), task_env=FakeTaskEnv(rollout_raises_at=2)
        ),
    }


@pytest.mark.parametrize("case", sorted(_cases()))
def test_eval_records_match_the_pre_refactor_fixture(case):
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))[case]
    record = _cases()[case]().to_json()
    record["duration_s"] = 0.0
    assert record == expected
