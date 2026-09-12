import numpy as np
import pytest

from icil_policies.common import oracles
from icil_policies.common.oracles import (
    ResampledQposReplay,
    next_targets,
    resampled_qpos_actions,
)
from icil_policies.common.resample import resample
from icil_policies.testing import assert_conversion_pinned, conversion_digest
from robotwin_icil import robotwin, tasks
from robotwin_icil.episode import EpisodeSpec, run_episode
from robotwin_icil.policy import ReplayPolicy
from robotwin_icil.records import Status
from synthetic import demonstration, endpose_row, observation

# One entry per ADAPTER_VERSION ever released; added, never edited.
PINS = {"1": "5df4811589dedc2b4d1c2498e80050b361eef0d631bf740455695f7332c4dac5"}


def _timed(steps, joint=None):
    """A demonstration whose first joint follows `joint(t)`, framed at physics steps `steps`."""
    times = np.asarray(steps) / 250
    qpos = np.zeros((len(times), 14))
    qpos[:, 0] = times if joint is None else joint(times)
    qpos[:, 6] = (times < times[-1] / 2).astype(float)  # the left gripper closes halfway
    return demonstration([endpose_row() for _ in times], times=times, qpos=qpos)


@pytest.fixture
def fake(monkeypatch):
    """The core's fake RoboTwin env, from the core's tests."""
    import pathlib

    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
    import fake_robotwin

    monkeypatch.setattr(robotwin, "unstable_error", lambda: fake_robotwin.FakeUnstable)
    return fake_robotwin


def test_next_targets_take_the_next_row_and_hold_the_last():
    states = np.arange(8.0).reshape(4, 2)
    np.testing.assert_array_equal(next_targets(states), [[2, 3], [4, 5], [6, 7], [6, 7]])
    np.testing.assert_array_equal(next_targets(states[:1]), states[:1])


def test_one_action_per_resampled_step():
    demo = _timed([0, 1, 16, 31, 40, 40, 41, 56, 71, 86, 100, 250])
    steps = resample(demo, 20.0)
    actions = resampled_qpos_actions(demo, 20.0)
    assert actions.shape == (len(steps), 14)
    np.testing.assert_array_equal(actions[:-1], steps.qpos[1:])
    np.testing.assert_array_equal(actions[-1], steps.qpos[-1])
    # 1 s of demonstration at 20 Hz is 21 steps, whatever its frame count.
    assert len(steps) == 21 and len(demo) == 12


def test_the_replay_plays_each_target_once_then_holds_the_last():
    demo = _timed(np.arange(0, 126, 5))  # 0.5 s, 26 frames
    policy = ResampledQposReplay()
    policy.reset()
    policy.set_demonstration(demo)
    expected = resampled_qpos_actions(demo)
    played = [policy.act(observation(endpose_row(), step=i))[0] for i in range(len(expected) + 3)]
    np.testing.assert_array_equal(played[: len(expected)], expected)
    for action in played[len(expected) :]:
        np.testing.assert_array_equal(action, expected[-1])
    assert policy.episode_info() == {"resampled_steps": 11, "calls": 14}
    assert policy.describe()["adapter_version"] == oracles.ADAPTER_VERSION


def test_a_rate_must_be_positive():
    for rate in (0, -20, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ResampledQposReplay(rate_hz=rate)


def test_the_conversion_is_pinned_to_its_version():
    demo = _timed([0, 1, 16, 31, 40, 40, 41, 56, 71], joint=np.sin)
    assert_conversion_pinned(oracles, conversion_digest(resampled_qpos_actions(demo)), PINS)


def test_it_solves_the_fake_task_from_the_same_scene(fake):
    spec = EpisodeSpec(
        episode=0,
        task=tasks.table()["place_object_basket"],
        global_seed=0,
        max_expert_attempts=5,
    )
    env = fake.FakeTaskEnv(expert_steps=60, step_lim=100)
    policy = ResampledQposReplay()
    record = run_episode(spec, policy, fake.FakeConfig(), task_env=env)
    assert record.status is Status.SCORED and record.success
    assert set(env.action_types) == {"qpos"}
    # 60 frames 1/250 s apart are 0.24 s: six 20 Hz steps, so five moves reach the target,
    # where the native replay spends one call per frame.
    assert record.steps == 5 and policy.episode_info()["resampled_steps"] == 6
    native = run_episode(
        spec,
        ReplayPolicy(),
        fake.FakeConfig(),
        task_env=fake.FakeTaskEnv(expert_steps=60, step_lim=100),
    )
    assert native.success and native.steps == 60
