import numpy as np
import pytest

from fake_robotwin import QPOS_DIM, FakeTaskEnv, FakeUnstable
from robotwin_icil import generate, robotwin
from robotwin_icil.demo import Demonstration, Frame, arms_moved
from robotwin_icil.generate import Attempt, Rejection
from robotwin_icil.scene import SceneFingerprint


def _demonstration() -> Demonstration:
    frames = tuple(
        Frame(index=i, images={}, qpos=np.full(QPOS_DIM, float(i)), endpose={}) for i in range(3)
    )
    return Demonstration(frames=frames, frequency=15)


def _fingerprint() -> SceneFingerprint:
    return SceneFingerprint(
        actors={},
        articulations={},
        articulation_roots={},
        cameras={},
        robot_qpos=np.zeros(QPOS_DIM),
    )


def _scripted(outcomes):
    """An attempt function that plays back a script of rejections, None meaning success."""
    calls = []

    def attempt_fn(task_env, seed, args, frequency, episode):
        calls.append(seed)
        rejection = outcomes[len(calls) - 1]
        if rejection is None:
            return Attempt(seed, None), _demonstration(), _fingerprint()
        return Attempt(seed, rejection), None, None

    return attempt_fn, calls


def test_scene_seeds_are_a_pure_function_of_run_seed_and_episode():
    assert generate.scene_seeds(42, 3, 5) == generate.scene_seeds(42, 3, 5)


def test_episodes_draw_independent_streams():
    # Episode 7's scene must not depend on how many seeds episodes 0-6 rejected.
    assert generate.scene_seeds(42, 0, 5) != generate.scene_seeds(42, 1, 5)
    assert generate.scene_seeds(42, 0, 8)[:5] == generate.scene_seeds(42, 0, 5)


def test_seeds_fit_numpy_seed_range():
    assert all(0 <= seed < 2**32 for seed in generate.scene_seeds(7, 0, 1000))


def test_generate_stops_at_the_first_success():
    attempt_fn, calls = _scripted([Rejection.UNSTABLE, Rejection.PLAN_FAILED, None, None])
    result = generate.generate(None, [10, 11, 12, 13], dict, 15, 0, attempt_fn=attempt_fn)
    assert result.ok and result.seed == 12
    assert calls == [10, 11, 12]
    assert [a.rejection for a in result.attempts] == [
        Rejection.UNSTABLE,
        Rejection.PLAN_FAILED,
        None,
    ]


def test_a_fully_rejected_budget_yields_no_demonstration():
    attempt_fn, calls = _scripted([Rejection.EXPERT_FAILED] * 3)
    result = generate.generate(None, [1, 2, 3], dict, 15, 0, attempt_fn=attempt_fn)
    assert not result.ok and result.seed is None and result.demonstration is None
    assert len(result.attempts) == 3


def test_every_attempt_gets_fresh_args():
    built = []

    def args_factory():
        built.append({})
        return built[-1]

    attempt_fn, _ = _scripted([Rejection.UNSTABLE, None])
    generate.generate(None, [1, 2], args_factory, 15, 0, attempt_fn=attempt_fn)
    assert len(built) == 2 and built[0] is not built[1]


def test_a_demonstrations_frequency_is_frames_per_second(monkeypatch):
    # RoboTwin records every `save_freq` physics steps of 1/250 s; 5 steps per frame is 50 fps,
    # not "5". Getting this wrong mis-times every demonstration a policy is handed.
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    result, demonstration, _ = generate.attempt(FakeTaskEnv(), 0, {"save_freq": 5}, 5, 0)
    assert result.rejection is None
    assert demonstration.frequency == pytest.approx(50.0)


@pytest.mark.parametrize("qpos_dim", [14, 16])
def test_a_demonstration_is_as_wide_as_the_robot_that_made_it(qpos_dim, monkeypatch):
    # Captured frames carry whatever `joint_action.vector` the scene's robot reports: 14 on
    # aloha-agilex, 16 on a dual Franka. Nothing in the capture path assumes one of them.
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    env = FakeTaskEnv(qpos_dim=qpos_dim)
    result, demonstration, initial = generate.attempt(env, 0, {"save_freq": 1}, 1, 0)
    assert result.rejection is None
    assert demonstration.qpos_dim == qpos_dim and demonstration.qpos().shape[1] == qpos_dim
    assert initial.robot_qpos.shape == (qpos_dim,)


def test_a_seed_whose_joints_read_non_finite_is_rejected_not_measured(monkeypatch):
    # The frame refuses the value while the expert runs, so the seed is a rejection with the
    # reason on record — not a demonstration that arms_moved would silently read as still.
    class NaNJointEnv(FakeTaskEnv):
        def get_obs(self):
            obs = super().get_obs()
            obs["joint_action"]["vector"][3] = np.nan
            return obs

    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    env = NaNJointEnv()
    result, demonstration, _ = generate.attempt(env, 0, {"save_freq": 1}, 1, 0)
    assert result.rejection is Rejection.EXPERT_ERROR and demonstration is None
    assert "non-finite" in result.detail
    assert env.closed == 1


@pytest.mark.parametrize("plan_fails", [False, True])
def test_an_attempt_without_images_renders_nothing_and_ends_the_same_way(plan_fails, monkeypatch):
    # The survey's attempt: no camera takes a picture, and the seed's outcome, frames and joints
    # are the ones the rendered attempt an episode makes would have recorded.
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    fails = {4} if plan_fails else set()
    rendered_env = FakeTaskEnv(moves=("left",), plan_fails_on=fails)
    plain_env = FakeTaskEnv(moves=("left",), plan_fails_on=fails)
    rendered, rendered_demo, _ = generate.attempt(rendered_env, 4, {"save_freq": 1}, 1, 0)
    plain, plain_demo, _ = generate.attempt(plain_env, 4, {"save_freq": 1}, 1, 0, images=False)

    assert plain == rendered and plain_env.get_obs_calls == 0 and plain_env.closed == 1
    if plan_fails:
        assert plain.rejection is Rejection.PLAN_FAILED and plain_demo is rendered_demo is None
        return
    assert rendered_env.get_obs_calls == len(rendered_demo) == len(plain_demo)
    np.testing.assert_array_equal(plain_demo.qpos(), rendered_demo.qpos())
    assert plain_demo.frequency == rendered_demo.frequency
    assert (plain_demo.cameras, rendered_demo.cameras) == ((), ("head_camera",))
    assert arms_moved(plain_demo) == arms_moved(rendered_demo) == ("left",)
