import numpy as np

from robotwin_icil import generate
from robotwin_icil.demo import BIMANUAL_QPOS_DIM, Demonstration, Frame
from robotwin_icil.generate import Attempt, Rejection
from robotwin_icil.scene import SceneFingerprint


def _demonstration() -> Demonstration:
    frames = tuple(
        Frame(index=i, images={}, qpos=np.full(BIMANUAL_QPOS_DIM, float(i)), endpose={})
        for i in range(3)
    )
    return Demonstration(frames=frames, frequency=15)


def _fingerprint() -> SceneFingerprint:
    return SceneFingerprint(
        actors={},
        articulations={},
        articulation_roots={},
        cameras={},
        robot_qpos=np.zeros(BIMANUAL_QPOS_DIM),
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
