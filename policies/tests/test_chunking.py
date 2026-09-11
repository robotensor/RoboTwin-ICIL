import numpy as np
import pytest

from icil_policies.common.chunking import ChunkExecutor
from robotwin_icil.policy import PolicyError


class _Model:
    """Returns chunk rows [plan, row] and records every history it was asked with."""

    def __init__(self, horizon=16):
        self.horizon = horizon
        self.histories = []

    def __call__(self, history):
        self.histories.append(list(history))
        plan = len(self.histories)
        return np.array([[plan, row] for row in range(self.horizon)], dtype=np.float64)


def test_the_first_history_is_padded_by_repeating_the_first_observation():
    model = _Model()
    executor = ChunkExecutor(model, n_obs=2, n_action=12)
    executor.act("o0")
    assert model.histories == [["o0", "o0"]]
    executor.act("o1")
    assert executor.history == ("o0", "o1")


def test_one_action_per_call_and_a_new_chunk_when_the_queue_empties():
    model = _Model()
    executor = ChunkExecutor(model, n_obs=2, n_action=12)
    actions = [executor.act(f"o{i}") for i in range(30)]
    # Chunks of 16 are planned at calls 0, 12 and 24; the first 12 rows of each are executed.
    assert [a.tolist() for a in actions[:13]] == [[1, r] for r in range(12)] + [[2, 0]]
    assert executor.plans == 3 and executor.pending == 6
    assert model.histories[1] == ["o11", "o12"] and model.histories[2] == ["o23", "o24"]


def test_a_history_longer_than_the_episode_so_far_stays_padded_at_its_start():
    model = _Model(horizon=1)
    executor = ChunkExecutor(model, n_obs=3, n_action=1)
    for i in range(4):
        executor.act(i)
    assert model.histories == [[0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3]]


def test_reset_forgets_the_episode():
    model = _Model()
    executor = ChunkExecutor(model, n_obs=2, n_action=4)
    executor.act("a")
    executor.reset()
    assert executor.history == () and executor.pending == 0 and executor.plans == 0
    executor.act("b")
    assert model.histories[-1] == ["b", "b"]


@pytest.mark.parametrize(
    ("chunk", "n_action"),
    [
        (np.zeros((3, 2)), 4),
        (np.zeros(16), 4),
        (np.float64(0.0), 4),
        (np.zeros((1, 16, 10)), 4),  # batch-first, as BPP's predict_action returns
        (np.zeros((1, 16, 10)), 1),  # would otherwise queue the whole chunk as one action
    ],
)
def test_a_chunk_too_short_or_not_rows_is_a_policy_error(chunk, n_action):
    executor = ChunkExecutor(lambda history: chunk, n_obs=1, n_action=n_action)
    with pytest.raises(PolicyError, match=f"at least {n_action} actions as the rows of a 2-D"):
        executor.act("o")


@pytest.mark.parametrize(
    ("n_obs", "n_action"), [(0, 1), (1, 0), (2.0, 1), (1, "8"), (True, 1), (1, True)]
)
def test_sizes_must_be_positive_integers(n_obs, n_action):
    with pytest.raises(ValueError, match="positive integer"):
        ChunkExecutor(_Model(), n_obs=n_obs, n_action=n_action)
