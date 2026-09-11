"""Run a chunking model one action per `act()` (plan 3.4, Cadence).

BPP and UniSkill predict a chunk of `Tp` actions from a history of `To` observations and execute
the first `Ta` of it. `ChunkExecutor` keeps that bookkeeping out of every adapter:

- the history holds the last `To` observations, oldest first; at the start of an episode it is
  padded by repeating the first observation, as BPP's own wrapper pads it (no warm-up steps,
  which would come out of the task's step budget);
- one action leaves the queue per `act()`, so the benchmark's step limit counts model steps;
- the model is asked for a new chunk only when the queue is empty, from the history as it is
  then, and the first `Ta` rows of the chunk are queued.

It holds inference-time state only; `reset()` clears all of it and must run with the policy's
own `reset()`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from robotwin_icil.policy import PolicyError


class ChunkExecutor:
    """A history of `n_obs` observations and a queue of `n_action` actions for a chunking model.

    `plan(history)` receives the history, a list of `n_obs` observations oldest first, and
    returns at least `n_action` actions as the rows of a 2-D array; a batch axis, as in (1, Tp,
    action), is refused rather than guessed away.
    """

    def __init__(
        self, plan: Callable[[Sequence[Any]], np.ndarray], n_obs: int, n_action: int
    ) -> None:
        for name, value in (("n_obs", n_obs), ("n_action", n_action)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self._plan = plan
        self.n_obs = n_obs
        self.n_action = n_action
        self._history: deque[Any] = deque(maxlen=n_obs)
        self._queue: deque[np.ndarray] = deque()
        self.plans = 0

    def reset(self) -> None:
        """Forget the episode: its observations, its queued actions and its plan count."""
        self._history.clear()
        self._queue.clear()
        self.plans = 0

    @property
    def history(self) -> tuple[Any, ...]:
        return tuple(self._history)

    @property
    def pending(self) -> int:
        """Actions still queued from the last chunk."""
        return len(self._queue)

    def act(self, observation: Any) -> np.ndarray:
        """Record the observation, re-plan if the queue is empty, and return the next action."""
        if not self._history:
            self._history.extend([observation] * self.n_obs)
        else:
            self._history.append(observation)
        if not self._queue:
            chunk = np.asarray(self._plan(list(self._history)))
            if chunk.ndim != 2 or len(chunk) < self.n_action:
                raise PolicyError(
                    f"the model returned a chunk of shape {chunk.shape}; expected at least "
                    f"{self.n_action} actions as the rows of a 2-D array"
                )
            self._queue.extend(chunk[: self.n_action])
            self.plans += 1
        return self._queue.popleft()
