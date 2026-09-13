"""The policy side of the protocol: reset, receive exactly one demonstration, observe, act.

A policy is frozen for the whole benchmark. It adapts only through the demonstration it is handed;
inference-time state (a KV cache, observation history, a recurrent state) is fine as long as
`reset()` clears it, so every episode starts from nothing.

`ICILPolicy` enforces that lifecycle so an adapter cannot get it wrong quietly: acting before it has
a demonstration, or being handed a second one within an episode, is an error rather than a silently
different benchmark.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from .demo import Demonstration

# Given to models that require a language input, so that what is measured is the demonstration
# and not the prompt. Never the task name or RoboTwin's per-task instruction.
NEUTRAL_INSTRUCTION = "Follow the demonstrated behavior."

ActionType = Literal["qpos", "ee"]

# The width of an action per action type, read off the live robot by `robotwin.action_dims`:
# {"qpos": 14, "ee": 16} on aloha-agilex, {"qpos": 16, "ee": 16} on two Franka arms.
ActionDims = Mapping[str, int]


class PolicyError(RuntimeError):
    """A policy was driven outside the reset -> demonstrate -> act lifecycle, or returned junk."""


@dataclass(frozen=True)
class Observation:
    """What the policy sees at one control step: the same modalities as a demonstration frame.

    Deliberately no task name, scene seed, success condition, object identities or actor handles.
    """

    step: int
    images: dict[str, np.ndarray]
    qpos: np.ndarray
    endpose: dict[str, Any] = field(default_factory=dict)
    instruction: str = NEUTRAL_INSTRUCTION


class ICILPolicy:
    """Base class for every policy the benchmark runs.

    Subclasses implement `_reset`, `_set_demonstration` and `_act`; the public methods keep the
    protocol: exactly one demonstration per episode, handed over before the first action.
    """

    name: ClassVar[str] = "icil"
    action_type: ClassVar[ActionType] = "qpos"

    def __init__(self) -> None:
        self._demonstration: Demonstration | None = None
        self._was_reset = False

    def reset(self) -> None:
        """Forget everything from the previous episode, its demonstration included."""
        self._demonstration = None
        self._was_reset = True
        self._reset()

    def set_demonstration(self, demonstration: Demonstration) -> None:
        if not self._was_reset:
            raise PolicyError(f"{self.name}: reset() must come before set_demonstration()")
        if self._demonstration is not None:
            raise PolicyError(f"{self.name}: an episode gets exactly one demonstration")
        self._demonstration = demonstration
        self._set_demonstration(demonstration)

    def act(self, observation: Observation, action_dims: ActionDims) -> np.ndarray:
        """The actions to execute before the next observation, shape (k, action_dim), k >= 1.

        `action_dims` is the robot's own width per action type, which the harness reads off the
        live arms every rollout. RoboTwin's `take_action` splits an action by the arms it has, so
        an action of another robot's width would be mis-read joint by joint rather than refused.
        """
        if self._demonstration is None:
            raise PolicyError(f"{self.name}: act() before set_demonstration()")
        expected = action_dims.get(self.action_type)
        if expected is None:
            raise PolicyError(
                f"{self.name}: no action width for action_type {self.action_type!r}; "
                f"the robot takes {sorted(action_dims)}"
            )
        actions = np.asarray(self._act(observation), dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != expected:
            raise PolicyError(
                f"{self.name}: act() returned shape {actions.shape}, expected (k, {expected}) "
                f"for action_type {self.action_type!r}"
            )
        if not np.all(np.isfinite(actions)):
            raise PolicyError(f"{self.name}: act() returned a non-finite action")
        return actions

    def describe(self) -> dict[str, Any]:
        """What the run manifest records about this policy: its name, model and checkpoint."""
        return {"policy": self.name, "action_type": self.action_type}

    def _reset(self) -> None:
        """Clear inference-time state. Called at the start of every episode."""

    def _set_demonstration(self, demonstration: Demonstration) -> None:
        """Consume the one demonstration, e.g. encode it into a context."""

    def _act(self, observation: Observation) -> np.ndarray:
        raise NotImplementedError


class DummyPolicy(ICILPolicy):
    """Holds the robot where it is. Exercises the loop; scores near zero by design."""

    name = "dummy"

    def _act(self, observation: Observation) -> np.ndarray:
        return observation.qpos


class ReplayPolicy(ICILPolicy):
    """Plays the demonstration's actions back verbatim, ignoring what it observes.

    Under Same Scene the rollout starts from the state the expert started from, so this is the
    harness's own upper bound. When it fails, the bug is in the reset, the action path or the
    success check — not in a model.
    """

    name = "replay"

    def _reset(self) -> None:
        self._actions: np.ndarray | None = None
        self._cursor = 0

    def _set_demonstration(self, demonstration: Demonstration) -> None:
        self._actions = demonstration.actions()
        self._cursor = 0

    def _act(self, observation: Observation) -> np.ndarray:
        assert self._actions is not None
        # Past the end, hold the final target: the expert's last state is where it succeeded.
        index = min(self._cursor, len(self._actions) - 1)
        self._cursor += 1
        return self._actions[index]


BUILTIN: dict[str, type[ICILPolicy]] = {"dummy": DummyPolicy, "replay": ReplayPolicy}


def make_policy(spec: str, **kwargs: Any) -> ICILPolicy:
    """A built-in name, or ``package.module:Class`` for an adapter living outside the core."""
    if spec in BUILTIN:
        return BUILTIN[spec](**kwargs)
    module_name, sep, class_name = spec.partition(":")
    if not sep:
        known = ", ".join(sorted(BUILTIN))
        raise PolicyError(f"unknown policy {spec!r}; built-ins are {known}, or give module:Class")
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise PolicyError(f"cannot load policy {spec!r}: {exc}") from exc
    if not (isinstance(cls, type) and issubclass(cls, ICILPolicy)):
        raise PolicyError(f"{spec!r} is not an ICILPolicy subclass")
    return cls(**kwargs)
