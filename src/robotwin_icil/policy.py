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
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from .demo import BIMANUAL_EE_DIM, BIMANUAL_QPOS_DIM, Demonstration

# Given to models that require a language input, so that what is measured is the demonstration
# and not the prompt. Never the task name or RoboTwin's per-task instruction.
NEUTRAL_INSTRUCTION = "Follow the demonstrated behavior."

ActionType = Literal["qpos", "ee"]
_ACTION_DIMS: dict[str, int] = {"qpos": BIMANUAL_QPOS_DIM, "ee": BIMANUAL_EE_DIM}


class PolicyError(RuntimeError):
    """A policy was driven outside the reset -> demonstrate -> act lifecycle, or returned junk."""


@dataclass(frozen=True)
class Observation:
    """What the policy sees at one control step: the same modalities as a demonstration frame.

    Deliberately no task name, scene seed, success condition, object identities or actor handles.
    `time_s` is simulated seconds since the rollout started, counted in physics steps, as a
    frame's `time_s` is since the expert started; None when the caller kept no clock.
    `gripper_joints` are the measured gripper joint positions per arm, as in a `Frame`; None
    when the caller read none.
    """

    step: int
    images: dict[str, np.ndarray]
    qpos: np.ndarray
    endpose: dict[str, Any] = field(default_factory=dict)
    instruction: str = NEUTRAL_INSTRUCTION
    time_s: float | None = None
    gripper_joints: dict[str, np.ndarray] | None = None


class ICILPolicy:
    """Base class for every policy the benchmark runs.

    Subclasses implement `_reset`, `_set_demonstration` and `_act`; the public methods keep the
    protocol: exactly one demonstration per episode, handed over before the first action.
    """

    name: ClassVar[str] = "icil"
    # Not a ClassVar: a policy whose action path depends on how it was configured sets this per
    # instance (`remote` takes it from the server it connected to), and one class per action
    # type would be a class per configuration.
    action_type: ActionType = "qpos"

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

    def act(self, observation: Observation) -> np.ndarray:
        """The actions to execute before the next observation, shape (k, action_dim), k >= 1."""
        if self._demonstration is None:
            raise PolicyError(f"{self.name}: act() before set_demonstration()")
        actions = np.asarray(self._act(observation), dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        expected = _ACTION_DIMS[self.action_type]
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

    # Optional hooks. Each has a no-op default, so a policy overrides only those it needs.

    def seed(self, seed: int) -> None:
        """Seed this episode's sampling. Called before every `reset()`, never with the scene seed.

        The seed comes from the policy's own stream, a function of the run's global seed and the
        episode index, so an episode samples the same way whether or not the run was resumed.
        RoboTwin seeds torch's global RNG whenever it builds a scene: draw noise from a generator
        of your own seeded here, never from a global RNG.
        """

    def episode_info(self) -> dict[str, Any]:
        """What this policy reports about the episode it just rolled out, e.g. the arm it drove or
        how many actions it clipped. Called after every rollout and stored in the episode record's
        `policy_info`; every value must be JSON-serialisable."""
        return {}

    def close(self) -> None:
        """Release what the policy holds: a model on the GPU, a connection to a model server.

        Called exactly once, when the run ends, however it ends.
        """

    def environment(self) -> dict[str, str]:
        """The policy's own software environment, as strings: python, torch, CUDA, the GPU, the
        commits of the model's repositories. Recorded beside the run; like the benchmark's own
        environment, it may differ on resume."""
        return {}

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
