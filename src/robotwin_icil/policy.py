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
import inspect
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from .demo import BIMANUAL_EE_DIM, BIMANUAL_QPOS_DIM, Demonstration

# Given to models that require a language input, so that what is measured is the demonstration
# and not the prompt. Never the task name or RoboTwin's per-task instruction.
NEUTRAL_INSTRUCTION = "Follow the demonstrated behavior."

ActionType = Literal["qpos", "ee"]
_ACTION_DIMS: dict[str, int] = {"qpos": BIMANUAL_QPOS_DIM, "ee": BIMANUAL_EE_DIM}

# The `describe()` convention: optional keys an adapter fills in so a run records what it ran.
# `check_description` holds them to their types; other keys are the adapter's own.
DESCRIPTION_KEYS = (
    "adapter",
    "adapter_version",
    "checkpoint",
    "checkpoint_sha256",
    "training_tasks",
    "camera_profile_required",
    "parameter_checksum",
)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


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
        # The hook first: a demonstration it refuses is not kept, so act() still refuses to run
        # and a corrected demonstration can still be handed over.
        self._set_demonstration(demonstration)
        self._demonstration = demonstration

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
        """What the run manifest records about this policy: its name, model and checkpoint.

        Extend the base description with the keys of `DESCRIPTION_KEYS` that apply; the runner
        holds it to `check_description` when a run starts.
        """
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

        `runner.run` calls it exactly once, when the run ends, however it ends.
        """

    def environment(self) -> dict[str, str]:
        """The policy's own software environment, as strings: python, torch, CUDA, the GPU, the
        commits of the model's repositories. Recorded in the run manifest as
        `policy_environment`; like the benchmark's `environment`, it may differ on resume."""
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
        self._actions = self._played(demonstration)
        self._cursor = 0

    def _act(self, observation: Observation) -> np.ndarray:
        assert self._actions is not None
        # Past the end, hold the final target: the expert's last state is where it succeeded.
        index = min(self._cursor, len(self._actions) - 1)
        self._cursor += 1
        return self._actions[index]

    @staticmethod
    def _played(demonstration: Demonstration) -> np.ndarray:
        """The demonstration's actions, one row per call, in this policy's `action_type`."""
        return demonstration.actions()


class ReplayEEPolicy(ReplayPolicy):
    """Plays the demonstration's end-effector targets back through `take_action('ee')`.

    The ceiling of the `ee` path for any model: each call re-plans both arms with CuRobo to the
    next frame's flange pose and commanded gripper value. Where it falls short of `replay`, the
    loss is in that path — planning, its goal tolerance, at least 31 physics steps per call —
    not in a model.
    """

    name = "replay_ee"
    action_type = "ee"

    @staticmethod
    def _played(demonstration: Demonstration) -> np.ndarray:
        return demonstration.ee_actions()


def json_mapping(data: Any, what: str) -> dict[str, Any]:
    """`data` as plain JSON: string keys, JSON values, no NaN or infinity.

    Raises `PolicyError` naming the offending key. What comes back went through JSON once, tuples
    turned into lists, so a record or manifest reads back equal to what was written.
    """
    if not isinstance(data, Mapping):
        raise PolicyError(f"{what} must be a mapping, not {type(data).__name__}")
    plain: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise PolicyError(f"{what}: key {key!r} is not a string")
        try:
            text = json.dumps(value, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise PolicyError(f"{what}: {key!r} is not JSON-serialisable: {exc}") from exc
        plain[key] = json.loads(text)
    return plain


def check_description(description: Any) -> None:
    """Hold a `describe()` result to the convention, or raise `PolicyError` saying what is wrong.

    It is a mapping of JSON values naming the policy under `policy`, and of the optional keys:
    `adapter` and `adapter_version` (str), `checkpoint` (str or None), `checkpoint_sha256` (64 hex
    digits or None), `training_tasks` (a list of task names, or "unknown"),
    `camera_profile_required` (a camera profile's name or None) and `parameter_checksum` (str or
    None). Any other key is the adapter's own.
    """
    json_mapping(description, "describe()")
    if not isinstance(description.get("policy"), str):
        raise PolicyError("describe() must name the policy under 'policy', as the base class does")
    what = f"{description['policy']}: describe()"

    def fail(key: str, expected: str) -> PolicyError:
        return PolicyError(f"{what}: {key!r} must be {expected}, not {description[key]!r}")

    for key in ("adapter", "adapter_version"):
        if key in description and not isinstance(description[key], str):
            raise fail(key, "a string")
    for key in ("checkpoint", "camera_profile_required", "parameter_checksum"):
        if description.get(key) is not None and not isinstance(description[key], str):
            raise fail(key, "a string or None")
    digest = description.get("checkpoint_sha256")
    if digest is not None and not (isinstance(digest, str) and _SHA256.fullmatch(digest)):
        raise fail("checkpoint_sha256", "64 hex digits or None")
    if "training_tasks" in description:
        tasks = description["training_tasks"]
        if tasks != "unknown" and not (
            isinstance(tasks, (list, tuple)) and all(isinstance(t, str) and t for t in tasks)
        ):
            raise fail("training_tasks", 'a list of task names or "unknown"')


BUILTIN: dict[str, type[ICILPolicy]] = {
    "dummy": DummyPolicy,
    "replay": ReplayPolicy,
    "replay_ee": ReplayEEPolicy,
}


def make_policy(spec: str, **kwargs: Any) -> ICILPolicy:
    """A built-in name, or ``package.module:Class`` for an adapter living outside the core.

    `kwargs` go to the class's constructor; one it does not take is a `PolicyError`, not a
    traceback. Every policy must also construct with none.
    """
    if spec in BUILTIN:
        return _construct(spec, BUILTIN[spec], kwargs)
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
    return _construct(spec, cls, kwargs)


def _construct(spec: str, cls: type[ICILPolicy], kwargs: dict[str, Any]) -> ICILPolicy:
    # Bound first, so a TypeError from inside the adapter's constructor keeps its traceback.
    try:
        inspect.signature(cls).bind(**kwargs)
    except TypeError as exc:
        raise PolicyError(f"policy {spec!r} does not take these arguments: {exc}") from exc
    return cls(**kwargs)
