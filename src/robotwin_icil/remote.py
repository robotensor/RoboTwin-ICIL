"""A policy served at an address: `robotwin-icil run-unit --policy-address ADDR --authkey-env NAME`.

A competitor's policy is untrusted code in a container of its own, so the benchmark cannot import
it. It is served by `python -m icil_policy.serve` (the `icil-policy` distribution), and
`RemotePolicy` is the `ICILPolicy` that drives it through `icil_policy.client.RemotePolicy`. That
client is imported only when a `RemotePolicy` is built: the benchmark imports, runs and is tested
without `icil_policy` installed.

What crosses the socket is public, under the names `prompt.npz` gives it:

    hello                 -> the policy's action_type ("qpos" or "ee") and its name
    reset(seed)           <- `EpisodeInfo.seed`: `unit.episode_seed` of the prompt's bytes, never
                             the scene seed
    prompt(arrays, info)  <- `prompt.arrays_from(demonstration)`, the very arrays `write_prompt`
                             writes (`frames_<camera>`, `qpos`, `endpose`, `actions`, `times`,
                             `frequency`), and info {"frequency", "cameras", "embodiment",
                             "action_dims"}; never `meta`
    act(observation)      <- `frames_<camera>`, `qpos` and `endpose` of the live robot
                          -> "action", (A,) or (H, A); `ICILPolicy.act` checks its width against
                             `action_dims[action_type]` and hands the rows to the rollout

How a remote failure ends the unit follows how `icil_policy.client` reports one. Every failure is
a `PolicyUnavailable`, whose `remote_type` is set exactly when the server sent an error reply:

- an error reply to `reset`, `prompt` or `act` is the policy raising. It has failed, as a local
  policy that raises has: `PolicyError`, so the unit is scored, success false, never void.
- anything else is a policy that could not be spoken to: nothing listened, the key was refused,
  `hello` was refused or unanswered (an error reply to `hello` is a policy the server could not
  build, and the server ends the session), an action type the benchmark cannot execute, a call
  past its timeout, a hang-up, a malformed reply. `PolicyUnreachable`: `evaluate` re-raises it and
  `run-unit` writes void with `void_cause` "policy", the client's message — ending with the
  server log's tail when `log_file` names the log — as the error.
- a message the client refuses before sending it (`WireError`) holds the benchmark's own arrays:
  `Unscorable`, void on the harness.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import numpy as np

from .demo import Demonstration
from .policy import ICILPolicy, Observation, PolicyError, PolicyUnreachable, Unscorable
from .prompt import FRAMES_PREFIX, PromptError, arrays_from, flatten_endpose

#: The action types `take_action` executes; a policy declaring another is refused at `hello`.
ACTION_TYPES = ("qpos", "ee")
#: The calls whose error reply is the policy's own failure. Not `hello`: an error reply to it is a
#: policy the server could not build, and the session is over.
POLICY_OPS = ("reset", "prompt", "act")
#: The shortest key the policy wire accepts.
MIN_AUTHKEY_BYTES = 16
#: How long reaching the address and authenticating may take. The orchestrator runs run-unit once
#: the policy listens, so a server that is not there by then is not coming.
CONNECT_TIMEOUT_S = 60.0
#: How long each of `hello`, `reset` and `prompt` may take: `hello` builds the policy, loading its
#: weights, and `prompt` hands it the demonstration to encode. That is work, not a hang, and the
#: unit's own wall clock, which the orchestrator enforces, bounds it all.
SETUP_TIMEOUT_S = 300.0
#: How long one `act` may take unless run-unit is told otherwise (`--act-timeout-s`).
ACT_TIMEOUT_S = 60.0


def authkey_from_env(name: str, environ: MutableMapping[str, str] | None = None) -> bytes:
    """The policy's key, held as hex in the environment variable `name`, which is then removed.

    Never a command-line argument, where any process on the host can read it; removed once read,
    so nothing run-unit starts (the video encoder is a subprocess) inherits it. `PolicyError`, not
    quoting the value, for a variable that is missing, not hex or under `MIN_AUTHKEY_BYTES`.
    """
    environ = os.environ if environ is None else environ
    text = environ.pop(name, None)
    if not text:
        raise PolicyError(f"--authkey-env {name}: the environment variable holds no key")
    try:
        key = bytes.fromhex(text)
    except ValueError:
        raise PolicyError(f"--authkey-env {name}: the key is not hex") from None
    if len(key) < MIN_AUTHKEY_BYTES:
        raise PolicyError(
            f"--authkey-env {name}: the key is {len(key)} bytes; at least {MIN_AUTHKEY_BYTES}"
        )
    return key


def observation_arrays(observation: Observation) -> dict[str, np.ndarray]:
    """One observation as a served policy receives it, named and laid out as a prompt's frame:
    `frames_<camera>` (h, w, 3), `qpos` (D,) and `endpose` (16,)."""
    arrays = {
        f"{FRAMES_PREFIX}{camera}": np.asarray(image)
        for camera, image in sorted(observation.images.items())
    }
    arrays["qpos"] = np.asarray(observation.qpos, dtype=np.float64)
    arrays["endpose"] = flatten_endpose(observation.endpose)
    return arrays


class RemotePolicy(ICILPolicy):
    """The policy served at `address`, reached with `authkey`; see the module docstring.

    Nothing is sent until the first `reset`, which connects and says `hello`. `close` says `close`
    and drops the connection, after which the server exits; run-unit calls it however the unit
    ended. `client_factory` stands in for `icil_policy.client.RemotePolicy` in tests.
    """

    name = "remote"

    def __init__(
        self,
        address: str,
        authkey: bytes,
        *,
        act_timeout_s: float = ACT_TIMEOUT_S,
        setup_timeout_s: float = SETUP_TIMEOUT_S,
        connect_timeout_s: float = CONNECT_TIMEOUT_S,
        log_file: str | os.PathLike[str] | None = None,
        client_factory: Any = None,
    ) -> None:
        super().__init__()
        try:
            from icil_policy.errors import PolicyUnavailable, WireError

            if client_factory is None:
                from icil_policy.client import RemotePolicy as client_factory
        except ImportError as exc:
            raise PolicyError(
                f"a policy served at an address needs icil-policy installed: {exc}"
            ) from exc
        for what, seconds in (
            ("act", act_timeout_s),
            ("setup", setup_timeout_s),
            ("connect", connect_timeout_s),
        ):
            if not seconds > 0:
                raise PolicyError(f"the {what} timeout must be positive, not {seconds!r}")
        self.address = str(address)
        self.act_timeout_s = float(act_timeout_s)
        self.setup_timeout_s = float(setup_timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        # Absolute before the RoboTwin seam moves the working directory, and never resolved: the
        # policy can write beside its log, and `icil_policy.logs.tail` refuses to follow a log
        # swapped for a link only while the path it opens still ends in that link.
        self.log_file = None if log_file is None else Path(os.path.abspath(log_file))
        #: The served policy's `action_type` and name, known once it has answered `hello`.
        self.action_type = None  # type: ignore[assignment]
        self.served_policy: str | None = None
        self._authkey = bytes(authkey)
        self._client_factory = client_factory
        self._unavailable = PolicyUnavailable
        self._wire_error = WireError
        self._client: Any = None

    def describe(self) -> dict[str, Any]:
        described: dict[str, Any] = {
            "policy": self.name,
            "action_type": self.action_type,
            "served_policy": self.served_policy,
        }
        if self.served_policy is not None:
            described["model"] = self.served_policy
        return described

    def close(self) -> None:
        """Say `close` and drop the connection, if there is one. Idempotent; raises nothing."""
        client, self._client = self._client, None
        if client is not None:
            client.close()

    def _reset(self) -> None:
        episode = self.episode
        if episode is None or episode.seed is None:
            raise Unscorable(
                "a served policy is reset with its episode's public facts and seed, which "
                "run-unit draws; it was reset without them"
            )
        if self._client is None:
            self._connect()
        self._call("reset", self.setup_timeout_s, episode.seed)

    def _set_demonstration(self, demonstration: Demonstration) -> None:
        episode = self.episode
        assert episode is not None  # `_reset` refused to go on without it
        try:
            arrays = arrays_from(demonstration)
        except PromptError as exc:
            raise Unscorable(
                f"the demonstration cannot be sent as a prompt's arrays: {exc}"
            ) from exc
        info = {
            "frequency": float(demonstration.frequency),
            "cameras": list(demonstration.cameras),
            "embodiment": episode.embodiment,
            "action_dims": {str(k): int(v) for k, v in episode.action_dims.items()},
        }
        self._call("prompt", self.setup_timeout_s, arrays, info)

    def _act(self, observation: Observation) -> np.ndarray:
        try:
            arrays = observation_arrays(observation)
        except PromptError as exc:
            raise Unscorable(f"the observation cannot be sent: {exc}") from exc
        reply = self._call("act", self.act_timeout_s, arrays)
        # A copy: the client's arrays are read-only views of the reply's bytes.
        return np.array(reply["action"], dtype=np.float64)

    def _connect(self) -> None:
        try:
            self._client = self._client_factory(
                self.address,
                self._authkey,
                timeout_s=self.connect_timeout_s,
                log_file=self.log_file,
            )
        except self._unavailable as exc:
            raise PolicyUnreachable(f"the policy is unreachable: {exc}") from exc
        hello = self._call("hello", self.setup_timeout_s)
        action_type = hello.get("action_type")
        if action_type not in ACTION_TYPES:
            self.close()
            raise PolicyUnreachable(
                f"the policy is unreachable: hello: it declares action_type {action_type!r}; "
                f"the benchmark executes {' or '.join(ACTION_TYPES)}"
            )
        self.action_type = action_type
        self.served_policy = str(hello.get("policy"))

    def _call(self, op: str, timeout_s: float, *args: Any) -> Any:
        """One client call under its own timeout, its failure mapped as the module docstring says."""
        client = self._client
        if client is None:
            raise PolicyUnreachable(f"the policy is unreachable: {op}: the connection is closed")
        method = {
            "hello": client.hello,
            "reset": client.reset,
            "prompt": client.set_demonstration,
            "act": client.act,
        }[op]
        # Each call is timed on its own: the client reads `timeout_s` when the call starts.
        client.timeout_s = timeout_s
        try:
            return method(*args)
        except self._wire_error as exc:
            raise Unscorable(f"the benchmark could not send {op} to the policy: {exc}") from exc
        except self._unavailable as exc:
            if exc.remote_type is not None and exc.op in POLICY_OPS:
                raise PolicyError(
                    f"the served policy answered {exc.op} with an error: {exc}"
                ) from exc
            raise PolicyUnreachable(f"the policy is unreachable: {exc}") from exc
