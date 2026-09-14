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

How long a policy may take is bounded three ways. Each call has its own timeout, which bounds a
hang. All the calls of a unit together — connecting, `hello`, `reset`, `prompt`, every `act` —
may take at most the policy's budget (`policy_budget_s`), which bounds a policy that answers each
call just in time: the call in flight when it runs out is cut short there, `PolicyUnreachable`,
void with `void_cause` "policy". Without it a slow policy would run the unit into its caller's
kill, and a unit killed while its policy was still alive is void for both sides of a duel, not the
slow side's loss. And given the unit's `deadline`, no call runs past it: a call cut short there
while the policy was still within its budget means the harness took more than the unit left it,
`Unscorable`, void on the harness, written before the caller's kill.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, MutableMapping
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
#: The dtype kinds an action may have: bool, signed and unsigned integers, floats.
NUMERIC_KINDS = ("b", "i", "u", "f")
#: The most rows of one action chunk that are kept. More than any task's step limit (the longest
#: in RoboTwin's `_eval_step_limit.yml` is 1700), so a row dropped here could never have run.
MAX_ACTION_ROWS = 4096
#: The longest name a served policy is recorded under: `hello`'s `policy` is whatever the server
#: says, and it goes into result.json twice.
MAX_NAME_CHARS = 200
#: How much of a remote failure's text reaches result.json: its start, which names the call and
#: the reason, and its end, which holds the server's log tail. `icil_policy.serve` bounds both; a
#: hostile server bounds neither.
REASON_HEAD_CHARS = 4000
REASON_TAIL_CHARS = 8192
#: How long reaching the address and authenticating may take. The orchestrator runs run-unit once
#: the policy listens, so a server that is not there by then is not coming.
CONNECT_TIMEOUT_S = 60.0
#: How long each of `hello`, `reset` and `prompt` may take: `hello` builds the policy, loading its
#: weights, and `prompt` hands it the demonstration to encode. That is work, not a hang; the
#: policy's budget bounds it together with every act.
SETUP_TIMEOUT_S = 300.0
#: How long one `act` may take unless run-unit is told otherwise (`--act-timeout-s`).
ACT_TIMEOUT_S = 60.0
#: How long saying `close` may take. By then the unit's result is written, and a policy that
#: stalls its close must not keep run-unit, and the unit, running past it.
CLOSE_TIMEOUT_S = 5.0
#: How long all the calls to a served policy may take together in one unit unless run-unit is
#: told otherwise (`--policy-budget-s`): connecting, `hello`, `reset`, `prompt` and every `act`,
#: not `close`. Half of the orchestrator's placeholder unit budget (600 s); the other half is the
#: harness's own, building the scene, stepping and rendering it, writing the clip.
POLICY_BUDGET_S = 300.0
#: What run-unit keeps back from `--unit-timeout-s` to finish once a call is cut short at the
#: deadline: close the scene, write the clip and result.json, close the policy.
RESULT_RESERVE_S = 30.0


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


def bounded(text: str, head: int, tail: int = 0) -> str:
    """`text` if it is at most `head + tail` characters; otherwise its first `head` and last `tail`
    characters, with how many were left out between them."""
    if len(text) <= head + tail:
        return text
    end = text[len(text) - tail :] if tail else ""
    return f"{text[:head]} [... {len(text) - head - tail} characters ...] {end}".rstrip()


def _reason(exc: BaseException) -> str:
    return bounded(str(exc), REASON_HEAD_CHARS, REASON_TAIL_CHARS)


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
    ended. `deadline` is a `clock()` reading no call runs past, or None. `client_factory` stands
    in for `icil_policy.client.RemotePolicy` and `clock` for `time.monotonic` in tests.
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
        close_timeout_s: float | None = None,
        policy_budget_s: float = POLICY_BUDGET_S,
        deadline: float | None = None,
        log_file: str | os.PathLike[str] | None = None,
        client_factory: Any = None,
        clock: Callable[[], float] = time.monotonic,
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
        close_timeout_s = CLOSE_TIMEOUT_S if close_timeout_s is None else close_timeout_s
        for what, seconds in (
            ("act", act_timeout_s),
            ("setup", setup_timeout_s),
            ("connect", connect_timeout_s),
            ("close", close_timeout_s),
        ):
            if not seconds > 0:
                raise PolicyError(f"the {what} timeout must be positive, not {seconds!r}")
        self.address = str(address)
        self.act_timeout_s = float(act_timeout_s)
        self.setup_timeout_s = float(setup_timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.close_timeout_s = float(close_timeout_s)
        if not policy_budget_s > 0:
            raise PolicyError(f"the policy's budget must be positive, not {policy_budget_s!r}")
        self.policy_budget_s = float(policy_budget_s)
        self.deadline = deadline
        #: How long the calls to the policy have taken so far, counted against its budget.
        self.policy_wall_s = 0.0
        self._clock = clock
        self._built = clock()
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
            "policy_wall_s": round(self.policy_wall_s, 3),
            "policy_budget_s": self.policy_budget_s,
        }
        if self.served_policy is not None:
            described["model"] = self.served_policy
        return described

    def close(self) -> None:
        """Say `close` and drop the connection, if there is one, within `close_timeout_s`, not
        whatever timeout the last call had. Idempotent; raises nothing."""
        client, self._client = self._client, None
        if client is not None:
            client.timeout_s = self.close_timeout_s
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
        return self._rows(reply.get("action"))

    def _rows(self, action: Any) -> np.ndarray:
        """The reply's action as rows of float64, checked before anything is copied.

        The client bounds a reply at a gigabyte, so an action converted first and checked after
        would cost run-unit up to eight times that (a bool or uint8 array of any shape): its width
        and dtype are held to the robot's here, as `ICILPolicy.act` holds them afterwards, and
        rows past `MAX_ACTION_ROWS`, which no episode can execute, are dropped.
        """
        episode = self.episode
        width = None if episode is None else episode.action_dims.get(self.action_type)
        shape = tuple(getattr(action, "shape", ()))
        rows_shape = (1, *shape) if len(shape) == 1 else shape
        kind = getattr(getattr(action, "dtype", None), "kind", None)
        if len(shape) not in (1, 2) or 0 in shape or shape[-1] != width:
            raise PolicyError(
                f"{self.name}: act() returned shape {rows_shape}, expected (k, {width}) "
                f"for action_type {self.action_type!r}"
            )
        if kind not in NUMERIC_KINDS:
            raise PolicyError(f"{self.name}: act() returned a {action.dtype} action, not numbers")
        rows = action.reshape(1, -1) if len(shape) == 1 else action[:MAX_ACTION_ROWS]
        # A copy: the client's arrays are read-only views of the reply's bytes.
        return np.array(rows, dtype=np.float64)

    def _connect(self) -> None:
        def connect(limit: float) -> Any:
            return self._client_factory(
                self.address, self._authkey, timeout_s=limit, log_file=self.log_file
            )

        self._client = self._timed("connect", self.connect_timeout_s, connect)
        hello = self._call("hello", self.setup_timeout_s)
        action_type = hello.get("action_type")
        if action_type not in ACTION_TYPES:
            self.close()
            declared = bounded(repr(action_type), MAX_NAME_CHARS)
            raise PolicyUnreachable(
                f"the policy is unreachable: hello: it declares action_type {declared}; "
                f"the benchmark executes {' or '.join(ACTION_TYPES)}"
            )
        self.action_type = action_type
        self.served_policy = bounded(str(hello.get("policy")), MAX_NAME_CHARS)

    def _call(self, op: str, timeout_s: float, *args: Any) -> Any:
        """One client call, given at most `timeout_s` and timed as `_timed` says."""
        client = self._client
        if client is None:
            raise PolicyUnreachable(f"the policy is unreachable: {op}: the connection is closed")
        method = {
            "hello": client.hello,
            "reset": client.reset,
            "prompt": client.set_demonstration,
            "act": client.act,
        }[op]

        def call(limit: float) -> Any:
            # Each call is timed on its own: the client reads `timeout_s` when the call starts.
            client.timeout_s = limit
            return method(*args)

        return self._timed(op, timeout_s, call)

    def _timed(self, op: str, timeout_s: float, call: Callable[[float], Any]) -> Any:
        """`call(limit)`, `limit` the least of `timeout_s`, what is left of the policy's budget and
        what is left before the deadline. Its wall time counts against the budget, and its failure
        is mapped as the module docstring says."""
        limit, bound = self._limit(timeout_s)
        if limit <= 0:
            raise self._out_of_time(op, bound)
        started = self._clock()
        try:
            return call(limit)
        except self._wire_error as exc:
            raise Unscorable(
                f"the benchmark could not send {op} to the policy: {_reason(exc)}"
            ) from exc
        except self._unavailable as exc:
            if exc.remote_type is not None and exc.op in POLICY_OPS:
                raise PolicyError(
                    f"the served policy answered {exc.op} with an error: {_reason(exc)}"
                ) from exc
            spent = self._clock() - started
            if bound != "call" and spent >= limit:
                raise self._out_of_time(op, bound, spent) from exc
            raise PolicyUnreachable(f"the policy is unreachable: {_reason(exc)}") from exc
        finally:
            self.policy_wall_s += self._clock() - started

    def _limit(self, timeout_s: float) -> tuple[float, str]:
        """How long the next call may take, and what sets that: "call", its own timeout;
        "budget", what is left of the policy's; "deadline", what is left of the unit's."""
        left = self.policy_budget_s - self.policy_wall_s
        until = math.inf if self.deadline is None else self.deadline - self._clock()
        if timeout_s <= min(left, until):
            return timeout_s, "call"
        if left <= until:
            return left, "budget"
        return until, "deadline"

    def _out_of_time(self, op: str, bound: str, spent: float = 0.0) -> Unscorable:
        """What ends a unit whose call the policy's budget or the unit's deadline cut short.

        The budget is the policy's own: void on the policy. At the deadline the policy was within
        its budget, so the rest of the unit's time went to the harness: void on the harness.
        """
        used = self.policy_wall_s + spent
        if bound == "budget":
            return PolicyUnreachable(
                f"the policy is unreachable: {op}: its calls took {used:.1f}s, its whole budget "
                f"of {self.policy_budget_s:g}s for the unit"
            )
        gone = self._clock() - self._built
        return Unscorable(
            f"the unit ran out of time during {op}: {gone:.1f}s gone, {gone - used:.1f}s of it "
            f"the harness's and {used:.1f}s the policy's, within its {self.policy_budget_s:g}s "
            "budget"
        )
