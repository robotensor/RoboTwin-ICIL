"""`python -m robotwin_icil.serve`: one policy, in its own Python environment, behind a socket.

    python -m robotwin_icil.serve --policy SPEC [--config FILE] [--policy-arg KEY=VALUE ...]
        --address PATH|HOST:PORT (--authkey-env NAME | --authkey-file FILE)

The server listens, accepts one authenticated client, builds the policy when that client says
hello, and answers each operation by calling the policy's public methods, so `ICILPolicy`'s
lifecycle checks run where the model lives. An operation that raises is answered with
`{"op": "error", "type", "message", "traceback"}` and serving goes on: the client decides what
an error means, and `RemotePolicy` stops the server. The server closes the policy and exits on
`shutdown` and when the client hangs up. It logs to stderr.

It needs the standard library, numpy and this package's own dependencies, not the simulator:
`RemotePolicy` puts the package on the path of the environment it spawns the server in.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections.abc import Callable
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

from . import protocol
from .demo import Demonstration, Frame
from .policy import ICILPolicy, Observation, make_policy, parse_policy_arg


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{stamp} robotwin_icil.serve[{os.getpid()}]: {message}", file=sys.stderr, flush=True)


class _HungUp(Exception):
    """The client went away."""


class Server:
    """Serves one connected client: builds the policy on hello, then one operation at a time."""

    def __init__(self, conn, spec: str, kwargs: dict[str, Any]) -> None:
        self.conn = conn
        self.spec = spec
        self.kwargs = kwargs
        self.policy: ICILPolicy | None = None
        self.closed = False
        # The demonstration being streamed: its other fields, its frame count, its frames so far.
        self._pending: tuple[dict[str, Any], int, list[Frame]] | None = None
        self._ops: dict[str, Callable[[dict[str, Any]], dict[str, Any] | None]] = {
            "describe": lambda fields: {"description": self._policy().describe()},
            "environment": lambda fields: {"environment": self._policy().environment()},
            "seed": self._seed,
            "reset": self._reset,
            "demo_begin": self._demo_begin,
            "demo_frames": self._demo_frames,
            "demo_end": self._demo_end,
            "act": self._act,
            "info": lambda fields: {"info": self._policy().episode_info()},
            "ping": lambda fields: None,
        }

    def run(self) -> int:
        """Serve until shutdown or hang-up; the exit status: 0, or 1 when hello or close failed."""
        try:
            return self._serve()
        except _HungUp:
            log("the client hung up")
            return 0
        finally:
            if self.policy is not None and not self.closed:
                self.closed = True
                try:
                    self.policy.close()
                except Exception:
                    log(f"close() raised:\n{traceback.format_exc()}")

    def _serve(self) -> int:
        message = self._receive()
        if message is None:
            return 1
        if message.op != "hello":
            self._fail("hello", protocol.ProtocolError(f"expected hello, got {message.op!r}"))
            return 1
        if not self._reply(message, self._hello):
            return 1
        while True:
            message = self._receive()
            if message is None:
                continue
            if message.op == "shutdown":
                return 0 if self._reply(message, self._shutdown) else 1
            op = self._ops.get(message.op)
            if op is None:
                self._fail(message.op, protocol.ProtocolError(f"unknown op {message.op!r}"))
            else:
                self._reply(message, op)

    def _receive(self) -> protocol.Message | None:
        """The next message, or None when it was malformed and answered with an error."""
        try:
            return protocol.receive(self.conn)
        except (EOFError, OSError):
            raise _HungUp from None
        except protocol.ProtocolError as exc:
            self._fail("receive", exc)
            return None

    def _reply(
        self,
        message: protocol.Message,
        handle: Callable[[dict[str, Any]], dict[str, Any] | None],
    ) -> bool:
        """Answer a message with what `handle` returns, or with the error it raised."""
        try:
            reply = protocol.encode(message.op, **(handle(message.fields) or {}))
        except Exception as exc:
            self._fail(message.op, exc)
            return False
        self._send(reply)
        return True

    def _fail(self, op: str, exc: Exception) -> None:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log(f"{op} failed:\n{trace.rstrip()}")
        self._send(
            protocol.encode("error", type=type(exc).__name__, message=str(exc), traceback=trace)
        )

    def _send(self, reply: protocol.Encoded) -> None:
        try:
            protocol.send_encoded(self.conn, reply)
        except OSError:
            raise _HungUp from None

    def _policy(self) -> ICILPolicy:
        assert self.policy is not None
        return self.policy

    def _hello(self, fields: dict[str, Any]) -> dict[str, Any]:
        version = fields.get("protocol_version")
        if version != protocol.PROTOCOL_VERSION:
            raise protocol.ProtocolError(
                f"the client speaks protocol version {version!r}, this server "
                f"{protocol.PROTOCOL_VERSION}"
            )
        log(f"building {self.spec} with {self.kwargs or 'no arguments'}")
        self.policy = make_policy(self.spec, **self.kwargs)
        description = self.policy.describe()
        log(f"serving {description.get('policy')!r}, action_type {self.policy.action_type!r}")
        return {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "action_type": self.policy.action_type,
            "description": description,
            "python": sys.executable,
        }

    def _seed(self, fields: dict[str, Any]) -> None:
        self._policy().seed(_field(fields, "seed", int))

    def _reset(self, fields: dict[str, Any]) -> None:
        self._pending = None
        self._policy().reset()

    def _demo_begin(self, fields: dict[str, Any]) -> None:
        count = _field(fields, "frames", int)
        self._pending = (_field(fields, "demonstration", dict), count, [])

    def _demo_frames(self, fields: dict[str, Any]) -> None:
        if self._pending is None:
            raise protocol.ProtocolError("demo_frames before demo_begin")
        frames = _field(fields, "frames", tuple)
        if not all(isinstance(frame, Frame) for frame in frames):
            raise protocol.ProtocolError("demo_frames carries frames only")
        _, count, received = self._pending
        received.extend(frames)
        if len(received) > count:
            raise protocol.ProtocolError(f"{len(received)} frames of a {count}-frame demonstration")

    def _demo_end(self, fields: dict[str, Any]) -> None:
        pending, self._pending = self._pending, None
        if pending is None:
            raise protocol.ProtocolError("demo_end before demo_begin")
        other, count, frames = pending
        if len(frames) != count:
            raise protocol.ProtocolError(f"{len(frames)} frames of a {count}-frame demonstration")
        self._policy().set_demonstration(Demonstration(frames=tuple(frames), **other))

    def _act(self, fields: dict[str, Any]) -> dict[str, Any]:
        return {"actions": self._policy().act(_field(fields, "observation", Observation))}

    def _shutdown(self, fields: dict[str, Any]) -> None:
        self.closed = True
        self._policy().close()
        log("shut down")


def _field(fields: dict[str, Any], name: str, cls: type) -> Any:
    value = fields.get(name)
    if not isinstance(value, cls) or (cls is int and isinstance(value, bool)):
        raise protocol.ProtocolError(f"field {name!r} must be {cls.__name__}, not {value!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m robotwin_icil.serve",
        description="Serve one ICIL policy to one benchmark client over a socket.",
    )
    parser.add_argument("--policy", required=True, help="a built-in name or module:Class")
    parser.add_argument(
        "--config", help="the policy's own config file, handed to it as config=PATH, resolved"
    )
    parser.add_argument(
        "--policy-arg",
        dest="policy_args",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a keyword argument for the policy, read as eval reads it; repeatable",
    )
    parser.add_argument(
        "--address", required=True, help="a Unix socket's path, or host:port to listen on TCP"
    )
    key = parser.add_mutually_exclusive_group(required=True)
    key.add_argument(
        "--authkey-env",
        metavar="NAME",
        help="the environment variable holding the key; removed from the environment once read",
    )
    key.add_argument("--authkey-file", metavar="FILE", help="a file holding the key")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    kwargs: dict[str, Any] = {}
    for item in args.policy_args:
        try:
            key, value = parse_policy_arg(item)
        except ValueError as exc:
            parser.error(f"--policy-arg: {exc}")
        if key in kwargs:
            parser.error(f"--policy-arg: {key!r} is given twice")
        kwargs[key] = value
    if args.config is not None:
        if "config" in kwargs:
            parser.error("--config and --policy-arg config=... both name a config")
        kwargs["config"] = str(Path(args.config).resolve())
    try:
        authkey = _authkey(args)
        address, family = protocol.parse_address(args.address)
        listener = Listener(address, family, authkey=authkey)
    except (OSError, ValueError, protocol.ProtocolError) as exc:
        log(f"cannot serve on {args.address}: {exc}")
        return 2
    with listener:
        log(f"listening on {args.address}")
        conn = _accept(listener)
    with conn:
        log("a client connected")
        return Server(conn, args.policy, kwargs).run()


def _authkey(args: argparse.Namespace) -> bytes:
    if args.authkey_env is not None:
        # Out of the environment, so neither the model nor anything it starts inherits it.
        key, where = os.environ.pop(args.authkey_env, ""), f"${args.authkey_env}"
    else:
        key, where = Path(args.authkey_file).read_text(encoding="utf-8"), args.authkey_file
    key = key.strip()
    if not key:
        raise ValueError(f"no authkey in {where}")
    return key.encode("utf-8")


def _accept(listener: Listener):
    """The first client that authenticates; any other is refused and the server listens on."""
    while True:
        try:
            return listener.accept()
        except AuthenticationError as exc:
            log(f"refused a client: {exc}")
        except (OSError, EOFError) as exc:
            log(f"a client failed to connect: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
