"""`RemotePolicy`, the built-in `remote`: a policy served from another Python environment.

    robotwin-icil eval --policy remote --policy-arg policy=mypkg.adapters:Model \\
        --policy-arg python=/opt/envs/model/bin/python --policy-arg config=configs/model.yml ...

Without an address it spawns `python -m robotwin_icil.serve` under the model environment's
interpreter, on a Unix socket in a private temporary directory, with a fresh 32-byte key handed
over in an environment variable, never on the command line. With `address=HOST:PORT` it connects
to a server already running, with the key read from `authkey_file`. It is an `ICILPolicy`
itself, so the lifecycle and action checks run here, in the simulator's process, as well as in
the server, where the model lives.

Every failure (an error reply, a timeout, a server that hung up or died) stops the server and
raises `PolicyError` naming the server's log and quoting its last lines. `close()` asks the
server to shut down and stops it all the same. No spawned server outlives its client.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, get_args

from . import protocol
from .demo import Demonstration
from .policy import ActionType, ICILPolicy, Observation, PolicyError, format_policy_arg

# The first hello loads the model, and one adapter's checkpoint alone is 6.9 GB.
STARTUP_TIMEOUT_S = 900.0
TIMEOUT_S = 120.0
# How long a server has to exit after SIGTERM, and after shutdown, before it is killed.
KILL_GRACE_S = 5.0
LOG_TAIL_LINES = 20
AUTHKEY_ENV = "ROBOTWIN_ICIL_AUTHKEY"
_PACKAGE = Path(__file__).resolve().parent


class _Server:
    """A spawned server: its process, its log and its private directory. Stopped once."""

    def __init__(self, directory: Path, log: Path) -> None:
        self.directory = directory
        self.log = log
        self.process: subprocess.Popen | None = None
        self.handle = None
        # The exit status the server reached on its own, before it was signalled; None if not.
        self.exited: int | None = None

    def stop(self) -> None:
        try:
            if self.process is not None:
                self.exited = self.process.poll()
                if self.exited is None:
                    _signal(self.process, signal.SIGTERM)
                    try:
                        self.process.wait(KILL_GRACE_S)
                    except subprocess.TimeoutExpired:
                        _signal(self.process, signal.SIGKILL)
                self.process.wait()
        finally:
            if self.handle is not None:
                self.handle.close()
            shutil.rmtree(self.directory, ignore_errors=True)


def _signal(process: subprocess.Popen, sig: int) -> None:
    # The server leads a process group of its own, so anything it started goes with it.
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        with contextlib.suppress(ProcessLookupError):
            process.send_signal(sig)


class RemotePolicy(ICILPolicy):
    """A policy served by `python -m robotwin_icil.serve`, spawned here or already running.

    `policy` is what the server serves: a built-in name or `module:Class`, importable in the
    model environment. `python` is that environment's interpreter (default: this one), `config`
    the policy's own config file, handed to it as `config=PATH`, and every other keyword argument
    is the policy's, passed as a `--policy-arg` that reads back as the value given. `log` is the
    server's log file, appended to; a temporary file when not given, kept after the run.
    `address` connects to a running server instead, with its key in `authkey_file`; the server
    then has its own policy, arguments and log. `startup_timeout` bounds the wait for a spawned
    server to listen and, separately, for hello, whose reply comes once the model is loaded;
    `timeout` bounds every other operation. Each bounds sending the request and, again, its reply.
    """

    name = "remote"

    def __init__(
        self,
        policy: str | None = None,
        python: str | None = None,
        config: str | os.PathLike | None = None,
        address: str | None = None,
        log: str | os.PathLike | None = None,
        authkey_file: str | os.PathLike | None = None,
        startup_timeout: float = STARTUP_TIMEOUT_S,
        timeout: float = TIMEOUT_S,
        **policy_kwargs: Any,
    ) -> None:
        super().__init__()
        self.startup_timeout = float(startup_timeout)
        self.timeout = float(timeout)
        self.address = address
        self._conn = None
        self._server: _Server | None = None
        self._stopper: weakref.finalize | None = None
        self._python: str | None = None
        self._failure: str | None = None
        self._closed = False
        if not (self.startup_timeout > 0 and self.timeout > 0):
            raise PolicyError(
                f"remote: timeouts must be positive, not {startup_timeout}, {timeout}"
            )
        try:
            if address is None:
                if policy is None:
                    raise PolicyError(
                        "remote: give policy=SPEC to serve it, or address=HOST:PORT of a running "
                        "server"
                    )
                if authkey_file is not None:
                    raise PolicyError(
                        "remote: authkey_file is for address=; a spawned server gets a fresh key"
                    )
                self._spawn(policy, python or sys.executable, config, log, policy_kwargs)
            else:
                given = [
                    name
                    for name, value in (
                        ("policy", policy),
                        ("python", python),
                        ("config", config),
                        ("log", log),
                    )
                    if value is not None
                ] + sorted(policy_kwargs)
                if given:
                    raise PolicyError(
                        f"remote: the server at {address} was started with its own policy; "
                        f"{', '.join(given)} cannot be given with address="
                    )
                if authkey_file is None:
                    raise PolicyError("remote: address= needs authkey_file=, the server's key")
                self._connect(address, authkey_file)
            self._hello()
        except BaseException:
            self._stop()
            raise

    # The lifecycle, forwarded. The base class checks it here; the server's policy checks it again.

    def seed(self, seed: int) -> None:
        self._call("seed", seed=int(seed))

    def _reset(self) -> None:
        self._call("reset")

    def _set_demonstration(self, demonstration: Demonstration) -> None:
        fields = protocol.demonstration_fields(demonstration)
        self._call("demo_begin", demonstration=fields, frames=len(demonstration.frames))
        for chunk in protocol.frame_chunks(demonstration.frames, protocol.DEMO_CHUNK_BYTES):
            self._call("demo_frames", frames=chunk)
        self._call("demo_end")

    def _act(self, observation: Observation) -> Any:
        return self._call("act", observation=observation).get("actions")

    def episode_info(self) -> Any:
        return self._call("info").get("info")

    def environment(self) -> Any:
        return self._call("environment").get("environment")

    def describe(self) -> dict[str, Any]:
        """The served policy's description, asked afresh each time, with `remote` beside it.

        `remote.address` is None for a spawned server: its socket's path changes every run, and
        the description is part of the run's identity.
        """
        description = self._call("describe").get("description")
        if not isinstance(description, dict):
            return description
        remote = {
            "python": self._python,
            "address": self.address,
            "protocol_version": protocol.PROTOCOL_VERSION,
        }
        return {**description, "remote": remote}

    def close(self) -> None:
        """Ask the server to close the policy and exit; stop it whatever it answers. Once."""
        if self._closed:
            return
        try:
            if self._conn is not None and self._failure is None:
                self._call("shutdown")
                if self._server is not None and self._server.process is not None:
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        self._server.process.wait(KILL_GRACE_S)
        finally:
            self._closed = True
            self._stop()

    # The connection.

    def _spawn(
        self,
        policy: str,
        python: str,
        config: str | os.PathLike | None,
        log: str | os.PathLike | None,
        policy_kwargs: dict[str, Any],
    ) -> None:
        argv = [python, "-m", "robotwin_icil.serve", "--policy", policy]
        if config is not None:
            argv += ["--config", str(Path(config).resolve())]
        for key, value in policy_kwargs.items():
            argv += ["--policy-arg", format_policy_arg(key, value)]
        if log is None:
            fd, log = tempfile.mkstemp(prefix="robotwin-icil-serve-", suffix=".log")
            os.close(fd)
        log = Path(log).resolve()
        log.parent.mkdir(parents=True, exist_ok=True)
        server = self._server = _Server(Path(tempfile.mkdtemp(prefix="robotwin-icil-")), log)
        self._stopper = weakref.finalize(self, server.stop)
        # The package alone goes on the path, not the directory holding it: installed, that is
        # the simulator environment's site-packages, whose pins are what the model avoids.
        path = server.directory / "path"
        path.mkdir()
        (path / "robotwin_icil").symlink_to(_PACKAGE, target_is_directory=True)
        socket_path = server.directory / "policy.sock"
        authkey = secrets.token_hex(32)
        pythonpath = os.pathsep.join(p for p in (str(path), os.environ.get("PYTHONPATH")) if p)
        env = {**os.environ, AUTHKEY_ENV: authkey, "PYTHONPATH": pythonpath}
        env["PYTHONUNBUFFERED"] = "1"  # so the log holds everything up to a crash
        argv += ["--address", str(socket_path), "--authkey-env", AUTHKEY_ENV]
        server.handle = log.open("ab")
        try:
            server.process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=server.handle,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            raise PolicyError(f"remote: cannot start a policy server with {python}: {exc}") from exc
        self._conn = self._listening(socket_path, authkey.encode())

    def _listening(self, socket_path: Path, authkey: bytes):
        """A connection to the spawned server, once it listens."""
        assert self._server is not None and self._server.process is not None
        deadline = time.monotonic() + self.startup_timeout
        while True:
            status = self._server.process.poll()
            if status is not None:
                raise self._failed(
                    f"the policy server exited with status {status} before it listened"
                )
            if socket_path.exists():
                try:
                    return _connect(
                        str(socket_path), "AF_UNIX", authkey, deadline - time.monotonic()
                    )
                except AuthenticationError as exc:
                    raise self._failed(f"the policy server refused the key: {exc}") from None
                except TimeoutError:
                    pass
                except (OSError, EOFError):
                    pass  # not accepting yet, or gone: the next poll says which
            if time.monotonic() > deadline:
                raise self._failed(
                    f"the policy server did not listen within {self.startup_timeout:g} s"
                )
            time.sleep(0.02)

    def _connect(self, address: str, authkey_file: str | os.PathLike) -> None:
        try:
            authkey = Path(authkey_file).read_text(encoding="utf-8").strip().encode("utf-8")
            target, family = protocol.parse_address(address)
        except (OSError, protocol.ProtocolError) as exc:
            raise PolicyError(f"remote: cannot connect to {address}: {exc}") from exc
        if not authkey:
            raise PolicyError(f"remote: no key in {authkey_file}")
        try:
            self._conn = _connect(target, family, authkey, self.startup_timeout)
        except AuthenticationError as exc:
            raise self._failed(f"the policy server at {address} refused the key: {exc}") from None
        except Exception as exc:
            raise self._failed(
                f"cannot connect to {address}: {type(exc).__name__}: {exc}"
            ) from None

    def _hello(self) -> None:
        reply = self._call(
            "hello", self.startup_timeout, protocol_version=protocol.PROTOCOL_VERSION
        )
        version = reply.get("protocol_version")
        if version != protocol.PROTOCOL_VERSION:
            raise self._failed(
                f"the policy server speaks protocol version {version!r}, this client "
                f"{protocol.PROTOCOL_VERSION}"
            )
        action_type = reply.get("action_type")
        description = reply.get("description")
        if action_type not in get_args(ActionType):
            raise self._failed(f"the policy server serves action_type {action_type!r}")
        if not (isinstance(description, dict) and isinstance(description.get("policy"), str)):
            raise self._failed(f"the policy server describes its policy as {description!r}")
        self.action_type = action_type
        self.name = f"{description['policy']} (remote)"
        self._python = reply.get("python")

    def _call(self, op: str, timeout: float | None = None, **fields: Any) -> dict[str, Any]:
        """Send one operation and return the fields of its reply; any failure stops the server."""
        if self._failure is not None:
            raise PolicyError(
                f"{self.name}: the policy server was stopped after an earlier failure: "
                f"{self._failure}"
            )
        if self._closed or self._conn is None:
            raise PolicyError(f"{self.name}: {op} after close()")
        timeout = self.timeout if timeout is None else timeout
        try:
            _send(self._conn, op, timeout, fields)
            reply = protocol.receive(self._conn, timeout)
        except _SendTimeout:
            raise self._failed(f"could not send {op} within {timeout:g} s") from None
        except TimeoutError:
            raise self._failed(f"no reply to {op} within {timeout:g} s") from None
        except (EOFError, OSError):
            raise self._failed(f"the policy server hung up during {op}", hung_up=True) from None
        except Exception as exc:
            raise self._failed(f"{op}: {type(exc).__name__}: {exc}") from None
        except BaseException:
            # Interrupted mid-message: the connection can no longer be trusted.
            self._failure = f"{op} was interrupted"
            self._stop()
            raise
        if reply.op == "error":
            raise self._failed(
                f"{op} failed in the policy server: {reply.fields.get('type')}: "
                f"{reply.fields.get('message')}",
                trace=reply.fields.get("traceback"),
            )
        if reply.op != op:
            raise self._failed(f"the policy server answered {op} with {reply.op!r}")
        return reply.fields

    def _failed(self, what: str, trace: Any = None, hung_up: bool = False) -> PolicyError:
        """Stop the server, remember why, and say so with the end of its log."""
        server = self._server
        if hung_up and server is not None and server.process is not None:
            # A server that died is still exiting when its socket closes: wait for its status.
            with contextlib.suppress(subprocess.TimeoutExpired):
                server.process.wait(1.0)
        self._failure = what
        self._stop()
        if server is not None:
            if server.exited is not None:
                what += f" (it exited with status {server.exited})"
            what += f"\npolicy server log {server.log}, last lines:\n{_tail(server.log)}"
        elif isinstance(trace, str):
            what += f"\npolicy server traceback:\n{trace.rstrip()}"
        # All of it: a later call raises this again, and its error may be the one shown, as the
        # frozen-policy audit's describe() is when a run stops on this failure.
        self._failure = what
        return PolicyError(f"{self.name}: {what}")

    def _stop(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            with contextlib.suppress(OSError):
                conn.close()
        if self._stopper is not None:
            self._stopper()


class _SendTimeout(Exception):
    """A request that could not be sent in time: the peer stopped reading."""


def _send(conn, op: str, timeout: float, fields: dict[str, Any]) -> None:
    """`protocol.send`, given a timeout, which `send_bytes` has not.

    A peer that stops reading (stopped, partitioned, stuck) blocks a large message once the
    socket's buffer fills, before any reply could time out. At the deadline the socket is shut
    down, which makes the blocked write fail; the connection is then done, as after any failure.
    """
    message = protocol.encode(op, **fields)  # a value that cannot be sent fails before the timer
    lock = threading.Lock()
    state = {"sent": False, "expired": False}

    def expire() -> None:
        # Under the lock throughout: once sent, the connection may be closed and its fd reused.
        with lock:
            if state["sent"]:
                return
            state["expired"] = True
            with contextlib.suppress(OSError):
                fd = os.dup(conn.fileno())
                try:
                    sock = socket.socket(fileno=fd)
                except OSError:
                    os.close(fd)
                    raise
                with sock:
                    sock.shutdown(socket.SHUT_RDWR)

    timer = threading.Timer(timeout, expire)
    timer.daemon = True
    timer.start()
    try:
        protocol.send_encoded(conn, message)
    except OSError:
        if not state["expired"]:
            raise
    finally:
        with lock:
            state["sent"] = True
        timer.cancel()
    if state["expired"]:
        raise _SendTimeout


def _connect(address, family: str, authkey: bytes, timeout: float):
    """`Client`, which authenticates with no timeout of its own, given one."""
    result: dict[str, Any] = {}

    def attempt() -> None:
        try:
            result["conn"] = Client(address, family, authkey=authkey)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=attempt, name="robotwin-icil-connect", daemon=True)
    thread.start()
    thread.join(max(0.0, timeout))
    if thread.is_alive():
        raise TimeoutError(f"no connection within {timeout:g} s")
    if "error" in result:
        raise result["error"]
    return result["conn"]


def _tail(log: Path, lines: int = LOG_TAIL_LINES) -> str:
    try:
        with log.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 64 * 1024))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return f"    (cannot read it: {exc})"
    tail = text.splitlines()[-lines:]
    return "\n".join(f"    {line}" for line in tail) if tail else "    (empty)"
