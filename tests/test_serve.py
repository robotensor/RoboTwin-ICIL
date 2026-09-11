"""`python -m robotwin_icil.serve`, driven through its command line and the protocol alone."""

import json
import os
import socket
import subprocess
import sys
import time
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
import pytest

from robotwin_icil import protocol
from test_protocol import FRAME_FIELDS, demonstration, observation

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src"
KEY = "not-a-secret"


def wait_for(condition, timeout=20.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.02)


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Served:
    def __init__(self, process, address, log):
        self.process, self.address, self.log = process, address, log

    def connect(self, key=KEY):
        address, family = protocol.parse_address(self.address)
        return Client(address, family, authkey=key.encode())

    def wait(self):
        return self.process.wait(timeout=20)

    def text(self):
        return self.log.read_text()


@pytest.fixture
def serve(tmp_path):
    """Start a server in a child process; every one still running at the end is killed."""
    started = []

    def start(*args, address=None, key_via="file"):
        address = address or str(tmp_path / "policy.sock")
        log = tmp_path / f"serve-{len(started)}.log"
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), str(TESTS)])}
        if key_via == "file":
            (tmp_path / "key").write_text(KEY + "\n")
            auth = ["--authkey-file", str(tmp_path / "key")]
        else:
            env["ROBOTWIN_ICIL_AUTHKEY"] = KEY
            auth = ["--authkey-env", "ROBOTWIN_ICIL_AUTHKEY"]
        with log.open("w") as handle:
            process = subprocess.Popen(
                [sys.executable, "-m", "robotwin_icil.serve", "--address", address, *auth, *args],
                env=env,
                cwd=tmp_path,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        served = Served(process, address, log)
        started.append(served)
        wait_for(lambda: "listening on" in log.read_text() or process.poll() is not None)
        return served

    yield start
    for served in started:
        if served.process.poll() is None:
            served.process.kill()
            served.process.wait()


def call(conn, op, **fields):
    protocol.send(conn, op, **fields)
    return protocol.receive(conn, timeout=20)


def ok(conn, op, **fields):
    reply = call(conn, op, **fields)
    assert reply.op == op, reply.fields.get("traceback", reply)
    return reply.fields


def hello(conn, version=protocol.PROTOCOL_VERSION):
    return call(conn, "hello", protocol_version=version)


def test_a_served_policy_runs_the_whole_lifecycle_over_the_protocol(serve, tmp_path):
    served = serve(
        "--policy", "served_policies:Echo", "--policy-arg", "temperature=0.5", "--config", "a.yml"
    )
    demo, obs = demonstration(frames=4), observation()
    with served.connect() as conn:
        reply = hello(conn)
        assert reply.op == "hello"
        assert reply.fields["protocol_version"] == protocol.PROTOCOL_VERSION
        assert reply.fields["action_type"] == "qpos"
        assert reply.fields["python"] == sys.executable
        kwargs = {"temperature": 0.5, "config": str(tmp_path / "a.yml")}
        assert reply.fields["description"] == {
            "policy": "echo",
            "action_type": "qpos",
            "kwargs": kwargs,
        }
        ok(conn, "seed", seed=11)
        ok(conn, "reset")
        ok(conn, "demo_begin", demonstration=protocol.demonstration_fields(demo), frames=4)
        for chunk in protocol.frame_chunks(demo.frames, max_bytes=300):
            ok(conn, "demo_frames", frames=chunk)
        ok(conn, "demo_end")
        actions = ok(conn, "act", observation=obs)["actions"]
        np.testing.assert_array_equal(actions, demo.actions()[:1])
        assert actions.dtype == np.float64
        assert ok(conn, "info")["info"] == {
            "seeds": [11],
            "acts": 1,
            "frames": 4,
            "time_s": obs.time_s,
            "images": sorted(obs.images),
            "gripper_joints": obs.gripper_joints["left"].tolist(),
        }
        assert ok(conn, "describe")["description"]["kwargs"] == kwargs
        assert ok(conn, "environment")["environment"]["python"] == sys.version.split()[0]
        ok(conn, "ping")
        ok(conn, "shutdown")
    assert served.wait() == 0
    assert "shut down" in served.text()


def test_an_operation_that_raises_is_answered_with_an_error_and_serving_goes_on(serve):
    served = serve("--policy", "served_policies:Unencodable")
    with served.connect() as conn:
        assert hello(conn).op == "hello"
        ok(conn, "reset")
        refused = call(conn, "act", observation=observation())
        assert refused.op == "error" and refused.fields["type"] == "PolicyError"
        assert "act() before set_demonstration()" in refused.fields["message"]
        assert "Traceback" in refused.fields["traceback"]
        unencodable = call(conn, "info")
        assert unencodable.op == "error"
        assert "object values cannot be sent" in unencodable.fields["message"]
        assert "unknown op 'teleport'" in call(conn, "teleport").fields["message"]
        assert "demo_end before demo_begin" in call(conn, "demo_end").fields["message"]
        assert "field 'seed' must be int, not '3'" in call(conn, "seed", seed="3").fields["message"]
        ok(conn, "ping")
    assert served.wait() == 0
    assert "act failed" in served.text()


@pytest.mark.parametrize(
    "header, frames, message",
    [
        (
            {
                "op": "act",
                "observation": {"$": "dataclass", "type": "Frame", "fields": FRAME_FIELDS},
                "arrays": [{"name": "q", "dtype": "<f8", "shape": [2, 7]}],
            },
            (b"\0" * 112,),
            "cannot build a Frame: DemonstrationError",
        ),
        (
            {"op": "info", "x": {"$": "dict", "items": [[[1], 2]]}, "arrays": []},
            (),
            "a dict key must be hashable",
        ),
    ],
)
def test_a_malformed_message_is_answered_with_an_error_and_serving_goes_on(
    serve, header, frames, message
):
    served = serve("--policy", "replay")
    with served.connect() as conn:
        assert hello(conn).op == "hello"
        conn.send_bytes(json.dumps(header).encode())
        for raw in frames:
            conn.send_bytes(raw)
        refused = protocol.receive(conn, timeout=20)
        assert refused.op == "error" and refused.fields["type"] == "ProtocolError"
        assert message in refused.fields["message"]
        ok(conn, "ping")
        ok(conn, "shutdown")
    assert served.wait() == 0


@pytest.mark.parametrize(
    "first, message",
    [
        (("hello", {"protocol_version": 0}), "the client speaks protocol version 0, this server"),
        (("act", {}), "expected hello, got 'act'"),
    ],
)
def test_a_client_that_does_not_say_hello_in_this_protocol_is_refused(serve, first, message):
    served = serve("--policy", "replay")
    with served.connect() as conn:
        reply = call(conn, first[0], **first[1])
        assert reply.op == "error" and message in reply.fields["message"]
    assert served.wait() == 1


@pytest.mark.parametrize(
    "spec, error, message",
    [
        ("served_policies:BrokenConstructor", "RuntimeError", "checkpoint not found"),
        ("no_such_module:Model", "PolicyError", "cannot load policy 'no_such_module:Model'"),
    ],
)
def test_a_policy_that_cannot_be_built_is_reported_in_reply_to_hello(serve, spec, error, message):
    served = serve("--policy", spec)
    with served.connect() as conn:
        reply = hello(conn)
    assert reply.op == "error" and reply.fields["type"] == error
    assert message in reply.fields["message"]
    assert served.wait() == 1


def test_the_server_exits_when_its_client_hangs_up(serve):
    served = serve("--policy", "replay")
    with served.connect() as conn:
        assert hello(conn).op == "hello"
    assert served.wait() == 0
    assert "the client hung up" in served.text()


def test_a_client_with_another_key_is_refused_and_the_server_listens_on(serve):
    served = serve("--policy", "replay", address=f"127.0.0.1:{free_port()}")
    with pytest.raises(AuthenticationError):
        served.connect(key="guessed")
    with served.connect() as conn:
        assert hello(conn).op == "hello"
        ok(conn, "shutdown")
    assert served.wait() == 0
    assert "refused a client" in served.text()


def test_a_key_from_the_environment_leaves_it_before_the_policy_is_built(serve):
    served = serve("--policy", "served_policies:Echo", key_via="env")
    with served.connect() as conn:
        assert hello(conn).op == "hello"
        assert ok(conn, "environment")["environment"]["authkey"] == "False"
        ok(conn, "shutdown")
    assert served.wait() == 0


@pytest.mark.parametrize(
    "args, message",
    [
        (["--policy-arg", "temperature"], "'temperature' is not KEY=VALUE"),
        (["--policy-arg", "a=1", "--policy-arg", "a=2"], "'a' is given twice"),
        (["--config", "a.yml", "--policy-arg", "config=b.yml"], "both name a config"),
        (["--authkey-file", "empty.key"], "no authkey in"),
    ],
)
def test_a_server_started_wrongly_exits_before_it_listens(tmp_path, args, message):
    (tmp_path / "empty.key").write_text("\n")
    if "--authkey-file" not in args:
        args = [*args, "--authkey-env", "UNSET_FOR_THIS_TEST"]
    result = subprocess.run(
        [sys.executable, "-m", "robotwin_icil.serve", "--policy", "replay"]
        + ["--address", str(tmp_path / "s.sock"), *args],
        env={**os.environ, "PYTHONPATH": str(SRC)},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert not (tmp_path / "s.sock").exists()
