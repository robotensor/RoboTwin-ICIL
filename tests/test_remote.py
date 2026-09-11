"""RemotePolicy against real policy servers, each a child process launched with this interpreter.

Every process a test spawns is recorded, and the test fails if any is still running at its end:
no server outlives its client, after `close()` or after any failure.
"""

import contextlib
import gc
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing.connection import Listener

import numpy as np
import pytest

import test_runner
from fake_robotwin import FakeConfig
from robotwin_icil import protocol, remote, runner
from robotwin_icil.policy import PolicyError, ReplayEEPolicy, make_policy
from robotwin_icil.records import RunDir
from robotwin_icil.remote import RemotePolicy
from test_protocol import demonstration, observation
from test_runner import quiet, spec
from test_serve import KEY, SRC, TESTS, free_port, wait_for

fake_sim = test_runner.fake_sim  # the runner tests' fixture: fake RoboTwin envs

REMOTE = {"python": sys.executable, "address": None, "protocol_version": protocol.PROTOCOL_VERSION}


@pytest.fixture(autouse=True)
def servers(monkeypatch):
    """Every process spawned during the test; none may still run when it ends."""
    monkeypatch.setenv("PYTHONPATH", str(TESTS))  # so servers import served_policies
    spawned = []
    popen = subprocess.Popen

    def recording(*args, **kwargs):
        process = popen(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", recording)
    yield spawned
    alive = [process for process in spawned if process.poll() is None]
    for process in alive:
        process.kill()
        process.wait()
    assert not alive, f"{len(alive)} server process(es) outlived the test"


def running(policy):
    return policy._server.process.poll() is None


def untimed(records):
    return [{**record.to_json(), "duration_s": 0.0} for record in records]


@pytest.mark.parametrize("name", ["replay", "dummy", "replay_ee"])
def test_a_served_policy_records_what_it_records_in_process(tmp_path, fake_sim, name):
    local = runner.run(
        spec(tmp_path, run_dir=tmp_path / "local"), make_policy(name), FakeConfig(), log=quiet
    )
    served_policy = RemotePolicy(policy=name, log=tmp_path / "serve.log")
    served = runner.run(
        spec(tmp_path, run_dir=tmp_path / "served"), served_policy, FakeConfig(), log=quiet
    )
    assert untimed(served) == untimed(local)
    assert all(record.success for record in local) is (name != "dummy")
    in_process = RunDir(tmp_path / "local").manifest().policy
    assert RunDir(tmp_path / "served").manifest().policy == {**in_process, "remote": REMOTE}
    # The run closed the policy: the server shut down on request and left nothing behind.
    assert served_policy._server.process.returncode == 0
    assert not served_policy._server.directory.exists()


def test_a_remote_policy_is_described_and_named_as_the_policy_it_serves(tmp_path):
    policy = RemotePolicy(policy="replay_ee", log=tmp_path / "serve.log")
    try:
        assert policy.action_type == "ee" and policy.name == "replay_ee (remote)"
        assert policy.describe() == {**ReplayEEPolicy().describe(), "remote": REMOTE}
    finally:
        policy.close()


def test_the_served_policy_gets_its_arguments_and_every_operation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(protocol, "DEMO_CHUNK_BYTES", 300)
    sent = []
    send = protocol.send
    monkeypatch.setattr(
        protocol, "send", lambda conn, op, **f: sent.append(op) or send(conn, op, **f)
    )
    policy = RemotePolicy(
        policy="served_policies:Echo",
        config="a.yml",
        temperature=0.5,
        revision="0123",
        path=tmp_path / "x",
    )
    try:
        assert policy.describe()["kwargs"] == {
            "config": str(tmp_path / "a.yml"),
            "temperature": 0.5,
            "revision": "0123",
            "path": str(tmp_path / "x"),
        }
        demo, obs = demonstration(frames=4), observation()
        policy.seed(5)
        policy.reset()
        policy.set_demonstration(demo)
        np.testing.assert_array_equal(policy.act(obs), demo.actions()[:1])
        assert policy.episode_info() == {
            "seeds": [5],
            "acts": 1,
            "frames": 4,
            "time_s": obs.time_s,
            "images": sorted(obs.images),
            "gripper_joints": obs.gripper_joints["left"].tolist(),
        }
        assert sent.count("demo_frames") > 1
        # The key went through the environment, which the server emptied before the model loaded.
        assert policy.environment()["authkey"] == "False"
        arguments = policy._server.process.args
        assert "--authkey-env" in arguments
        assert not any(re.fullmatch("[0-9a-f]{64}", argument) for argument in arguments)
        assert remote.AUTHKEY_ENV not in os.environ
    finally:
        policy.close()


def test_a_server_that_crashes_mid_episode_stops_the_run_and_the_run_resumes(tmp_path, fake_sim):
    log = tmp_path / "serve.log"
    crashing = RemotePolicy(policy="served_policies:Crashing", crash_at=9, log=log)
    with pytest.raises(PolicyError) as raised:
        runner.run(spec(tmp_path), crashing, FakeConfig(), log=quiet)
    message = str(raised.value)
    assert "the policy server hung up during act (it exited with status 3)" in message
    assert f"policy server log {log}, last lines:" in message and "crashing now" in message
    assert not running(crashing)
    assert [record.episode for record in RunDir(tmp_path / "run").records()] == [0]

    # The same command again resumes from episodes.jsonl.
    healthy = RemotePolicy(policy="served_policies:Crashing", crash_at=0, log=log)
    records = runner.run(spec(tmp_path), healthy, FakeConfig(), log=quiet)
    assert [record.episode for record in records] == [0, 1, 2, 3]
    assert all(record.success for record in records)


def test_a_server_that_hangs_times_out_and_is_stopped(tmp_path):
    policy = RemotePolicy(policy="served_policies:Hanging", timeout=0.5, log=tmp_path / "serve.log")
    policy.reset()
    policy.set_demonstration(demonstration())
    started = time.monotonic()
    with pytest.raises(PolicyError, match=r"no reply to act within 0.5 s") as raised:
        policy.act(observation())
    assert time.monotonic() - started < remote.KILL_GRACE_S  # terminated, not waited out
    assert "hanging in act" in str(raised.value)
    assert not running(policy)
    with pytest.raises(PolicyError, match="stopped after an earlier failure: no reply to act"):
        policy.describe()
    policy.close()  # nothing left to stop, and nothing to raise


def test_a_wrong_action_shape_is_a_policy_error_and_stops_the_server(tmp_path):
    policy = RemotePolicy(policy="served_policies:WrongShape", log=tmp_path / "serve.log")
    policy.reset()
    policy.set_demonstration(demonstration())
    with pytest.raises(PolicyError, match=r"act\(\) returned shape \(1, 5\), expected \(k, 14\)"):
        policy.act(observation())
    assert not running(policy)


def test_an_error_in_the_server_is_raised_here_with_the_end_of_its_log(tmp_path):
    policy = RemotePolicy(policy="served_policies:Unencodable", log=tmp_path / "serve.log")
    with pytest.raises(PolicyError) as raised:
        policy.episode_info()
    message = str(raised.value)
    assert "info failed in the policy server: ProtocolError: info.handle: object values" in message
    assert "Traceback" in message  # the server logged it, and the log's tail is quoted
    assert not running(policy)


def test_a_policy_the_server_cannot_build_is_a_policy_error_and_leaves_no_server(tmp_path, servers):
    log = tmp_path / "serve.log"
    with pytest.raises(PolicyError) as raised:
        RemotePolicy(policy="served_policies:BrokenConstructor", log=log)
    message = str(raised.value)
    assert "hello failed in the policy server: RuntimeError: checkpoint not found" in message
    assert f"policy server log {log}" in message
    assert [process.poll() is not None for process in servers] == [True]


def test_close_shuts_the_server_down_once(tmp_path):
    policy = RemotePolicy(policy="replay", log=tmp_path / "serve.log")
    directory = policy._server.directory
    assert directory.exists()  # its socket is gone: the server stops listening once connected
    policy.close()
    policy.close()
    assert policy._server.process.returncode == 0 and not directory.exists()
    assert "shut down" in (tmp_path / "serve.log").read_text()
    with pytest.raises(PolicyError, match=r"describe after close\(\)"):
        policy.describe()


def test_a_policy_dropped_without_close_takes_its_server_with_it(tmp_path):
    policy = RemotePolicy(policy="replay", log=tmp_path / "serve.log")
    process = policy._server.process
    del policy
    gc.collect()
    assert process.poll() is not None


def test_without_a_log_the_server_logs_to_a_temporary_file_that_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    policy = RemotePolicy(policy="replay")
    log = policy._server.log
    policy.close()
    assert log.parent == tmp_path and log.name.startswith("robotwin-icil-serve-")
    assert "listening on" in log.read_text()
    assert [path.name for path in tmp_path.iterdir()] == [log.name]  # the socket's directory went


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "give policy=SPEC"),
        ({"policy": "replay", "authkey_file": "k"}, "authkey_file is for address="),
        (
            {"address": "127.0.0.1:1", "policy": "replay", "authkey_file": "k"},
            "policy cannot be given with address=",
        ),
        (
            {"address": "127.0.0.1:1", "temperature": 0.5, "authkey_file": "k"},
            "temperature cannot be given with address=",
        ),
        ({"address": "127.0.0.1:1"}, "address= needs authkey_file="),
        ({"policy": "replay", "tags": [1, 2]}, "list values cannot be passed"),
        ({"policy": "replay", "timeout": 0}, "timeouts must be positive"),
    ],
)
def test_arguments_that_cannot_work_are_refused_before_a_server_starts(servers, kwargs, message):
    with pytest.raises(PolicyError, match=message):
        RemotePolicy(**kwargs)
    assert servers == []


def test_an_interpreter_that_does_not_exist_is_a_policy_error(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(PolicyError, match="cannot start a policy server with /nonexistent/python"):
        RemotePolicy(policy="replay", python="/nonexistent/python", log=tmp_path / "serve.log")
    assert [path.name for path in tmp_path.iterdir()] == ["serve.log"]


def test_remote_is_a_built_in(tmp_path):
    policy = make_policy("remote", policy="dummy", log=str(tmp_path / "serve.log"))
    try:
        assert isinstance(policy, RemotePolicy) and policy.describe()["policy"] == "dummy"
    finally:
        policy.close()
    with pytest.raises(PolicyError, match="built-ins are dummy, remote, replay, replay_ee"):
        make_policy("nope")


def test_a_running_server_is_reached_over_tcp_with_a_key_from_a_file(tmp_path):
    address = f"127.0.0.1:{free_port()}"
    (tmp_path / "key").write_text(KEY + "\n")
    (tmp_path / "guessed").write_text("guessed")
    log = tmp_path / "serve.log"
    with log.open("w") as handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "robotwin_icil.serve", "--policy", "replay"]
            + ["--address", address, "--authkey-file", str(tmp_path / "key")],
            env={**os.environ, "PYTHONPATH": str(SRC)},
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    wait_for(lambda: "listening on" in log.read_text())
    with pytest.raises(PolicyError, match=f"the policy server at {address} refused the key"):
        RemotePolicy(address=address, authkey_file=tmp_path / "guessed")
    policy = RemotePolicy(address=address, authkey_file=tmp_path / "key")
    assert policy.describe()["remote"] == {**REMOTE, "address": address}
    demo = demonstration()
    policy.reset()
    policy.set_demonstration(demo)
    np.testing.assert_array_equal(policy.act(observation()), demo.actions()[:1])
    policy.close()
    assert process.wait(timeout=20) == 0


def test_an_out_of_date_server_refuses_a_newer_client(tmp_path):
    (tmp_path / "key").write_text(KEY)
    address = tmp_path / "old.sock"
    process = subprocess.Popen(
        [sys.executable, str(TESTS / "old_serve.py"), "--policy", "replay"]
        + ["--address", str(address), "--authkey-file", str(tmp_path / "key")],
        env={**os.environ, "PYTHONPATH": str(SRC)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_for(lambda: address.exists() or process.poll() is not None)
    with pytest.raises(PolicyError) as raised:
        RemotePolicy(address=str(address), authkey_file=tmp_path / "key")
    assert "the client speaks protocol version 1, this server 0" in str(raised.value)
    assert "policy server traceback:" in str(raised.value)  # no local log: the reply's own
    assert process.wait(timeout=20) == 1


HELLO = {
    "protocol_version": protocol.PROTOCOL_VERSION,
    "action_type": "qpos",
    "description": {"policy": "fake", "action_type": "qpos"},
    "python": "fake",
}


@contextlib.contextmanager
def fake_server(tmp_path, answer):
    """A server in a thread, answering each message with `answer(op, fields) -> (op, fields)`."""
    address = str(tmp_path / "fake.sock")
    (tmp_path / "fake.key").write_text(KEY)
    listener = Listener(address, "AF_UNIX", authkey=KEY.encode())
    ops = []

    def serve():
        with listener, listener.accept() as conn:
            while True:
                try:
                    message = protocol.receive(conn, timeout=20)
                except (EOFError, OSError):
                    return
                ops.append(message.op)
                op, fields = answer(message.op, message.fields)
                protocol.send(conn, op, **fields)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield address, tmp_path / "fake.key", ops
    thread.join(10)
    assert not thread.is_alive(), "the client never hung up"


def test_a_server_of_another_protocol_version_is_refused(tmp_path):
    with fake_server(tmp_path, lambda op, fields: ("hello", {**HELLO, "protocol_version": 0})) as (
        address,
        key,
        ops,
    ):
        with pytest.raises(PolicyError, match="the policy server speaks protocol version 0, this"):
            RemotePolicy(address=address, authkey_file=key)
    assert ops == ["hello"]


@pytest.mark.parametrize(
    "actions, message",
    [
        (np.zeros(5), r"fake \(remote\): act\(\) returned shape \(1, 5\), expected \(k, 14\)"),
        (np.full((2, 14), np.nan), r"fake \(remote\): act\(\) returned a non-finite action"),
    ],
)
def test_actions_are_checked_here_as_well_as_in_the_server(tmp_path, actions, message):
    def answer(op, fields):
        if op == "hello":
            return "hello", HELLO
        return op, {"actions": actions} if op == "act" else {}

    with fake_server(tmp_path, answer) as (address, key, ops):
        policy = RemotePolicy(address=address, authkey_file=key)
        policy.reset()
        policy.set_demonstration(demonstration())
        with pytest.raises(PolicyError, match=message):
            policy.act(observation())
        policy.close()
    assert ops[-1] == "shutdown"
