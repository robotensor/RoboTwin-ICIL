"""run-unit against a real `python -m icil_policy.serve`, through the command line.

The replay example served at an address, a server killed mid-episode, a policy that hangs or
raises, one the server cannot build, and every byte run-unit sends a policy. The simulator is the
fake one; the server, the client and the socket are real. Skips without icil-policy.
"""

import contextlib
import json
import os
import shutil
import socket
import struct
import tempfile
import threading

import pytest

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import cli, prompt, records, robotwin, unit

pytest.importorskip("icil_policy")
pytest.importorskip("imageio_ffmpeg")

TASK = "click_bell"
SEED = 11


@pytest.fixture(autouse=True)
def fake_sim(monkeypatch):
    envs = {"next": lambda name: FakeTaskEnv()}
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "SceneConfig", FakeConfig)
    monkeypatch.setattr(robotwin, "load_task", lambda name: envs["next"](name))
    return envs


def materialize(tmp_path, seeds=(SEED,)):
    argv = ["materialize", "--task", TASK, "--out", str(tmp_path / "prompt")]
    for seed in seeds:
        argv += ["--scene-seed", str(seed)]
    assert cli.main(argv) == 0
    return tmp_path / "prompt" / "prompt.npz"


def run_unit(tmp_path, path, served, *extra, address=None):
    argv = ["run-unit", "--prompt", str(path), "--out", str(tmp_path / "run")]
    argv += ["--policy-address", address or served.address, "--authkey-env", served.authkey_env]
    argv += ["--policy-log", str(served.log_file), *extra]
    assert cli.main(argv) == 0
    return json.loads((tmp_path / "run" / "result.json").read_text())


def test_the_replay_example_served_at_an_address_succeeds(tmp_path, serve_policy, replay_manifest):
    path = materialize(tmp_path)
    served = serve_policy(replay_manifest)
    result = run_unit(tmp_path, path, served)
    assert result["success"] is True and result["void"] is False and result["void_cause"] is None
    assert result["steps"] == FakeTaskEnv().expert_steps and result["error"] is None
    assert result["policy"] == "remote" and result["served_policy"] == "replay.policy:ReplayPolicy"
    # The client said close however the unit ended, and the server went with it.
    assert served.process.wait(timeout=15) == 0
    # The key was read once and removed, so nothing run-unit started inherited it.
    assert served.authkey_env not in os.environ


def test_a_server_killed_mid_episode_voids_the_unit_on_the_policy(
    tmp_path, fake_sim, serve_policy, replay_manifest
):
    path = materialize(tmp_path)
    served = serve_policy(replay_manifest)

    class KillsTheServer(FakeTaskEnv):
        def take_action(self, action, action_type="qpos"):
            super().take_action(action, action_type)
            if self.take_action_cnt == 2:
                served.process.kill()
                served.process.wait()

    fake_sim["next"] = lambda name: KillsTheServer()
    result = run_unit(tmp_path, path, served)
    assert result["void"] is True and result["void_cause"] == "policy"
    assert result["success"] is None and result["steps"] is None
    assert result["error"].startswith("the policy is unreachable: act: the policy went away")
    assert "--- policy log (tail) ---" in result["error"] and "listening on" in result["error"]
    assert served.process.poll() is not None


def test_a_log_the_policy_swapped_for_a_link_adds_nothing_to_the_error(
    tmp_path, fake_sim, serve_policy, replay_manifest
):
    # The policy writes its own log's directory: once it listens it can put a link to a file of
    # the benchmark's host (here, this process's environment) where its log was.
    path = materialize(tmp_path)
    served = serve_policy(replay_manifest)
    os.unlink(served.log_file)
    os.symlink("/proc/self/environ", served.log_file)

    class KillsTheServer(FakeTaskEnv):
        def take_action(self, action, action_type="qpos"):
            super().take_action(action, action_type)
            if self.take_action_cnt == 2:
                served.process.kill()
                served.process.wait()

    fake_sim["next"] = lambda name: KillsTheServer()
    result = run_unit(tmp_path, path, served)
    assert result["void"] is True and result["void_cause"] == "policy"
    assert result["error"].startswith("the policy is unreachable: act: the policy went away")
    assert "policy log (tail)" not in result["error"] and "PATH=" not in result["error"]


def test_a_policy_that_hangs_is_void_at_its_act_timeout_and_its_server_exits(
    tmp_path, serve_policy, policy_repo
):
    path = materialize(tmp_path)
    served = serve_policy(policy_repo(where="hang", at=2, sleep_s=300))
    result = run_unit(tmp_path, path, served, "--act-timeout-s", "1")
    assert result["void"] is True and result["void_cause"] == "policy"
    assert result["error"].startswith("the policy is unreachable: act: no answer within 1s")
    # No server outlives its unit: the client hung up mid-call, and the server exits on that.
    assert served.process.wait(timeout=15) == 0


@pytest.mark.parametrize("where", ["reset", "prompt", "act"])
def test_a_policy_that_answers_with_an_error_fails_the_unit(
    tmp_path, serve_policy, policy_repo, where
):
    path = materialize(tmp_path)
    served = serve_policy(policy_repo(where=where, at=2))
    result = run_unit(tmp_path, path, served)
    assert result["success"] is False and result["void"] is False and result["void_cause"] is None
    assert f"the served policy answered {where} with an error" in result["detail"]
    assert f"boom in {where}" in result["detail"]
    assert result["steps"] == (2 if where == "act" else 0)
    assert served.process.wait(timeout=15) == 0


def test_a_policy_the_server_cannot_build_voids_on_the_policy(tmp_path, serve_policy, policy_repo):
    path = materialize(tmp_path)
    served = serve_policy(policy_repo(where="init"))
    result = run_unit(tmp_path, path, served)
    assert result["void"] is True and result["void_cause"] == "policy"
    assert result["error"].startswith("the policy is unreachable: hello: RuntimeError")
    assert "refuses to be built" in result["error"] and result["served_policy"] is None
    assert served.process.wait(timeout=15) == 1


class Relay:
    """A Unix socket between run-unit and the server that keeps every byte run-unit sends."""

    def __init__(self, target):
        self.directory = tempfile.mkdtemp(prefix="robotwin-icil-relay-")
        self.address = os.path.join(self.directory, "relay.sock")
        self.sent = bytearray()
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(self.address)
        self._listener.listen(1)
        self._thread = threading.Thread(target=self._relay, args=(target,), daemon=True)
        self._thread.start()

    def _relay(self, target):
        client, _ = self._listener.accept()
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        upstream.connect(target)
        back = threading.Thread(target=self._pump, args=(upstream, client, None), daemon=True)
        back.start()
        self._pump(client, upstream, self.sent)
        back.join()
        for sock in (client, upstream, self._listener):
            sock.close()

    @staticmethod
    def _pump(source, sink, keep):
        try:
            while chunk := source.recv(1 << 16):
                if keep is not None:
                    keep.extend(chunk)
                sink.sendall(chunk)
        except OSError:
            pass
        finally:
            for sock in (source, sink):
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)

    def frames(self):
        """Every frame sent, as `multiprocessing.connection` frames them: authentication first."""
        self._thread.join(timeout=15)
        data, frames, at = bytes(self.sent), [], 0
        while at < len(data):
            (size,) = struct.unpack("!i", data[at : at + 4])
            at += 4
            if size == -1:
                (size,) = struct.unpack("!Q", data[at : at + 8])
                at += 8
            frames.append(data[at : at + size])
            at += size
        return frames

    def close(self):
        shutil.rmtree(self.directory, ignore_errors=True)


def _names(value):
    """Every key of a JSON value, however deep."""
    if isinstance(value, dict):
        return set(value) | set().union(*(_names(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_names(v) for v in value))
    return set()


def test_nothing_of_meta_and_no_scene_seed_crosses_the_wire(
    tmp_path, fake_sim, serve_policy, replay_manifest
):
    # Distinctive candidates, the first rejected, so meta names a chosen seed and a candidate.
    rejected, chosen = 918273645, 918273646
    fake_sim["next"] = lambda name: FakeTaskEnv(unstable_seeds={rejected})
    path = materialize(tmp_path, seeds=(rejected, chosen))
    fake_sim["next"] = lambda name: FakeTaskEnv()
    _, meta = prompt.read_raw(path)
    assert meta["scene_seed"] == chosen and meta["expert"]["scene_seeds"] == [rejected, chosen]

    served = serve_policy(replay_manifest)
    relay = Relay(served.address)
    try:
        result = run_unit(tmp_path, path, served, address=relay.address)
        frames = relay.frames()
    finally:
        relay.close()
    assert result["success"] is True

    headers = [json.loads(f) for f in frames if f.startswith(b'{"protocol":')]
    assert [h["op"] for h in headers][:3] == ["hello", "reset", "prompt"]
    assert headers[-1]["op"] == "close"
    names = set().union(*(_names(h["fields"]) | {a["name"] for a in h["arrays"]} for h in headers))
    # No key of meta names anything sent, but the two info publishes by the protocol's own
    # definition, and those hold the robot's name and the camera list, not meta's values.
    assert names & set(meta) == {"embodiment", "cameras"} and "meta" not in names
    [prompt_header] = [h for h in headers if h["op"] == "prompt"]
    assert prompt_header["fields"]["info"]["embodiment"] == meta["embodiment"]["name"]

    sent = b"".join(frames)
    [reset_header] = [h for h in headers if h["op"] == "reset"]
    assert reset_header["fields"]["seed"] == unit.episode_seed(prompt.sha256_of(path))
    for seed in (rejected, chosen):
        assert reset_header["fields"]["seed"] != seed
        for encoded in (str(seed).encode(), struct.pack("<q", seed), struct.pack("<i", seed)):
            assert encoded not in sent, seed
    secrets_ = [
        meta["scene"]["sha256"],
        result["prompt_sha256"],
        TASK,
        records.git_commit(robotwin.REPO_ROOT)[:40],
    ]
    for secret in secrets_:
        assert secret.encode() not in sent, secret
    for digest in (meta["scene"]["sha256"], result["prompt_sha256"]):
        assert bytes.fromhex(digest) not in sent
