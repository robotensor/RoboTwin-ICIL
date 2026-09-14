"""Fixtures shared by the pure tests and the simulator tests: a policy served at an address.

`serve_policy` runs the real `python -m icil_policy.serve` in a subprocess, so the benchmark is
tested against the server a competition runs, not against a stand-in. It needs the `icil-policy`
distribution, which the benchmark's CI cannot install: every test using it skips there.
"""

import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

#: The environment variable a served policy's key travels in, in tests.
AUTHKEY_ENV = "ROBOTWIN_ICIL_TEST_POLICY_KEY"

#: A policy that answers like the replay example, except where it is told to raise or hang.
FLAKY_POLICY = """
import time

import numpy as np


class Flaky:
    action_type = "qpos"

    def __init__(self, where="", at=2, sleep_s=0.0):
        if where == "init":
            raise RuntimeError("the policy refuses to be built")
        self.where, self.at, self.sleep_s = where, at, sleep_s
        self.actions, self.step = None, 0

    def reset(self, seed):
        self.step = 0
        if self.where == "reset":
            raise RuntimeError("boom in reset")

    def set_demonstration(self, arrays, info):
        if self.where == "prompt":
            raise RuntimeError("boom in prompt")
        self.actions = np.array(arrays["actions"])

    def act(self, observation):
        if self.step == self.at and self.where == "act":
            raise RuntimeError("boom in act")
        if self.step == self.at and self.where == "hang":
            time.sleep(self.sleep_s)
        if self.where == "slow":
            time.sleep(self.sleep_s)
        k = min(self.step, len(self.actions) - 1)
        self.step += 1
        return {"action": self.actions[k]}

    def close(self):
        if self.where == "close":
            time.sleep(self.sleep_s)
"""


@dataclass
class Served:
    """One served policy: its process, the address it listens on and the log it writes."""

    process: subprocess.Popen
    address: str
    log_file: Path
    directory: Path
    authkey_env: str = AUTHKEY_ENV


@pytest.fixture
def replay_manifest():
    """The `icil.yaml` of icil-policy's replay example: under `ICIL_POLICY_EXAMPLES`, or beside
    the installed package's source tree."""
    icil_policy = pytest.importorskip("icil_policy")
    root = os.environ.get("ICIL_POLICY_EXAMPLES")
    examples = Path(root) if root else Path(icil_policy.__file__).resolve().parents[2] / "examples"
    manifest = examples / "replay_policy" / "icil.yaml"
    if not manifest.is_file():
        pytest.skip(f"no icil-policy replay example at {manifest}; set ICIL_POLICY_EXAMPLES")
    return manifest


@pytest.fixture
def policy_repo(tmp_path):
    """`policy_repo(**kwargs)`: the `icil.yaml` of a repository serving `FLAKY_POLICY`, built
    with `kwargs` (`where`: init, reset, prompt, act, hang, slow or close; `at`: the act step;
    `sleep_s`, how long `hang` stalls that act, `slow` every act and `close` closing)."""

    def write(**kwargs):
        root = tmp_path / "flaky-repo"
        (root / "flaky").mkdir(parents=True, exist_ok=True)
        (root / "flaky" / "__init__.py").write_text("")
        (root / "flaky" / "policy.py").write_text(FLAKY_POLICY)
        manifest = root / "icil.yaml"
        manifest.write_text(f"api: 1\npolicy: flaky.policy:Flaky\nkwargs: {json.dumps(kwargs)}\n")
        return manifest

    return write


@pytest.fixture
def serve_policy(monkeypatch):
    """`serve_policy(manifest)`: `python -m icil_policy.serve` listening on a Unix socket, its key in
    `AUTHKEY_ENV` for run-unit to read. Every server is killed, if still alive, afterwards."""
    pytest.importorskip("icil_policy")
    started: list[Served] = []

    def start(manifest, *, idle_timeout_s=120.0):
        # A socket path holds about a hundred bytes; pytest's tmp_path can be longer.
        directory = Path(tempfile.mkdtemp(prefix="robotwin-icil-"))
        authkey = secrets.token_bytes(32)
        address, log_file = directory / "policy.sock", directory / "policy.log"
        argv = [sys.executable, "-m", "icil_policy.serve", "--manifest", str(manifest)]
        argv += ["--address", str(address), "--authkey-env", AUTHKEY_ENV]
        argv += ["--log-file", str(log_file), "--idle-timeout-s", str(idle_timeout_s)]
        env = {**os.environ, AUTHKEY_ENV: authkey.hex(), "PYTHONDONTWRITEBYTECODE": "1"}
        process = subprocess.Popen(
            argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )
        served = Served(process, str(address), log_file, directory)
        started.append(served)
        deadline = time.monotonic() + 30.0
        while not address.exists():
            if process.poll() is not None:
                pytest.fail(f"the policy server exited {process.returncode} before listening")
            if time.monotonic() > deadline:
                pytest.fail("the policy server did not listen within 30s")
            time.sleep(0.05)
        monkeypatch.setenv(AUTHKEY_ENV, authkey.hex())
        return served

    yield start
    for served in started:
        if served.process.poll() is None:
            served.process.kill()
        served.process.wait()
        shutil.rmtree(served.directory, ignore_errors=True)
