"""The run path, without the simulator: the policy seam and the prompt round trip.

`run_unit` itself builds a RoboTwin scene, so it is exercised against the real simulator
separately. What is checked here is everything on the way in - which policy a unit runs, and what
a `Demonstration` rebuilt from a published prompt still carries - because that is where a duel
silently turns into `void` rather than a verdict.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from icil_benchmark_robotwin import BENCHMARK
from icil_benchmark_robotwin.cli import build_parser
from icil_benchmark_robotwin.plugin import PROMPT_NAME
from icil_benchmark_robotwin.prompt import CHANNELS, channel_of, dump, load
from icil_benchmark_robotwin.run import _demonstration, _policy
from icil_benchmark_robotwin.units import derive_units

from robotwin_icil.demo import ARMS, Demonstration, Frame
from robotwin_icil.policy import PolicyError
from robotwin_icil.remote import RemotePolicy

KEY = "0" * 64
UNIT = derive_units(seed_material="duel-1", count=1, suite="v1")[0].as_dict()


# ---------------------------------------------------------------- demonstrations and prompts


def demonstration(frames: int = 4, *, measured: bool = True) -> Demonstration:
    def frame(i: int) -> Frame:
        pose = np.array([0.2 + 0.01 * i, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0])
        return Frame(
            index=i,
            images={"head_camera": np.full((4, 4, 3), i, dtype=np.uint8)},
            qpos=np.full(14, float(i)),
            endpose={
                "left_endpose": pose,
                "left_gripper": 1.0,
                "right_endpose": pose + 0.5,
                "right_gripper": 0.0,
            },
            gripper_joints=(
                {"left": np.array([0.01 * i, 0.01 * i + 0.001]), "right": np.array([0.04, 0.041])}
                if measured
                else None
            ),
        )

    return Demonstration(frames=tuple(frame(i) for i in range(frames)), frequency=16.0)


def written(tmp_path: Path, **kwargs) -> dict:
    dump(demonstration(**kwargs), tmp_path / PROMPT_NAME, task="click_bell", scene_seed=7)
    return load(tmp_path / PROMPT_NAME)


def test_a_prompt_carries_the_measured_gripper_joints(tmp_path):
    """RoboTwin's gripper value in `qpos` and `endpose` is the *command*, which reads closed while
    the fingers rest on an object. Where the fingers actually are is proprioception, it is what a
    model trained on `gripper_states` reads, and only the prompt can carry it."""
    doc = written(tmp_path)
    source = demonstration()
    assert doc["gripper_joints"].shape == (len(source), len(ARMS), 2)
    rebuilt = _demonstration(doc)
    for frame, original in zip(rebuilt.frames, source.frames, strict=True):
        for arm in ARMS:
            assert frame.gripper_joints[arm] == pytest.approx(original.gripper_joints[arm])


def test_the_measurement_is_proprioception_so_a_view_can_drop_it_with_the_rest(tmp_path):
    """The orchestrator's view allows or drops whole channels; an array in none of them would
    never reach a policy at all."""
    assert channel_of("gripper_joints") == "proprio"
    assert "gripper_joints" in BENCHMARK.info()["demo_channels"]["proprio"]
    assert set(BENCHMARK.info()["demo_channels"]) == set(CHANNELS)


def test_a_prompt_written_before_the_measurement_still_rebuilds(tmp_path):
    """Older published prompts have no such array, and are read as the demonstrations they are:
    with no measurement, rather than with a fabricated one."""
    doc = written(tmp_path, measured=False)
    assert "gripper_joints" not in doc
    rebuilt = _demonstration(doc)
    assert all(frame.gripper_joints is None for frame in rebuilt.frames)


def test_a_rebuilt_demonstration_is_the_one_that_was_published(tmp_path):
    source = demonstration()
    rebuilt = _demonstration(written(tmp_path))
    assert len(rebuilt) == len(source) and rebuilt.cameras == source.cameras
    assert rebuilt.qpos() == pytest.approx(source.qpos())
    assert rebuilt.endposes() == pytest.approx(source.endposes())
    assert rebuilt.times() == pytest.approx(source.times())
    assert rebuilt.images("head_camera").tolist() == source.images("head_camera").tolist()


def test_a_prompt_with_the_measurement_still_verifies(tmp_path):
    written(tmp_path)
    verdict = BENCHMARK.verify_prompt(path=str(tmp_path), unit={"task": "click_bell"})
    assert verdict["ok"], verdict["problems"]


# ---------------------------------------------------------------- which policy a unit runs


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def served(tmp_path):
    """A policy server the "orchestrator" already runs, as `--policy-address` assumes."""
    address = f"127.0.0.1:{free_port()}"
    authkey = tmp_path / "authkey"
    authkey.write_text(KEY + "\n")
    log = tmp_path / "serve.log"
    with log.open("w") as handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "robotwin_icil.serve", "--policy", "replay"]
            + ["--address", address, "--authkey-file", str(authkey)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ},
        )
    deadline = time.monotonic() + 30
    while "listening on" not in log.read_text():
        assert process.poll() is None, log.read_text()
        assert time.monotonic() < deadline, "the policy server never listened"
        time.sleep(0.02)
    try:
        yield address, str(authkey)
    finally:
        process.terminate()
        process.wait(10)


def test_a_policy_address_reaches_a_policy_the_orchestrator_is_serving(served, tmp_path):
    """The whole reason the plugin exists: the entrant's weights run in their own environment,
    behind their own server, and the simulator never imports them."""
    address, authkey = served
    policy = _policy(address, None, authkey)
    try:
        assert isinstance(policy, RemotePolicy)
        assert policy.describe()["policy"] == "replay"
        assert policy.action_type == "qpos"
        # It is an ICILPolicy here too, so a unit is checked on both sides of the socket.
        policy.seed(3)
        policy.reset()
        policy.set_demonstration(_demonstration(written(tmp_path)))
        actions = policy.act(_observation())
        assert actions.shape == (1, 14)
    finally:
        policy.close()


def _observation():
    from robotwin_icil.policy import Observation

    return Observation(
        step=0, images={"head_camera": np.zeros((4, 4, 3), np.uint8)}, qpos=np.zeros(14)
    )


def test_a_served_policy_needs_the_key_the_orchestrator_handed_both_sides(served, tmp_path):
    address, _ = served
    wrong = tmp_path / "guessed"
    wrong.write_text("guessed")
    with pytest.raises(PolicyError, match="refused the key|cannot connect"):
        _policy(address, None, str(wrong))
    with pytest.raises(PolicyError, match="address= needs authkey_file="):
        _policy(address, None, None)


def test_an_importable_policy_still_runs_in_this_process():
    policy = _policy(None, "replay", None)
    assert policy.describe() == {"policy": "replay", "action_type": "qpos"}
    policy.close()


def test_a_unit_runs_one_policy_and_the_record_says_which():
    with pytest.raises(RuntimeError, match="are alternatives"):
        _policy("/work/p.sock", "replay", None)
    with pytest.raises(RuntimeError, match="one of --policy-address or --policy is required"):
        _policy(None, None, None)


def test_the_key_file_reaches_the_cli_the_plugin_names():
    """A server on TCP needs its key, and the orchestrator passes it through `**extra`."""
    argv = BENCHMARK.run_command(
        unit=UNIT,
        prompt="/prompt/u0/prompt.npz",
        out_dir="/work/u0",
        policy_address="127.0.0.1:9000",
        authkey_file="/work/authkey",
    )
    args = build_parser().parse_args([str(a) for a in argv[1:]])
    assert args.policy_address == "127.0.0.1:9000"
    assert args.authkey_file == "/work/authkey"
    assert args.policy is None
