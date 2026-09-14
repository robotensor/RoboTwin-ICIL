"""A policy served at an address: what the adapter sends it, and what each way it fails costs.

The adapter is driven through a fake `icil_policy.client.RemotePolicy` here, which fails exactly
where a test says; `test_served_policy.py` drives a real server. Tests that need `icil_policy`'s
exception types skip without it (the benchmark's CI cannot install it); the rest run everywhere.
"""

import subprocess
import sys
import tracemalloc

import numpy as np
import pytest
import yaml

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import prompt, remote, robotwin, unit
from robotwin_icil.policy import EpisodeInfo, Observation, PolicyError, Unscorable

TASK = "click_bell"
SEED = 11
KEY = bytes(range(32))
REPLAY = {"protocol": 1, "action_type": "qpos", "policy": "replay.policy:ReplayPolicy"}


@pytest.fixture(autouse=True)
def _fake_sim(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "SceneConfig", FakeConfig)


@pytest.fixture
def errors():
    """`icil_policy.errors`, whose exceptions the adapter maps; the test skips without it."""
    return pytest.importorskip("icil_policy.errors")


class Script:
    """What the fake client answers, where it fails, and every client it built."""

    def __init__(self, *, fail=None, hello=REPLAY, chunk=1, reshape=None):
        self.fail = dict(fail or {})
        self.hello = dict(hello)
        self.chunk = chunk
        self.reshape = reshape
        self.clients = []

    def factory(self, address, authkey, *, timeout_s, log_file=None):
        return FakeClient(self, address, authkey, timeout_s, log_file)


class FakeClient:
    """Stands in for `icil_policy.client.RemotePolicy`: replays the demonstration's actions, as the
    replay example does, and records each call with the timeout it was made under."""

    def __init__(self, script, address, authkey, timeout_s, log_file):
        script.clients.append(self)
        self.script = script
        self.connected = (address, authkey, timeout_s, log_file)
        if "connect" in script.fail:
            raise script.fail["connect"]
        self.timeout_s = timeout_s
        self.calls = []
        self.closed = False

    def _record(self, op, **fields):
        self.calls.append((op, self.timeout_s, fields))
        if op in self.script.fail:
            raise self.script.fail[op]

    def hello(self):
        self._record("hello")
        return dict(self.script.hello)

    def reset(self, seed):
        self._record("reset", seed=seed)
        self._step = 0

    def set_demonstration(self, arrays, info):
        self._record("prompt", arrays=dict(arrays), info=dict(info))
        self._actions = np.array(arrays["actions"])

    def act(self, observation):
        self._record("act", arrays=dict(observation))
        rows = [min(self._step + k, len(self._actions) - 1) for k in range(self.script.chunk)]
        self._step += self.script.chunk
        action = self._actions[rows] if self.script.chunk > 1 else self._actions[rows[0]]
        if self.script.reshape is not None:
            action = self.script.reshape(action)
        return {"action": action}

    def close(self):
        self.closed = True
        self.closed_within = self.timeout_s

    def ops(self):
        return [op for op, _, _ in self.calls]

    def call(self, op):
        return next(fields for name, _, fields in self.calls if name == op)


def materialized(tmp_path):
    out = tmp_path / "prompt"
    done = unit.materialize(TASK, SEED, FakeConfig(), out, task_env=FakeTaskEnv(), video=False)
    assert done.ok
    return out / "prompt.npz"


def run(tmp_path, script, **options):
    path = materialized(tmp_path)
    policy = remote.RemotePolicy("/run/policy.sock", KEY, client_factory=script.factory, **options)
    result = unit.run_unit(path, policy, tmp_path / "run", task_env=FakeTaskEnv(), video=False)
    return path, policy, result


def test_a_served_replay_is_sent_the_prompts_own_arrays_and_public_info(tmp_path, errors):
    script = Script()
    path, policy, result = run(tmp_path, script, act_timeout_s=7.0, log_file="policy.log")
    assert result["success"] is True and result["void"] is False and result["void_cause"] is None
    assert result["policy"] == "remote" and result["served_policy"] == REPLAY["policy"]
    assert result["model"] == REPLAY["policy"]

    [client] = script.clients
    assert client.connected == (
        "/run/policy.sock",
        KEY,
        remote.CONNECT_TIMEOUT_S,
        tmp_path.cwd() / "policy.log",
    )
    assert client.ops()[:3] == ["hello", "reset", "prompt"] and set(client.ops()[3:]) == {"act"}
    # Each call under its own timeout: setting a policy up is work, an act has a budget.
    timeouts = {op: seconds for op, seconds, _ in client.calls}
    setup = remote.SETUP_TIMEOUT_S
    assert timeouts == {"hello": setup, "reset": setup, "prompt": setup, "act": 7.0}

    # The demonstration is the file's own arrays, name for name and byte for byte; meta is not.
    on_disk, _ = prompt.read_raw(path)
    sent = client.call("prompt")
    assert sorted(sent["arrays"]) == sorted(on_disk) and "meta" not in sent["arrays"]
    for name, array in on_disk.items():
        assert sent["arrays"][name].dtype == array.dtype, name
        assert np.array_equal(sent["arrays"][name], array), name
    assert sent["info"] == {
        "frequency": float(on_disk["frequency"]),
        "cameras": ["head_camera"],
        "embodiment": "fake-arms",
        "action_dims": {"qpos": 14, "ee": 16},
    }
    # The episode's seed is drawn from the prompt's bytes, never the scene seed.
    assert client.call("reset") == {"seed": unit.episode_seed(prompt.sha256_of(path))}
    observation = client.call("act")["arrays"]
    assert {name: a.shape for name, a in observation.items()} == {
        "frames_head_camera": (16, 16, 3),
        "qpos": (14,),
        "endpose": (16,),
    }

    policy.close()
    policy.close()
    assert client.closed


@pytest.mark.parametrize("op", ["reset", "prompt", "act"])
def test_an_error_reply_is_the_policys_failure_never_a_void(tmp_path, errors, op):
    # The policy answered, by raising: what it did is its result, as a local policy that raises.
    failure = errors.PolicyUnavailable(
        f"{op}: RuntimeError: boom", op=op, remote_type="RuntimeError", log_tail="policy log"
    )
    _, policy, result = run(tmp_path, Script(fail={op: failure}))
    assert result["success"] is False and result["void"] is False and result["void_cause"] is None
    assert result["error"] is None and result["steps"] == 0
    assert (
        f"the served policy answered {op} with an error: {op}: RuntimeError: boom"
        in (result["detail"])
    )


def test_close_has_its_own_short_timeout_whatever_the_last_call_had(tmp_path, errors):
    # A unit whose policy raised in reset has its result; the close after it must not wait out
    # the 300s reset had.
    failure = errors.PolicyUnavailable("reset: RuntimeError: boom", op="reset", remote_type="E")
    script = Script(fail={"reset": failure})
    _, policy, result = run(tmp_path, script)
    assert result["success"] is False
    [client] = script.clients
    assert client.timeout_s == remote.SETUP_TIMEOUT_S
    policy.close()
    assert client.closed and client.closed_within == remote.CLOSE_TIMEOUT_S


@pytest.mark.parametrize(
    ("op", "remote_type", "message"),
    [
        ("connect", None, "connect: nothing listened at /run/policy.sock within 60s"),
        ("connect", None, "connect: the server at /run/policy.sock refused this key"),
        # An error reply to hello is a policy the server could not build: the session is over.
        ("hello", "RuntimeError", "hello: RuntimeError: the policy refuses to be built"),
        ("hello", None, "hello: no answer within 300s"),
        ("reset", None, "reset: the policy went away: EOFError"),
        ("prompt", None, "prompt: malformed reply: header is not a JSON object"),
        ("act", None, "act: no answer within 7s"),
        ("act", None, "act: the reply holds no usable 'action' (shape None)"),
    ],
)
def test_a_policy_that_cannot_be_spoken_to_voids_on_the_policy(
    tmp_path, errors, op, remote_type, message
):
    failure = errors.PolicyUnavailable(
        message, op=op, remote_type=remote_type, log_tail="serve[1] the server's last words"
    )
    _, policy, result = run(tmp_path, Script(fail={op: failure}))
    assert result["void"] is True and result["void_cause"] == "policy"
    assert result["success"] is None and result["steps"] is None
    assert result["error"].startswith(f"the policy is unreachable: {message}")
    assert "the server's last words" in result["error"]  # the log tail travels with the reason


def test_a_policy_log_swapped_for_a_link_is_never_followed(tmp_path, errors, monkeypatch):
    # The policy can write beside its log. A path resolved when run-unit starts would open the
    # link's target, and the client's no-follow open would never see a link to refuse.
    logs = pytest.importorskip("icil_policy.logs")
    secret = tmp_path / "host-secret"
    secret.write_text("HOST-SECRET-TOKEN\n")
    (tmp_path / "logs").mkdir()
    link = tmp_path / "logs" / "policy.log"
    link.symlink_to(secret)
    monkeypatch.chdir(tmp_path)

    policy = remote.RemotePolicy("/run/policy.sock", KEY, log_file="logs/policy.log")
    assert policy.log_file == link and policy.log_file.is_absolute()
    assert logs.tail(policy.log_file) == ""


def test_what_a_hostile_server_says_is_bounded_before_it_reaches_the_result(tmp_path, errors):
    # icil_policy.serve cuts its messages short; a server that is not icil_policy.serve need not.
    name = {**REPLAY, "policy": "P" * 3_000_000}
    raised = errors.PolicyUnavailable(
        "reset: RoboTwinError: " + "M" * 1_000_000,
        op="reset",
        remote_type="RoboTwinError",
        log_tail="T" * 2_000_000,
    )
    _, _, failed = run(tmp_path / "failed", Script(hello=name, fail={"reset": raised}))
    assert failed["success"] is False and failed["void"] is False
    assert len(failed["served_policy"]) == len(failed["model"]) <= remote.MAX_NAME_CHARS + 40
    assert failed["served_policy"].startswith("P" * 100)
    detail = failed["detail"]
    assert len(detail) < 13_000 and "reset: RoboTwinError: MMM" in detail
    assert detail.endswith("T" * 100) and "characters ...]" in detail
    assert (tmp_path / "failed" / "run" / "result.json").stat().st_size < 64_000

    lost = errors.PolicyUnavailable("act: the policy went away", op="act", log_tail="T" * 2_000_000)
    _, _, void = run(tmp_path / "void", Script(hello=name, fail={"act": lost}))
    assert void["void_cause"] == "policy" and len(void["error"]) < 13_000
    assert void["error"].startswith("the policy is unreachable: act: the policy went away")

    torque = {**REPLAY, "action_type": "t" * 3_000_000}
    _, _, refused = run(tmp_path / "refused", Script(hello=torque))
    assert refused["void_cause"] == "policy" and len(refused["error"]) < 1_000


def test_an_action_type_the_benchmark_cannot_execute_is_refused_at_hello(tmp_path, errors):
    script = Script(hello={**REPLAY, "action_type": "torque"})
    _, policy, result = run(tmp_path, script)
    assert result["void"] is True and result["void_cause"] == "policy"
    assert "declares action_type 'torque'" in result["error"]
    [client] = script.clients
    assert client.ops() == ["hello"] and client.closed


def test_a_chunk_of_actions_is_executed_row_by_row(tmp_path, errors):
    script = Script(chunk=4)
    _, _, result = run(tmp_path, script)
    assert result["success"] is True and result["steps"] == 6
    [client] = script.clients
    assert client.ops().count("act") == 2


@pytest.mark.parametrize(
    ("hello", "reshape", "reason"),
    [
        (REPLAY, lambda action: action[:13], "act() returned shape (1, 13), expected (k, 14)"),
        ({**REPLAY, "action_type": "ee"}, None, "expected (k, 16) for action_type 'ee'"),
    ],
)
def test_an_action_of_the_wrong_width_fails_the_unit(tmp_path, errors, hello, reshape, reason):
    _, _, result = run(tmp_path, Script(hello=hello, reshape=reshape))
    assert result["success"] is False and result["void"] is False
    assert reason in result["detail"]


def acting(tmp_path, script):
    """A served policy reset and handed its demonstration, and an observation to act on."""
    policy = remote.RemotePolicy("/run/policy.sock", KEY, client_factory=script.factory)
    demonstration, _ = prompt.read_prompt(materialized(tmp_path))
    env = FakeTaskEnv()
    env.setup_demo(seed=SEED)
    raw = robotwin.observation(env)
    dims = {"qpos": 14, "ee": 16}
    policy.reset(EpisodeInfo(embodiment="fake-arms", action_dims=dims, seed=1))
    policy.set_demonstration(demonstration)
    observation = Observation(
        step=0, images=raw["images"], qpos=raw["qpos"], endpose=raw["endpose"]
    )
    return policy, observation, dims


def test_an_oversized_action_is_refused_before_it_is_copied(tmp_path, errors):
    # 8 MiB on the wire, 64 MiB as float64: converted first, it would cost run-unit eight times
    # what the policy paid to send it, up to the client's gigabyte reply bound.
    huge = np.zeros((1, 1 << 23), dtype=bool)
    policy, observation, dims = acting(tmp_path, Script(reshape=lambda action: huge))
    tracemalloc.start()
    try:
        with pytest.raises(PolicyError, match=r"shape \(1, 8388608\), expected \(k, 14\)"):
            policy.act(observation, dims)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 << 20


def test_rows_no_episode_can_execute_are_dropped_before_the_copy(tmp_path, errors):
    chunk = Script(reshape=lambda action: np.tile(action, (100_000, 1)))
    policy, observation, dims = acting(tmp_path, chunk)
    assert policy.act(observation, dims).shape == (remote.MAX_ACTION_ROWS, 14)


def test_no_task_runs_more_steps_than_an_action_chunk_keeps():
    limits = robotwin.ROBOTWIN_ROOT / "env_cfg" / "task_config" / "_eval_step_limit.yml"
    if not limits.is_file():
        pytest.skip(f"no RoboTwin checkout at {robotwin.ROBOTWIN_ROOT}")
    assert max(yaml.safe_load(limits.read_text()).values()) < remote.MAX_ACTION_ROWS


def test_a_message_the_client_refuses_to_send_voids_on_the_harness(tmp_path, errors):
    refused = errors.WireError("array 'qpos': dtype object cannot be sent")
    _, _, result = run(tmp_path, Script(fail={"act": refused}))
    assert result["void"] is True and result["void_cause"] == "harness"
    assert "could not send act" in result["error"]


def test_a_served_policy_is_not_reset_without_its_episodes_seed(errors):
    script = Script()
    policy = remote.RemotePolicy("/run/policy.sock", KEY, client_factory=script.factory)
    with pytest.raises(Unscorable, match="seed"):
        policy.reset(EpisodeInfo(embodiment="fake-arms", action_dims={"qpos": 14}, seed=None))
    assert script.clients == []  # nothing connected


def test_a_served_policy_needs_icil_policy_installed(monkeypatch):
    for name in ("icil_policy", "icil_policy.errors", "icil_policy.client"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(PolicyError, match="needs icil-policy installed"):
        remote.RemotePolicy("/run/policy.sock", KEY)


def test_the_benchmark_imports_without_icil_policy():
    # The adapter imports the client only when one is built, so the core never needs it.
    code = (
        "import sys; import robotwin_icil.cli, robotwin_icil.remote, robotwin_icil.unit; "
        "assert 'icil_policy' not in sys.modules, sorted(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_authkey_is_read_as_hex_from_the_environment_and_removed_from_it():
    environ = {"POLICY_KEY": KEY.hex(), "OTHER": "kept"}
    assert remote.authkey_from_env("POLICY_KEY", environ) == KEY
    assert environ == {"OTHER": "kept"}


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (None, "holds no key"),
        ("", "holds no key"),
        ("zz" * 16, "not hex"),
        ("ab" * 15, "the key is 15 bytes; at least 16"),
    ],
)
def test_a_missing_or_short_authkey_is_refused_without_quoting_it(value, reason):
    environ = {} if value is None else {"POLICY_KEY": value}
    with pytest.raises(PolicyError) as refused:
        remote.authkey_from_env("POLICY_KEY", environ)
    assert reason in str(refused.value)
    assert not value or value not in str(refused.value)
