import dataclasses
import json
import multiprocessing
import multiprocessing.connection
import struct
import threading

import numpy as np
import pytest

from robotwin_icil import protocol
from robotwin_icil.demo import Demonstration, Frame
from robotwin_icil.policy import Observation
from robotwin_icil.protocol import ProtocolError


def roundtrip(op, **fields):
    """Send a message down a pipe and receive it at the other end, as a server and client do."""
    a, b = multiprocessing.Pipe()
    with a, b:
        # A sender of its own: a large message fills the pipe before the receiver reads it.
        sender = threading.Thread(target=protocol.send, args=(a, op), kwargs=fields)
        sender.start()
        message = protocol.receive(b, timeout=10)
        sender.join()
    return message


def frame(i: int) -> Frame:
    rng = np.random.default_rng(i)
    return Frame(
        index=i,
        images={
            "head_camera": rng.integers(0, 256, (6, 8, 3), dtype=np.uint8),
            "left_camera": rng.integers(0, 256, (4, 5, 3), dtype=np.uint8),
        },
        qpos=rng.normal(size=14),
        endpose={
            "left_endpose": [float(x) for x in rng.normal(size=7)],
            "left_gripper": float(rng.uniform()),
            "right_endpose": rng.normal(size=7).astype(np.float32),
            "right_gripper": np.float64(rng.uniform()),
        },
        time_s=0.004 * (i + 1),
        gripper_joints={"left": rng.uniform(-0.01, 0.045, 2), "right": rng.uniform(size=2)},
    )


def demonstration(frames: int = 3) -> Demonstration:
    return Demonstration(
        frames=tuple(frame(i) for i in range(frames)),
        frequency=250 / 15,
        cameras=("left_camera", "head_camera"),
    )


def observation() -> Observation:
    rng = np.random.default_rng(7)
    return Observation(
        step=5,
        images={"head_camera": rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)},
        qpos=rng.normal(size=14),
        endpose={"left_endpose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], "left_gripper": 0.5},
        instruction="Pick it up.",
        time_s=1.236,
        gripper_joints={"left": np.array([0.01, 0.01]), "right": np.array([0.045, 0.044])},
    )


def assert_bit_equal(sent, received, path="value"):
    """Equal in type, dtype, shape and every byte, however deep."""
    assert type(received) is type(sent), f"{path}: {type(received)} is not {type(sent)}"
    if isinstance(sent, (np.ndarray, np.generic)):
        assert received.dtype == sent.dtype and received.shape == sent.shape, path
        assert received.tobytes() == sent.tobytes(), path
    elif isinstance(sent, float):
        assert struct.pack("<d", received) == struct.pack("<d", sent), path
    elif isinstance(sent, dict):
        assert list(received) == list(sent), path
        for key in sent:
            assert_bit_equal(sent[key], received[key], f"{path}.{key}")
    elif isinstance(sent, (list, tuple)):
        assert len(received) == len(sent), path
        for i, (a, b) in enumerate(zip(sent, received, strict=True)):
            assert_bit_equal(a, b, f"{path}[{i}]")
    elif dataclasses.is_dataclass(sent):
        for f in dataclasses.fields(sent):
            assert_bit_equal(getattr(sent, f.name), getattr(received, f.name), f"{path}.{f.name}")
    else:
        assert received == sent, path


@pytest.mark.parametrize("value", [frame(0), demonstration(), observation()])
def test_the_samples_set_every_field(value):
    # A field added to Frame, Demonstration or Observation fails here until the samples set it,
    # and so is pinned to round-trip by the tests below.
    for f in dataclasses.fields(value):
        if f.default is not dataclasses.MISSING:
            assert getattr(value, f.name) != f.default, f.name
        if f.default_factory is not dataclasses.MISSING:
            assert getattr(value, f.name) != f.default_factory(), f.name


def test_a_demonstration_round_trips_bit_equal():
    demo = demonstration()
    assert_bit_equal(demo, roundtrip("demo", demonstration=demo).fields["demonstration"])


def test_a_demonstration_streamed_in_chunks_round_trips_bit_equal():
    demo = demonstration(frames=7)
    chunks = list(protocol.frame_chunks(demo.frames, max_bytes=500))
    assert len(chunks) > 2
    begin = roundtrip("demo_begin", demonstration=protocol.demonstration_fields(demo))
    frames = [
        f for chunk in chunks for f in roundtrip("demo_frames", frames=chunk).fields["frames"]
    ]
    rebuilt = Demonstration(frames=tuple(frames), **begin.fields["demonstration"])
    assert_bit_equal(demo, rebuilt)


def test_an_observation_round_trips_bit_equal():
    obs = observation()
    received = roundtrip("act", observation=obs).fields["observation"]
    assert_bit_equal(obs, received)
    received.images["head_camera"][0, 0, 0] += 1  # writable, as it would be in one process


def test_values_keep_their_type_and_every_bit():
    values = {
        "none": None,
        "flags": [True, False],
        "big": 10**30,
        "floats": [0.1, -0.0, 1e-300, float("nan"), float("inf"), -float("inf")],
        "text": 'naïve "quoted" ☃',
        "tuple": (1, (2.5, "x"), []),
        "keys": {0: "int", (1, 2): "tuple", "s": {"nested": np.arange(3)}},
        "scalars": [np.float32(0.25), np.int16(-3), np.bool_(True), np.uint64(2**63)],
        "empty": np.zeros((0, 3), dtype=np.float32),
        "zero_d": np.array(7, dtype=np.int8),
        "strided": np.arange(24, dtype=np.float64).reshape(4, 6)[::2, 1::2],
    }
    for name in ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32"):
        values[name] = np.ones((2, 3), dtype=name)
    for name in ("uint64", "float16", "float32", "float64"):
        values[name] = np.linspace(0, 1, 6, dtype=name).reshape(3, 2)
    received = roundtrip("values", **values).fields
    expected = {**values, "strided": np.ascontiguousarray(values["strided"])}
    assert_bit_equal(expected, received)


def test_a_big_endian_array_arrives_little_endian_with_its_values():
    sent = np.arange(5, dtype=">f8")
    received = roundtrip("values", array=sent).fields["array"]
    assert received.dtype == np.dtype("<f8")
    np.testing.assert_array_equal(received, sent)


@pytest.mark.parametrize(
    "value, message",
    [
        (np.array([1, "x"], dtype=object), "an object array cannot be sent"),
        ({"deep": [np.array([None])]}, r"deep\[0\]: an object array cannot be sent"),
        (np.array([1 + 2j]), "a complex128 array cannot be sent"),
        (np.array(["text"]), "array cannot be sent"),
        (np.array(["2024-01-01"], dtype="datetime64[D]"), "array cannot be sent"),
        (np.zeros(2, dtype=[("a", "<f4")]), "array cannot be sent"),
        (np.longdouble(1.0), "array cannot be sent"),
        ({1, 2}, "set values cannot be sent"),
        (object(), "object values cannot be sent"),
    ],
)
def test_what_the_protocol_does_not_carry_is_refused_before_anything_is_sent(value, message):
    with pytest.raises(ProtocolError, match=message):
        protocol.encode("act", value=value)


def test_a_dataclass_outside_the_protocol_is_refused():
    @dataclasses.dataclass
    class Handle:
        actor: str

    with pytest.raises(ProtocolError, match="Handle values cannot be sent"):
        protocol.encode("act", value=Handle("cube"))


def test_reserved_field_names_are_refused():
    with pytest.raises(ProtocolError, match="'arrays' is not a field name"):
        protocol.encode("act", arrays=[])


# A Frame's fields, with its qpos in array frame 0.
FRAME_FIELDS = {
    "index": 0,
    "images": {"$": "dict", "items": []},
    "qpos": {"$": "array", "index": 0},
    "endpose": {"$": "dict", "items": []},
}


def raw_message(header, *frames):
    """What a peer sends when it does not go through `protocol.send`."""
    a, b = multiprocessing.Pipe()
    a.send_bytes(json.dumps(header).encode() if not isinstance(header, bytes) else header)
    for raw in frames:
        a.send_bytes(raw)
    return a, b


@pytest.mark.parametrize(
    "header, frames, message",
    [
        (b"not json", (), "the header is not JSON"),
        ([1, 2], (), "the header is a list, not an object"),
        ({"arrays": []}, (), "needs a string 'op'"),
        (
            {
                "op": "act",
                "x": {"$": "array", "index": 0},
                "arrays": [{"name": "x", "dtype": "|O", "shape": [1]}],
            },
            (b"\0" * 8,),
            "dtype '|O' is not one the protocol carries",
        ),
        (
            {
                "op": "act",
                "x": {"$": "array", "index": 0},
                "arrays": [{"name": "x", "dtype": "<f8", "shape": [2]}],
            },
            (b"\0" * 8,),
            "8 bytes for shape \\(2,\\) of float64, expected 16",
        ),
        ({"op": "act", "x": {"$": "array", "index": 3}, "arrays": []}, (), "no array 3"),
        ({"op": "act", "x": {"$": "pickle"}, "arrays": []}, (), "unknown tag 'pickle'"),
        (
            {"op": "act", "x": {"$": "dataclass", "type": "Popen", "fields": {}}, "arrays": []},
            (),
            "no dataclass 'Popen' in the protocol",
        ),
        (
            {
                "op": "act",
                "x": {"$": "dataclass", "type": "Frame", "fields": {"seed": 3}},
                "arrays": [],
            },
            (),
            "cannot build a Frame",
        ),
        (
            {
                "op": "act",
                "x": {"$": "dataclass", "type": "Frame", "fields": FRAME_FIELDS},
                "arrays": [{"name": "q", "dtype": "<f8", "shape": [2, 7]}],
            },
            (b"\0" * 112,),
            r"cannot build a Frame: DemonstrationError: frame 0: qpos has shape \(2, 7\)",
        ),
        (
            {
                "op": "act",
                "x": {"$": "dataclass", "type": "Frame", "fields": {**FRAME_FIELDS, "qpos": [1]}},
                "arrays": [],
            },
            (),
            "cannot build a Frame: AttributeError",
        ),
        (
            {"op": "act", "x": {"$": "dict", "items": [[[1], 2]]}, "arrays": []},
            (),
            "a dict key must be hashable, not a list",
        ),
    ],
)
def test_a_malformed_message_is_refused(header, frames, message):
    a, b = raw_message(header, *frames)
    with a, b, pytest.raises(ProtocolError, match=message):
        protocol.receive(b, timeout=1)


def test_a_message_refused_for_an_array_is_read_whole():
    header = {
        "op": "act",
        "x": {"$": "array", "index": 0},
        "y": {"$": "array", "index": 1},
        "arrays": [
            {"name": "x", "dtype": "|O", "shape": [1]},
            {"name": "y", "dtype": "<f8", "shape": [1]},
        ],
    }
    a, b = raw_message(header, b"\0" * 8, b"\0" * 8)
    with a, b:
        protocol.send(a, "ping")
        with pytest.raises(ProtocolError, match=r"dtype '\|O'"):
            protocol.receive(b, timeout=1)
        assert protocol.receive(b, timeout=1) == protocol.Message("ping", {})


def test_receive_times_out_and_reports_eof():
    a, b = multiprocessing.Pipe()
    with b:
        with pytest.raises(TimeoutError):
            protocol.receive(b, timeout=0.05)
        a.close()
        with pytest.raises(EOFError):
            protocol.receive(b, timeout=1)


def test_nothing_is_pickled(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Connection.send and recv pickle; the protocol must not use them")

    monkeypatch.setattr(multiprocessing.connection.Connection, "send", refuse)
    monkeypatch.setattr(multiprocessing.connection.Connection, "recv", refuse)
    assert_bit_equal(
        observation(), roundtrip("act", observation=observation()).fields["observation"]
    )


def test_frame_chunks_keep_order_and_hold_at_least_one_frame():
    frames = demonstration(frames=5).frames
    size = protocol._nbytes(frames[0])
    assert [len(c) for c in protocol.frame_chunks(frames, max_bytes=2 * size)] == [2, 2, 1]
    assert [len(c) for c in protocol.frame_chunks(frames, max_bytes=1)] == [1] * 5
    assert list(protocol.frame_chunks(frames)) == [frames]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("127.0.0.1:5000", (("127.0.0.1", 5000), "AF_INET")),
        ("gpu-box:41000", (("gpu-box", 41000), "AF_INET")),
        ("/tmp/robotwin-icil/policy.sock", ("/tmp/robotwin-icil/policy.sock", "AF_UNIX")),
        ("policy.sock", ("policy.sock", "AF_UNIX")),
        ("./policy:1", ("./policy:1", "AF_UNIX")),
    ],
)
def test_an_address_is_host_and_port_or_a_socket_path(text, expected):
    assert protocol.parse_address(text) == expected


def test_an_empty_address_is_refused():
    with pytest.raises(ProtocolError):
        protocol.parse_address("")
