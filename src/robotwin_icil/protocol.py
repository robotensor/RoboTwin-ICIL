"""The wire protocol between the benchmark and a policy served in its own Python environment.

Model stacks and RoboTwin pin libraries that conflict, so a model runs behind
`python -m robotwin_icil.serve` in an environment of its own and the simulator reaches it through
`RemotePolicy`. Both ends speak this module, which needs only the standard library and numpy.

Transport is `multiprocessing.connection`, whose frames carry their length and have no size cap,
over a Unix socket or TCP, authenticated with a shared key. Only `send_bytes` and `recv_bytes`
are used: `Connection.send` and `recv` pickle, and unpickling runs code. A message is one JSON
header frame, `{"op": ..., <fields>, "arrays": [{"name", "dtype", "shape"}, ...]}`, then one raw
frame per array, in that order. Field values are JSON, with tagged objects for what JSON lacks:
`{"$": "array" | "scalar" | "float", "index": i}` for a numpy array, a numpy scalar or a
non-finite float (its bytes are array frame `i`), `{"$": "tuple" | "dict", "items": ...}`, and
`{"$": "dataclass", "type": ..., "fields": ...}` for the benchmark's own `Demonstration`, `Frame`
and `Observation`. Their fields are walked with `dataclasses.fields`, so a field added to them
travels without a change here. Arrays are bool, integer or float; any other dtype, object above
all, is refused at both ends.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .demo import Demonstration, Frame
from .policy import Observation

# Raised whenever a message changes shape: the client and the server must speak the same one.
PROTOCOL_VERSION = 1

# The demonstration is streamed in chunks of about this many bytes of arrays, at least one frame
# each, so no single message holds a whole 300 MB demonstration.
DEMO_CHUNK_BYTES = 32 << 20

# Every dtype an array may have on the wire, little-endian. Anything else is refused.
DTYPES = frozenset(
    np.dtype(name).newbyteorder("<").str
    for name in (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    )
)

# The dataclasses a message may carry, by name.
DATACLASSES: dict[str, type] = {cls.__name__: cls for cls in (Demonstration, Frame, Observation)}


class ProtocolError(RuntimeError):
    """A value that cannot be sent, or a message that arrived malformed."""


@dataclass(frozen=True)
class Message:
    """One decoded message: its operation and its fields, arrays and dataclasses rebuilt."""

    op: str
    fields: dict[str, Any]


@dataclass(frozen=True)
class Encoded:
    """A message ready to send, so that encoding cannot fail halfway through sending it."""

    header: bytes
    arrays: tuple[np.ndarray, ...]


def encode(op: str, **fields: Any) -> Encoded:
    """Encode a message, or raise `ProtocolError` naming the value that cannot be sent."""
    arrays: list[np.ndarray] = []
    specs: list[dict[str, Any]] = []
    header: dict[str, Any] = {"op": op}
    for key, value in fields.items():
        if key in ("op", "arrays"):
            raise ProtocolError(f"{key!r} is not a field name a message may use")
        header[key] = _encode(value, arrays, specs, key)
    header["arrays"] = specs
    text = json.dumps(header, separators=(",", ":"), allow_nan=False)
    return Encoded(text.encode("utf-8"), tuple(arrays))


def send(conn, op: str, **fields: Any) -> None:
    """Encode and send one message on a `multiprocessing.connection.Connection`."""
    send_encoded(conn, encode(op, **fields))


def send_encoded(conn, message: Encoded) -> None:
    conn.send_bytes(message.header)
    for array in message.arrays:
        conn.send_bytes(array.tobytes())


def receive(conn, timeout: float | None = None) -> Message:
    """The next message; `TimeoutError` if it has not arrived within `timeout` seconds.

    EOF is the connection's own `EOFError`; a malformed message is a `ProtocolError`.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    raw = _receive_frame(conn, deadline, timeout)
    try:
        header = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise ProtocolError(f"the header is not JSON: {exc}") from None
    if not isinstance(header, dict):
        raise ProtocolError(f"the header is a {type(header).__name__}, not an object")
    op = header.pop("op", None)
    specs = header.pop("arrays", None)
    if not isinstance(op, str) or not isinstance(specs, list):
        raise ProtocolError("the header needs a string 'op' and a list 'arrays'")
    # Every frame first: a message refused for one of its arrays is still read whole, so the
    # next message starts where it should.
    raws = [_receive_frame(conn, deadline, timeout) for _ in specs]
    arrays = [_array(spec, raw) for spec, raw in zip(specs, raws, strict=True)]
    return Message(op, {key: _decode(value, arrays, key) for key, value in header.items()})


def parse_address(text: str) -> tuple[str | tuple[str, int], str]:
    """`host:port` as a TCP address, anything else as a Unix socket's path; with its family.

    A socket path with a colon in it needs a slash to read as one: `./policy:1`.
    """
    host, sep, port = text.rpartition(":")
    if sep and host and port.isdigit() and "/" not in text:
        return (host, int(port)), "AF_INET"
    if not text:
        raise ProtocolError("an address is a Unix socket's path or host:port")
    return text, "AF_UNIX"


def demonstration_fields(demonstration: Demonstration) -> dict[str, Any]:
    """Every field of a demonstration but its frames, which are streamed after it."""
    return {
        f.name: getattr(demonstration, f.name)
        for f in dataclasses.fields(demonstration)
        if f.name != "frames"
    }


def frame_chunks(
    frames: Sequence[Frame], max_bytes: int = DEMO_CHUNK_BYTES
) -> Iterator[tuple[Frame, ...]]:
    """The frames in order, in chunks holding at most `max_bytes` of arrays or a single frame."""
    chunk: list[Frame] = []
    size = 0
    for frame in frames:
        nbytes = _nbytes(frame)
        if chunk and size + nbytes > max_bytes:
            yield tuple(chunk)
            chunk, size = [], 0
        chunk.append(frame)
        size += nbytes
    if chunk:
        yield tuple(chunk)


def _receive_frame(conn, deadline: float | None, timeout: float | None) -> bytes:
    # `poll` with no time left still reports what has already arrived.
    if deadline is not None and not conn.poll(max(0.0, deadline - time.monotonic())):
        raise TimeoutError(f"nothing arrived within {timeout} s")
    return conn.recv_bytes()


def _encode(value: Any, arrays: list, specs: list, path: str) -> Any:
    if isinstance(value, np.ndarray):
        return {"$": "array", "index": _add(value, arrays, specs, path)}
    if isinstance(value, np.generic) and not isinstance(value, str):
        return {"$": "scalar", "index": _add(np.asarray(value), arrays, specs, path)}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # JSON has no NaN or infinity; a 0-d array carries them bit for bit.
        if math.isfinite(value):
            return value
        return {"$": "float", "index": _add(np.asarray(value), arrays, specs, path)}
    if isinstance(value, list):
        return [_encode(item, arrays, specs, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, tuple):
        items = [_encode(item, arrays, specs, f"{path}[{i}]") for i, item in enumerate(value)]
        return {"$": "tuple", "items": items}
    if isinstance(value, dict):
        items = [
            [
                _encode(key, arrays, specs, f"{path}.<key>"),
                _encode(item, arrays, specs, _at(path, key)),
            ]
            for key, item in value.items()
        ]
        return {"$": "dict", "items": items}
    if type(value) in DATACLASSES.values():
        fields = {
            f.name: _encode(getattr(value, f.name), arrays, specs, f"{path}.{f.name}")
            for f in dataclasses.fields(value)
        }
        return {"$": "dataclass", "type": type(value).__name__, "fields": fields}
    raise ProtocolError(f"{path}: {type(value).__name__} values cannot be sent")


def _at(path: str, key: Any) -> str:
    return f"{path}.{key}" if isinstance(key, str) else f"{path}[{key!r}]"


def _add(array: np.ndarray, arrays: list, specs: list, path: str) -> int:
    dtype = array.dtype
    if dtype.hasobject:
        raise ProtocolError(f"{path}: an object array cannot be sent")
    little = dtype.newbyteorder("<") if dtype.byteorder != "|" else dtype
    if little.str not in DTYPES:
        raise ProtocolError(
            f"{path}: a {dtype} array cannot be sent; only bool, integer and float arrays can"
        )
    # C order, as `tobytes()` writes it; unlike `ascontiguousarray`, keeps a 0-d array 0-d.
    array = np.asarray(array, dtype=little, order="C")
    arrays.append(array)
    specs.append({"name": path, "dtype": little.str, "shape": list(array.shape)})
    return len(arrays) - 1


def _array(spec: Any, raw: bytes) -> np.ndarray:
    if not isinstance(spec, dict):
        raise ProtocolError(f"an array spec is a {type(spec).__name__}, not an object")
    name, dtype, shape = spec.get("name"), spec.get("dtype"), spec.get("shape")
    if dtype not in DTYPES:
        raise ProtocolError(f"array {name!r}: dtype {dtype!r} is not one the protocol carries")
    if not (
        isinstance(shape, list)
        and all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in shape)
    ):
        raise ProtocolError(f"array {name!r}: shape {shape!r} is not a list of sizes")
    dtype = np.dtype(dtype)
    expected = math.prod(shape) * dtype.itemsize
    if len(raw) != expected:
        raise ProtocolError(
            f"array {name!r}: {len(raw)} bytes for shape {tuple(shape)} of {dtype}, "
            f"expected {expected}"
        )
    # A copy, so the receiver holds a writable array, as it would in one process.
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _decode(value: Any, arrays: list[np.ndarray], path: str) -> Any:
    if isinstance(value, list):
        return [_decode(item, arrays, f"{path}[{i}]") for i, item in enumerate(value)]
    if not isinstance(value, dict):
        return value
    tag = value.get("$")
    if tag in ("array", "scalar", "float"):
        index = value.get("index")
        if not (isinstance(index, int) and 0 <= index < len(arrays)):
            raise ProtocolError(f"{path}: no array {index!r} in this message")
        array = arrays[index]
        if tag == "array":
            return array
        if array.ndim != 0:
            raise ProtocolError(f"{path}: a {tag} needs a 0-d array, not shape {array.shape}")
        return float(array[()]) if tag == "float" else array[()]
    if tag == "tuple":
        items = _items(value, path)
        return tuple(_decode(item, arrays, f"{path}[{i}]") for i, item in enumerate(items))
    if tag == "dict":
        decoded = {}
        for pair in _items(value, path):
            if not (isinstance(pair, list) and len(pair) == 2):
                raise ProtocolError(f"{path}: a dict item is a [key, value] pair")
            key = _decode(pair[0], arrays, f"{path}.<key>")
            item = _decode(pair[1], arrays, _at(path, key))
            try:
                decoded[key] = item
            except TypeError:
                raise ProtocolError(
                    f"{path}: a dict key must be hashable, not a {type(key).__name__}"
                ) from None
        return decoded
    if tag == "dataclass":
        cls = DATACLASSES.get(value.get("type"))
        fields = value.get("fields")
        if cls is None or not isinstance(fields, dict):
            raise ProtocolError(f"{path}: no dataclass {value.get('type')!r} in the protocol")
        kwargs = {name: _decode(item, arrays, f"{path}.{name}") for name, item in fields.items()}
        try:
            return cls(**kwargs)
        except Exception as exc:
            # Wrong fields, or fields its own checks refuse: the sender's fault either way.
            raise ProtocolError(
                f"{path}: cannot build a {cls.__name__}: {type(exc).__name__}: {exc}"
            ) from None
    raise ProtocolError(f"{path}: unknown tag {tag!r}")


def _items(value: dict, path: str) -> list:
    items = value.get("items")
    if not isinstance(items, list):
        raise ProtocolError(f"{path}: {value.get('$')} items must be a list")
    return items


def _nbytes(value: Any) -> int:
    """Bytes of every array a value holds, however deep."""
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(item) for item in value)
    if type(value) in DATACLASSES.values():
        return sum(_nbytes(getattr(value, f.name)) for f in dataclasses.fields(value))
    return 0
