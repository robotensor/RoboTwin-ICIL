"""Checks every adapter's tests share: its `ADAPTER_VERSION`, and conversion math pinned to it.

Each adapter module declares a module-level `ADAPTER_VERSION`, a dotted string of integers such
as "1" or "1.2". It names the adapter's conversion math: how a `Demonstration` and each
`Observation` become model inputs, and how model outputs become RoboTwin actions. Changing that
math bumps it, so two runs under one version converted alike (the run records it through
`describe()`, see #37).

`assert_conversion_pinned` makes a forgotten bump a test failure. An adapter's test runs its
conversion on fixed inputs, takes `conversion_digest` of the outputs, and keeps a table of the
digest each version produced; outputs that change under an unchanged version fail the test.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from types import ModuleType

import numpy as np

VERSION_PATTERN = re.compile(r"[0-9]+(\.[0-9]+)*")


def adapter_version(module: ModuleType) -> str:
    """The module's `ADAPTER_VERSION`, after checking it is a dotted string of integers."""
    version = getattr(module, "ADAPTER_VERSION", None)
    if version is None:
        raise AssertionError(f"{module.__name__} declares no ADAPTER_VERSION")
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise AssertionError(
            f"{module.__name__}.ADAPTER_VERSION is {version!r}; "
            "expected a dotted string of integers such as '1' or '1.2'"
        )
    return version


def conversion_digest(*arrays: np.ndarray, decimals: int = 6) -> str:
    """sha256 of arrays' shapes and values, floats rounded to `decimals` places.

    Rounding keeps the digest stable across platforms and numpy versions that differ in the
    last bits of a float; -0.0 and 0.0 digest alike.
    """
    digest = hashlib.sha256()
    for array in arrays:
        array = np.asarray(array)
        if array.dtype.kind == "f":
            array = np.round(array.astype("<f8"), decimals) + 0.0
        elif array.dtype.kind in "biu":
            array = array.astype("<i8")
        else:
            raise TypeError(f"cannot digest an array of dtype {array.dtype}")
        digest.update(f"{array.dtype.str}{array.shape}".encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def assert_conversion_pinned(module: ModuleType, digest: str, pins: Mapping[str, str]) -> None:
    """Fail unless `pins` maps the module's `ADAPTER_VERSION` to `digest`.

    `pins` is the adapter test's own table, version -> digest, one entry per version ever
    released; entries are added, never edited.
    """
    version = adapter_version(module)
    pinned = pins.get(version)
    if pinned is None:
        raise AssertionError(
            f"no digest pinned for {module.__name__} ADAPTER_VERSION {version!r}; "
            f"add {version!r}: {digest!r} to the pins"
        )
    if pinned != digest:
        raise AssertionError(
            f"{module.__name__}'s conversion outputs changed under ADAPTER_VERSION {version!r} "
            f"(pinned {pinned[:12]}, now {digest[:12]}): bump ADAPTER_VERSION and pin the new "
            "digest under the new version"
        )
