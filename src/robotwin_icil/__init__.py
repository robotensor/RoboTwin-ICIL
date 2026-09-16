"""One-demonstration in-context imitation learning benchmark on RoboTwin 2.0."""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path

__all__ = ["__version__", "source_digest", "source_sha256"]

__version__ = "0.1.0"

#: The files a source digest covers: every module and data file (`tasks.yml`) of the package.
SOURCE_SUFFIXES = (".py", ".yml")


def source_digest(root: str | Path) -> str:
    """sha256 over the source files under `root`: each one's path relative to `root` and the
    sha256 of its bytes, in path order. Compiled caches are not source."""
    root = Path(root)
    files = sorted(
        (path.relative_to(root).as_posix(), path)
        for path in root.rglob("*")
        if path.suffix in SOURCE_SUFFIXES and path.is_file() and "__pycache__" not in path.parts
    )
    digest = hashlib.sha256()
    for name, path in files:
        digest.update(name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@cache
def source_sha256() -> str:
    """The digest of this package's own source: which benchmark code is running, however it was
    installed. A git commit names none for a wheel and hides an uncommitted edit; this does
    neither. The competition plugin passes its own to `materialize` and `run-unit`
    (`--expect-source-sha256`), which refuse to run under any other, and both record theirs."""
    return source_digest(Path(__file__).resolve().parent)
