"""Nothing in the benchmark or its competition plugin reaches the network.

`CLAUDE.md` says results are plain files and not hosted artifacts. A competition built on this
benchmark does publish - but that is the competition's business, and the line is that neither the
core nor the plugin uploads, fetches or needs a network. This checks it rather than trusting it,
because it is the kind of rule that erodes one convenient import at a time.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: Reaching any of these means a module talks to something over a network.
NETWORK = {
    "urllib",
    "urllib3",
    "http",
    "requests",
    "httpx",
    "aiohttp",
    "huggingface_hub",
    "boto3",
    "botocore",
    "ftplib",
    "smtplib",
    "telnetlib",
    "xmlrpc",
}

#: `multiprocessing.connection` legitimately uses sockets: it is how a policy in another Python
#: environment is reached, over a Unix socket on a shared mount, with no network involved.
ALLOWED = {"socket", "multiprocessing"}


def _roots() -> list[Path]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return [parent / "src" / "robotwin_icil", parent / "competition" / "src"]
    return []


def _imports(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_neither_the_core_nor_the_plugin_imports_a_network_client():
    roots = [r for r in _roots() if r.exists()]
    if not roots:
        pytest.skip("not a source checkout")
    offenders: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            hits = _imports(ast.parse(path.read_text())) & NETWORK
            offenders.extend(f"{path.name}: {name}" for name in sorted(hits))
    assert not offenders, (
        f"a benchmark run needs no network; publishing is the competition's business: {offenders}"
    )


def test_the_allowance_is_narrow_and_deliberate():
    """`socket` and `multiprocessing` are permitted, and this says why, so a future reader does
    not widen the rule by accident."""
    assert not ALLOWED & NETWORK
