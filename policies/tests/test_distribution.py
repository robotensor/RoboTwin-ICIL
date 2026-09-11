"""The policies distribution stays outside the core, and its toolkit stays numpy-only."""

import ast
import sys
from pathlib import Path

import icil_policies.common

POLICIES = Path(__file__).resolve().parents[1]
CORE = POLICIES.parent / "src" / "robotwin_icil"
COMMON = POLICIES / "src" / "icil_policies" / "common"
# What icil_policies.common may import beyond the standard library.
COMMON_THIRD_PARTY = {"numpy", "robotwin_icil", "icil_policies"}


def _imported_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_core_never_imports_the_policies():
    for path in sorted(CORE.rglob("*.py")):
        assert "icil_policies" not in _imported_roots(path), path


def test_the_core_names_no_model():
    for path in sorted(CORE.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yml"}:
            text = path.read_text().lower()
            for name in ("bpp", "uniskill", "libero"):
                assert name not in text, f"{path} names {name}"


def test_common_imports_numpy_and_the_core_only():
    for path in sorted(COMMON.rglob("*.py")):
        extra = _imported_roots(path) - COMMON_THIRD_PARTY - set(sys.stdlib_module_names)
        assert not extra, f"{path.name} imports {sorted(extra)}"


def test_common_is_importable():
    assert icil_policies.common.__doc__
