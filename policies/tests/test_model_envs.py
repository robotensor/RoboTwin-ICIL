"""The model environments' lockfiles and their installer, `scripts/install_policy_env.sh`.

The installer itself is never run here: it downloads gigabytes. These check what can go wrong
without a download: a loose pin, a moving branch, a script that does not parse, and extras
that drift from the locks.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

POLICIES = Path(__file__).resolve().parents[1]
INSTALLER = POLICIES.parent / "scripts" / "install_policy_env.sh"
ENVS = sorted(p.name for p in (POLICIES / "envs").iterdir() if p.is_dir())
EXACT = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.\-]*)==([A-Za-z0-9_.+!\-]+)")
COMMIT = re.compile(r"[0-9a-f]{40}")

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _name(requirement: str) -> str:
    return re.sub(r"[-_.]+", "-", requirement).lower()


def _pins(path: Path) -> dict[str, str]:
    """Package -> version of a requirements file whose every line must be an exact pin."""
    pins = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = EXACT.fullmatch(line)
        assert match, f"{path.name}: {line!r} is not an exact pin"
        pins[_name(match[1])] = match[2]
    return pins


def _shell(name: str, expression: str) -> list[str]:
    """Words of `expression` after sourcing an environment's pins.env in bash."""
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'set -eu; source "$1"; printf "%s\\n" {expression}',
            "_",
            POLICIES / "envs" / name / "pins.env",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def test_there_is_one_lock_directory_per_model():
    assert ENVS == ["bpp", "uniskill"]


@pytest.mark.parametrize("name", ENVS)
def test_every_requirement_is_an_exact_pin(name):
    for file in ("requirements.lock", "constraints.txt"):
        assert _pins(POLICIES / "envs" / name / file)


@needs_bash
@pytest.mark.parametrize("name", ENVS)
def test_every_repository_is_pinned_to_a_full_commit(name):
    text = (POLICIES / "envs" / name / "pins.env").read_text()
    commits = re.findall(r"^\w+_COMMIT=(\S+)$", text, flags=re.MULTILINE)
    assert commits
    for commit in commits:
        assert COMMIT.fullmatch(commit), commit
    for ref in re.findall(r"git\+\S+?@([^\s\"')]+)", text):
        assert COMMIT.fullmatch(ref), f"a git requirement pinned to {ref!r}, not a commit"
    for torch_pin in _shell(name, '"${TORCH[@]}"'):
        assert EXACT.fullmatch(torch_pin), torch_pin


@needs_bash
def test_the_installer_parses_and_refuses_unknown_environments():
    subprocess.run(["bash", "-n", INSTALLER], check=True)
    for args in ([], ["nope"], ["bpp", "uniskill"]):
        result = subprocess.run(["bash", INSTALLER, *args], capture_output=True, text=True)
        assert result.returncode == 2 and "usage" in result.stderr, args


@needs_bash
@pytest.mark.skipif(sys.version_info < (3, 11), reason="reads pyproject.toml with tomllib")
@pytest.mark.parametrize("name", ENVS)
def test_the_extras_mirror_the_locks(name):
    import tomllib

    extras = tomllib.loads((POLICIES / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    locked = _pins(POLICIES / "envs" / name / "requirements.lock")
    locked |= _pins(POLICIES / "envs" / name / "constraints.txt")
    locked |= {_name(m[1]): m[2] for m in map(EXACT.fullmatch, _shell(name, '"${TORCH[@]}"'))}
    for requirement in extras[name]:
        match = EXACT.fullmatch(requirement)
        assert match, f"extra {name}: {requirement!r} is not an exact pin"
        assert locked.get(_name(match[1])) == match[2], (
            f"extra {name}: {requirement} not in the locks"
        )
