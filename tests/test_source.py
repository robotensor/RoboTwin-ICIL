"""The benchmark's source digest: which code runs, however it was installed."""

import shutil
from pathlib import Path

import robotwin_icil

PACKAGE = Path(robotwin_icil.__file__).resolve().parent


def test_the_digest_is_the_packages_own_source_wherever_it_lies(tmp_path):
    copy = tmp_path / "elsewhere" / "robotwin_icil"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert robotwin_icil.source_digest(copy) == robotwin_icil.source_sha256()
    assert len(robotwin_icil.source_sha256()) == 64


def test_compiled_caches_and_other_files_are_not_source(tmp_path):
    copy = tmp_path / "robotwin_icil"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    before = robotwin_icil.source_digest(copy)
    (copy / "__pycache__").mkdir()
    (copy / "__pycache__" / "unit.cpython-310.py").write_text("stale")
    (copy / "notes.txt").write_text("not source")
    assert robotwin_icil.source_digest(copy) == before


def test_a_changed_renamed_or_added_file_changes_the_digest(tmp_path):
    copy = tmp_path / "robotwin_icil"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    before = robotwin_icil.source_digest(copy)

    unit = copy / "unit.py"
    unit.write_text(unit.read_text() + "\n# edited\n")
    edited = robotwin_icil.source_digest(copy)
    assert edited != before

    unit.rename(copy / "unit2.py")
    assert robotwin_icil.source_digest(copy) not in (before, edited)

    (copy / "unit2.py").rename(unit)
    (copy / "tasks_extra.yml").write_text("extra: {}\n")
    assert robotwin_icil.source_digest(copy) not in (before, edited)
