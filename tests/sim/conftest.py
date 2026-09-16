import pytest

from robotwin_icil.robotwin import ROBOTWIN_ROOT


def pytest_collection_modifyitems(config, items):
    # A missing asset download should read as "not installed", not as forty failures.
    if (ROBOTWIN_ROOT / "assets" / "objects").is_dir():
        return
    skip = pytest.mark.skip(reason="RoboTwin assets not installed; run scripts/install_robotwin.sh")
    for item in items:
        # By the mark, not `item.keywords`: those hold this folder's name, `sim`, for every test
        # in it, and would skip a plain test here that needs no simulator.
        if item.get_closest_marker("sim") is not None:
            item.add_marker(skip)
