import pytest

from robotwin_icil.robotwin import ROBOTWIN_ROOT


def pytest_collection_modifyitems(config, items):
    # A missing asset download should read as "not installed", not as a row of failures.
    if (ROBOTWIN_ROOT / "assets" / "objects").is_dir():
        return
    skip = pytest.mark.skip(reason="RoboTwin assets not installed; run scripts/install_robotwin.sh")
    for item in items:
        if "sim" in item.keywords:
            item.add_marker(skip)
