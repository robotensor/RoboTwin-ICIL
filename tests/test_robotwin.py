"""The RoboTwin seam's error reporting, exercised without importing RoboTwin."""

import types

import pytest

from robotwin_icil import robotwin


@pytest.fixture
def importing(monkeypatch):
    """Make `load_task`'s import raise, or return, whatever a test says."""
    monkeypatch.setattr(robotwin, "_ensure_importable", lambda: None)

    def use(outcome):
        def import_module(name):
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(robotwin.importlib, "import_module", import_module)

    return use


def test_a_missing_task_module_is_an_unknown_task(importing):
    importing(ModuleNotFoundError("No module named 'envs.nope'", name="envs.nope"))
    with pytest.raises(robotwin.RoboTwinError, match="has no task 'nope'"):
        robotwin.load_task("nope")


def test_a_missing_dependency_is_reported_as_itself(importing):
    # Upstream imports its motion planner at module level; a broken install must not read as a typo.
    importing(ModuleNotFoundError("No module named 'curobo'", name="curobo"))
    with pytest.raises(robotwin.RoboTwinError, match="failed: No module named 'curobo'"):
        robotwin.load_task("place_object_basket")


def test_a_failing_import_inside_a_dependency_is_reported_as_itself(importing):
    importing(ImportError("cannot import name 'CuroboPlanner' from 'envs.robot.planner'"))
    with pytest.raises(robotwin.RoboTwinError, match="CuroboPlanner.*see docs/install.md"):
        robotwin.load_task("place_object_basket")


def test_a_module_without_the_task_class_is_named(importing):
    importing(types.ModuleType("envs.place_object_basket"))
    with pytest.raises(robotwin.RoboTwinError, match="defines no class 'place_object_basket'"):
        robotwin.load_task("place_object_basket")


def test_the_task_class_is_instantiated(importing):
    class place_object_basket:  # noqa: N801 - RoboTwin names classes after tasks
        pass

    importing(types.SimpleNamespace(place_object_basket=place_object_basket))
    assert isinstance(robotwin.load_task("place_object_basket"), place_object_basket)
