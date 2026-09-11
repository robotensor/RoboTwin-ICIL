"""The RoboTwin seam: error reporting and config resolution, exercised without importing RoboTwin."""

import types

import pytest
import yaml

from robotwin_icil import camera_profiles, robotwin


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


def test_gpu_exhaustion_is_recognised_without_torch():
    class OutOfMemoryError(RuntimeError):
        pass

    assert robotwin.gpu_exhausted(OutOfMemoryError("anything"))
    assert robotwin.gpu_exhausted(RuntimeError("CUDA out of memory. Tried to allocate 2 MiB"))
    assert not robotwin.gpu_exhausted(AssertionError("target_pose cannot be None for move action."))


@pytest.fixture
def fake_checkout(tmp_path, monkeypatch):
    """The handful of RoboTwin config files `SceneConfig.resolve` reads, in a scratch checkout."""
    from test_camera_profiles import STATIC_CAMERA_LIST

    config_dir = tmp_path / "env_cfg" / "task_config"
    config_dir.mkdir(parents=True)
    embodiment = tmp_path / "assets" / "embodiments" / "aloha-agilex"
    embodiment.mkdir(parents=True)
    files = {
        config_dir / "demo_clean.yml": {
            "embodiment": ["aloha-agilex"],
            "camera": {"head_camera_type": "D435", "collect_head_camera": True},
            "domain_randomization": {"random_head_camera_dis": 0},
        },
        config_dir / "_embodiment_config.yml": {
            "aloha-agilex": {"file_path": "./assets/embodiments/aloha-agilex/"}
        },
        config_dir / "_camera_config.yml": {
            name: {"fovy": 45, "w": 320, "h": 180}
            for name in ("L515", "Large_L515", "D435", "Large_D435")
        },
        embodiment / "config.yml": {"static_camera_list": STATIC_CAMERA_LIST},
    }
    for path, data in files.items():
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(robotwin, "ROBOTWIN_ROOT", tmp_path)
    monkeypatch.setattr(robotwin, "_ensure_importable", lambda: None)
    return tmp_path


def static_names(args, side="left"):
    return [c["name"] for c in args[f"{side}_embodiment_config"]["static_camera_list"]]


def test_resolve_applies_the_camera_profile(fake_checkout):
    assert static_names(robotwin.SceneConfig().resolve("click_bell")) == [
        "head_camera",
        "front_camera",
    ]
    args = robotwin.SceneConfig(camera_profile="far_side").resolve("click_bell")
    assert static_names(args) == ["head_camera", "far_side_camera"]
    assert static_names(args, "right") == ["head_camera", "front_camera"]


def test_resolve_checks_camera_types_against_robotwin(fake_checkout):
    config_file = fake_checkout / "env_cfg" / "task_config" / "_camera_config.yml"
    config_file.write_text(yaml.safe_dump({"D435": {}}), encoding="utf-8")
    assert robotwin.camera_types() == {"D435"}
    with pytest.raises(camera_profiles.ProfileError, match="camera type 'L515'"):
        robotwin.SceneConfig(camera_profile="far_side").resolve("click_bell")


def test_overrides_still_have_the_last_word(fake_checkout):
    config = robotwin.SceneConfig(camera_profile="far_side", overrides={"camera": {}})
    assert config.resolve("click_bell")["camera"] == {}


def test_an_unknown_camera_profile_fails_before_any_scene():
    with pytest.raises(camera_profiles.ProfileError, match="unknown camera profile"):
        robotwin.SceneConfig(camera_profile="nope")
