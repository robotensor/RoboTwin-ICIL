"""The RoboTwin seam's config resolution and error reporting, exercised without importing RoboTwin."""

import sys
import types

import pytest
import yaml

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


def test_gpu_exhaustion_is_recognised_without_torch():
    class OutOfMemoryError(RuntimeError):
        pass

    assert robotwin.gpu_exhausted(OutOfMemoryError("anything"))
    assert robotwin.gpu_exhausted(RuntimeError("CUDA out of memory. Tried to allocate 2 MiB"))
    assert not robotwin.gpu_exhausted(AssertionError("target_pose cannot be None for move action."))


def test_render_manifests_left_in_the_env_are_used(tmp_path, monkeypatch):
    import os

    from robotwin_icil import robotwin

    share = tmp_path / "share" / "robotwin-icil"
    icd = share / "vulkan" / "icd.d" / "nvidia_icd.json"
    egl = share / "glvnd" / "egl_vendor.d" / "10_nvidia.json"
    for manifest in (icd, egl):
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(robotwin.sys, "prefix", str(tmp_path))
    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", "/explicit/10_nvidia.json")
    robotwin.use_env_render_manifests()
    assert os.environ["VK_ICD_FILENAMES"] == str(icd)
    assert os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] == "/explicit/10_nvidia.json"


def test_no_render_manifests_in_the_env_leaves_the_environment_alone(tmp_path, monkeypatch):
    import os

    from robotwin_icil import robotwin

    monkeypatch.setattr(robotwin.sys, "prefix", str(tmp_path))
    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)
    monkeypatch.delenv("__EGL_VENDOR_LIBRARY_FILENAMES", raising=False)
    robotwin.use_env_render_manifests()
    assert "VK_ICD_FILENAMES" not in os.environ
    assert "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ


def test_oidn_stays_below_compute_capability_10():
    assert robotwin.denoiser_for((8, 6), None) is None
    assert robotwin.denoiser_for(None, None) is None


def test_oidn_is_turned_off_on_blackwell():
    assert robotwin.denoiser_for((10, 0), None) == "none"
    assert robotwin.denoiser_for((12, 0), None) == "none"


def test_the_denoiser_override_wins_and_is_checked():
    assert robotwin.denoiser_for((12, 0), "oidn") is None
    assert robotwin.denoiser_for((8, 6), "none") == "none"
    with pytest.raises(robotwin.RoboTwinError, match="must be 'oidn' or 'none'"):
        robotwin.denoiser_for((12, 0), "optix")


def _fake_sapien(monkeypatch):
    calls = []
    render = types.ModuleType("sapien.render")
    render.set_ray_tracing_denoiser = calls.append
    sapien = types.ModuleType("sapien")
    sapien.render = render
    monkeypatch.setitem(sys.modules, "sapien", sapien)
    monkeypatch.setitem(sys.modules, "sapien.render", render)
    monkeypatch.delenv("ROBOTWIN_ICIL_DENOISER", raising=False)
    return render, calls


def test_robotwins_oidn_request_becomes_none_on_blackwell(monkeypatch):
    render, calls = _fake_sapien(monkeypatch)
    monkeypatch.setattr(robotwin, "_gpu_capability", lambda: (12, 0))
    robotwin.use_supported_denoiser()
    robotwin.use_supported_denoiser()  # once per process: no second wrapper
    render.set_ray_tracing_denoiser("oidn")
    render.set_ray_tracing_denoiser("optix")
    assert calls == ["none", "optix"]


def test_the_denoiser_is_left_alone_where_oidn_runs(monkeypatch):
    render, calls = _fake_sapien(monkeypatch)
    original = render.set_ray_tracing_denoiser
    monkeypatch.setattr(robotwin, "_gpu_capability", lambda: (8, 6))
    robotwin.use_supported_denoiser()
    assert render.set_ray_tracing_denoiser is original


def test_no_sapien_leaves_the_denoiser_alone(monkeypatch):
    monkeypatch.setitem(sys.modules, "sapien", None)
    robotwin.use_supported_denoiser()


def test_a_lost_gpu_device_is_recognised():
    assert robotwin.gpu_lost(RuntimeError("vk::Device::waitForFences: ErrorDeviceLost"))
    assert robotwin.gpu_lost(RuntimeError("VK_ERROR_DEVICE_LOST"))
    assert not robotwin.gpu_lost(RuntimeError("simulator exploded"))
    assert not robotwin.gpu_lost(AssertionError("target_pose cannot be None for move action."))


@pytest.fixture
def robotwin_root(tmp_path, monkeypatch):
    """A RoboTwin checkout in miniature: a task config, the embodiment index, and their assets."""
    config_dir = tmp_path / "env_cfg" / "task_config"
    config_dir.mkdir(parents=True)
    (config_dir / "demo_clean.yml").write_text(
        yaml.safe_dump({"embodiment": ["aloha-agilex"], "camera": {"head_camera_type": "D435"}})
    )
    (config_dir / "_embodiment_config.yml").write_text(
        yaml.safe_dump(
            {
                name: {"file_path": f"./assets/embodiments/{name}/"}
                for name in ("aloha-agilex", "franka-panda", "piper")
            }
        )
    )
    for name, dual_arm in (("aloha-agilex", True), ("franka-panda", False), ("piper", False)):
        asset = tmp_path / "assets" / "embodiments" / name
        asset.mkdir(parents=True)
        (asset / "config.yml").write_text(yaml.safe_dump({"dual_arm": dual_arm}))
    monkeypatch.setattr(robotwin, "ROBOTWIN_ROOT", tmp_path)
    monkeypatch.setattr(robotwin, "_ensure_importable", lambda: None)
    return tmp_path


def test_the_default_embodiment_is_the_task_configs_own(robotwin_root):
    args = robotwin.SceneConfig().resolve("click_bell")
    assert args["embodiment"] == ["aloha-agilex"] and args["embodiment_name"] == "aloha-agilex"
    assert args["dual_arm_embodied"] is True and "embodiment_dis" not in args
    assert (
        args["left_robot_file"] == args["right_robot_file"] == "./assets/embodiments/aloha-agilex/"
    )
    assert args["left_embodiment_config"] == {"dual_arm": True}


def test_franka_is_two_arms_the_documented_distance_apart(robotwin_root):
    # RoboTwin builds a single-arm robot twice, `[left, right, distance]`; its guide's dual-Franka
    # example puts them 0.8 m apart.
    args = robotwin.SceneConfig(embodiment="franka-panda").resolve("click_bell")
    assert args["embodiment"] == ["franka-panda", "franka-panda", 0.8]
    assert args["embodiment_dis"] == robotwin.FRANKA_ARM_DISTANCE_M == 0.8
    assert args["dual_arm_embodied"] is False and args["embodiment_name"] == "franka-panda"
    assert (
        args["left_robot_file"] == args["right_robot_file"] == "./assets/embodiments/franka-panda/"
    )
    assert args["right_embodiment_config"] == {"dual_arm": False}


def test_choosing_aloha_explicitly_is_the_default(robotwin_root):
    assert robotwin.SceneConfig(embodiment="aloha-agilex").resolve("click_bell") == (
        robotwin.SceneConfig().resolve("click_bell")
    )


def test_an_unknown_embodiment_is_refused_with_the_known_ones(robotwin_root):
    with pytest.raises(robotwin.RoboTwinError, match="unknown embodiment 'ur5-wsg'") as exc:
        robotwin.SceneConfig(embodiment="ur5-wsg").resolve("click_bell")
    assert "aloha-agilex, franka-panda" in str(exc.value) and "piper" in str(exc.value)


def test_a_task_config_naming_an_unshipped_robot_is_refused(robotwin_root):
    config = robotwin_root / "env_cfg" / "task_config" / "demo_clean.yml"
    config.write_text(yaml.safe_dump({"embodiment": ["piper", "ur5-wsg", 0.6]}))
    with pytest.raises(robotwin.RoboTwinError, match="'ur5-wsg' is not in _embodiment_config.yml"):
        robotwin.SceneConfig().resolve("click_bell")
    config.write_text(yaml.safe_dump({"embodiment": ["piper", "franka-panda"]}))
    with pytest.raises(robotwin.RoboTwinError, match="1 or 3 entries"):
        robotwin.SceneConfig().resolve("click_bell")


def test_embodiment_names_read_back_what_the_flag_takes():
    assert robotwin.embodiment_name(["aloha-agilex"]) == "aloha-agilex"
    assert robotwin.embodiment_name(["franka-panda", "franka-panda", 0.8]) == "franka-panda"
    assert robotwin.embodiment_name(["piper", "franka-panda", 0.6]) == "piper+franka-panda"
