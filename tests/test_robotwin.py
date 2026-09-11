"""The RoboTwin seam's error reporting, exercised without importing RoboTwin."""

import sys
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
