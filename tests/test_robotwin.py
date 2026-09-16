"""The RoboTwin seam's config resolution and error reporting, exercised without importing RoboTwin."""

import subprocess
import sys
import types

import numpy as np
import pytest
import yaml

from fake_robotwin import FakeTaskEnv
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


@pytest.mark.parametrize(
    ("stdout", "raises", "held"),
    [
        ("1234, 5120\n4242, 23000\n", None, [(1234, 5120), (4242, 23000)]),
        ("", None, []),
        ("[N/A], [N/A]\n", None, None),
        ("", FileNotFoundError("nvidia-smi"), None),
        ("", subprocess.TimeoutExpired("nvidia-smi", 10), None),
    ],
)
def test_gpu_processes_are_who_nvidia_smi_says_holds_memory(monkeypatch, stdout, raises, held):
    def run(argv, **kwargs):
        assert argv[:2] == ["nvidia-smi", "--query-compute-apps=pid,used_memory"]
        assert kwargs["timeout"] <= 10  # a lost device must not hang the result being written
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(robotwin.subprocess, "run", run)
    expected = None if held is None else [{"pid": p, "used_mib": m} for p, m in held]
    assert robotwin.gpu_processes() == expected


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


def test_an_embodiment_override_is_refused_in_favour_of_the_field():
    # `overrides` are applied last, after the robot's URDFs, arm distance and name were derived
    # from `embodiment`; one that replaced the list would record a robot the scene never built.
    with pytest.raises(robotwin.RoboTwinError, match="SceneConfig.embodiment"):
        robotwin.SceneConfig(overrides={"embodiment": ["piper", "piper", 0.6]})
    assert robotwin.SceneConfig(overrides={"render_freq": 0}).overrides == {"render_freq": 0}


def test_embodiment_names_read_back_what_the_flag_takes():
    assert robotwin.embodiment_name(["aloha-agilex"]) == "aloha-agilex"
    assert robotwin.embodiment_name(["franka-panda", "franka-panda", 0.8]) == "franka-panda"
    assert robotwin.embodiment_name(["piper", "franka-panda", 0.6]) == "piper+franka-panda@0.6"


def test_two_arms_at_another_distance_are_another_robot_by_name(robotwin_root):
    # `--embodiment franka-panda` fixes the distance, so its name need not say it; a task config
    # that stands the same arms elsewhere is another scene, and every record must tell them apart.
    assert robotwin.embodiment_name(["franka-panda", "franka-panda", 0.6]) == "franka-panda@0.6"
    assert robotwin.embodiment_name(["piper", "piper", 0.8]) == "piper@0.8"
    config = robotwin_root / "env_cfg" / "task_config" / "demo_clean.yml"
    config.write_text(yaml.safe_dump({"embodiment": ["franka-panda", "franka-panda", 0.6]}))
    assert robotwin.SceneConfig().resolve("click_bell")["embodiment_name"] == "franka-panda@0.6"


def _captured(images, qpos_dim=14, crazy_random_light=False, data_type=None):
    """Run the fake expert on seed 3 inside `capture`, as `generate.attempt` does."""
    env = FakeTaskEnv(qpos_dim=qpos_dim)
    kwargs = {"save_freq": 1} if data_type is None else {"save_freq": 1, "data_type": data_type}
    env.setup_demo(seed=3, **kwargs)
    env.crazy_random_light = crazy_random_light
    with robotwin.capture(env, 1, images=images) as frames:
        env.play_once()
    return env, frames


@pytest.mark.parametrize("qpos_dim", [14, 16])
def test_a_capture_without_images_records_the_same_frames_and_never_calls_get_obs(qpos_dim):
    rendered_env, rendered = _captured(images=True, qpos_dim=qpos_dim)
    env, frames = _captured(images=False, qpos_dim=qpos_dim)

    assert env.get_obs_calls == 0 and env.update_renders == 0
    assert rendered_env.get_obs_calls == len(rendered) > 2  # the default still renders each frame
    assert [f.index for f in frames] == [f.index for f in rendered]
    for plain, full in zip(frames, rendered, strict=True):
        assert plain.images == {} and list(full.images) == ["head_camera"]
        np.testing.assert_array_equal(plain.qpos, full.qpos)
        assert plain.qpos.dtype == full.qpos.dtype == np.float64
        assert plain.endpose == full.endpose
    assert set(frames[-1].endpose) == {
        "left_endpose",
        "left_gripper",
        "right_endpose",
        "right_gripper",
    }
    assert frames[0].endpose != frames[-1].endpose  # read afresh at every frame


def test_a_capture_without_images_leaves_out_an_endpose_the_config_does_not_ask_for():
    # `get_obs` fills endpose only where `data_type` asks for it; the frame must not differ there.
    no_endpose = {"rgb": True, "qpos": True}
    _, rendered = _captured(images=True, data_type=no_endpose)
    _, frames = _captured(images=False, data_type=no_endpose)
    assert [f.endpose for f in frames] == [f.endpose for f in rendered] == [{}] * len(rendered)


def test_a_capture_without_images_redraws_crazy_lights_where_get_obs_would():
    # `_update_render` draws light colours from numpy's global RNG under `crazy_random_light`;
    # skipping it would shift every later draw the expert makes.
    rendered_env, rendered = _captured(images=True, crazy_random_light=True)
    env, frames = _captured(images=False, crazy_random_light=True)
    assert env.get_obs_calls == 0
    assert env.update_renders == rendered_env.update_renders == len(frames) == len(rendered)


def test_a_capture_puts_the_env_back_either_way():
    for images in (True, False):
        env = FakeTaskEnv()
        env.setup_demo(seed=3, save_freq=7)
        take_picture = env._take_picture
        with robotwin.capture(env, 1, images=images):
            assert env.save_data is True and env.save_freq == 1
        assert env._take_picture == take_picture
        assert (env.save_data, env.save_freq) == (False, 7)


def built(**kwargs):
    from fake_robotwin import FakeTaskEnv

    env = FakeTaskEnv(**kwargs)
    env.setup_demo(seed=0)
    return env


def test_the_clock_counts_physics_steps_from_where_it_is_installed():
    from fake_robotwin import SETTLE_STEPS

    env = built()
    assert env.scene.stepped == SETTLE_STEPS  # RoboTwin's settle, inside setup_demo
    with robotwin.clock(env) as ticks:
        assert ticks.steps == 0 and ticks.seconds == 0.0
        env.scene.step()
        env.scene.step()
    assert ticks.steps == 2 and ticks.seconds == pytest.approx(2 / 250)
    assert env.scene.stepped == SETTLE_STEPS + 2  # every counted step still ran


def test_the_clock_sees_steps_taken_inside_the_env():
    env = built(physics_per_action=3)
    with robotwin.clock(env) as ticks:
        env.take_action(env.target)
    assert ticks.steps == 3


def test_leaving_the_clock_restores_the_scenes_own_step():
    env = built()
    with robotwin.clock(env) as ticks:
        assert "step" in vars(env.scene)
    assert "step" not in vars(env.scene)
    env.scene.step()
    assert ticks.steps == 0


def test_the_clock_is_removed_when_the_block_raises():
    env = built()
    with pytest.raises(RuntimeError, match="expert exploded"):
        with robotwin.clock(env) as ticks:
            env.scene.step()
            raise RuntimeError("expert exploded")
    assert "step" not in vars(env.scene)
    env.scene.step()
    assert ticks.steps == 1


def test_nested_clocks_both_count_and_unwind_in_order():
    env = built()
    with robotwin.clock(env) as outer:
        env.scene.step()
        with robotwin.clock(env) as inner:
            env.scene.step()
        env.scene.step()
    assert (outer.steps, inner.steps) == (3, 1)
    assert "step" not in vars(env.scene)


def test_a_scene_that_cannot_be_shadowed_is_refused():
    class Frozen:
        __slots__ = ()

        def get_timestep(self):
            return 1 / 250

        def step(self):
            pass

    with pytest.raises(robotwin.RoboTwinError, match="cannot count the physics steps"):
        with robotwin.clock(types.SimpleNamespace(scene=Frozen())):
            pass


def _primitive(env, physics_steps, save_freq):
    """One motion primitive, recording as `Base_Task.take_dense_action` does: a frame before its
    first physics step, one after every `save_freq`-th step from the first, one after its last."""
    env._take_picture()
    for control_idx in range(physics_steps):
        env.scene.step()
        if control_idx % save_freq == 0:
            env._take_picture()
    env._take_picture()


def _recorded(primitives, clocked):
    env = built()
    if clocked:
        with robotwin.clock(env) as ticks, robotwin.capture(env, 2, ticks) as frames:
            for steps in primitives:
                _primitive(env, steps, 2)
    else:
        with robotwin.capture(env, 2) as frames:
            for steps in primitives:
                _primitive(env, steps, 2)
    return frames


@pytest.mark.parametrize(
    ("primitives", "physics_steps"),
    [((4, 4), [0, 1, 3, 4, 4, 5, 7, 8]), ((4, 5), [0, 1, 3, 4, 4, 5, 7, 9, 9])],
)
def test_the_clock_times_upstreams_frames_and_never_adds_one(primitives, physics_steps):
    # The clock only says when each frame was taken; how many there are is upstream's recording.
    # Two frames share a time where one primitive hands over to the next, and one physics step
    # more in a primitive can add a frame: why one scene's expert, whose CuRobo trajectories are
    # not the same length every run, can record 77 frames in one run and 78 in the next.
    timed = _recorded(primitives, clocked=True)
    assert len(_recorded(primitives, clocked=False)) == len(timed) == len(physics_steps)
    assert [frame.time_s for frame in timed] == pytest.approx([s / 250 for s in physics_steps])
