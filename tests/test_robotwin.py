"""The RoboTwin seam: error reporting and config resolution, exercised without importing RoboTwin."""

import types

import numpy as np
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
    from test_cameras import STATIC_CAMERA_LIST

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


def test_overrides_are_held_to_the_camera_guard(fake_checkout):
    from test_cameras import SIDE_CAMERA, STATIC_CAMERA_LIST

    toggled = robotwin.SceneConfig(overrides={"camera": {"collect_head_camera": False}})
    with pytest.raises(camera_profiles.ProfileError, match="SceneConfig.overrides toggles"):
        toggled.resolve("click_bell")
    added = {"static_camera_list": [*STATIC_CAMERA_LIST, SIDE_CAMERA]}
    for profile in ("stock", "far_side"):
        config = robotwin.SceneConfig(
            camera_profile=profile, overrides={"left_embodiment_config": added}
        )
        with pytest.raises(camera_profiles.ProfileError, match="creates from 2 to 3"):
            config.resolve("click_bell")


def test_an_unknown_camera_profile_fails_before_any_scene():
    with pytest.raises(camera_profiles.ProfileError, match="unknown camera profile"):
        robotwin.SceneConfig(camera_profile="nope")


@pytest.fixture
def fake_task(monkeypatch):
    from fake_robotwin import FakeTaskEnv, FakeUnstable

    envs = []

    def load_task(name):
        envs.append(FakeTaskEnv(unstable_seeds={7}, setup_raises_on={8}))
        return envs[-1]

    monkeypatch.setattr(robotwin, "load_task", load_task)
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "free_gpu", lambda: None)
    return envs


def test_a_snapshot_is_every_camera_of_one_scene(fake_task):
    from fake_robotwin import FakeConfig

    images = robotwin.snapshot("click_bell", 3, FakeConfig())
    assert set(images) == {"head_camera"} and images["head_camera"].shape == (16, 16, 3)
    assert fake_task[0].setups == [3] and fake_task[0].task_names == ["click_bell"]
    assert fake_task[0].closed == 1


def test_a_snapshot_of_a_scene_that_will_not_build_says_why(fake_task):
    from fake_robotwin import FakeConfig

    with pytest.raises(robotwin.RoboTwinError, match="seed 7 does not settle"):
        robotwin.snapshot("click_bell", 7, FakeConfig())
    with pytest.raises(robotwin.RoboTwinError, match="building click_bell seed 8 failed"):
        robotwin.snapshot("click_bell", 8, FakeConfig())
    assert [env.closed for env in fake_task] == [1, 1]


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


def test_an_observation_reads_where_the_fingers_are_not_the_command():
    from fake_robotwin import FINGER_OPEN, FINGER_STOP

    env = built()
    env.qpos[6], env.qpos[13] = 1.0, 0.0  # left commanded open, right closed on an object
    obs = robotwin.observation(env)
    assert obs["endpose"]["right_gripper"] == 0.0  # RoboTwin's value is the command
    np.testing.assert_allclose(obs["gripper_joints"]["left"], [FINGER_OPEN] * 2)
    np.testing.assert_allclose(obs["gripper_joints"]["right"], [FINGER_STOP] * 2)


def test_captured_frames_carry_endposes_and_gripper_joints():
    from fake_robotwin import endpose_of, finger

    env = built()
    with robotwin.capture(env, save_freq=1) as frames:
        env.play_once()
    demonstration = robotwin.demonstration_from(frames, frequency=250.0)
    last = demonstration.frames[-1]
    np.testing.assert_allclose(last.gripper_joints["left"], [finger(env.qpos[6])] * 2)
    np.testing.assert_allclose(demonstration.endposes()[-1, :7], endpose_of(env.qpos[:6]))
    assert demonstration.endposes()[-1, 15] == env.qpos[13]
