"""A duck-typed stand-in for a RoboTwin task env, to test the protocol without a simulator.

It implements exactly the surface the benchmark touches — `setup_demo`, `play_once`,
`check_success`, `take_action`, `get_obs`, `close_env`, and what `robotwin.fingerprint` and
`robotwin.robot_state` read — and counts the calls that would render (`get_obs`) or redraw the
lights (`_update_render`). Like RoboTwin, it builds its scene as a pure function of the seed: a
cube placement and the joint target the expert must reach are drawn from
`np.random.default_rng(seed)`.
"""

from types import SimpleNamespace

import numpy as np

# aloha-agilex's joint vector: six joints and a gripper per arm. A dual Franka is 16.
QPOS_DIM = 14


class FakeUnstable(Exception):
    """Stands in for RoboTwin's `UnStableError`."""


class OutOfMemoryError(RuntimeError):
    """Named like torch's: that is how the benchmark recognises it without importing torch."""


class FakeConfig:
    """Stands in for `robotwin.SceneConfig`; takes the same keywords the CLI passes."""

    head_camera = None
    overrides = None

    def __init__(self, task_config="fake", save_freq=1, embodiment="fake-arms"):
        self.task_config = task_config
        self.save_freq = save_freq
        self.embodiment = embodiment

    def resolve(self, task_name=None):
        # Like SceneConfig, no embodiment (the CLI without --embodiment) is the config's own robot.
        robot = self.embodiment or "fake-arms"
        return {
            "save_freq": self.save_freq,
            "task_name": task_name,
            "embodiment": [robot],
            "embodiment_name": robot,
        }


class _Pose:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)):
        self.p = np.asarray(p, dtype=float)
        self.q = np.asarray(q, dtype=float)


class _Actor:
    def __init__(self, name, pose):
        self._name, self._pose = name, pose

    def get_name(self):
        return self._name

    def get_pose(self):
        return self._pose


class FakeTaskEnv:
    def __init__(
        self,
        *,
        unstable_seeds=(),
        plan_fails_on=(),
        expert_misses_on=(),
        drift=False,
        rollout_raises_at=None,
        setup_raises_on=(),
        broken_setup=False,
        expert_raises_on=(),
        oom_on_play=(),
        device_lost_on_play=(),
        rollout_error="simulator exploded",
        step_lim=50,
        expert_steps=6,
        qpos_dim=QPOS_DIM,
        moves=("left", "right"),
    ):
        self.unstable_seeds = set(unstable_seeds)
        self.plan_fails_on = set(plan_fails_on)
        self.expert_misses_on = set(expert_misses_on)
        self.drift = drift
        self.rollout_raises_at = rollout_raises_at
        self.setup_raises_on = set(setup_raises_on)
        self.broken_setup = broken_setup
        self.expert_raises_on = set(expert_raises_on)
        self.oom_on_play = set(oom_on_play)
        self.device_lost_on_play = set(device_lost_on_play)
        self.rollout_error = rollout_error
        self.step_lim_setting = step_lim
        self.expert_steps = expert_steps
        self.qpos_dim = qpos_dim
        self.moves = set(moves)  # the arms the expert drives; the others stay at their start
        self.save_data = False
        self.save_freq = None
        self.get_obs_calls = 0  # each one would ray-trace every camera
        self.update_renders = 0
        self.builds: dict[int, int] = {}
        self.setups: list[int] = []
        self.task_names: list[str | None] = []
        self.closed = 0

    def setup_demo(self, now_ep_num=0, seed=0, is_test=False, **kwargs):
        self.setups.append(seed)
        self.task_names.append(kwargs.get("task_name"))
        if seed in self.unstable_seeds:
            raise FakeUnstable(f"objects unstable in seed {seed}")
        if self.broken_setup or seed in self.setup_raises_on:
            raise RuntimeError(f"planner failed to construct for seed {seed}")
        self.builds[seed] = self.builds.get(seed, 0) + 1
        rng = np.random.default_rng(seed)
        self.target = rng.uniform(-1.0, 1.0, self.qpos_dim)
        half = self.qpos_dim // 2
        for arm, joints in (("left", slice(0, half)), ("right", slice(half, None))):
            if arm not in self.moves:
                self.target[joints] = 0.0  # never leaves the starting qpos below
        cube = rng.uniform(-0.3, 0.3, 3)
        if self.drift and self.builds[seed] > 1:
            cube = cube + 0.01  # what an unseeded RNG in scene construction would do
        self.seed = seed
        # demo_clean's: rgb, qpos and endpose. `get_obs` fills endpose only when asked, as upstream.
        self.data_type = kwargs.get("data_type", {"rgb": True, "qpos": True, "endpose": True})
        self.qpos = np.zeros(self.qpos_dim)
        # Like RoboTwin, each arm reports its joints then its gripper; the vector is left + right.
        half = self.qpos_dim // 2
        self.scene = SimpleNamespace(
            get_all_actors=lambda: [
                _Actor("table", _Pose([0.0, 0.0, 0.74])),
                _Actor("cube", _Pose(cube)),
            ],
            get_all_articulations=lambda: [],
            get_timestep=lambda: 1 / 250,
        )
        self.cameras = SimpleNamespace(
            get_config=lambda: {"head_camera": {"extrinsic_cv": np.eye(4)[:3]}}
        )
        self.robot = SimpleNamespace(
            get_left_arm_jointState=lambda: list(self.qpos[:half]),
            get_right_arm_jointState=lambda: list(self.qpos[half:]),
            get_left_gripper_val=lambda: float(self.qpos[half - 1]),
            get_right_gripper_val=lambda: float(self.qpos[-1]),
            left_urdf_path="./assets/embodiments/fake/fake.urdf",
            right_urdf_path="./assets/embodiments/fake/fake.urdf",
            is_dual_arm=True,
        )
        self.info = {"texture_info": {"wall_texture": 0, "table_texture": 0}}
        self.crazy_random_light = False
        self.table_z_bias = 0.0
        self.plan_success = True
        self.eval_success = False
        self.take_action_cnt = 0
        self.step_lim = self.step_lim_setting
        self.save_freq = kwargs.get("save_freq")

    def _take_picture(self):
        raise AssertionError("the benchmark must intercept _take_picture, not let upstream pickle")

    def play_once(self):
        if self.seed in self.oom_on_play:
            raise OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB.")
        if self.seed in self.device_lost_on_play:
            raise RuntimeError("vk::Device::waitForFences: ErrorDeviceLost")
        if self.seed in self.expert_raises_on:
            raise AssertionError("target_pose cannot be None for move action.")
        if self.seed in self.plan_fails_on:
            self.plan_success = False
            return {}
        steps = self.expert_steps // 2 if self.seed in self.expert_misses_on else self.expert_steps
        self._take_picture()
        for k in range(1, steps + 1):
            self.qpos = self.target * k / self.expert_steps
            self._take_picture()
        return {"info": {"{A}": "cube"}}

    def check_success(self):
        return bool(np.allclose(self.qpos, self.target, atol=1e-9))

    def get_arm_pose(self, arm_tag):
        # A pose that follows the arm's first joints, so two frames' endposes differ as it moves.
        half = self.qpos_dim // 2
        joints = self.qpos[:half] if arm_tag == "left" else self.qpos[half:]
        return [float(value) for value in joints[:3]] + [1.0, 0.0, 0.0, 0.0]

    def _update_render(self):
        self.update_renders += 1

    def get_obs(self):
        # Upstream's order: sync the renderer, take every camera's picture, then read the robot.
        self.get_obs_calls += 1
        self._update_render()
        endpose = {}
        if self.data_type.get("endpose", False):
            endpose = {
                "left_endpose": self.get_arm_pose("left"),
                "left_gripper": self.robot.get_left_gripper_val(),
                "right_endpose": self.get_arm_pose("right"),
                "right_gripper": self.robot.get_right_gripper_val(),
            }
        return {
            "observation": {"head_camera": {"rgb": np.zeros((16, 16, 3), dtype=np.uint8)}},
            "joint_action": {"vector": self.qpos.copy()},
            "endpose": endpose,
        }

    def take_action(self, action, action_type="qpos"):
        if self.take_action_cnt == self.step_lim or self.eval_success:
            return
        if self.rollout_raises_at is not None and self.take_action_cnt == self.rollout_raises_at:
            raise RuntimeError(self.rollout_error)
        self.take_action_cnt += 1
        self.qpos = np.asarray(action, dtype=float)
        if self.check_success():
            self.eval_success = True

    def close_env(self, clear_cache=False):
        self.closed += 1
