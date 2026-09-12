"""A duck-typed stand-in for a RoboTwin task env, to test the protocol without a simulator.

It implements exactly the surface the benchmark touches — `setup_demo`, `play_once`,
`check_success`, `take_action`, `get_obs`, `close_env`, and what `robotwin.fingerprint` reads — and,
like RoboTwin, builds its scene as a pure function of the seed: a cube placement and the joint
target the expert must reach are drawn from `np.random.default_rng(seed)`.

Physics advances only through `scene.step()`, as in RoboTwin: the scene settles for a few steps in
`setup_demo`, the expert records a frame every `save_freq` steps, and `take_action` runs
`physics_per_action` steps.

The robot is shaped like aloha-agilex: one articulation for both arms, each with six arm joints
and a finger joint plus its mimic. Its end-effector pose is a stand-in forward kinematics that
maps six arm joints to `[x, y, z, qw, qx, qy, qz]` and back exactly, so an `ee` action reaches
the joints whose endpose it names, as RoboTwin's round trip on aloha-agilex does.
"""

import math
from types import SimpleNamespace

import numpy as np

QPOS_DIM = 14
EE_DIM = 16
# RoboTwin settles a new scene for 2500 physics steps inside `setup_demo`; a few stand in for them.
SETTLE_STEPS = 10
# aloha-agilex's `gripper_scale`: finger positions, in metres, for gripper values 0 and 1.
FINGER_CLOSED, FINGER_OPEN = -0.01, 0.045
# The fingers close on an object this far apart: measured, they stop there whatever the command.
FINGER_STOP = 0.01


def endpose_of(arm_qpos):
    """The fake's forward kinematics: six arm joints -> a flange pose, inverted by `joints_of`."""
    vector = 0.5 * np.asarray(arm_qpos[3:6], dtype=float)  # |vector| < 1 for joints in [-1, 1]
    w = math.sqrt(max(0.0, 1.0 - float(vector @ vector)))
    return [*map(float, arm_qpos[:3]), w, *map(float, vector)]


def joints_of(endpose):
    endpose = np.asarray(endpose, dtype=float)
    return np.concatenate([endpose[:3], 2.0 * endpose[4:7]])


def finger(gripper_value):
    """Where a finger joint is for a commanded gripper value: an object stops it closing."""
    return max(FINGER_CLOSED + gripper_value * (FINGER_OPEN - FINGER_CLOSED), FINGER_STOP)


class FakeUnstable(Exception):
    """Stands in for RoboTwin's `UnStableError`."""


class OutOfMemoryError(RuntimeError):
    """Named like torch's: that is how the benchmark recognises it without importing torch."""


class FakeConfig:
    task_config = "fake"
    save_freq = 1
    head_camera = None
    overrides = None
    camera_profile = "stock"

    def resolve(self, task_name=None):
        return {"save_freq": self.save_freq, "task_name": task_name}


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


class _Scene:
    """What the benchmark reads off RoboTwin's scene.

    `step` is a method of the class, as it is on SAPIEN's Python `Scene` wrapper, so an instance
    attribute can shadow it and deleting that attribute brings it back.
    """

    def __init__(self, cube):
        self.cube = cube
        self.stepped = 0

    def get_all_actors(self):
        return [_Actor("table", _Pose([0.0, 0.0, 0.74])), _Actor("cube", _Pose(self.cube))]

    def get_all_articulations(self):
        return []

    def get_timestep(self):
        return 1 / 250

    def step(self):
        self.stepped += 1


class _Joint:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


class _Articulation:
    """aloha-agilex's robot: both arms in one articulation, `get_qpos()` per active joint."""

    def __init__(self, env):
        self._env = env
        self._joints = [_Joint(f"f{side}_joint{k}") for side in "lr" for k in range(1, 9)]

    def get_active_joints(self):
        return list(self._joints)

    def get_qpos(self):
        q = self._env.qpos
        return np.concatenate([q[:6], [finger(q[6])] * 2, q[7:13], [finger(q[13])] * 2]).astype(
            np.float32
        )


class _Robot:
    """What the benchmark reads off RoboTwin's `Robot`: joint state and the gripper joints."""

    def __init__(self, env):
        self._env = env
        self.left_entity = self.right_entity = _Articulation(env)
        joints = self.left_entity.get_active_joints()
        self.left_gripper = [(joints[6], 1.0, 0.0), (joints[7], 1.0, 0.0)]
        self.right_gripper = [(joints[14], 1.0, 0.0), (joints[15], 1.0, 0.0)]

    def get_left_arm_jointState(self):
        return list(self._env.qpos[:7])

    def get_right_arm_jointState(self):
        return list(self._env.qpos[7:])


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
        physics_per_action=1,
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
        self.physics_per_action = physics_per_action
        self.save_data = False
        self.save_freq = None
        self.builds: dict[int, int] = {}
        self.setups: list[int] = []
        self.task_names: list[str | None] = []
        self.closed = 0
        self.action_types: list[str] = []
        # Closes of a scene whose `step` was still shadowed: a clock left running.
        self.closed_while_clocked = 0

    def setup_demo(self, now_ep_num=0, seed=0, is_test=False, **kwargs):
        self.setups.append(seed)
        self.task_names.append(kwargs.get("task_name"))
        if seed in self.unstable_seeds:
            raise FakeUnstable(f"objects unstable in seed {seed}")
        if self.broken_setup or seed in self.setup_raises_on:
            raise RuntimeError(f"planner failed to construct for seed {seed}")
        self.builds[seed] = self.builds.get(seed, 0) + 1
        rng = np.random.default_rng(seed)
        self.target = rng.uniform(-1.0, 1.0, QPOS_DIM)
        cube = rng.uniform(-0.3, 0.3, 3)
        if self.drift and self.builds[seed] > 1:
            cube = cube + 0.01  # what an unseeded RNG in scene construction would do
        self.seed = seed
        self.qpos = np.zeros(QPOS_DIM)
        self.scene = _Scene(cube)
        for _ in range(SETTLE_STEPS):
            self.scene.step()
        self.cameras = SimpleNamespace(
            get_config=lambda: {"head_camera": {"extrinsic_cv": np.eye(4)[:3]}}
        )
        self.robot = _Robot(self)
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
            for _ in range(self.save_freq or 1):
                self.scene.step()
            self._take_picture()
        return {"info": {"{A}": "cube"}}

    def check_success(self):
        return bool(np.allclose(self.qpos, self.target, atol=1e-9))

    def get_obs(self):
        return {
            "observation": {"head_camera": {"rgb": np.zeros((16, 16, 3), dtype=np.uint8)}},
            "joint_action": {"vector": self.qpos.copy()},
            "endpose": {
                "left_endpose": endpose_of(self.qpos[:6]),
                "left_gripper": float(self.qpos[6]),
                "right_endpose": endpose_of(self.qpos[7:13]),
                "right_gripper": float(self.qpos[13]),
            },
        }

    def take_action(self, action, action_type="qpos"):
        if self.take_action_cnt == self.step_lim or self.eval_success:
            return
        if self.rollout_raises_at is not None and self.take_action_cnt == self.rollout_raises_at:
            raise RuntimeError(self.rollout_error)
        self.take_action_cnt += 1
        self.action_types.append(action_type)
        self.qpos = self._joint_target(action, action_type)
        for _ in range(self.physics_per_action):
            self.scene.step()
        if self.check_success():
            self.eval_success = True

    @staticmethod
    def _joint_target(action, action_type):
        """Where the joints go: a qpos action is the target, an `ee` action is inverted to one."""
        action = np.asarray(action, dtype=float)
        width = {"qpos": QPOS_DIM, "ee": EE_DIM}[action_type]
        if action.shape != (width,):
            raise ValueError(f"{action_type} action has shape {action.shape}, expected ({width},)")
        if action_type == "qpos":
            return action
        left, right = action[:8], action[8:]
        return np.concatenate([joints_of(left[:7]), left[7:], joints_of(right[:7]), right[7:]])

    def close_env(self, clear_cache=False):
        self.closed += 1
        scene = getattr(self, "scene", None)
        if scene is not None and "step" in vars(scene):
            self.closed_while_clocked += 1
