"""A duck-typed stand-in for a RoboTwin task env, to test the protocol without a simulator.

It implements exactly the surface the benchmark touches — `setup_demo`, `play_once`,
`check_success`, `take_action`, `get_obs`, `close_env`, and what `robotwin.fingerprint` reads — and,
like RoboTwin, builds its scene as a pure function of the seed: a cube placement and the joint
target the expert must reach are drawn from `np.random.default_rng(seed)`.
"""

from types import SimpleNamespace

import numpy as np

QPOS_DIM = 14


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
        step_lim=50,
        expert_steps=6,
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
        self.step_lim_setting = step_lim
        self.expert_steps = expert_steps
        self.save_data = False
        self.save_freq = None
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
        self.target = rng.uniform(-1.0, 1.0, QPOS_DIM)
        cube = rng.uniform(-0.3, 0.3, 3)
        if self.drift and self.builds[seed] > 1:
            cube = cube + 0.01  # what an unseeded RNG in scene construction would do
        self.seed = seed
        self.qpos = np.zeros(QPOS_DIM)
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
            get_left_arm_jointState=lambda: list(self.qpos[:7]),
            get_right_arm_jointState=lambda: list(self.qpos[7:]),
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

    def get_obs(self):
        return {
            "observation": {"head_camera": {"rgb": np.zeros((16, 16, 3), dtype=np.uint8)}},
            "joint_action": {"vector": self.qpos.copy()},
            "endpose": {},
        }

    def take_action(self, action, action_type="qpos"):
        if self.take_action_cnt == self.step_lim or self.eval_success:
            return
        if self.rollout_raises_at is not None and self.take_action_cnt == self.rollout_raises_at:
            raise RuntimeError("simulator exploded")
        self.take_action_cnt += 1
        self.qpos = np.asarray(action, dtype=float)
        if self.check_success():
            self.eval_success = True

    def close_env(self, clear_cache=False):
        self.closed += 1
