"""A materialized prompt and its unit, built without the simulator.

The prompt goes through the benchmark's own writers — `unit.build_meta` and `prompt.write_prompt`
— so what `verify_prompt` reads is laid out exactly as `robotwin-icil materialize` writes it.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from robotwin_icil import prompt, unit
from robotwin_icil.demo import Demonstration, Frame
from robotwin_icil.robotwin import EMBODIMENTS
from robotwin_icil.scene import SceneFingerprint

QPOS_DIMS = {"aloha-agilex": 14, "franka-panda": 16}
CANDIDATES = [5, 11, 17, 23]


def demonstration(qpos_dim, frames=4):
    endpose = {
        "left_endpose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        "left_gripper": 1.0,
        "right_endpose": [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0],
        "right_gripper": 0.0,
    }
    return Demonstration(
        frames=tuple(
            Frame(
                index=i,
                images={"head_camera": np.full((8, 10, 3), i, dtype=np.uint8)},
                qpos=np.full(qpos_dim, i / 10.0),
                endpose=endpose,
                time_s=i / 15.0,
            )
            for i in range(frames)
        ),
        frequency=250 / 15,
    )


@pytest.fixture
def make_prompt(tmp_path):
    """`make_prompt(task=, scene_seed=, embodiment=, qpos_dim=)`: the path of a prompt.npz."""

    def make(task="click_bell", scene_seed=11, embodiment="franka-panda", qpos_dim=None, name="p"):
        qpos_dim = qpos_dim or QPOS_DIMS[embodiment]
        demo = demonstration(qpos_dim)
        initial = SceneFingerprint(
            actors={"cube": np.array([0.1, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])},
            articulations={},
            articulation_roots={},
            cameras={"head_camera": np.eye(4)[:3]},
            robot_qpos=np.zeros(qpos_dim),
        )
        config = SimpleNamespace(
            embodiment=embodiment,
            task_config="demo_clean",
            save_freq=15,
            head_camera=None,
            overrides=None,
        )
        args = {"embodiment_name": embodiment, "embodiment": list(EMBODIMENTS[embodiment])}
        # As materialize records it: the unit's candidates tried in order up to the one kept, every
        # earlier one rejected. A seed that is no candidate keeps build_meta's one-seed record.
        expert = None
        if scene_seed in CANDIDATES:
            tried = CANDIDATES[: CANDIDATES.index(scene_seed) + 1]
            attempts = [{"seed": s, "rejection": "unstable", "detail": "moved"} for s in tried[:-1]]
            attempts.append({"seed": scene_seed, "rejection": None, "detail": ""})
            rejected = {"unstable": len(tried) - 1} if len(tried) > 1 else {}
            expert = {"scene_seeds": list(CANDIDATES), "attempts": attempts, "rejections": rejected}
        meta = unit.build_meta(task, scene_seed, config, args, demo, initial, expert)
        path = tmp_path / name / "prompt.npz"
        prompt.write_prompt(path, demo, meta)
        return path

    return make


@pytest.fixture
def franka_unit():
    """A unit of the Franka suite whose candidates hold the prompt `make_prompt` writes."""
    return {
        "task": "click_bell",
        "task_label": "Click bell",
        "suite": "franka_1arm",
        "category": "press_push",
        "instance_params": {
            "scene_seed": None,
            "scene_seeds": list(CANDIDATES),
            "embodiment": "franka-panda",
        },
    }
