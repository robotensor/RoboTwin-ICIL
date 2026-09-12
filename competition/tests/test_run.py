"""The run path, without the simulator: what a demonstration still carries when it is rebuilt.

`run_unit` itself builds a RoboTwin scene, so it is exercised against the real simulator
separately. What is checked here is everything on the way in, because that is where a duel
silently turns into `void` rather than a verdict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from icil_benchmark_robotwin import BENCHMARK
from icil_benchmark_robotwin.plugin import PROMPT_NAME
from icil_benchmark_robotwin.prompt import CHANNELS, channel_of, dump, load
from icil_benchmark_robotwin.run import _demonstration

from robotwin_icil.demo import ARMS, Demonstration, Frame

# ---------------------------------------------------------------- demonstrations and prompts


def demonstration(frames: int = 4, *, measured: bool = True) -> Demonstration:
    def frame(i: int) -> Frame:
        pose = np.array([0.2 + 0.01 * i, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0])
        return Frame(
            index=i,
            images={"head_camera": np.full((4, 4, 3), i, dtype=np.uint8)},
            qpos=np.full(14, float(i)),
            endpose={
                "left_endpose": pose,
                "left_gripper": 1.0,
                "right_endpose": pose + 0.5,
                "right_gripper": 0.0,
            },
            gripper_joints=(
                {"left": np.array([0.01 * i, 0.01 * i + 0.001]), "right": np.array([0.04, 0.041])}
                if measured
                else None
            ),
        )

    return Demonstration(frames=tuple(frame(i) for i in range(frames)), frequency=16.0)


def written(tmp_path: Path, **kwargs) -> dict:
    dump(demonstration(**kwargs), tmp_path / PROMPT_NAME, task="click_bell", scene_seed=7)
    return load(tmp_path / PROMPT_NAME)


def test_a_prompt_carries_the_measured_gripper_joints(tmp_path):
    """RoboTwin's gripper value in `qpos` and `endpose` is the *command*, which reads closed while
    the fingers rest on an object. Where the fingers actually are is proprioception, it is what a
    model trained on `gripper_states` reads, and only the prompt can carry it."""
    doc = written(tmp_path)
    source = demonstration()
    assert doc["gripper_joints"].shape == (len(source), len(ARMS), 2)
    rebuilt = _demonstration(doc)
    for frame, original in zip(rebuilt.frames, source.frames, strict=True):
        for arm in ARMS:
            assert frame.gripper_joints[arm] == pytest.approx(original.gripper_joints[arm])


def test_the_measurement_is_proprioception_so_a_view_can_drop_it_with_the_rest(tmp_path):
    """The orchestrator's view allows or drops whole channels; an array in none of them would
    never reach a policy at all."""
    assert channel_of("gripper_joints") == "proprio"
    assert "gripper_joints" in BENCHMARK.info()["demo_channels"]["proprio"]
    assert set(BENCHMARK.info()["demo_channels"]) == set(CHANNELS)


def test_a_prompt_written_before_the_measurement_still_rebuilds(tmp_path):
    """Older published prompts have no such array, and are read as the demonstrations they are:
    with no measurement, rather than with a fabricated one."""
    doc = written(tmp_path, measured=False)
    assert "gripper_joints" not in doc
    rebuilt = _demonstration(doc)
    assert all(frame.gripper_joints is None for frame in rebuilt.frames)


def test_a_rebuilt_demonstration_is_the_one_that_was_published(tmp_path):
    source = demonstration()
    rebuilt = _demonstration(written(tmp_path))
    assert len(rebuilt) == len(source) and rebuilt.cameras == source.cameras
    assert rebuilt.qpos() == pytest.approx(source.qpos())
    assert rebuilt.endposes() == pytest.approx(source.endposes())
    assert rebuilt.times() == pytest.approx(source.times())
    assert rebuilt.images("head_camera").tolist() == source.images("head_camera").tolist()


def test_a_prompt_with_the_measurement_still_verifies(tmp_path):
    written(tmp_path)
    verdict = BENCHMARK.verify_prompt(path=str(tmp_path), unit={"task": "click_bell"})
    assert verdict["ok"], verdict["problems"]
