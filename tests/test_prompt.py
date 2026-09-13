"""prompt.npz round-trips a demonstration exactly, and never lets `meta` reach a policy."""

import dataclasses
import json

import numpy as np
import pytest

from fake_robotwin import FakeTaskEnv, FakeUnstable
from robotwin_icil import generate, prompt, robotwin
from robotwin_icil.demo import Demonstration, Frame

META = {"task": "click_bell", "scene_seed": 7, "secret": {"digest": "abc"}}


@pytest.fixture(autouse=True)
def _fake_unstable(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)


def captured(qpos_dim=14, save_freq=3) -> Demonstration:
    """A demonstration as the expert path really produces it: clocked, with endposes."""
    result, demonstration, _ = generate.attempt(
        FakeTaskEnv(qpos_dim=qpos_dim), 0, {"save_freq": save_freq}, save_freq, 0
    )
    assert result.rejection is None
    return demonstration


def test_the_channel_map_has_the_published_shape():
    assert prompt.CHANNELS == {
        "video": ("frames_",),
        "proprio": ("qpos", "endpose"),
        "actions": ("actions",),
    }
    assert prompt.PREFIX_CHANNELS == ("video",)
    assert prompt.METADATA == ("times", "frequency")
    assert prompt.PRIVILEGED == "meta"


def test_every_array_written_belongs_to_a_channel():
    arrays = prompt.arrays_from(captured())
    assert {name: prompt.channel_of(name) for name in arrays} == {
        "frames_head_camera": "video",
        "qpos": "proprio",
        "endpose": "proprio",
        "actions": "actions",
        "times": "metadata",
        "frequency": "metadata",
    }
    assert prompt.channel_of("meta") == "privileged"
    assert prompt.channel_of("scene_seed") is None


@pytest.mark.parametrize("qpos_dim", [14, 16])
def test_a_prompt_round_trips_every_array_and_dtype(tmp_path, qpos_dim):
    original = captured(qpos_dim=qpos_dim)
    path = tmp_path / prompt.PROMPT_FILE
    sha = prompt.write_prompt(path, original, META)
    assert sha == prompt.sha256_of(path) and len(sha) == 64

    arrays, meta = prompt.read_raw(path)
    assert meta == META and "meta" not in arrays
    written = prompt.arrays_from(original)
    assert set(arrays) == set(written)
    for name, array in written.items():
        np.testing.assert_array_equal(arrays[name], array)
        assert arrays[name].dtype == array.dtype, name
    assert arrays["frames_head_camera"].dtype == np.uint8
    assert arrays["qpos"].dtype == arrays["endpose"].dtype == arrays["times"].dtype == np.float64
    assert arrays["qpos"].shape == (len(original), qpos_dim)
    assert arrays["endpose"].shape == (len(original), 16)
    assert arrays["actions"].shape == (len(original) - 1, qpos_dim)
    assert arrays["frequency"].shape == () and float(arrays["frequency"]) == original.frequency

    restored, meta = prompt.read_prompt(path)
    assert meta == META
    assert restored.cameras == original.cameras and restored.frequency == original.frequency
    assert restored.qpos_dim == qpos_dim and len(restored) == len(original)
    np.testing.assert_array_equal(restored.qpos(), original.qpos())
    np.testing.assert_array_equal(restored.actions(), original.actions())
    np.testing.assert_array_equal(restored.times(), original.times())
    np.testing.assert_array_equal(restored.images("head_camera"), original.images("head_camera"))
    assert restored.images("head_camera").dtype == np.uint8
    assert [f.endpose for f in restored.frames] == [f.endpose for f in original.frames]
    assert [f.index for f in restored.frames] == [f.index for f in original.frames]
    assert restored.timed


def test_endposes_flatten_per_arm_as_take_action_reads_them():
    endpose = {
        "left_endpose": [1, 2, 3, 1, 0, 0, 0],
        "left_gripper": 0.5,
        "right_endpose": [4, 5, 6, 0, 1, 0, 0],
        "right_gripper": 1.0,
    }
    row = prompt.flatten_endpose(endpose)
    assert row.shape == (16,) and row.dtype == np.float64
    np.testing.assert_array_equal(row, [1, 2, 3, 1, 0, 0, 0, 0.5, 4, 5, 6, 0, 1, 0, 0, 1.0])
    assert prompt.unflatten_endpose(row) == {
        "left_endpose": [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        "left_gripper": 0.5,
        "right_endpose": [4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0],
        "right_gripper": 1.0,
    }


def test_a_demonstration_without_endposes_cannot_be_a_prompt(tmp_path):
    # RoboTwin records endposes only with data_type.endpose on; a prompt without them would hand
    # one policy less than another, so it is refused rather than padded.
    frames = tuple(
        Frame(index=i, images={"cam": np.zeros((2, 2, 3), np.uint8)}, qpos=np.ones(14), endpose={})
        for i in range(2)
    )
    with pytest.raises(prompt.PromptError, match="left_endpose"):
        prompt.write_prompt(tmp_path / "p.npz", Demonstration(frames=frames, frequency=1), {})


def test_meta_never_reaches_a_policy(tmp_path):
    path = tmp_path / prompt.PROMPT_FILE
    prompt.write_prompt(path, captured(), META)
    arrays, _ = prompt.read_raw(path)
    assert "meta" not in prompt.policy_arrays({**arrays, "meta": "{}"})

    demonstration, _ = prompt.read_prompt(path)
    # What a policy gets is the Demonstration alone; nothing on it or its frames names a seed,
    # a task or a digest.
    assert {f.name for f in dataclasses.fields(demonstration)} == {"frames", "frequency", "cameras"}
    assert {f.name for f in dataclasses.fields(demonstration.frames[0])} == {
        "index",
        "images",
        "qpos",
        "endpose",
        "time_s",
    }
    text = json.dumps(
        [
            {k: v for k, v in dataclasses.asdict(frame).items() if k != "images"}
            for frame in demonstration.frames
        ],
        default=lambda value: value.tolist(),
    )
    for privileged in ("click_bell", "scene_seed", "abc", "meta"):
        assert privileged not in text


def test_a_tampered_prompt_is_refused(tmp_path):
    path = tmp_path / prompt.PROMPT_FILE
    prompt.write_prompt(path, captured(), META)
    arrays, meta = prompt.read_raw(path)

    shifted = {**arrays, "actions": arrays["actions"] + 1e-3}
    with pytest.raises(prompt.PromptError, match="actions are not the next frame"):
        prompt.demonstration_from(shifted)
    with pytest.raises(prompt.PromptError, match="no 'qpos'"):
        prompt.demonstration_from({k: v for k, v in arrays.items() if k != "qpos"})
    with pytest.raises(prompt.PromptError, match="frames_<camera>"):
        prompt.demonstration_from({k: v for k, v in arrays.items() if not k.startswith("frames_")})
    with pytest.raises(prompt.PromptError, match="times has shape"):
        prompt.demonstration_from({**arrays, "times": arrays["times"][:-1]})

    np.savez(path, **arrays)  # no meta at all
    with pytest.raises(prompt.PromptError, match="no meta"):
        prompt.read_raw(path)
    with pytest.raises(prompt.PromptError, match="cannot read"):
        prompt.read_raw(tmp_path / "missing.npz")


def _written(tmp_path):
    path = tmp_path / prompt.PROMPT_FILE
    prompt.write_prompt(path, captured(), META)
    return path


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda data: b"", id="empty"),
        pytest.param(lambda data: data[: len(data) // 2], id="truncated"),
        pytest.param(lambda data: data[:200], id="header-only"),
        pytest.param(lambda data: b"not a zip at all" * 64, id="garbage"),
    ],
)
def test_a_corrupt_prompt_file_is_a_prompt_error(tmp_path, corrupt):
    # An interrupted copy or write leaves exactly these; numpy and zipfile raise a zoo of errors
    # for them, and every one must reach the caller as a prompt that cannot be read.
    path = _written(tmp_path)
    path.write_bytes(corrupt(path.read_bytes()))
    with pytest.raises(prompt.PromptError, match="cannot read"):
        prompt.read_prompt(path)


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (
            lambda a: {**a, "frames_head_camera": a["frames_head_camera"].astype(np.float32)},
            "frames are float32, expected uint8",
        ),
        (
            lambda a: {**a, "frames_head_camera": a["frames_head_camera"][..., :2]},
            "expected \\(7, h, w, 3\\)",
        ),
        (lambda a: {**a, "qpos": a["qpos"].astype(str)}, "qpos is <U"),
        (lambda a: {**a, "qpos": a["qpos"].astype(np.float32)}, "qpos is float32"),
        (lambda a: {**a, "endpose": a["endpose"].astype(np.int64)}, "endpose is int64"),
        (lambda a: {**a, "times": a["times"].astype(np.float32)}, "times is float32"),
        (lambda a: {**a, "frequency": np.asarray([20.0, 20.0])}, "frequency has shape \\(2,\\)"),
        (lambda a: {**a, "frequency": np.asarray("fast")}, "frequency is <U"),
        (lambda a: {**a, "frequency": np.asarray(np.inf)}, "frequency is inf"),
    ],
)
def test_arrays_outside_the_published_dtypes_are_refused(tmp_path, edit, reason):
    arrays, _ = prompt.read_raw(_written(tmp_path))
    assert len(arrays["qpos"]) == 7
    with pytest.raises(prompt.PromptError, match=reason):
        prompt.demonstration_from(edit(arrays))
