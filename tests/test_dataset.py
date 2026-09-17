import io
import json
import zipfile
from pathlib import Path

import h5py
import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image

from robotwin_icil import cli, dataset, dataset_release, tasks


@pytest.fixture
def trajectory(tmp_path):
    path = tmp_path / "source.hdf5"
    n = 3
    qpos = np.arange(n * 14, dtype=np.float32).reshape(n, 14) / 100
    actions = np.concatenate([qpos[1:], qpos[-1:]])
    with h5py.File(path, "w") as h5:
        for name, values in (("state", qpos), ("action", actions)):
            for key, start, stop in zip(
                dataset.JOINT_KEYS, (0, 6, 7, 13), (6, 7, 13, 14), strict=True
            ):
                h5.create_dataset(f"{name}/{key}", data=values[:, start:stop])
            for arm in ("left", "right"):
                poses = np.zeros((n, 7), dtype=np.float32)
                poses[:, 3] = 1
                h5.create_dataset(f"{name}/{arm}_ee_poses", data=poses)
        h5.create_dataset("additional_info/frequency", data=np.int32(15))
        for camera in dataset.CAMERAS:
            encoded = []
            for i in range(n):
                frame = np.full((240, 320, 3), [230 - i * 10, 10, 30], dtype=np.uint8)
                stream = io.BytesIO()
                Image.fromarray(frame).save(stream, format="JPEG")
                encoded.append(stream.getvalue())
            h5.create_dataset(f"vision/{camera}/colors", data=np.asarray(encoded, dtype="S"))
            h5.create_dataset(
                f"vision/{camera}/shape", data=np.array([240, 320, 3], dtype=np.int32)
            )
            h5.create_dataset(
                f"vision/{camera}/intrinsic_matrix",
                data=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
            )
            h5.create_dataset(
                f"vision/{camera}/extrinsics_matrix",
                data=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
            )
    return path


def test_loader_preserves_rgb_action_alignment_and_unknown_time(trajectory):
    loaded = dataset.load_trajectory(trajectory)
    assert loaded.qpos.shape == loaded.actions.shape == (3, 14)
    assert loaded.qpos.dtype == np.float32
    assert loaded.endpose.shape == (3, 16)
    assert loaded.simulated_frame_times is None
    assert loaded.source_frequency_value == 15
    assert set(loaded.rgb) == set(dataset.CAMERAS.values())
    assert np.array_equal(loaded.actions[:-1], loaded.qpos[1:])
    pixel = loaded.rgb["head_camera"][0, 0, 0]
    assert pixel[0] > 220 and pixel[2] < 40
    query = loaded.query(1)
    assert len(query["qpos"]) == 2
    assert all(len(frames) == 2 for frames in query["rgb"].values())
    assert "actions" not in query and "task" not in query and "scene_seed" not in query
    assert query["instruction"] == dataset.NEUTRAL_INSTRUCTION
    with pytest.raises(dataset.DatasetError):
        loaded.query(3)


def test_camera_selection_rejects_privileged_extra_view(trajectory):
    assert dataset.load_trajectory(trajectory, cameras=("head_camera",)).rgb.keys() == {
        "head_camera"
    }
    with pytest.raises(dataset.DatasetError):
        dataset.load_trajectory(trajectory, cameras=("cam_third_view",))


@pytest.mark.parametrize("change", ["shift", "nan", "camera"])
def test_invalid_alignment_nonfinite_and_camera_shape_fail(trajectory, change):
    with h5py.File(trajectory, "r+") as h5:
        if change == "shift":
            h5["action/left_arm_joint_states"][0] = 0
        elif change == "nan":
            h5["state/right_arm_joint_states"][0] = np.nan
        else:
            h5["vision/cam_head/shape"][0] = 480
    with pytest.raises(dataset.DatasetError):
        dataset.load_trajectory(trajectory)


def test_video_encodes_exact_source_rows_in_rgb(trajectory, tmp_path):
    path = tmp_path / "head.mp4"
    with h5py.File(trajectory) as h5:
        dataset_release.encode_camera(h5, "cam_head", path)
    assert imageio_ffmpeg.count_frames_and_secs(str(path))[0] == 3
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        first = np.frombuffer(next(reader), np.uint8).reshape(240, 320, 3)
        assert metadata["size"] == (320, 240) and metadata["fps"] == 30
        assert first[0, 0, 0] > 220 and first[0, 0, 2] < 40
    finally:
        reader.close()


def release_index(root, trajectory):
    rows = [
        {"task": task, "episode_index": i, "hdf5": trajectory.name}
        for task in tasks.table().tasks
        for i in range(50)
    ]
    (root / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_released_dataset_is_every_episode_with_causal_training_inputs(trajectory, tmp_path):
    release_index(tmp_path, trajectory)
    train = dataset.ReleasedDataset(tmp_path)
    assert len(train) == 2500
    same_scene = train.training_sample(0, 1)
    assert len(same_scene["inputs"]["demonstration"]["qpos"]) == 3
    assert len(same_scene["inputs"]["observation"]["qpos"]) == 2
    assert "actions" not in same_scene["inputs"]["observation"]
    with pytest.raises(dataset.DatasetError):
        train.training_sample(2500, 1)
    assert train.episodes[0]["task"] == train.episodes[1]["task"]
    sample = train.training_pair(0, 1, 1)
    assert sample["target_actions"].shape == (1, 14)
    assert len(sample["inputs"]["observation"]["qpos"]) == 2
    assert "actions" not in sample["inputs"]["observation"]
    assert "task" not in sample["inputs"]["demonstration"]
    with pytest.raises(dataset.DatasetError):
        train.training_pair(0, 0, 0)
    with pytest.raises(dataset.DatasetError):
        train.training_pair(0, 50, 0)


def test_release_rejects_missing_duplicate_and_escaping_episodes(trajectory, tmp_path):
    release_index(tmp_path, trajectory)
    index_path = tmp_path / "episodes.jsonl"
    lines = index_path.read_text().splitlines()
    index_path.write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(dataset.DatasetError, match="unique"):
        dataset.ReleasedDataset(tmp_path)
    index_path.write_text("\n".join([lines[0], *lines]) + "\n")
    with pytest.raises(dataset.DatasetError, match="unique"):
        dataset.ReleasedDataset(tmp_path)
    shifted = [json.loads(line) for line in lines]
    shifted[0]["episode_index"] = 50
    index_path.write_text("".join(json.dumps(row) + "\n" for row in shifted))
    with pytest.raises(dataset.DatasetError, match="0-49"):
        dataset.ReleasedDataset(tmp_path)
    release_index(tmp_path, trajectory)
    ds = dataset.ReleasedDataset(tmp_path)
    ds.episodes[0]["hdf5"] = "../outside.hdf5"
    with pytest.raises(dataset.DatasetError, match="escapes"):
        ds.path(0)


def test_conversion_preserves_hdf5_and_resumes_without_reencoding(
    trajectory, tmp_path, monkeypatch
):
    archive_path = tmp_path / "task.zip"
    prefix = "click_bell/aloha_agilex/"
    raw = trajectory.read_bytes()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for i in range(50):
            archive.writestr(prefix + f"data/episode_{i:07d}.hdf5", raw)
            archive.writestr(prefix + f"instruction/episode_{i:07d}.json", "{}")
        archive.writestr(prefix + "seed.txt", "\n".join(map(str, range(50))))
        archive.writestr(prefix + "scene_info.json", "{}")
    calls = []

    def encode(_h5, camera, path):
        calls.append(camera)
        path.write_bytes(b"test video")

    monkeypatch.setattr(dataset_release, "encode_camera", encode)
    out = tmp_path / "release"
    converted = dataset_release.convert_task("click_bell", archive_path, out)
    assert len(converted["episodes"]) == 50 and len(calls) == 150
    assert (out / converted["episodes"][0]["hdf5"]).read_bytes() == raw
    assert dataset_release.convert_task("click_bell", archive_path, out) == converted
    assert len(calls) == 150
    (out / converted["episodes"][0]["videos"]["head_camera"]).write_bytes(b"corrupt")
    dataset_release.convert_task("click_bell", archive_path, out)
    assert len(calls) == 300


def test_dataset_cli_requires_pinned_download_revision():
    parser = cli.build_parser()
    args = parser.parse_args(["dataset", "download", "--out", "/tmp/data", "--revision", "a" * 40])
    assert args.workers == 4
    with pytest.raises(dataset.DatasetError, match="commit SHA"):
        dataset_release.download_release(Path("/tmp/data"), "main")
    with pytest.raises(SystemExit):
        parser.parse_args(["dataset", "sample", "/tmp/data", "--split", "train"])
