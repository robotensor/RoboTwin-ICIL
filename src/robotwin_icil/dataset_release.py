"""Copy pinned clean trajectories and encode aligned RGB videos, without simulation."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import (
    CAMERAS,
    EPISODES_PER_TASK,
    TASKS,
    DatasetError,
    ReleasedDataset,
    decode_rgb,
    dependencies,
    inspect,
)
from .records import write_json

FORMAT = "robotwin-icil-hdf5-multiview-v1"
VIDEO_FPS = 30
DATASET_REPO = "robotensor/robotwin-icil-aloha-clean"
SOURCE_REPO = "TianxingChen/RoboTwin2.0"
SOURCE_REVISION = "981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042"
EPISODES = TASKS * EPISODES_PER_TASK
# The Hub keeps only what the release inventories, plus the `.gitattributes` it writes itself.
HUB_EXTRAS = frozenset({".gitattributes"})


def check_revision(revision: str) -> None:
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise DatasetError("--revision must be a full lowercase Hugging Face commit SHA")


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def schema() -> dict[str, Any]:
    return {
        "format": FORMAT,
        "embodiment": "aloha-agilex",
        "source_rgb_encoding": "standard RGB JPEG at pinned source revision",
        "cameras": CAMERAS,
        "rgb_shape": [240, 320, 3],
        "rgb_dtype": "uint8",
        "qpos_shape": ["N", 14],
        "qpos_dtype": "float32",
        "qpos_order": ["left_joints[6]", "left_gripper", "right_joints[6]", "right_gripper"],
        "qpos_semantics": "joint drive targets / gripper commands, not measured encoder positions",
        "joint_units": "radians",
        "gripper_units": "normalized opening",
        "action_semantics": "absolute next-sampled joint/gripper targets; already shifted in source",
        "endpose_shape": ["N", 16],
        "endpose_order": [
            "left_xyz",
            "left_quaternion_wxyz",
            "left_gripper",
            "right_xyz",
            "right_quaternion_wxyz",
            "right_gripper",
        ],
        "pose_units": "metres and quaternion wxyz in upstream world coordinates",
        "rows": "N image/state/action transitions; terminal wrist frame unavailable",
        "video": {
            "codec": "H.264",
            "pixel_format": "yuv420p",
            "fps": VIDEO_FPS,
            "fps_semantics": "presentation only, not physical capture rate",
            "alignment": "exactly N frames per camera, frame i corresponds to HDF5 row i",
            "lossy": True,
            "training_rgb_authority": "original HDF5 JPEG bytes",
        },
        "simulated_frame_times": None,
        "timing_source": "unavailable in archived source",
        "source_frequency_semantics": "upstream stores save_freq, not verified physical Hz",
        "policy_language": "Follow the demonstrated behavior.",
        "privileged_metadata": ["task", "scene_seed", "scene_info", "instructions", "calibration"],
        "evaluation_prompts": "generated at runtime by materialize; archives are not prompt.npz",
    }


def encode_camera(h5, camera: str, path: Path) -> None:
    import imageio_ffmpeg

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.mp4")
    writer = imageio_ffmpeg.write_frames(
        str(temporary),
        (320, 240),
        fps=VIDEO_FPS,
        codec="libx264",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        quality=None,
        output_params=["-crf", "18", "-preset", "fast", "-threads", "1", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for value in h5[f"vision/{camera}/colors"]:
            frame = decode_rgb(value)
            if frame.shape != (240, 320, 3):
                raise DatasetError(f"{camera}: JPEG dimensions disagree with schema")
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()
    count, _ = imageio_ffmpeg.count_frames_and_secs(str(temporary))
    if count != len(h5[f"vision/{camera}/colors"]):
        raise DatasetError(f"{path}: encoded frame count mismatch")
    temporary.replace(path)


def convert_task(task: str, archive_path: Path, out: Path) -> dict:
    h5py, _ = dependencies()
    done = out / ".build" / f"{task}.json"
    archive_sha = file_digest(archive_path)
    if done.exists():
        cached = json.loads(done.read_text())
        if cached["archive_sha256"] == archive_sha and all(
            (out / name).is_file() and file_digest(out / name) == sha
            for name, sha in cached["files"].items()
        ):
            return cached
    prefix = f"{task}/aloha_agilex/"
    rows, files = [], {}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        hdf5s = sorted(n for n in names if n.startswith(prefix + "data/") and n.endswith(".hdf5"))
        expected = [prefix + f"data/episode_{i:07d}.hdf5" for i in range(50)]
        if hdf5s != expected:
            raise DatasetError(f"{task}: source must contain episodes 0000000 through 0000049")
        seeds = [int(s) for s in archive.read(prefix + "seed.txt").split()]
        if len(seeds) != 50:
            raise DatasetError(f"{task}: expected 50 source scene seeds")
        for source_name in hdf5s:
            index = int(Path(source_name).stem.removeprefix("episode_"))
            relative = Path("episodes") / task / f"episode_{index:07d}"
            directory = out / relative
            directory.mkdir(parents=True, exist_ok=True)
            hdf5 = directory / "trajectory.hdf5"
            temporary = hdf5.with_suffix(".partial")
            with archive.open(source_name) as source, temporary.open("wb") as dest:
                shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
            temporary.replace(hdf5)
            videos = {}
            with h5py.File(hdf5, "r") as h5:
                info = inspect(h5)
                for camera, public in CAMERAS.items():
                    video = directory / f"{public}.mp4"
                    encode_camera(h5, camera, video)
                    videos[public] = video.relative_to(out).as_posix()
            instruction = prefix + f"instruction/episode_{index:07d}.json"
            instruction_path = directory / "instruction.json"
            instruction_path.write_bytes(archive.read(instruction))
            row = {
                "task": task,
                "episode_index": index,
                "scene_seed": seeds[index],
                "embodiment": "aloha-agilex",
                "hdf5": hdf5.relative_to(out).as_posix(),
                "videos": videos,
                "instruction": instruction_path.relative_to(out).as_posix(),
                **info,
                "video_fps": VIDEO_FPS,
                "simulated_frame_times": None,
                "source_archive": f"dataset/{task}/demo_clean.zip",
                "source_hdf5": source_name,
            }
            rows.append(row)
            for path in (
                hdf5,
                instruction_path,
                *(directory / f"{c}.mp4" for c in CAMERAS.values()),
            ):
                files[path.relative_to(out).as_posix()] = file_digest(path)
        for name in ("scene_info.json", "seed.txt"):
            target = out / "source_metadata" / task / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(prefix + name))
            files[target.relative_to(out).as_posix()] = file_digest(target)
    result = {
        "task": task,
        "archive_sha256": archive_sha,
        "archive_bytes": archive_path.stat().st_size,
        "episodes": rows,
        "files": files,
    }
    done.parent.mkdir(parents=True, exist_ok=True)
    write_json(done, result)
    return result


def card() -> str:
    return f"""---
license: mit
pretty_name: RoboTwin ICIL Aloha Clean
task_categories:
- robotics
tags:
- robotics
- imitation-learning
- in-context-learning
- robotwin
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: episodes.jsonl
---
# RoboTwin ICIL Aloha Clean

Training data for [RoboTwin-ICIL](https://github.com/robotensor/RoboTwin-ICIL), a
one-demonstration in-context imitation learning benchmark, in {FORMAT} format.

## Contents

50 tasks x 50 trajectories = 2,500 aloha-agilex trajectories, all of them training data. Each
episode has an HDF5 trajectory, three aligned 320x240 H.264 videos (head, left wrist, right wrist)
and an instruction JSON. The HDF5 holds joint commands, actions, end-effector poses, camera
calibration and a third-view camera. The three public RGB streams are the policy camera contract.
The dataset viewer rows are `episodes.jsonl`, one per trajectory, pointing at its HDF5 and videos.

There is no validation or test split: policies are scored in simulation, not on stored
trajectories. Evaluation scenes use scene seeds from 100000 upwards, which none of these
trajectories use.

## Representation

See `schema.json`, `manifest.json`, `episodes.jsonl` and `checksums.json`.
Joint state/action is float32 width 14: left six joints, left gripper, right six joints,
right gripper. State is drive targets/gripper commands, not measured encoder positions.
Actions are absolute next-row targets, already shifted. EE pose is width 16, with xyz,
quaternion wxyz and gripper for each arm. Decode the HDF5 JPEGs as standard RGB, without channel
reversal.

Videos have exactly N frames aligned to the N HDF5 transition rows. They are lossy convenience
views; HDF5 JPEGs remain authoritative training RGB. Their 30 fps is presentation timing only.
Simulation timestamps are unavailable, and the stored frequency value is not verified physical Hz.
No terminal wrist frame or initial scene fingerprint exists.

## Evaluation

`robotwin-icil standard-eval` scores a frozen policy on all 50 tasks, 100 episodes each, in a clean
and a randomized setting. Each episode generates an expert demonstration in simulation, rebuilds
the identical scene and gives the policy that demonstration as context. Stored trajectories are
not usable as evaluation `prompt.npz` files. Task/seed/instruction/calibration/scene metadata must
not reach the policy. Use the neutral instruction: "Follow the demonstrated behavior."

## License

MIT; see `LICENSE`, which carries the copyright notice of RoboTwin 2.0, the simulation platform
these trajectories come from. `provenance.json` records their origin. Simulator and object assets
are not included.
"""


def write_metadata(out: Path, episodes: list[dict], provenance: dict) -> dict:
    """Write every root metadata file and the checksum inventory for converted `episodes`."""
    import imageio_ffmpeg

    # Conversions cached by an earlier build carry a `split` field this release no longer has.
    episodes = sorted(
        ({k: v for k, v in row.items() if k != "split"} for row in episodes),
        key=lambda row: (row["task"], row["episode_index"]),
    )
    index = out / "episodes.jsonl"
    with index.with_suffix(".tmp").open("w") as file:
        for row in episodes:
            file.write(json.dumps(row, sort_keys=True) + "\n")
    index.with_suffix(".tmp").replace(index)
    write_json(out / "schema.json", schema())
    write_json(out / "provenance.json", provenance)
    (out / "README.md").write_text(card())
    manifest = {
        "format": FORMAT,
        "dataset_repo": DATASET_REPO,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "tasks": len({row["task"] for row in episodes}),
        "episodes": len(episodes),
        "videos": len(episodes) * 3,
        "rows": sum(r["rows"] for r in episodes),
        "ffmpeg_version": imageio_ffmpeg.get_ffmpeg_version(),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def checksum_inventory(out: Path, files: dict[str, str]) -> None:
    files = dict(files)
    for path in out.iterdir():
        if path.is_file() and path.name != "checksums.json":
            files[path.name] = file_digest(path)
    write_json(out / "checksums.json", dict(sorted(files.items())))


def _task_names() -> list[str]:
    from .tasks import table

    return sorted(table().tasks)


def build_release(out: Path, cache: Path, workers: int = 4) -> dict:
    if workers < 1:
        raise DatasetError("--workers must be positive")
    out, cache = out.resolve(), cache.resolve()
    if out == cache or cache.is_relative_to(out) or out.is_relative_to(cache):
        raise DatasetError("source cache and release output must be separate directories")
    from huggingface_hub import snapshot_download

    snapshot_download(
        SOURCE_REPO,
        repo_type="dataset",
        revision=SOURCE_REVISION,
        allow_patterns=["dataset/*/demo_clean.zip", "README.md", "LICENSE*"],
        local_dir=str(cache),
        max_workers=workers,
    )
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("dataset-license.txt"), out / "LICENSE")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(convert_task, task, cache / f"dataset/{task}/demo_clean.zip", out): task
            for task in _task_names()
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"converted {len(results)}/50: {result['task']} (50 episodes, 150 videos)",
                flush=True,
            )
    results.sort(key=lambda r: r["task"])
    provenance = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "archives": [
            {k: r[k] for k in ("task", "archive_sha256", "archive_bytes")} for r in results
        ],
        "hdf5_conversion": "none; original bytes preserved",
        "license_source": "RoboTwin upstream LICENSE at simulator revision 96c1feab536306b50c26af200044fcdf126e8904",
        "video_conversion": "source JPEG RGB frames; libx264 CRF 18 fast yuv420p 30 presentation fps",
        "source_dataset_card": (cache / "README.md").read_text(),
    }
    episodes = [row for result in results for row in result["episodes"]]
    manifest = write_metadata(out, episodes, provenance)
    checksum_inventory(out, {name: sha for r in results for name, sha in r["files"].items()})
    return manifest


def verify_release(root: Path, videos: bool = False) -> dict:
    root = root.resolve()
    dataset = ReleasedDataset(root)
    release = json.loads((root / "manifest.json").read_text())
    if (
        release.get("format") != FORMAT
        or json.loads((root / "schema.json").read_text()) != schema()
    ):
        raise DatasetError("unexpected dataset format/schema")
    if (
        release.get("tasks") != TASKS
        or release.get("episodes") != EPISODES
        or release.get("videos") != 3 * EPISODES
    ):
        raise DatasetError("release counts do not match 50 tasks x 50 episodes")
    checksums = json.loads((root / "checksums.json").read_text())
    for name, expected in checksums.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_digest(path) != expected:
            raise DatasetError(f"checksum failed: {name}")
    h5py, _ = dependencies()
    rows = 0
    for row in dataset.episodes:
        hdf5 = root / row["hdf5"]
        names = [row["hdf5"], row["instruction"], *row["videos"].values()]
        if any(name not in checksums for name in names) or set(row["videos"]) != set(
            CAMERAS.values()
        ):
            raise DatasetError("episode files missing from checksum inventory")
        with h5py.File(hdf5, "r") as h5:
            if inspect(h5) != {k: row[k] for k in ("rows", "source_frequency_value")}:
                raise DatasetError(f"{hdf5}: index disagrees with HDF5")
        rows += row["rows"]
        if videos:
            import imageio_ffmpeg

            for video in row["videos"].values():
                count, _ = imageio_ffmpeg.count_frames_and_secs(str(root / video))
                if count != row["rows"]:
                    raise DatasetError(f"{video}: frame count differs from HDF5 rows")
    if rows != release["rows"]:
        raise DatasetError("row count disagrees with manifest")
    return {**release, "verified_files": len(checksums), "verified_video_frames": videos}


def upload_release(root: Path, workers: int = 4) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    verified = verify_release(root)
    api = HfApi()
    api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        folder_path=str(root),
        num_workers=workers,
        allow_patterns=[
            "*.json",
            "*.jsonl",
            "README.md",
            "LICENSE",
            "episodes/**",
            "source_metadata/**",
        ],
        ignore_patterns=[".build/**", ".cache/**", "*.partial*", "*.tmp"],
    )
    expected = set(json.loads((root / "checksums.json").read_text())) | {"checksums.json"}
    revision = api.dataset_info(DATASET_REPO).sha
    remote = set(api.list_repo_files(DATASET_REPO, repo_type="dataset", revision=revision))
    # A file an earlier release published and this one does not would otherwise stay on main.
    if stale := sorted(remote - expected - HUB_EXTRAS):
        api.delete_files(
            repo_id=DATASET_REPO,
            delete_patterns=stale,
            repo_type="dataset",
            commit_message="(chore): remove files the release no longer inventories",
        )
        revision = api.dataset_info(DATASET_REPO).sha
        remote = set(api.list_repo_files(DATASET_REPO, repo_type="dataset", revision=revision))
    if remote - HUB_EXTRAS != expected:
        raise DatasetError(
            f"Hub release differs from the local one: {len(expected - remote)} missing, "
            f"{len(remote - expected - HUB_EXTRAS)} extra"
        )
    for name in ("manifest.json", "checksums.json"):
        path = Path(hf_hub_download(DATASET_REPO, name, repo_type="dataset", revision=revision))
        if file_digest(path) != file_digest(root / name):
            raise DatasetError(f"uploaded {name} does not match the local release")
    return {
        "repo": DATASET_REPO,
        "revision": revision,
        "verified_files": len(expected),
        "manifest": verified,
    }


def verify_revision(revision: str) -> dict:
    """Refuse nonexistent, partial or incompatible Hub releases."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    check_revision(revision)
    try:
        data = {}
        for name in ("manifest.json", "schema.json", "checksums.json"):
            path = hf_hub_download(DATASET_REPO, name, repo_type="dataset", revision=revision)
            data[name] = json.loads(Path(path).read_text())
        manifest = data["manifest.json"]
        if (
            data["schema.json"] != schema()
            or manifest.get("format") != FORMAT
            or manifest.get("source_revision") != SOURCE_REVISION
            or manifest.get("tasks") != TASKS
            or manifest.get("episodes") != EPISODES
            or manifest.get("videos") != 3 * EPISODES
        ):
            raise DatasetError("Hub release is incompatible with this dataset format")
        remote = set(HfApi().list_repo_files(DATASET_REPO, repo_type="dataset", revision=revision))
        if not set(data["checksums.json"]) <= remote:
            raise DatasetError("Hub release is only partially uploaded")
        return manifest
    except (OSError, ValueError, HfHubHTTPError) as exc:
        raise DatasetError(f"cannot validate dataset revision {revision}: {exc}") from exc


def download_release(out: Path, revision: str, workers: int = 4) -> str:
    from huggingface_hub import snapshot_download

    check_revision(revision)
    return snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=str(out.resolve()),
        max_workers=workers,
    )
