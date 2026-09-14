import dataclasses

import numpy as np
import pytest

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import robotwin, tasks, video
from robotwin_icil.episode import EpisodeSpec, run_episode
from robotwin_icil.policy import ReplayPolicy
from robotwin_icil.records import Status

imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")

SPEC = EpisodeSpec(
    episode=0, task=tasks.table()["place_object_basket"], global_seed=0, max_expert_attempts=5
)


@pytest.fixture(autouse=True)
def _fake_unstable(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)


def frames_in(path) -> int:
    return imageio_ffmpeg.count_frames_and_secs(str(path))[0]


def test_odd_frame_sizes_are_padded_not_rejected(tmp_path):
    video.write_mp4(tmp_path / "clip.mp4", [np.zeros((17, 23, 3), dtype=np.uint8)] * 3, fps=10)
    assert frames_in(tmp_path / "clip.mp4") == 3


def test_malformed_clips_are_refused(tmp_path):
    with pytest.raises(video.VideoError):
        video.write_mp4(tmp_path / "clip.mp4", [], fps=10)
    frames = [np.zeros((16, 16, 3), dtype=np.uint8), np.zeros((32, 16, 3), dtype=np.uint8)]
    with pytest.raises(video.VideoError):
        video.write_mp4(tmp_path / "clip.mp4", frames, fps=10)


def test_an_episode_writes_both_clips(tmp_path):
    env = FakeTaskEnv()
    run_episode(
        SPEC, ReplayPolicy(), FakeConfig(), task_env=env, video=video.EpisodeVideo(tmp_path)
    )
    assert frames_in(tmp_path / video.DEMONSTRATION_CLIP) == env.expert_steps + 1
    # One frame per observation the policy was given, plus the final state after the verdict.
    assert frames_in(tmp_path / video.EVALUATION_CLIP) == env.expert_steps + 1


def test_video_changes_nothing_that_is_scored(tmp_path):
    plain = run_episode(SPEC, ReplayPolicy(), FakeConfig(), task_env=FakeTaskEnv())
    filmed = run_episode(
        SPEC,
        ReplayPolicy(),
        FakeConfig(),
        task_env=FakeTaskEnv(),
        video=video.EpisodeVideo(tmp_path),
    )
    assert dataclasses.replace(plain, duration_s=0) == dataclasses.replace(filmed, duration_s=0)


def test_a_drifted_scene_still_leaves_its_first_frame(tmp_path):
    record = run_episode(
        SPEC,
        ReplayPolicy(),
        FakeConfig(),
        task_env=FakeTaskEnv(drift=True),
        video=video.EpisodeVideo(tmp_path),
    )
    assert record.status is Status.INVALID
    assert frames_in(tmp_path / video.EVALUATION_CLIP) == 1


def test_a_clip_that_fails_to_write_is_a_note_not_an_outcome(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise video.VideoError("disk full")

    monkeypatch.setattr(video, "write_mp4", broken)
    record = run_episode(
        SPEC,
        ReplayPolicy(),
        FakeConfig(),
        task_env=FakeTaskEnv(),
        video=video.EpisodeVideo(tmp_path),
    )
    assert record.status is Status.SCORED and record.success
    assert "video: VideoError: disk full" in record.detail


def test_the_evaluation_clip_can_be_named_and_paced_without_a_demonstration_clip(tmp_path):
    clip = video.EpisodeVideo(tmp_path, evaluation_clip="evaluation.mp4", fps=25.0)
    for _ in range(3):
        clip.observe({"head_camera": np.zeros((16, 16, 3), dtype=np.uint8)})
    clip.finish()
    assert not (tmp_path / video.EVALUATION_CLIP).exists()
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(tmp_path / "evaluation.mp4"))
    assert frames == 3 and seconds == pytest.approx(3 / 25, abs=0.01)
