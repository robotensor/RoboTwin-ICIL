"""Side-by-side evidence: the demonstration and the evaluation rollout as two clips.

Watching `demonstration.mp4` next to `evaluation_same_scene.mp4` is the fastest way to catch a
reset, camera or action-conversion bug. Recording reads only frames the episode already has — the
demonstration's captured images and the observations the policy was given — plus one frame once
the outcome is decided, so enabling it cannot change a scene or a score.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .demo import Demonstration

DEMONSTRATION_CLIP = "demonstration.mp4"
EVALUATION_CLIP = "evaluation_same_scene.mp4"
DEFAULT_CAMERA = "head_camera"


class VideoError(RuntimeError):
    """A clip could not be written."""


def write_mp4(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    """Encode rgb frames as H.264. Odd sizes are edge-padded to even, as yuv420p requires."""
    path = Path(path)
    if not frames:
        raise VideoError(f"{path.name}: no frames to write")
    shape = np.asarray(frames[0]).shape
    if len(shape) != 3 or shape[2] != 3:
        raise VideoError(f"{path.name}: frames must be (h, w, 3), got {shape}")
    if any(np.asarray(frame).shape != shape for frame in frames):
        raise VideoError(f"{path.name}: frames change size mid-clip")
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise VideoError("video needs imageio-ffmpeg: pip install 'robotwin-icil[video]'") from exc

    pad = ((0, shape[0] % 2), (0, shape[1] % 2), (0, 0))
    size = (shape[1] + shape[1] % 2, shape[0] + shape[0] % 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size,
        fps=fps,
        codec="libx264",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    writer.send(None)
    try:
        for frame in frames:
            padded = np.pad(np.asarray(frame, dtype=np.uint8), pad, mode="edge")
            writer.send(np.ascontiguousarray(padded))
    finally:
        writer.close()


class EpisodeVideo:
    """One episode's two clips, written side by side into its directory.

    `evaluation_clip` names the rollout's file: a run directory's episode keeps
    `evaluation_same_scene.mp4`; an evaluation from a saved prompt writes `evaluation.mp4` next
    to it. `fps` is the rate a rollout clip plays at when no demonstration clip is written first.
    """

    def __init__(
        self,
        directory: Path,
        camera: str = DEFAULT_CAMERA,
        evaluation_clip: str = EVALUATION_CLIP,
        fps: float = 10.0,
    ) -> None:
        self.directory = Path(directory)
        self.camera = camera
        self.evaluation_clip = evaluation_clip
        self.fps = float(fps)
        self._frames: list[np.ndarray] = []

    def demonstration(self, demonstration: Demonstration) -> None:
        """Write what the expert did; the evaluation clip plays at the same rate."""
        self.fps = float(demonstration.frequency)
        if self.camera in demonstration.cameras:
            frames = list(demonstration.images(self.camera))
            write_mp4(self.directory / DEMONSTRATION_CLIP, frames, self.fps)

    def observe(self, images: dict[str, np.ndarray]) -> None:
        frame = images.get(self.camera)
        if frame is not None:
            self._frames.append(np.asarray(frame))

    def finish(self) -> None:
        """Write what the policy did, from every observation it was given."""
        if self._frames:
            write_mp4(self.directory / self.evaluation_clip, self._frames, self.fps)
        self._frames = []
