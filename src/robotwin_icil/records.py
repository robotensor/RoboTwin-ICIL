"""Episode records and run manifests: everything a run leaves behind, besides optional videos.

A run directory holds `manifest.json` (what was run, against which commits and configs) and
`episodes.jsonl` (one line per episode, appended and flushed as each finishes). Reporting reads
only these files, so a finished run is re-reported without a simulator, and an interrupted run
resumes by skipping the episodes already on disk.
"""

from __future__ import annotations

import enum
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .arms import ONE, TWO

SAME_SCENE = "same_scene"

MANIFEST = "manifest.json"
EPISODES = "episodes.jsonl"


class RecordError(ValueError):
    """A record or run directory is malformed, or does not match the run it claims to continue."""


class Status(str, enum.Enum):
    # The expert produced a demonstration, the reset reproduced its scene, the policy rolled out.
    SCORED = "scored"
    # No seed in the attempt budget produced a successful expert demonstration.
    REJECTED = "rejected"
    # The evaluation scene did not match the demonstration's, or the harness itself failed.
    INVALID = "invalid"


@dataclass(frozen=True)
class EpisodeRecord:
    episode: int
    evaluation_setting: str
    skill_category: str
    task: str
    scene_seed: int | None
    status: Status
    success: bool | None
    steps: int
    step_limit: int | None
    demonstration_frames: int
    expert_generation_attempts: int
    rejections: dict[str, int]
    scene_max_error: float
    model: str
    # The robot, as `robotwin.embodiment_name` names it (the `--embodiment` choice, or the arms and
    # their distance): a 14-wide aloha-agilex episode and a 16-wide franka-panda one are not the
    # same measurement, and neither are two Frankas at different distances.
    embodiment: str
    checkpoint: str | None = None
    detail: str = ""
    duration_s: float = 0.0
    # One example detail per rejection reason: why seeds were rejected, not only how often.
    rejection_details: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = Status(self.status)
        object.__setattr__(self, "status", status)
        if status is Status.SCORED and not isinstance(self.success, bool):
            raise RecordError(f"episode {self.episode}: a scored episode needs a boolean success")
        if status is not Status.SCORED and self.success is not None:
            raise RecordError(f"episode {self.episode}: only scored episodes carry a success")

    @property
    def scored(self) -> bool:
        return self.status is Status.SCORED

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EpisodeRecord:
        # Before a run could choose its robot, every episode ran on aloha-agilex, the embodiment of
        # every shipped task config; records from then still load, and say so.
        return cls(**{"embodiment": "aloha-agilex", **data})


@dataclass(frozen=True)
class RunManifest:
    """Enough to reproduce any single episode of the run on its own."""

    global_seed: int
    evaluation_setting: str
    tasks: tuple[str, ...]
    episodes: int
    max_expert_attempts: int
    policy: dict[str, Any]
    benchmark_commit: str | None
    robotwin_commit: str | None
    benchmark_config: dict[str, Any]
    robotwin_config: dict[str, Any]
    environment: dict[str, str] = field(default_factory=dict)
    # "1" when the run asked for one-arm tasks only. Runs recorded before the field existed ran
    # whatever they named, which is what "2" means, so they load unchanged.
    arms: str = TWO
    # `generate.SEED_STREAMS`; runs recorded before the field existed drew independent streams.
    seed_stream: str = "independent"

    def __post_init__(self) -> None:
        if self.arms not in (ONE, TWO):
            raise RecordError(
                f"a run asks for arms {ONE} or {TWO}; the manifest says {self.arms!r}"
            )
        if self.seed_stream not in ("independent", "robotwin"):
            raise RecordError(f"unknown seed stream {self.seed_stream!r}")

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["tasks"] = list(self.tasks)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RunManifest:
        # Older manifests named a suite as well as the exact task list. Only the task list
        # determines what ran, so discard that obsolete label when reading legacy results.
        fields = {key: value for key, value in data.items() if key != "suite"}
        return cls(**{**fields, "tasks": tuple(data["tasks"])})

    def identity(self) -> dict[str, Any]:
        """The fields a resumed run must share with the run it continues."""
        data = self.to_json()
        data.pop("environment", None)
        return data


class RunDir:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()

    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST

    @property
    def episodes_path(self) -> Path:
        return self.path / EPISODES

    def episode_dir(self, episode: int) -> Path:
        return self.path / f"episode_{episode:05d}"

    def start(self, manifest: RunManifest) -> None:
        """Begin a run, or continue one: a directory already holding a different run is refused."""
        self.path.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            existing = self.manifest()
            if existing.identity() != manifest.identity():
                raise RecordError(
                    f"{self.path} already holds a different run; choose a new --run-dir"
                )
            return
        write_json(self.manifest_path, manifest.to_json(), sort_keys=True)

    def manifest(self) -> RunManifest:
        if not self.manifest_path.exists():
            raise RecordError(f"{self.path} has no {MANIFEST}; is it a run directory?")
        return RunManifest.from_json(json.loads(self.manifest_path.read_text(encoding="utf-8")))

    def append(self, record: EpisodeRecord) -> None:
        line = json.dumps(record.to_json(), sort_keys=True)
        with self.episodes_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def records(self) -> list[EpisodeRecord]:
        if not self.episodes_path.exists():
            return []
        records = []
        for number, line in enumerate(
            self.episodes_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                records.append(EpisodeRecord.from_json(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                # A torn final line is what a killed run leaves; anything earlier is corruption.
                if number == len(self.episodes_path.read_text(encoding="utf-8").splitlines()):
                    break
                raise RecordError(f"{self.episodes_path}:{number}: {exc}") from exc
        episodes = [record.episode for record in records]
        if len(set(episodes)) != len(episodes):
            raise RecordError(f"{self.episodes_path} records an episode twice")
        return records

    def completed(self) -> set[int]:
        return {record.episode for record in self.records()}


def write_json(path: Path, data: Any, *, sort_keys: bool = False) -> None:
    """Replace `path` with `data` as JSON, whole or not at all.

    The text goes to a sibling `.tmp` and is renamed into place, so a process killed mid-write
    leaves the previous file rather than a truncated one.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8")
    tmp.replace(path)


def is_checkout_top(path: Path) -> bool:
    """Whether `path` is the top of a git checkout (a worktree's or a submodule's included).

    A benchmark installed from a wheel lives inside whatever directory the environment is in, and
    that may be some other repository, or a submodule directory left empty may sit inside the
    benchmark's: git would answer for the repository around it.
    """
    try:
        top = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(top) and Path(top).resolve() == Path(path).resolve()


def git_commit(path: Path) -> str | None:
    """HEAD of the checkout at `path`, suffixed `-dirty` when it has uncommitted changes; None
    unless `path` is the top of a checkout (`is_checkout_top`)."""
    if not is_checkout_top(path):
        return None
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}-dirty" if dirty else head


def environment() -> dict[str, str]:
    return {"python": sys.version.split()[0], "platform": platform.platform()}
