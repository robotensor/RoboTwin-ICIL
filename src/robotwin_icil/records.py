"""Episode records and run manifests: everything a run leaves behind, besides optional videos.

A run directory holds `manifest.json` (what was run, against which commits and configs) and
`episodes.jsonl` (one line per episode, appended and flushed as each finishes). Reporting reads
only these files, so a finished run is re-reported without a simulator, and an interrupted run
resumes by skipping the episodes already on disk. A run whose policy failed the frozen-policy audit
also holds `audit.json`, and is then neither resumed nor reported.
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

from . import camera_profiles

SAME_SCENE = "same_scene"

MANIFEST = "manifest.json"
EPISODES = "episodes.jsonl"
# Written only when the policy's parameters changed during the run; its absence is a pass, or a
# run that predates the audit.
AUDIT = "audit.json"


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
    checkpoint: str | None = None
    detail: str = ""
    duration_s: float = 0.0
    # One example detail per rejection reason: why seeds were rejected, not only how often.
    rejection_details: dict[str, str] = field(default_factory=dict)
    # Physics steps the rollout ran; 0 when nothing was rolled out, and in records written before
    # the benchmark counted them. `steps` counts `take_action` calls, each of many physics steps.
    physics_steps: int = 0
    # What the policy's `episode_info()` reported after the rollout; empty when nothing was rolled
    # out, for policies that report nothing, and in records written before policies could.
    policy_info: dict[str, Any] = field(default_factory=dict)
    # The policy's `action_type`, "qpos" or "ee": the action path the rollout went through. None
    # in records written before the benchmark recorded it.
    action_type: str | None = None
    # The arms the demonstration moved, from `Demonstration.arms_moved()`. Metadata for analysis,
    # never a report slice: the protocol has no difficulty tiers. None when no demonstration was
    # generated, and in records written before the benchmark recorded it.
    demonstration_arms: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        status = Status(self.status)
        object.__setattr__(self, "status", status)
        if self.demonstration_arms is not None:
            # JSON reads a tuple back as a list; a record compares equal to what was written.
            object.__setattr__(self, "demonstration_arms", tuple(self.demonstration_arms))
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
        return cls(**data)


@dataclass(frozen=True)
class RunManifest:
    """Enough to reproduce any single episode of the run on its own."""

    global_seed: int
    evaluation_setting: str
    suite: str | None
    tasks: tuple[str, ...]
    episodes: int
    max_expert_attempts: int
    policy: dict[str, Any]
    benchmark_commit: str | None
    robotwin_commit: str | None
    benchmark_config: dict[str, Any]
    robotwin_config: dict[str, Any]
    environment: dict[str, str] = field(default_factory=dict)
    # The policy's own `environment()`: its python, torch, CUDA, GPU and model commits. Empty for
    # policies that report none, and in manifests written before policies could.
    policy_environment: dict[str, str] = field(default_factory=dict)
    # The keyword arguments the policy was built with (`eval --policy-arg`). Part of the run's
    # identity; manifests written before it existed built their policy with none.
    policy_config: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["tasks"] = list(self.tasks)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RunManifest:
        return cls(**{**data, "tasks": tuple(data["tasks"])})

    def identity(self) -> dict[str, Any]:
        """The fields a resumed run must share with the run it continues.

        Manifests written before camera profiles existed ran RoboTwin's own cameras and recorded
        neither a profile nor `static_cameras`, so they read as `stock`. Under `stock` the static
        cameras are the embodiment's own, implied by `embodiment` and the RoboTwin commit as
        before, so they are recorded for the reader but compared only under another profile.
        The machine may differ between a run and its resumption, so neither the benchmark's
        environment nor the policy's is compared. The policy's description and `policy_config`
        are: a resume with another adapter version or other policy arguments is another run.
        """
        data = self.to_json()
        data.pop("environment", None)
        data.pop("policy_environment", None)
        stock = camera_profiles.get(camera_profiles.STOCK).identity()
        benchmark_config = data["benchmark_config"]
        benchmark_config.setdefault("camera_profile", stock)
        if benchmark_config["camera_profile"] == stock:
            data["robotwin_config"].pop("static_cameras", None)
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

    @property
    def audit_path(self) -> Path:
        return self.path / AUDIT

    def episode_dir(self, episode: int) -> Path:
        return self.path / f"episode_{episode:05d}"

    def start(self, manifest: RunManifest) -> None:
        """Begin a run, or continue one: a directory already holding a different run is refused,
        and so is one whose run failed the frozen-policy audit."""
        self.check_audit()
        self.path.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            existing = self.manifest()
            if existing.identity() != manifest.identity():
                raise RecordError(
                    f"{self.path} already holds a different run; choose a new --run-dir"
                )
            return
        _write_json(self.manifest_path, manifest.to_json())

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

    def fail_audit(self, detail: dict[str, Any]) -> None:
        """Record that the policy's parameters changed during this run.

        The error that follows ends with the process; this file does not, so a later resume or
        report still knows the episodes on disk came from a policy that was not frozen.
        """
        _write_json(self.audit_path, {"frozen": False, **detail})

    def check_audit(self) -> None:
        """Raise `RecordError` if this run failed the frozen-policy audit."""
        if self.audit_path.exists():
            raise RecordError(
                f"{self.path} failed the frozen-policy audit (see {AUDIT}): the policy's "
                "parameters changed during the run, so its episodes are neither resumed nor "
                "reported; choose a new --run-dir"
            )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def git_commit(path: Path) -> str | None:
    """HEAD of the checkout at `path`, suffixed `-dirty` when it has uncommitted changes."""
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
