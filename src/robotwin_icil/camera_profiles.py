"""Camera profiles: swap RoboTwin's static cameras without changing any seed's scene.

Profiles are data (`cameras.yml`, beside `tasks.yml`); this module loads them and applies one to
RoboTwin's resolved `args`. It is pure — no RoboTwin import — so the guard below is tested in CI.

RoboTwin builds every static camera with `create_camera`, which draws four numbers from numpy's
global RNG after `np.random.seed(seed)` and before `load_actors`. The number of static cameras
created is therefore part of every seed's scene, while their names, types and poses are not. A
profile may replace a camera in its slot, and `apply` refuses anything that would change that
count, and anything that would let one camera overwrite another. There is no override: a profile
that needs one is a different benchmark, and every earlier seed's scene would silently move.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROFILES_PATH = Path(__file__).with_name("cameras.yml")

STOCK = "stock"
HEAD_CAMERA = "head_camera"
WRIST_CAMERAS = ("left_camera", "right_camera")
DEFAULT_VIDEO_CAMERA = HEAD_CAMERA
# RoboTwin's own default for `camera.head_camera_type` (`envs/camera/camera.py`).
DEFAULT_HEAD_CAMERA_TYPE = "D435"

_PROFILE_KEYS = frozenset({"replace", "camera", "video_camera"})
_CAMERA_KEYS = ("name", "type", "position", "forward", "left")
_VECTOR_KEYS = ("position", "forward", "left")
_CAMERA_TYPE_ARGS = ("head_camera_type", "wrist_camera_type")
# `create_camera` normalises `forward` and `left` but never orthogonalises them.
_PERPENDICULAR_TOL = 1e-3


class ProfileError(ValueError):
    """A camera profile is malformed, unknown, or would change the scenes RoboTwin builds."""


@dataclass(frozen=True)
class Profile:
    """One named profile: static cameras replaced by name, and `camera` args merged in."""

    name: str
    replace: dict[str, dict[str, Any]] = field(default_factory=dict)
    camera: dict[str, Any] = field(default_factory=dict)
    video_camera: str = DEFAULT_VIDEO_CAMERA

    def definition(self) -> dict[str, Any]:
        """Everything the profile changes, with defaults filled in and vectors as floats."""
        return {
            "replace": copy.deepcopy(self.replace),
            "camera": copy.deepcopy(self.camera),
            "video_camera": self.video_camera,
        }

    @property
    def sha256(self) -> str:
        """A digest of `definition()` that ignores key order and `1` vs `1.0` spellings."""
        canonical = json.dumps(self.definition(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def identity(self) -> dict[str, str]:
        """What a run manifest records: two runs compare equal only under the same definition."""
        return {"name": self.name, "sha256": self.sha256}


def parse(name: str, raw: Any) -> Profile:
    """Validate one profile's shape. Whether it keeps the scenes is `apply`'s question."""
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ProfileError(f"camera profile {name!r} must be a mapping, got {type(raw).__name__}")
    if unknown := sorted(set(raw) - _PROFILE_KEYS):
        raise ProfileError(
            f"camera profile {name!r} has unknown keys {unknown}; "
            f"known keys: {sorted(_PROFILE_KEYS)}"
        )

    replace_raw = raw.get("replace") or {}
    if not isinstance(replace_raw, Mapping):
        raise ProfileError(f"camera profile {name!r}: 'replace' must map camera names to cameras")
    replace = {str(old): _parse_camera(name, str(old), entry) for old, entry in replace_raw.items()}

    camera = raw.get("camera") or {}
    if not isinstance(camera, Mapping) or not all(isinstance(key, str) for key in camera):
        raise ProfileError(f"camera profile {name!r}: 'camera' must be a mapping of camera args")
    for key in _CAMERA_TYPE_ARGS:
        if key in camera and not isinstance(camera[key], str):
            raise ProfileError(f"camera profile {name!r}: camera.{key} must be a camera type")

    video_camera = raw.get("video_camera", DEFAULT_VIDEO_CAMERA)
    if not isinstance(video_camera, str) or not video_camera:
        raise ProfileError(f"camera profile {name!r}: 'video_camera' must be a camera name")
    return Profile(
        name=name, replace=replace, camera=copy.deepcopy(dict(camera)), video_camera=video_camera
    )


def _parse_camera(profile: str, old: str, entry: Any) -> dict[str, Any]:
    where = f"camera profile {profile!r}, replacement for {old!r}"
    if not isinstance(entry, Mapping):
        raise ProfileError(f"{where}: must be a mapping of {list(_CAMERA_KEYS)}")
    if missing := [key for key in _CAMERA_KEYS if key not in entry]:
        raise ProfileError(f"{where}: missing {missing}")
    if extra := sorted(set(entry) - set(_CAMERA_KEYS)):
        raise ProfileError(f"{where}: unknown keys {extra}")
    for key in ("name", "type"):
        if not isinstance(entry[key], str) or not entry[key]:
            raise ProfileError(f"{where}: {key!r} must be a non-empty string")
    camera: dict[str, Any] = {"name": entry["name"], "type": entry["type"]}
    for key in _VECTOR_KEYS:
        camera[key] = _vector(f"{where}: {key!r}", entry[key])
    for key in ("forward", "left"):
        if math.hypot(*camera[key]) == 0.0:
            raise ProfileError(f"{where}: {key!r} must not be zero")
    forward, left = (_unit(camera["forward"]), _unit(camera["left"]))
    if abs(sum(f * g for f, g in zip(forward, left, strict=True))) > _PERPENDICULAR_TOL:
        raise ProfileError(f"{where}: 'forward' and 'left' must be perpendicular")
    return camera


def _vector(where: str, value: Any) -> list[float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
        or not all(math.isfinite(x) for x in value)
    ):
        raise ProfileError(f"{where} must be three finite numbers, got {value!r}")
    return [float(x) for x in value]


def _unit(vector: list[float]) -> list[float]:
    norm = math.hypot(*vector)
    return [x / norm for x in vector]


def load(path: Path = PROFILES_PATH) -> dict[str, Profile]:
    """Every profile in a `cameras.yml`, in file order; `stock` must exist and change nothing."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    profiles_raw = raw.get("profiles") if isinstance(raw, Mapping) else None
    if not isinstance(profiles_raw, Mapping):
        raise ProfileError(f"{Path(path).name} must hold a 'profiles' mapping")
    profiles = {str(name): parse(str(name), body) for name, body in profiles_raw.items()}
    if STOCK not in profiles:
        raise ProfileError(f"{Path(path).name} has no {STOCK!r} profile")
    if profiles[STOCK].definition() != Profile(STOCK).definition():
        raise ProfileError(f"the {STOCK!r} profile is RoboTwin's own cameras and changes nothing")
    return profiles


@lru_cache(maxsize=1)
def _profiles() -> dict[str, Profile]:
    return load()


def names() -> tuple[str, ...]:
    return tuple(_profiles())


def get(name: str) -> Profile:
    try:
        return _profiles()[name]
    except KeyError:
        known = ", ".join(_profiles())
        raise ProfileError(f"unknown camera profile {name!r}; known profiles: {known}") from None


def _camera_list(args: Mapping[str, Any]) -> list[dict[str, Any]]:
    left = args.get("left_embodiment_config")
    cameras = left.get("static_camera_list") if isinstance(left, Mapping) else None
    if not isinstance(cameras, list):
        raise ProfileError("args carry no left_embodiment_config.static_camera_list")
    return cameras


def _collects_head(camera: Mapping[str, Any]) -> bool:
    return bool(camera.get("collect_head_camera", True))


def _created(cameras: list[dict[str, Any]], camera: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The static cameras RoboTwin will build, each one drawing from numpy's global RNG.

    Mirrors `Camera.load_camera`: `head_camera` is built only while `collect_head_camera` is
    set; every other entry is always built.
    """
    return [c for c in cameras if c.get("name") != HEAD_CAMERA or _collects_head(camera)]


def static_cameras(args: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The static cameras RoboTwin builds from these args, as it builds them.

    RoboTwin reads the list from the left embodiment config only, and gives `head_camera` the
    type in `camera.head_camera_type`. None when the args carry no embodiment config.
    """
    try:
        cameras = _camera_list(args)
    except ProfileError:
        return None
    camera = args.get("camera") or {}
    created = copy.deepcopy(_created(cameras, camera))
    for entry in created:
        if entry.get("name") == HEAD_CAMERA:
            entry["type"] = camera.get("head_camera_type", DEFAULT_HEAD_CAMERA_TYPE)
    return created


def apply(
    args: Mapping[str, Any],
    profile: str | Profile,
    camera_types: Collection[str] | None = None,
) -> dict[str, Any]:
    """New `args` with the profile applied; `args` itself is left untouched.

    Each replaced camera takes the slot of the one it replaces, in a deep copy of
    `left_embodiment_config` (RoboTwin's `Camera` reads the static camera list from the left
    config only), and the profile's `camera` mapping is merged into `args['camera']`. Refused,
    with `ProfileError`: a camera that does not exist, a camera type outside `camera_types`, a
    change in how many static cameras RoboTwin creates, toggling `collect_head_camera`, touching
    `head_camera`, duplicate camera names, or a `video_camera` the profile does not render.
    """
    profile = get(profile) if isinstance(profile, str) else profile
    where = f"camera profile {profile.name!r}"
    before = _camera_list(args)
    names_before = [entry.get("name") for entry in before]
    camera_before = dict(args.get("camera") or {})
    camera_after = {**camera_before, **copy.deepcopy(profile.camera)}

    after = copy.deepcopy(before)
    for old, new in profile.replace.items():
        if names_before.count(old) != 1:
            state = "does not exist" if old not in names_before else "is not unique"
            raise ProfileError(
                f"{where} replaces {old!r}, which {state}; the static cameras are {names_before}"
            )
        after[names_before.index(old)] = copy.deepcopy(new)

    if camera_types is not None:
        introduced = [entry["type"] for entry in profile.replace.values()]
        introduced += [profile.camera[key] for key in _CAMERA_TYPE_ARGS if key in profile.camera]
        for camera_type in introduced:
            if camera_type not in camera_types:
                raise ProfileError(
                    f"{where} uses camera type {camera_type!r}; RoboTwin's types are "
                    f"{sorted(camera_types)}"
                )

    if HEAD_CAMERA in profile.replace:
        raise ProfileError(f"{where} replaces {HEAD_CAMERA}; it may not be replaced or renamed")
    _refuse_scene_changes(where, before, camera_before, after, camera_after)
    created_after = _created(after, camera_after)

    if profile.video_camera != DEFAULT_VIDEO_CAMERA:
        rendered = [entry.get("name") for entry in created_after]
        if camera_after.get("collect_wrist_camera", True):
            rendered += list(WRIST_CAMERAS)
        if profile.video_camera not in rendered:
            raise ProfileError(
                f"{where} films {profile.video_camera!r}, which it does not render; "
                f"it renders {rendered}"
            )

    left = copy.deepcopy(args["left_embodiment_config"])
    left["static_camera_list"] = after
    return {**args, "camera": camera_after, "left_embodiment_config": left}


def _refuse_scene_changes(
    where: str,
    before: list[dict[str, Any]],
    camera_before: Mapping[str, Any],
    after: list[dict[str, Any]],
    camera_after: Mapping[str, Any],
) -> None:
    """Refuse static cameras that move a seed's scene, or that let one camera overwrite another."""
    if _collects_head(camera_before) != _collects_head(camera_after):
        raise ProfileError(
            f"{where} toggles camera.collect_head_camera, which changes how many cameras draw "
            "from the RNG before the scene is built, and so every seed's scene"
        )
    created_before, created_after = _created(before, camera_before), _created(after, camera_after)
    if len(created_before) != len(created_after):
        raise ProfileError(
            f"{where} changes the number of static cameras RoboTwin creates from "
            f"{len(created_before)} to {len(created_after)}; each one draws from the RNG before "
            "the scene is built, so every seed's scene would change"
        )
    heads_before = [i for i, entry in enumerate(before) if entry.get("name") == HEAD_CAMERA]
    heads_after = [i for i, entry in enumerate(after) if entry.get("name") == HEAD_CAMERA]
    if len(heads_after) > len(heads_before):
        raise ProfileError(f"{where} names another camera {HEAD_CAMERA}")
    if heads_before != heads_after:
        raise ProfileError(f"{where} moves or removes {HEAD_CAMERA}")
    names_after = [entry.get("name") for entry in after]
    if duplicates := sorted({name for name in names_after if names_after.count(name) > 1}):
        raise ProfileError(
            f"{where} leaves two cameras named {duplicates}; RoboTwin keys cameras by name, so "
            "one would overwrite the other"
        )
    # `Camera.get_config()` and `get_rgba()` store the wrist cameras under these names before the
    # static cameras, so a static camera named like one silently replaces its config and image.
    # Reserved even while `collect_wrist_camera` is off, so a later toggle cannot collide.
    if wrist := [name for name in names_after if name in WRIST_CAMERAS]:
        raise ProfileError(
            f"{where} names a static camera {wrist[0]!r}, like a wrist camera; RoboTwin keys "
            "cameras by name, so one would overwrite the other"
        )


def check_scene_preserved(
    before: Mapping[str, Any], after: Mapping[str, Any], where: str = "the resolved args"
) -> None:
    """Refuse `after` args whose static cameras would build other scenes than `before`'s.

    `apply` holds a profile to this; `SceneConfig.resolve` holds its `overrides` to it too, once
    they have had the last word, so the guard has no way around it. Raises `ProfileError`.
    """
    _refuse_scene_changes(
        where,
        _camera_list(before),
        before.get("camera") or {},
        _camera_list(after),
        after.get("camera") or {},
    )
