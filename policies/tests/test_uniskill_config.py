import hashlib
from pathlib import Path

import pytest

from icil_policies.uniskill.config import (
    load_config,
    model_path,
    verify_sha256,
)
from icil_policies.uniskill.conversion import Augmentation
from robotwin_icil.policy import PolicyError


def test_the_packaged_defaults_load(monkeypatch, tmp_path):
    monkeypatch.setenv("ICIL_HOME", str(tmp_path))
    config = load_config()
    assert config.checkpoint is None and config.checkpoint_path is None
    assert config.skills.k == 20 and config.skills.rate_hz == 20.0
    assert config.skills.camera == "far_side_camera" and config.skills.crop == 180
    assert config.skills.isd_path == tmp_path / "models" / "uniskill" / "idm.pth"
    assert config.skills.depth_dir == tmp_path / "models" / "depth-anything-v2-small"
    assert config.augmentation == Augmentation()
    assert config.as_dict()["augmentation"]["crop_scale"] == [0.9, 1.0]


def test_a_file_overrides_only_the_keys_it_names(tmp_path):
    path = tmp_path / "mine.yml"
    path.write_text("device: cpu\nskills:\n  batch_size: 4\naugmentation:\n  variants: 2\n")
    config = load_config(path, checkpoint="/abs/policy.pth")
    assert config.device == "cpu" and config.skills.batch_size == 4
    assert config.skills.k == 20 and config.augmentation.variants == 2
    assert str(config.checkpoint_path) == "/abs/policy.pth"


@pytest.mark.parametrize(
    "text",
    [
        "devise: cpu\n",
        "skills:\n  kk: 3\n",
        "skills: 3\n",
        "- a list\n",
        "augmentation:\n  brightness: 2\n",
    ],
)
def test_unknown_or_invalid_settings_are_refused(tmp_path, text):
    path = tmp_path / "bad.yml"
    path.write_text(text)
    with pytest.raises(PolicyError):
        load_config(path)


def test_model_paths_are_absolute(monkeypatch, tmp_path):
    monkeypatch.setenv("ICIL_HOME", str(tmp_path))
    assert model_path("/x/y") == Path("/x/y")
    assert model_path("~/y") == Path.home() / "y"
    assert model_path("a/b") == tmp_path / "models" / "a" / "b"


def test_sha256_is_checked(tmp_path):
    path = tmp_path / "w.bin"
    path.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    assert verify_sha256(path, digest, "weights") == digest
    assert verify_sha256(path, None, "weights") == digest
    with pytest.raises(PolicyError, match="expected"):
        verify_sha256(path, "0" * 64, "weights")
    with pytest.raises(PolicyError, match="not found"):
        verify_sha256(tmp_path / "missing", None, "weights")
