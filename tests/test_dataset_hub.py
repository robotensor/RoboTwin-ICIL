import json

import huggingface_hub
import pytest

from robotwin_icil import dataset_release as release
from robotwin_icil import tasks
from robotwin_icil.dataset import DatasetError
from robotwin_icil.records import write_json

REVISION = "a" * 40


def fake_release(root):
    manifest = {
        "format": release.FORMAT,
        "source_revision": release.SOURCE_REVISION,
        "tasks": 50,
        "episodes": 2500,
        "videos": 7500,
    }
    write_json(root / "manifest.json", manifest)
    write_json(root / "schema.json", release.schema())
    write_json(
        root / "checksums.json", {"manifest.json": release.file_digest(root / "manifest.json")}
    )
    return manifest


def hub(monkeypatch, root, missing=False):
    calls = []

    def download(repo_id, filename, **kwargs):
        assert repo_id == release.DATASET_REPO
        assert kwargs["revision"] == REVISION
        assert kwargs["repo_type"] == "dataset"
        calls.append(filename)
        return str(root / filename)

    class API:
        def list_repo_files(self, *args, **kwargs):
            return [] if missing else [p.name for p in root.iterdir() if p.is_file()]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    return calls


def test_commit_pinned_preflight_accepts_compatible_complete_metadata(tmp_path, monkeypatch):
    manifest = fake_release(tmp_path)
    calls = hub(monkeypatch, tmp_path)
    assert release.verify_revision(REVISION) == manifest
    assert calls == ["manifest.json", "schema.json", "checksums.json"]


def test_preflight_rejects_partial_upload(tmp_path, monkeypatch):
    fake_release(tmp_path)
    hub(monkeypatch, tmp_path, missing=True)
    with pytest.raises(DatasetError, match="partially"):
        release.verify_revision(REVISION)


def test_preflight_rejects_changed_source_or_a_mutable_revision(tmp_path, monkeypatch):
    manifest = fake_release(tmp_path)
    hub(monkeypatch, tmp_path)
    manifest["source_revision"] = "b" * 40
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(DatasetError, match="incompatible"):
        release.verify_revision(REVISION)
    with pytest.raises(DatasetError, match="commit SHA"):
        release.verify_revision("main")


def test_upload_verifies_excludes_caches_and_removes_files_the_release_dropped(
    tmp_path, monkeypatch
):
    manifest = fake_release(tmp_path)
    hub(monkeypatch, tmp_path)
    operations = []
    # What an earlier release left on the Hub: a split index this release no longer has.
    remote = {".gitattributes", "splits.json", "manifest.json", "checksums.json"}

    class API:
        def create_repo(self, *args, **kwargs):
            operations.append(("create", kwargs))

        def upload_large_folder(self, **kwargs):
            operations.append(("upload", kwargs))

        def dataset_info(self, *args):
            return type("Info", (), {"sha": REVISION})()

        def list_repo_files(self, *args, **kwargs):
            return sorted(remote)

        def delete_files(self, **kwargs):
            operations.append(("delete", kwargs))
            remote.difference_update(kwargs["delete_patterns"])

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(release, "verify_release", lambda root: manifest)
    result = release.upload_release(tmp_path)
    assert result["revision"] == REVISION
    upload = dict(operations)["upload"]
    assert upload["repo_id"] == release.DATASET_REPO
    assert ".build/**" in upload["ignore_patterns"] and ".cache/**" in upload["ignore_patterns"]
    assert not any("zip" in p for p in upload["allow_patterns"])
    assert dict(operations)["delete"]["delete_patterns"] == ["splits.json"]


def converted(task, archive, root):
    rows = [
        # A conversion cached before splits were removed still carries the field.
        {"task": task, "episode_index": index, "split": "train", "rows": 2}
        for index in range(50)
    ]
    return {"task": task, "archive_sha256": "a", "archive_bytes": 1, "episodes": rows, "files": {}}


def test_build_release_indexes_every_episode_as_one_training_set(tmp_path, monkeypatch):
    out, cache = tmp_path / "release", tmp_path / "cache"
    cache.mkdir()
    (cache / "README.md").write_text("---\nlicense: mit\n---\n")
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: calls.append(k))
    monkeypatch.setattr(release, "convert_task", converted)
    built = release.build_release(out, cache, workers=2)
    assert (built["tasks"], built["episodes"], built["videos"]) == (50, 2500, 7500)
    assert "split_counts" not in built and "profile" not in built
    assert calls[0]["revision"] == release.SOURCE_REVISION
    assert "Tianxing Chen" in (out / "LICENSE").read_text()
    rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 2500 and not any("split" in row for row in rows)
    assert {row["task"] for row in rows} == set(tasks.table().tasks)
    checksums = json.loads((out / "checksums.json").read_text())
    assert {"LICENSE", "README.md", "episodes.jsonl", "manifest.json"} <= set(checksums)
    assert not (out / "splits.json").exists()
    front_matter = (out / "README.md").read_text().split("---\n")[1]
    assert "  - split: train\n    path: episodes.jsonl\n" in front_matter
    assert "validation" not in front_matter and "test" not in front_matter
