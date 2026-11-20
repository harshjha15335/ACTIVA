"""Synthetic, offline unit tests for M2 private-artifact tooling (no model/data)."""
from pathlib import Path
import json
import sys
import zipfile

import numpy as np
import pytest
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "code")]

from contracts import ClipFeats, Importance, VideoInfo
from perception_cache import read_cache, write_cache
from scripts.check_m2_tensor_hash import (sha256_file, sha256_tensors,
                                          inspect_checkpoint)
from scripts.package_m2_cache import create_package, validate_member


def test_tensor_digest_is_content_based_not_safetensors_metadata(tmp_path):
    tensors = {"weight": np.array([[1.0, -2.0]], np.float32),
               "bias": np.array([0.125], np.float32)}
    expected = sha256_tensors(tensors)
    assert expected == sha256_tensors(dict(reversed(list(tensors.items()))))
    assert expected != sha256_tensors({"weight": np.array([[1.0, 2.0]], np.float32),
                                      "bias": tensors["bias"]})

    first, second = tmp_path / "first.safetensors", tmp_path / "second.safetensors"
    save_file(tensors, str(first), metadata={"origin": "one"})
    save_file(tensors, str(second), metadata={"origin": "two"})
    assert sha256_file(first) != sha256_file(second)
    assert inspect_checkpoint(first, expected)["tensor_match"]
    assert inspect_checkpoint(second, expected)["tensor_match"]
    assert not inspect_checkpoint(first, "0" * 64)["tensor_match"]


@pytest.mark.parametrize("raw", [
    "../clip/a.npz", "clip/../a.npz", "importance/a.npz",
    "/clip/a.npz", "C:/clip/a.npz", "",
])
def test_archive_member_rejects_unsafe_paths(raw):
    with pytest.raises(ValueError):
        validate_member(raw, "clip")


def test_windows_cache_path_is_portable():
    assert validate_member(r"clip\something.npz", "clip") == "clip/something.npz"


def test_private_handoff_zip_roundtrip_preserves_hashes_and_redacts_paths(tmp_path):
    video = tmp_path / "private-machine-dir" / "original-video.mp4"
    video.parent.mkdir()
    video.write_bytes(b"synthetic bytes only: not a real video")
    info = VideoInfo("toy-handoff", str(video), 60.0, 30, 64, 48)
    times = np.arange(0, 60, 0.5, dtype=np.float64)
    feats = np.zeros((len(times), 16), dtype=np.float32)
    feats[:, 0] = 1
    clip = ClipFeats(times, feats, "SYNTHETIC-test-only")
    imp = Importance(times, np.full(len(times), 0.5, dtype=np.float32))
    manifest_path = write_cache(tmp_path / "dev_cache", info, clip, imp,
                                {"kind": "synthetic-unit-test"})
    original = json.loads(manifest_path.read_text())
    original_hash = sha256_file(manifest_path)
    archive_path = tmp_path / "handoff.zip"

    result = create_package(manifest_path, archive_path)
    assert result["portable_roundtrip_verified"]
    assert result["feature_shape"] == [len(times), 16]
    assert sha256_file(manifest_path) == original_hash
    assert archive_path.is_file()

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        assert len(members) == 4
        assert all("private-machine-dir" not in member for member in members)
        assert "SHA256SUMS.json" in members
        receipt = json.loads(archive.read("SHA256SUMS.json"))
        assert receipt["original_manifest_sha256"] == original_hash
        assert receipt["model_name"] == "SYNTHETIC-test-only"
        transferred = json.loads(archive.read(result["cache_manifest_in_zip"]))
        assert transferred["input_sha256"] == original["input_sha256"]
        assert transferred["video"]["path"] == "source_media_not_included/original-video.mp4"
        assert "private-machine-dir" not in json.dumps(transferred)
        assert transferred["clip"]["sha256"] == original["clip"]["sha256"]

    with pytest.raises(FileExistsError):
        create_package(manifest_path, archive_path)


def test_package_refuses_corrupted_npz(tmp_path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"toy")
    info = VideoInfo("toy2", str(video), 40.0, 30, 64, 48)
    times = np.arange(0, 40, 0.5)
    feats = np.zeros((len(times), 8), dtype=np.float32)
    feats[:, 0] = 1
    imp = Importance(times, np.ones(len(times), dtype=np.float32))
    manifest = write_cache(tmp_path / "dev_cache", info, ClipFeats(times, feats, "synthetic"), imp, {})
    original = json.loads(manifest.read_text())
    with (manifest.parent.parent / original["clip"]["path"]).open("ab") as stream:
        stream.write(b"corruption")
    output = tmp_path / "invalid.zip"
    with pytest.raises(ValueError, match="checksum"):
        create_package(manifest, output)
    assert not output.exists()
