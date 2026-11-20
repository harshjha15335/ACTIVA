#!/usr/bin/env python3
"""Package an existing genuine M2 development cache for private M3 handoff.

Example:
    python scripts/package_m2_cache.py \
      --manifest dev_cache/manifests/vtest_full.abc.def.json \
      --output tmp/m2/vtest_full_handoff.zip

The ZIP has manifests/, clip/, importance/ and SHA256SUMS.json. It does NOT
contain the video, model checkpoint, Samsung kit or ground truth. Cache payload
hashes and model/source identity are preserved. A local video path is redacted
in the transport copy; the input manifest itself is NEVER modified.
Share privately and provide the ZIP's printed SHA256 to the recipient.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from perception_cache import read_cache  # noqa: E402


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_member(raw: str, bucket: str) -> str:
    """Translate Windows relative paths to portable and safe ZIP member names."""
    if not isinstance(raw, str) or not raw:
        raise ValueError("Cache member path must be a nonempty string")
    normalized = raw.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if (normalized.startswith("/") or ":" in normalized
            or len(parts) < 2 or parts[0] != bucket
            or any(part in ("", ".", "..") for part in parts)):
        raise ValueError(f"Unsafe {bucket} cache path: {raw!r}")
    return "/".join(parts)


def portable_manifest(original: dict) -> dict:
    """Preserve content hashes while preventing local absolute path disclosure."""
    data = json.loads(json.dumps(original))
    for section in ("clip", "importance"):
        data[section]["path"] = validate_member(data[section]["path"], section)
    source = str(data["video"]["path"]).replace("\\", "/").split("/")[-1]
    if not source or source in (".", ".."):
        source = "video-not-included.mp4"
    data["video"]["path"] = f"source_media_not_included/{source}"
    return data


def create_package(manifest_path: Path, destination: Path) -> dict:
    manifest_path = Path(manifest_path).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing artifact: {destination}")
    if destination.suffix.lower() != ".zip":
        raise ValueError("Output filename must end in .zip")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"M2 cache manifest not found: {manifest_path}")

    # Verify model, payload SHA256s, normalized float32 features and PTS first.
    info, clip, imp, original = read_cache(manifest_path)
    root = manifest_path.parent.parent
    exported = portable_manifest(original)
    members = {}
    for kind in ("clip", "importance"):
        # On the author's local OS, the paths are validated by read_cache().
        original_path = (root / original[kind]["path"]).resolve()
        member = exported[kind]["path"]
        if not original_path.is_relative_to(root):
            raise ValueError("Cache path escapes its development root")
        contents = original_path.read_bytes()
        if sha256_bytes(contents) != original[kind]["sha256"]:
            raise ValueError(f"{kind} payload hash does not match manifest")
        members[member] = contents

    relative_manifest = f"manifests/{manifest_path.name}"
    members[relative_manifest] = (
        json.dumps(exported, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    receipt = {
        "format": "activa-m2-private-handoff-v1",
        "manifest_member": relative_manifest,
        "original_manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "exported_manifest_sha256": sha256_bytes(members[relative_manifest]),
        "input_video_sha256": original["input_sha256"],
        "model_name": clip.model_name,
        "features_shape": list(clip.feats.shape),
        "importance_shape": list(imp.values.shape),
        "source_video_included": False,
        "model_checkpoint_included": False,
        "organizer_kit_included": False,
        "file_sha256": {name: sha256_bytes(payload) for name, payload in sorted(members.items())},
        "notice": (
            "Development-only M2 cache; source path redacted. Original model and "
            "video hashes preserved; no real-data accuracy or official metric claimed."
        ),
    }
    members["SHA256SUMS.json"] = (
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".zip",
                                         prefix=".m2-handoff-", delete=False) as temporary:
            temp_path = Path(temporary.name)
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as out:
            for name, content in sorted(members.items()):
                entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o600 << 16
                out.writestr(entry, content)

        # Extract to a throwaway directory and test the receiver's exact reader.
        with tempfile.TemporaryDirectory(prefix="activa-m2-accept-") as temporary:
            folder = Path(temporary)
            with zipfile.ZipFile(temp_path) as archive:
                if sorted(archive.namelist()) != sorted(members):
                    raise ValueError("Unexpected ZIP members")
                for name in members:
                    if sha256_bytes(archive.read(name)) != sha256_bytes(members[name]):
                        raise ValueError(f"ZIP payload verification failed: {name}")
                archive.extractall(folder)
            restored_info, restored_clip, restored_imp, restored = read_cache(folder / relative_manifest)
            if (restored["input_sha256"] != original["input_sha256"]
                    or restored_clip.model_name != clip.model_name
                    or restored_clip.feats.shape != clip.feats.shape
                    or restored_imp.values.shape != imp.values.shape
                    or restored_info.video_id != info.video_id):
                raise ValueError("Receiver cache roundtrip changed provenance or dimensions")
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return {
        "archive": str(destination),
        "archive_sha256": sha256_bytes(destination.read_bytes()),
        "cache_manifest_in_zip": relative_manifest,
        "feature_shape": list(clip.feats.shape),
        "model_name": clip.model_name,
        "video_sha256": original["input_sha256"],
        "portable_roundtrip_verified": True,
        "share_method": "private transfer only; not a public GitHub commit",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(create_package(args.manifest, args.output), indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
