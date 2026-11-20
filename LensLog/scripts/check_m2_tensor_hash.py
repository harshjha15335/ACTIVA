#!/usr/bin/env python3
"""Check local ACTIVA CLIP tensors against M4's content digest, without uploading weights.

Run inside the ACTIVA checkout:
    python scripts/check_m2_tensor_hash.py

The *file* SHA256 may differ across machines due to safetensors serialization.
The *tensor-content* SHA256 hashes sorted tensor names, dtypes, shapes and bytes
exactly as specified by M4 in issue #4 (2026-11-12). This is an offline test,
not a substitute for checkpoint provenance or empirical model evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_TENSOR_SHA256 = "4423d64febb7bfb00ddc8c77fd2c38b9047d5183c0e45f14e56847295ff8fedf"
EXPECTED_SOURCE_SHA256 = "46683a32721d5c68911153698992361285d20ca690bb4f317c11e45c03d798fa"


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_tensors(tensors) -> str:
    """Match M4's published sorted name/dtype/shape/bytes algorithm exactly."""
    digest = hashlib.sha256()
    for key in sorted(tensors):
        tensor = tensors[key]
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tensor.shape).encode())
        digest.update(tensor.tobytes())
    return digest.hexdigest()


def inspect_checkpoint(path: Path, expected: str = EXPECTED_TENSOR_SHA256) -> dict:
    from safetensors.numpy import load_file

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Local checkpoint not found: {path}")
    tensors = load_file(str(path))
    file_hash = sha256_file(path)
    content_hash = sha256_tensors(tensors)
    report = {
        "format": "activa-m2-tensor-hash-report-v1",
        "filename": path.name,
        "tensor_count": len(tensors),
        "file_sha256": file_hash,
        "tensor_content_sha256": content_hash,
        "reference_tensor_sha256": expected,
        "tensor_match": content_hash == expected,
        "manifest_present": False,
        "manifest_file_match": None,
        "raw_source_sha256": None,
        "raw_source_match": None,
    }
    manifest = path.with_suffix(".manifest.json")
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        report.update(
            manifest_present=True,
            manifest_file_match=data.get("sha256") == file_hash,
            raw_source_sha256=data.get("source_sha256"),
            raw_source_match=data.get("source_sha256") == EXPECTED_SOURCE_SHA256,
        )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/ViT-B-32-laion400m_e32.safetensors"))
    parser.add_argument("--expected", default=EXPECTED_TENSOR_SHA256,
                        help="Expected full 64-character tensor SHA256 from M4")
    parser.add_argument("--report", type=Path, help="Optional local JSON report path")
    args = parser.parse_args(argv)
    if len(args.expected) != 64 or any(x not in "0123456789abcdef" for x in args.expected):
        parser.error("--expected must be a lowercase 64-character SHA256")
    try:
        report = inspect_checkpoint(args.weights, args.expected)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"BLOCKED: {error}")
        return 2
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["tensor_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
