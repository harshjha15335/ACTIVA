#!/usr/bin/env python3
"""Preliminary inventory overlap check; cannot certify re-encoded content safe."""
import argparse
import csv
import json
from pathlib import Path

def inventory(path):
    rows = list(csv.DictReader(Path(path).open()))
    for row in rows:
        if not row.get("video_id") or not row.get("sha256") or len(row["sha256"]) != 64 or any(c not in "0123456789abcdefABCDEF" for c in row["sha256"]):
            raise ValueError("Inventory requires video_id and hexadecimal sha256 columns.")
    if len({r["video_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate video IDs in inventory.")
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public_inventory")
    parser.add_argument("evaluation_inventory")
    parser.add_argument("output")
    parser.add_argument("--public-fingerprints")
    parser.add_argument("--evaluation-fingerprints")
    args = parser.parse_args()
    public, evaluation = inventory(args.public_inventory), inventory(args.evaluation_inventory)
    ids = {r["video_id"] for r in evaluation}
    hashes = {r["sha256"].lower() for r in evaluation}
    results = [{"video_id": r["video_id"], "status": "evaluation-overlap" if r["video_id"] in ids or r["sha256"].lower() in hashes else "unresolved", "training_allowed": False} for r in public]
    candidates = []
    if bool(args.public_fingerprints) != bool(args.evaluation_fingerprints):
        parser.error("Supply both fingerprint inventories or neither.")
    if args.public_fingerprints:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
        from leakage_fingerprints import possible_content_match
        fingerprints = []
        for path in [args.public_fingerprints, args.evaluation_fingerprints]:
            data = json.loads(Path(path).read_text())
            if data.get("format") != "activa-fingerprint-inventory-v1":
                raise ValueError("Unsupported fingerprint inventory.")
            fingerprints.append(data["videos"])
        for first in fingerprints[0]:
            for second in fingerprints[1]:
                if possible_content_match(first, second):
                    candidates.append({"public_id": first["video_id"], "evaluation_id": second["video_id"],
                                       "status": "unresolved-content-candidate", "training_allowed": False})
    Path(args.output).write_text(json.dumps({"content_review_required": True, "videos": results,
                                           "near_duplicate_candidates": candidates}, indent=2) + "\n")

if __name__ == "__main__":
    main()
