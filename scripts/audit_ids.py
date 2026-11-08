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
    args = parser.parse_args()
    public, evaluation = inventory(args.public_inventory), inventory(args.evaluation_inventory)
    ids = {r["video_id"] for r in evaluation}
    hashes = {r["sha256"].lower() for r in evaluation}
    results = [{"video_id": r["video_id"], "status": "evaluation-overlap" if r["video_id"] in ids or r["sha256"].lower() in hashes else "unresolved", "training_allowed": False} for r in public]
    Path(args.output).write_text(json.dumps({"content_review_required": True, "videos": results}, indent=2) + "\n")

if __name__ == "__main__":
    main()
