#!/usr/bin/env python3
"""Dry-run non-manifest FPVSum source candidates without granting leakage clearance.

Requires locally held official kit/manifest.csv and the pinned upstream
FPVSum_videolist.csv downloaded by the existing video downloader.

Example (local only, no media downloaded):
  python scripts/m2_fpvsum_candidate_dry_run.py \
    --manifest kit/manifest.csv \
    --inventory data/videos/fpvsum/FPVSum_videolist.csv \
    --output tmp/m2/fpvsum_candidates.json

Optional --check-online invokes yt-dlp metadata checks (no videos downloaded).
This tool cannot resolve hidden re-encodes or grant a "reviewed-safe" label.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys

def _read_inventory(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"video_name", "youtube_id"} <= set(reader.fieldnames or []):
            raise ValueError("Inventory requires video_name and youtube_id")
        rows = {}
        for row in reader:
            name = (row.get("video_name") or "").strip()
            source = (row.get("youtube_id") or "").strip()
            if not name or not source or name in rows:
                raise ValueError("Missing/duplicate inventory name or source")
            rows[name] = source
        if not rows:
            raise ValueError("Empty FPVSum inventory")
    return rows

def _manifest_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"video_id", "dataset"} <= set(reader.fieldnames or []):
            raise ValueError("Official manifest requires video_id and dataset")
        return {row["video_id"].strip() for row in reader
                if row["dataset"].strip() == "fpvsum"}

def _online(source: str) -> str:
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--skip-download",
           "--dump-single-json", "--no-warnings", "--socket-timeout", "10",
           "--retries", "1", f"https://www.youtube.com/watch?v={source}"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            return "unavailable-or-check-failed"
        data = json.loads(p.stdout)
        return "metadata-available" if data.get("id") == source else "id-mismatch"
    except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return "check-failed"

def evaluate(inventory_path: Path, manifest_path: Path, check_online: bool = False) -> dict:
    inventory = _read_inventory(inventory_path)
    manifest = _manifest_ids(manifest_path)
    missing = manifest - inventory.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} manifest IDs from upstream inventory")
    protected = {inventory[name] for name in manifest}
    by_source = {}
    for name, source in inventory.items():
        if name not in manifest and source not in protected:
            by_source.setdefault(source, []).append(name)
    # Select one representative per original YouTube source; alternate filenames
    # are not independent validation videos.
    selected = {sorted(names)[0]: source for source, names in by_source.items()}
    alias_groups = {source: sorted(names) for source, names in by_source.items()
                    if len(names) > 1}
    excluded_by_shared_source = sum(
        name not in manifest and source in protected
        for name, source in inventory.items())
    online = {}
    if check_online:
        with ThreadPoolExecutor(max_workers=4) as pool:
            statuses = pool.map(_online, selected.values())
            online = dict(zip(selected.keys(), statuses))
    return {
        "format": "activa-fpvsum-source-id-dry-run-v1",
        "manifest_fpvsum_ids": len(manifest),
        "upstream_inventory_records": len(inventory),
        "shared_youtube_source_exclusions": excluded_by_shared_source,
        "distinct_source_id_candidates": len(selected),
        "duplicate_candidate_source_groups": alias_groups,
        "metadata_online_confirmed": sum(v == "metadata-available" for v in online.values())
            if check_online else None,
        "metadata_online_checked": check_online,
        "human_source_disjointness_review": "NOT DONE",
        "training_or_tuning_permission_granted": False,
        "warning": "Source-ID screening only. Resolve alternate uploads/re-encodes manually before use.",
        "candidates": [{"video_id": name, "youtube_id": source,
                        "online": online.get(name, "not-checked"),
                        "clearance": "UNRESOLVED"}
                       for name, source in sorted(selected.items())],
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-online", action="store_true")
    args = parser.parse_args()
    try:
        report = evaluate(args.inventory, args.manifest, args.check_online)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items()
                          if key in {"distinct_source_id_candidates",
                                     "metadata_online_confirmed",
                                     "human_source_disjointness_review",
                                     "training_or_tuning_permission_granted"}}, indent=2))
    except (OSError, ValueError) as err:
        print(f"BLOCKED: {err}", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
