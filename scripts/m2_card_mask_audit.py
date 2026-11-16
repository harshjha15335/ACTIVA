#!/usr/bin/env python3
"""Offline card-mask coverage audit of M2 development caches (NOT quality evidence).

The command checks the classifier proposal on existing genuine cached CLIP
vectors. It cannot determine whether a flagged frame is actually a title card:
that requires source-disjoint video verification and human frame review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from importance import card_frame_mask  # noqa: E402
from perception_cache import read_cache  # noqa: E402
from perception_options import settings  # noqa: E402


def audit(manifests: list[Path]) -> list[dict]:
    cfg = settings()
    rows: list[dict] = []
    for path in manifests:
        info, clip, _, source = read_cache(path)
        mask = card_frame_mask(clip.feats, cfg)
        count = int(mask.sum())
        fraction = count / len(mask)
        rows.append({
            "video_id": info.video_id,
            "source_sha256": source["input_sha256"],
            "cached_checkpoint": clip.model_name,
            "sample_count": len(mask),
            "predicted_card_frames": count,
            "predicted_card_fraction": fraction,
            "candidate_epic_gate_under_1pct": fraction <= 0.01,
            "predicted_card_pts_seconds": [float(t) for t in clip.times[mask]],
            "note": "Algorithmic output only; no human frame-ground-truth validation.",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True,
                        help="One existing, private activa-m2-cache-v1 manifest; repeat for all ten")
    parser.add_argument("--output", type=Path, required=True,
                        help="Local JSON report path under ignored dev_cache/ or tmp/")
    args = parser.parse_args()
    paths = [p.resolve() for p in args.manifest]
    if len(set(paths)) != len(paths):
        parser.error("Duplicate manifest paths; supply each source once.")
    try:
        rows = audit(paths)
        report = {
            "format": "activa-card-mask-rate-audit-v1",
            "quality_metrics_measured": False,
            "human_frame_review_done": False,
            "fpvsum_disjointness_verified": False,
            "go_no_go": "NO-GO pending the issue #27 real-video and selector requirements",
            "videos": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output),
                          "n": len(rows),
                          "candidate_1pct_pass_count": sum(r["candidate_epic_gate_under_1pct"] for r in rows)},
                         indent=2))
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
