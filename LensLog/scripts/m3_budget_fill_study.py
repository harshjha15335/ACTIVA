#!/usr/bin/env python3
"""Fixed #34/#37 budget-fill experiment. Never imported by inference.

Extend existing serialized J clips right then left, last clip first. The safety
fill is diagnostic; near-limit fill is the sole predeclared candidate. Load the
four held-out sources only after the six-tune gate and an exclusive lock write.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "code"), str(ROOT / "scripts")]
import numpy as np
import config
from contracts import CategoryResult, Highlight
from eval.metrics import coverage, union_duration
from main import normalize_highlights
from optimizer import SelectorSettings, select_variant
import schema
from scripts.run_ablation import digest, load_study, thin_actions

DENSITIES = (2, 3.5, 6, "all")
RULE = ("near_limit must strictly improve mean serialized half-overlap coverage at every "
        "density on tune, then at every density on held-out; all geometry checks pass. "
        "Otherwise retain production J. safety_fill is diagnostic only.")
METHOD = ("Serialize current J through normalize_highlights and schema.to_output; extend "
          "existing clips last-to-first, right then left, with one-tick gaps. No new clips. "
          "safety_fill target=max(serialized baseline ticks,floor(BUDGET_SAFETY*duration*scale)); "
          "near_limit target=floor(BUDGET_RATIO*min(actual,rounded duration)*scale)-1.")


def near_limit_ticks(duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Expected positive finite duration")
    scale = 10**config.OUTPUT_DECIMALS
    return math.floor(config.BUDGET_RATIO*min(duration, round(duration, config.OUTPUT_DECIMALS))*scale + 1e-9)-1


def extend_intervals(highlights, duration, target_ticks):
    """Pure tick arithmetic: preserve every baseline frame and add no new clip."""
    if not highlights or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Expected nonempty valid baseline and duration")
    if not isinstance(target_ticks, int):
        raise ValueError("Target must be integer ticks")
    scale = 10**config.OUTPUT_DECIMALS
    limit = math.floor(duration*scale + 1e-7)
    slots = []
    for h in highlights:
        if not all(math.isfinite(v) for v in (h.start, h.end, h.importance)):
            raise ValueError("Expected finite highlights")
        a, b = round(h.start*scale), round(h.end*scale)
        if (abs(a-h.start*scale) > 1e-6 or abs(b-h.end*scale) > 1e-6 or
                not 0 <= a < b <= limit or b-a < math.ceil(config.MIN_CLIP_SECONDS*scale) or
                slots and slots[-1][1] >= a):
            raise ValueError("Expected grid-aligned disjoint baseline clips >=minimum length")
        slots.append([a, b])
    remaining = target_ticks-sum(b-a for a, b in slots)
    if remaining < 0:
        raise ValueError("Target would shrink baseline")
    for i in reversed(range(len(slots))):
        right = limit if i+1 == len(slots) else slots[i+1][0]-1
        take = min(remaining, right-slots[i][1])
        slots[i][1] += take
        remaining -= take
        left = 0 if i == 0 else slots[i-1][1]+1
        take = min(remaining, slots[i][0]-left)
        slots[i][0] -= take
        remaining -= take
    return [Highlight(a/scale, b/scale, h.importance) for (a, b), h in zip(slots, highlights)]


def export_highlights(info, highlights):
    normalized = normalize_highlights(highlights, info)
    if not normalized:
        raise ValueError("Empty production highlights cannot be a budget-fill baseline")
    output = schema.to_output(info, CategoryResult("cooking", {"cooking": 1}), [], normalized,
                              team_id="development-study")
    return [Highlight(*row["segment"], h.importance) for row, h in zip(output["highlights"], normalized)]


def geometry_checks(info, highlights):
    seconds = sum(h.end-h.start for h in highlights)
    scale = 10**config.OUTPUT_DECIMALS
    return {"nonempty": bool(highlights),
            "finite": all(math.isfinite(v) for h in highlights for v in (h.start, h.end)),
            "in_range": all(0 <= h.start < h.end <= info.duration for h in highlights),
            "literal_minimum_length": all(h.end-h.start >= config.MIN_CLIP_SECONDS for h in highlights),
            "sorted_disjoint": all(a.end < b.start for a, b in zip(highlights, highlights[1:])),
            "grid_aligned": all(abs(v*scale-round(v*scale)) <= 1e-6 for h in highlights for v in (h.start, h.end)),
            "actual_duration_budget": seconds <= config.BUDGET_RATIO*info.duration,
            "output_duration_budget": seconds <= config.BUDGET_RATIO*round(info.duration, config.OUTPUT_DECIMALS)}


def run_study(cases):
    cases = list(cases)
    if not cases or any(c["dataset"] != "epic_kitchens_100" for c in cases):
        raise ValueError("Budget study requires EPIC cases")
    reports = {}
    scale = 10**config.OUTPUT_DECIMALS
    for density in DENSITIES:
        records = []
        for c in cases:
            info = c["info"]
            actions = thin_actions(c["actions"], info.duration, density)
            raw = select_variant(info, actions, c["imp"], c["clip"], mode="J")
            baseline = export_highlights(info, raw)
            used = sum(round((h.end-h.start)*scale) for h in baseline)
            targets = {"safety_fill": max(used, math.floor(config.BUDGET_SAFETY*info.duration*scale+1e-9)),
                       "near_limit": near_limit_ticks(info.duration)}
            variants = {"baseline": baseline}
            for name, target in targets.items():
                variants[name] = export_highlights(info, extend_intervals(baseline, info.duration, target))
            for variant, highlights in variants.items():
                seconds = union_duration(highlights)
                records.append({"video_id": info.video_id, "variant": variant,
                                "coverage_proxy": coverage(c["gt_actions"], highlights, "frac>=0.5"),
                                "raw_baseline_coverage_proxy": coverage(c["gt_actions"], raw, "frac>=0.5"),
                                "geometry": geometry_checks(info, highlights), "clips": len(highlights),
                                "highlight_seconds": seconds, "added_seconds": seconds-union_duration(baseline),
                                "fraction_of_video": seconds/info.duration,
                                "allowed_budget_fill": seconds/(config.BUDGET_RATIO*info.duration),
                                "target_ticks": used if variant == "baseline" else targets[variant],
                                "retained_actions": len(actions), "duration_sec": info.duration,
                                "highlights": [asdict(h) for h in highlights], "provenance": c["provenance"]})
        summary = {}
        for variant in ("baseline", "safety_fill", "near_limit"):
            rows = [r for r in records if r["variant"] == variant]
            summary[variant] = {"coverage_proxy": float(np.mean([r["coverage_proxy"] for r in rows])),
                                "invalid_selections": sum(not all(r["geometry"].values()) for r in rows),
                                "mean_allowed_budget_fill": float(np.mean([r["allowed_budget_fill"] for r in rows])),
                                "mean_added_seconds": float(np.mean([r["added_seconds"] for r in rows]))}
        reports[str(density)] = {"summary": summary, "records": records}
    return {"format": "activa-m3-budget-fill-v1", "official_metrics_verified": False,
            "coverage_rule": "frac>=0.5", "method": METHOD, "rule": RULE,
            "selector_settings": asdict(SelectorSettings()), "budget_ratio": config.BUDGET_RATIO,
            "budget_safety": config.BUDGET_SAFETY, "videos": len(cases), "densities": reports}


def passes_gate(report):
    return all(d["summary"]["near_limit"]["coverage_proxy"] > d["summary"]["baseline"]["coverage_proxy"]+1e-12
               and all(s["invalid_selections"] == 0 for s in d["summary"].values())
               for d in report["densities"].values()) and set(report["densities"]) == {"2", "3.5", "6", "all"}


def write_exclusive(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True, help="Exactly six tune cases")
    p.add_argument("--held-out-study", help="Only loaded if the tune gate passes")
    p.add_argument("--review", required=True)
    p.add_argument("--official-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ("tune.json", "selection-lock.json", "held-out.json", "decision.json")):
        p.error("Refusing to overwrite an existing study stage")
    sources = {name: digest(ROOT/name) for name in ("scripts/m3_budget_fill_study.py", "scripts/run_ablation.py",
               "code/optimizer.py", "code/candidates.py", "code/schema.py", "code/main.py", "eval/metrics.py", "code/config.py")}
    cases = list(load_study(args.study, args.review, args.official_manifest))
    if len(cases) != 6:
        p.error("The fixed study requires exactly six tune sources")
    tune = run_study(cases)
    tune["source_sha256"] = sources
    tune["study_sha256"], tune["review_sha256"] = digest(args.study), digest(args.review)
    write_exclusive(out/"tune.json", tune)
    lock = {"format": "activa-m3-budget-lock-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": METHOD, "rule": RULE, "tune_sha256": digest(out/"tune.json"),
            "source_sha256": sources, "tune_pass": passes_gate(tune),
            "held_out_status": "new fill condition not evaluated",
            "exposure": "Existing full-density/density held-out results known from #30/#32; partition reuse."}
    write_exclusive(out/"selection-lock.json", lock)
    decision = {"tune_pass": lock["tune_pass"], "held_out_evaluated": False, "held_out_pass": None,
                "production_change": False, "decision": "retain production J", "rule": RULE,
                "selection_lock_sha256": digest(out/"selection-lock.json"),
                "platform": platform.platform(), "python": platform.python_version(),
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    if lock["tune_pass"] and args.held_out_study:
        other = list(load_study(args.held_out_study, args.review, args.official_manifest))
        if (len(other) != 4 or len({c["info"].video_id for c in cases+other}) != 10 or
                len({c["info"].video_id.split("_")[0] for c in cases+other}) != 10 or
                len({c["provenance"]["input_sha256"] for c in cases+other}) != 10):
            raise ValueError("Expected four distinct participant/source-disjoint held-out cases")
        held = run_study(other)
        held["source_sha256"] = sources
        held["study_sha256"] = digest(args.held_out_study)
        write_exclusive(out/"held-out.json", held)
        decision.update(held_out_evaluated=True, held_out_pass=passes_gate(held))
        if decision["held_out_pass"]:
            decision["decision"] = "candidate meets EPIC gate; M4 integration/approval required, production unchanged"
    elif lock["tune_pass"]:
        decision["decision"] = "tune gate passed; held-out confirmation missing, production unchanged"
    write_exclusive(out/"decision.json", decision)
    print(json.dumps({"decision": decision, "tune": {d: r["summary"] for d, r in tune["densities"].items()}}, indent=2))


if __name__ == "__main__":
    main()
