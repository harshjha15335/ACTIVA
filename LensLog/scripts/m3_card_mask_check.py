#!/usr/bin/env python3
"""M3 tune-only no-harm check for an explicitly supplied M2 candidate source.

The source is loaded under a separate module name; inference files stay unchanged.
Requires real local CLIP weights matching the cache identity. No frame review or
FPVSum quality claim is inferred from this check.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "code"), str(ROOT / "scripts")]
import numpy as np
from perception_options import settings
from run_ablation import digest, load_study, run_density_cases


def validate_sources(tune, other):
    if len(tune) != 6 or len(other) != 4:
        raise ValueError("The fixed #28 check requires all six tune and four mask-rate-only sources")
    all_cases = tune + other
    if any(c["dataset"] != "epic_kitchens_100" for c in all_cases):
        raise ValueError("The fixed #28 check requires EPIC sources")
    if (len({c["info"].video_id for c in all_cases}) != 10 or
            len({c["provenance"]["input_sha256"] for c in all_cases}) != 10):
        raise ValueError("Tune and mask-rate-only sources must have distinct IDs and source hashes")


def validate_selector_identity(baseline):
    for source in ("code/optimizer.py", "code/candidates.py", "eval/metrics.py"):
        if baseline.get("source_sha256", {}).get(source) != digest(ROOT/source):
            raise ValueError("Baseline selector/scorer source changed: "+source)


def validate_study_settings(current, baseline):
    # JSON round trips turn dataclass tuple fields (alphas) into lists.
    if (json.dumps(current["variants"], sort_keys=True) != json.dumps(baseline["variants"], sort_keys=True)
            or current["thinning"] != baseline["thinning"]):
        raise ValueError("Baseline settings/thinning changed")


def main():
    from clip_backend import model_identifier
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--tune-study", required=True)
    parser.add_argument("--other-study", required=True, help="Mask-rate audit only; no coverage scored")
    parser.add_argument("--review", required=True)
    parser.add_argument("--official-manifest", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.candidate_commit) != 40 or any(c not in "0123456789abcdef" for c in args.candidate_commit):
        parser.error("Supply the full candidate commit SHA")
    spec = importlib.util.spec_from_file_location("m3_reviewed_importance", args.candidate_source)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    cfg = settings()
    if cfg.baseline != "novelty":
        parser.error("This check is fixed to the existing novelty baseline")
    identity = model_identifier(cfg.model, cfg.weights)
    tune = list(load_study(args.tune_study, args.review, args.official_manifest))
    other = list(load_study(args.other_study, args.review, args.official_manifest))
    validate_sources(tune, other)
    masked_cases, disabled_cases, audits = [], [], []
    for index, case in enumerate(tune + other):
        info, clip = case["info"], case["clip"]
        if identity != clip.model_name:
            raise ValueError("Text encoder checkpoint must exactly match cached image encoder")
        _, normalized = candidate.validate_clip(info, clip)
        mask = candidate.card_frame_mask(normalized, cfg)
        os.environ["ACTIVA_CARD_MASK"] = "0"
        disabled = candidate.predict_importance(info, clip)
        os.environ["ACTIVA_CARD_MASK"] = "1"
        enabled = candidate.predict_importance(info, clip)
        audits.append({"video_id": info.video_id, "source_sha256": case["provenance"]["input_sha256"],
                       "split": "tune" if index < len(tune) else "mask-rate-only",
                       "frames": len(mask), "masked_frames": int(mask.sum()),
                       "masked_fraction": float(np.mean(mask)), "epic_rate_gate": bool(np.mean(mask) <= .01),
                       "disabled_vs_cache_max_abs": float(np.max(np.abs(disabled.values-case["imp"].values))),
                       "enabled_vs_disabled_max_abs": float(np.max(np.abs(enabled.values-disabled.values)))})
        if index < len(tune):
            masked_cases.append({**case, "imp": enabled})
            disabled_cases.append({**case, "imp": disabled})
    baseline = json.loads(Path(args.baseline_report).read_text())
    validate_selector_identity(baseline)
    if (baseline["format"] != "activa-m3-density-study-v1" or baseline["videos"] != len(tune)
            or set(baseline["densities"]) != {"2", "3.5", "6", "all"}):
        raise ValueError("Expected fixed four-density tune baseline")
    expected_ids = {c["info"].video_id for c in tune}
    for density in baseline["densities"].values():
        if {r["video_id"] for r in density["records"]} != expected_ids:
            raise ValueError("Baseline must contain exactly the tune sources")
        for row in density["records"]:
            current = next(c for c in tune if c["info"].video_id == row["video_id"])
            if row["provenance"] != current["provenance"]:
                raise ValueError("Baseline input provenance changed")
    enabled = run_density_cases(masked_cases, coverage_rule=baseline["coverage_rule"])
    validate_study_settings(enabled, baseline)
    disabled = run_density_cases(disabled_cases, coverage_rule=baseline["coverage_rule"])
    comparisons = {}
    for density, report in enabled["densities"].items():
        before = baseline["densities"][density]["summary"]["J0.6"]["coverage_proxy"]
        after = report["summary"]["J0.6"]["coverage_proxy"]
        off = disabled["densities"][density]["summary"]["J0.6"]["coverage_proxy"]
        comparisons[density] = {"baseline_coverage": before, "masked_coverage": after,
                                "recomputed_disabled_coverage": off,
                                "delta_percentage_points": (after-before)*100,
                                "isolated_mask_delta_percentage_points": (after-off)*100,
                                "disabled_vs_cache_delta_percentage_points": (off-before)*100,
                                "no_harm_gate": abs(after-before) <= .005+1e-12 and abs(after-off) <= .005+1e-12}
    result = {"format": "activa-m3-card-mask-check-v1", "candidate_commit": args.candidate_commit,
              "candidate_source_sha256": digest(args.candidate_source), "checkpoint_identity": identity,
              "baseline_report_sha256": digest(args.baseline_report), "audits": audits,
              "comparisons": comparisons, "epic_rate_pass": all(r["epic_rate_gate"] for r in audits),
              "selector_no_harm_pass": all(r["no_harm_gate"] for r in comparisons.values()),
              "human_frame_review_done": False, "fpvsum_quality_measured": False,
              "go_no_go": "NO-GO until all #27 human/non-manifest gates pass",
              "masked_tune_report": enabled, "recomputed_disabled_tune_report": disabled}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("tune_report")}, indent=2))


if __name__ == "__main__":
    main()
