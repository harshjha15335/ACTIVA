#!/usr/bin/env python3
"""M3 U/I/A/J experiments. Synthetic acceptance is never a benchmark.

Real studies require a human-reviewed leakage file bound to the official manifest.
This development tool may read caches; production optimizer.py never does.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "code")]
import numpy as np
import config
from contracts import Action, ClipFeats, Importance, VideoInfo
from eval.metrics import action_map, budget_ok, coverage, highlight_f1, union_duration
from optimizer import SelectorSettings, select_variant


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def actions_from_json(rows):
    result = []
    for row in rows:
        start, end = row["segment"] if "segment" in row else (row["start"], row["end"])
        result.append(Action(row["verb"], row["noun"], start, end, row.get("score", 1.0)))
    return result


def synthetic_cases():
    """Handwritten controls; labels and feature vectors are explicitly synthetic."""
    for dataset, peak in (("epic_kitchens_100", 90), ("fpvsum", 100)):
        times = np.arange(0, 120, .5)
        values = np.zeros(len(times), np.float32)
        values[np.abs(times - peak) < 3] = 1
        feats = np.zeros((len(times), 512), np.float32)
        feats[:, 0] = 1
        info = VideoInfo("synthetic_" + dataset, "not_a_real_video", 120, 30, 64, 64)
        predicted = [Action("cut", "tomato", 8, 12, .95), Action("cut", "tomato", 28, 32, .9),
                     Action("wash", "dish", 48, 52, .85)] if dataset == "epic_kitchens_100" else []
        gt = [Action("cut", "tomato", 8, 12, 1), Action("wash", "dish", 48, 52, 1)]
        yield dict(info=info, clip=ClipFeats(times, feats, "SYNTHETIC-not-pretrained"),
                   imp=Importance(times, values), actions=predicted, gt_actions=gt,
                   dataset=dataset, human_times=times, human_importance=values,
                   gt_label="other", pred_label="other", provenance={"synthetic": True})


def load_study(path, review_path, official_manifest):
    from fpvsum_data import load_review
    from perception_cache import read_cache
    study = json.loads(Path(path).read_text())
    if study.get("format") != "activa-m3-study-v1" or not study.get("videos"):
        raise ValueError("Expected nonempty activa-m3-study-v1 videos")
    review = load_review(review_path, official_manifest)
    import csv
    with Path(official_manifest).open(newline="") as stream:
        evaluation_ids = {row["video_id"] for row in csv.DictReader(stream)}
    seen_ids, seen_hashes = set(), set()
    for row in study["videos"]:
        info, clip, imp, manifest = read_cache(row["m2_manifest"])
        decision = review.get(info.video_id, {})
        if (info.video_id in evaluation_ids or info.video_id in seen_ids
                or decision.get("status") != "reviewed-safe"
                or decision.get("sha256") != manifest["input_sha256"]
                or manifest["input_sha256"] in seen_hashes):
            raise ValueError("Duplicate, evaluation-overlap or unreviewed video in development study")
        seen_ids.add(info.video_id)
        seen_hashes.add(manifest["input_sha256"])
        dataset = row["dataset"]
        if dataset not in ("epic_kitchens_100", "fpvsum"):
            raise ValueError("Unknown development dataset")
        action_data = json.loads(Path(row["predictions"]).read_text())
        checkpoint = action_data.get("checkpoint_sha256", "")
        if (action_data["input_sha256"] != manifest["input_sha256"]
                or len(checkpoint) != 64 or any(c not in "0123456789abcdef" for c in checkpoint)
                or action_data.get("source") != "predicted"):
            raise ValueError("Predicted action provenance must match the source video")
        annotation = json.loads(Path(row["annotations"]).read_text())
        if annotation["input_sha256"] != manifest["input_sha256"]:
            raise ValueError("Annotation provenance must match the source video")
        predicted_rows = action_data["actions"]
        if any("score" not in item for item in predicted_rows):
            raise ValueError("Predicted actions need explicit confidence")
        predictions = actions_from_json(predicted_rows)
        truth = actions_from_json(annotation.get("actions", []))
        if any(not 0 <= a.start < a.end <= info.duration for a in predictions + truth):
            raise ValueError("Prediction/annotation intervals must be inside the source video")
        if annotation.get("times") is not None and any(not 0 <= t < info.duration for t in annotation["times"]):
            raise ValueError("Human annotation timestamps must be source-frame times inside the video")
        yield dict(info=info, clip=clip, imp=imp, actions=predictions,
                   gt_actions=truth, dataset=dataset,
                   human_times=annotation.get("times"), human_importance=annotation.get("importance"),
                   gt_label=annotation.get("label"), pred_label=row.get("pred_label"),
                   provenance={"input_sha256": manifest["input_sha256"], "model": clip.model_name,
                               "m2_manifest_sha256": digest(row["m2_manifest"]),
                               "predictions_sha256": digest(row["predictions"]),
                               "checkpoint_sha256": action_data["checkpoint_sha256"],
                               "annotations_sha256": digest(row["annotations"])})


def run_cases(cases, *, coverage_rule, synthetic=False, oracle=False):
    cases = list(cases)
    if not cases:
        raise ValueError("Ablation requires at least one video")
    records = []
    for case in cases:
        info = case["info"]
        actions = case["gt_actions"] if oracle else case["actions"]
        for variant in ("U", "I", "A", "J"):
            started = time.perf_counter()
            highlights = select_variant(info, actions, case["imp"], case["clip"], mode=variant)
            elapsed = time.perf_counter() - started
            cov, f1, acc = None, None, None
            if case["dataset"] == "epic_kitchens_100":
                cov = coverage(case["gt_actions"], highlights, coverage_rule)
            if case["dataset"] == "fpvsum":
                if (case["human_times"] is None) != (case["human_importance"] is None):
                    raise ValueError("F1 needs both aligned human times and importance")
                if case["human_times"] is not None:
                    f1 = highlight_f1(case["human_importance"], case["human_times"], highlights)["f1"]
                if case["gt_label"] is not None and case["pred_label"] is not None:
                    acc = float(case["gt_label"] == case["pred_label"])
            cut = []
            for action in case["actions"]:
                overlap = sum(max(0, min(action.end, h.end)-max(action.start, h.start)) for h in highlights)
                cut.append(0 < overlap < action.end-action.start-1e-9)
            records.append({"video_id": info.video_id, "dataset": case["dataset"], "variant": variant,
                            "coverage_proxy": cov, "f1_proxy": f1, "label_correct": acc,
                            "strict_budget_ok": budget_ok(highlights, info.duration),
                            "short_video_policy_exception": info.duration*config.BUDGET_RATIO < config.MIN_CLIP_SECONDS,
                            "highlight_seconds": union_duration(highlights), "clips": len(highlights),
                            "cut_action_fraction": float(np.mean(cut)) if cut else 0.0,
                            "runtime_seconds": elapsed, "highlights": [asdict(h) for h in highlights],
                            "provenance": case["provenance"]})
    gt = {c["info"].video_id: c["gt_actions"] for c in cases if c["dataset"] == "epic_kitchens_100"}
    pred = {c["info"].video_id: c["actions"] for c in cases if c["dataset"] == "epic_kitchens_100"}
    maps = action_map(gt, pred) if gt else None  # Always actual detector AP, including oracle selection experiments.
    summary = {}
    for variant in ("U", "I", "A", "J"):
        rows = [r for r in records if r["variant"] == variant]
        def mean(key):
            values = [r[key] for r in rows if r[key] is not None]
            return float(np.mean(values)) if values else None
        cov, f1, acc = mean("coverage_proxy"), mean("f1_proxy"), mean("label_correct")
        total = None
        violations = sum(not r["strict_budget_ok"] for r in rows)
        epic_rows = [r for r in rows if r["dataset"] == "epic_kitchens_100"]
        fpv_rows = [r for r in rows if r["dataset"] == "fpvsum"]
        complete = (epic_rows and fpv_rows and all(r["coverage_proxy"] is not None for r in epic_rows)
                    and all(r["f1_proxy"] is not None and r["label_correct"] is not None for r in fpv_rows))
        if all(v is not None for v in (cov, f1, acc, maps)) and not violations and complete:
            total = .4*cov + .25*maps["pair"]["mAP"] + .25*f1 + .1*acc
        summary[variant] = {"coverage_proxy": cov, "f1_proxy": f1, "label_accuracy": acc,
                            "weighted_total_proxy": total, "strict_budget_violations": violations,
                            "short_policy_exceptions": sum(r["short_video_policy_exception"] for r in rows),
                            "mean_clips": mean("clips"), "mean_cut_fraction": mean("cut_action_fraction"),
                            "mean_runtime_seconds": mean("runtime_seconds")}
        summary[variant]["metric_video_counts"] = {key: sum(r[key] is not None for r in rows)
                                                      for key in ("coverage_proxy", "f1_proxy", "label_correct")}
    return {"format": "activa-m3-ablation-v1", "synthetic_only": synthetic,
            "official_metrics_verified": False, "coverage_rule": coverage_rule,
            "selector_inputs": "oracle_gt" if oracle else "predicted",
            "settings": asdict(SelectorSettings()), "seed": 1108,
            "conventions": {"intervals": "half-open seconds", "f1": "annotation grid, ceil(.15*N), earliest ties",
                            "aggregation": "coverage mean on EPIC, F1/label mean on FPVSum, pooled detector AP on EPIC"},
            "videos": len(cases), "dataset_counts": {d: sum(c["dataset"] == d for c in cases)
                                                       for d in ("epic_kitchens_100", "fpvsum")},
            "action_map": maps, "summary": summary, "records": records,
            "platform": platform.platform(), "python": platform.python_version(),
            "hardware": platform.processor()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--study")
    parser.add_argument("--review")
    parser.add_argument("--official-manifest")
    parser.add_argument("--coverage-rule", required=True)
    parser.add_argument("--oracle", action="store_true", help="Oracle selector-input experiment, reported separately")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.study and not (args.review and args.official_manifest):
        parser.error("Real studies need --review and --official-manifest")
    cases = synthetic_cases() if args.synthetic else load_study(args.study, args.review, args.official_manifest)
    result = run_cases(cases, coverage_rule=args.coverage_rule, synthetic=args.synthetic, oracle=args.oracle)
    result["command"] = subprocess.list2cmdline([sys.executable, *sys.argv])
    result["source_sha256"] = {p: digest(ROOT / p) for p in ("code/optimizer.py", "code/candidates.py", "eval/metrics.py", "scripts/run_ablation.py")}
    if args.study:
        result["study_sha256"], result["review_sha256"] = digest(args.study), digest(args.review)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
