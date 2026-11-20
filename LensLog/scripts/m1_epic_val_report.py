"""M1 report-only action metrics on the EPIC-100 validation annotations (decision D-a, #25/#26).

    python scripts/m1_epic_val_report.py --outputs outputs \
        --annotations <path outside the repo>/EPIC_100_validation.csv \
        --classes-dir weights/m1 --out <path outside the repo>/m1_epic_val_report.json \
        --run-post-freeze

Run this ONCE, after the final outputs are committed and hashed (21-22 Nov). It is
report-only: nothing it prints may feed back into code, settings or outputs. Without
`--run-post-freeze` it only checks its inputs (which output files match annotated videos,
how many GT rows) and scores nothing. The annotations stay outside the repo; the script
refuses an annotations or report path inside it.

What it reports, pooled over every output whose video_id has validation annotations:
  - verb / noun / pair mAP at tIoU 0.1:0.1:0.5 (eval.metrics.action_map, which uses the
    vendored ActionFormer AP in eval/_vendor/actionformer_metrics.py), pooled across videos
    before AP, as ActionFormer's EPIC evaluation does;
  - highlight coverage of the GT pairs under 'any', 'frac>=0.5' (half) and 'frac>=1.0'
    (full) overlap, mean over videos;
  - SHA-256 of the annotations file and of every scored output, and the git commit.
GT class ids are mapped to keys with EPIC_100_{verb,noun}_classes.csv, the same tables
the engine uses for its predictions. GT rows with stop <= start are skipped and counted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT)]

TIOUS = (0.1, 0.2, 0.3, 0.4, 0.5)
COVERAGE_RULES = {"any": "any", "half": "frac>=0.5", "full": "frac>=1.0"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def timestamp_to_sec(s: str) -> float:
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def load_class_keys(path: Path) -> dict[int, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["id"]): r["key"] for r in csv.DictReader(f)}


def load_gt(annotations: Path, verbs: dict[int, str], nouns: dict[int, str]):
    """video_id -> list of GT actions (segment, verb, noun); plus the count of skipped rows."""
    gt, skipped = {}, 0
    with open(annotations, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            start, end = timestamp_to_sec(r["start_timestamp"]), timestamp_to_sec(r["stop_timestamp"])
            if not end > start:
                skipped += 1
                continue
            gt.setdefault(r["video_id"], []).append({
                "segment": [start, end], "verb": verbs[int(r["verb_class"])], "noun": nouns[int(r["noun_class"])]})
    return gt, skipped


def load_outputs(outputs: Path) -> dict[str, tuple[Path, dict]]:
    found = {}
    for p in sorted(outputs.glob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and isinstance(doc.get("video_id"), str) and "actions" in doc:
            found[doc["video_id"]] = (p, doc)
    return found


def inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def report(outputs: Path, annotations: Path, classes_dir: Path, score: bool) -> dict:
    verbs = load_class_keys(classes_dir / "EPIC_100_verb_classes.csv")
    nouns = load_class_keys(classes_dir / "EPIC_100_noun_classes.csv")
    gt, skipped = load_gt(annotations, verbs, nouns)
    outs = load_outputs(outputs)
    matched = sorted(set(gt) & set(outs))
    rep = {
        "annotations": {"file": annotations.name, "sha256": sha256(annotations), "videos": len(gt),
                        "gt_rows": sum(map(len, gt.values())), "skipped_zero_length_rows": skipped},
        "outputs": {"dir": str(outputs), "files_read": len(outs), "matched_videos": len(matched),
                    "annotated_videos_without_output": sorted(set(gt) - set(outs)),
                    "sha256": {v: sha256(outs[v][0]) for v in matched}},
        "git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
        "tious": list(TIOUS),
        "scored": False,
    }
    if not score:
        return rep

    from eval.metrics import action_map, coverage

    preds = {v: outs[v][1]["actions"] for v in matched}
    gts = {v: gt[v] for v in matched}
    m = action_map(gts, preds, TIOUS)
    rep["mAP"] = {k: {"mAP": m[k]["mAP"], "per_tiou": m[k]["per_tiou"]} for k in ("verb", "noun", "pair")}
    cov = {name: [coverage(gts[v], outs[v][1].get("highlights", []), rule) for v in matched]
           for name, rule in COVERAGE_RULES.items()}
    rep["highlight_coverage_mean"] = {k: (sum(x) / len(x) if x else 0.0) for k, x in cov.items()}
    rep["per_video"] = [{"video_id": v, "n_pred": len(preds[v]), "n_gt": len(gts[v]),
                         **{f"coverage_{k}": cov[k][i] for k in cov}} for i, v in enumerate(matched)]
    rep["scored"] = True
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--annotations", required=True, help="EPIC_100_validation.csv, kept outside the repo")
    ap.add_argument("--classes-dir", default=str(ROOT / "weights" / "m1"))
    ap.add_argument("--out", help="report JSON path, outside the repo (default: print only)")
    ap.add_argument("--run-post-freeze", action="store_true",
                    help="actually score; only after the final outputs are committed and hashed")
    args = ap.parse_args(argv)

    annotations = Path(args.annotations)
    if inside_repo(annotations):
        sys.exit("annotations must stay outside the repo")
    if args.out and inside_repo(Path(args.out)):
        sys.exit("write the report outside the repo")
    rep = report(Path(args.outputs), annotations, Path(args.classes_dir), args.run_post_freeze)
    text = json.dumps(rep, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    if not rep["scored"]:
        print("inputs checked; nothing scored (pass --run-post-freeze after the final outputs are committed)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
