"""M1 tuning data for M3 (#15/#16): real action predictions + canonical GT on the 10 agreed
EPIC-100 *train* videos (none in the Samsung manifest). Dev tooling; never used by run.sh.

On a machine with the M1 assets (`scripts/setup_m1_assets.py`), the kit in `kit/` and a GPU:

    python scripts/download_videos.py epic --manifest kaggle/m1_tuning_manifest.csv --out data/tuning_videos
    ACTIVA_DEVICE=cuda python kaggle/m1_tuning_export.py --videos data/tuning_videos --out dev_cache/m1_tuning

Per video it writes <out>/<video_id>.json with:
  predictions  `actions.detect_actions(info)` output: segment, verb, noun, score (+ class ids)
  gt           EPIC_100_train.csv rows mapped with the SAME class-id -> key tables as the predictions
  source       file MD5 + SHA-256, duration, fps, size, rotation
  engine       commit, checkpoint SHA-256s (weights/m1/ASSETS.json), FrameSource / Preprocess /
               grid / pairing settings, ActionFormer test_cfg, device, runtime
plus SUMMARY.json and a local proxy mAP (eval.metrics.action_map) per split. The local scorer is
a proxy, not the organisers' one. ActionFormer was trained on EPIC-100 train, including these
videos, so action quality here is optimistic.
"""
import argparse
import csv
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT)]

TUNE = ["P01_06", "P02_04", "P08_06", "P21_03", "P24_04", "P26_21"]
HELD_OUT = ["P03_19", "P11_15", "P22_13", "P30_06"]
ANNOTATIONS_COMMIT = "ea8b40457a400c3fffa1c7f406ef3dc169cc2522"   # as checked by M3 on #15
TRAIN_CSV_URL = ("https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/"
                 f"{ANNOTATIONS_COMMIT}/EPIC_100_train.csv")
TRAIN_CSV_SHA256 = "a3a7ef2e397bd8af12bd0496b09d61b1f2c1f3f6e1726fa86af50a27f4855e87"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            md5.update(b)
            sha.update(b)
    return md5.hexdigest(), sha.hexdigest()


def timestamp(s):
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def train_csv(out):
    path = out / "EPIC_100_train.csv"
    if not path.exists():
        urllib.request.urlretrieve(TRAIN_CSV_URL, path)
    got = hashes(path)[1]
    if got != TRAIN_CSV_SHA256:
        raise SystemExit(f"{path}: SHA-256 {got} != expected {TRAIN_CSV_SHA256}")
    return path


def ground_truth(csv_path, videos, verb_keys, noun_keys):
    """EPIC train rows -> {video_id: [gt]} with keys from the same class tables as the predictions."""
    gt = {v: [] for v in videos}
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["video_id"] in gt:
                vi, ni = int(r["verb_class"]), int(r["noun_class"])
                gt[r["video_id"]].append({
                    "segment": [round(timestamp(r["start_timestamp"]), 3), round(timestamp(r["stop_timestamp"]), 3)],
                    "verb": verb_keys[vi], "noun": noun_keys[ni], "verb_class": vi, "noun_class": ni,
                    "narration_id": r["narration_id"], "narration": r["narration"]})
    for v in videos:
        if not gt[v]:
            raise SystemExit(f"no ground truth for {v} in {csv_path}")
        gt[v].sort(key=lambda g: g["segment"][0])
    return gt


def find_video(root, vid):
    hits = [p for p in Path(root).rglob("*") if p.is_file() and p.stem == vid and p.suffix.lower() == ".mp4"]
    if not hits:
        raise SystemExit(f"{vid}.mp4 not found under {root}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder holding the 10 MP4s (any depth)")
    ap.add_argument("--out", default=str(ROOT / "dev_cache" / "m1_tuning"))
    ap.add_argument("--only", nargs="*", help="subset of video IDs (default: all 10)")
    ap.add_argument("--gt-only", action="store_true", help="write GT + source metadata only, no model run")
    args = ap.parse_args()

    import actions
    import config
    from action_config import GRID, PAIRING, engine_assets
    from slowfast_features import FrameSource, Preprocess

    videos = args.only or TUNE + HELD_OUT
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assets = engine_assets()
    verb_keys = actions.load_class_keys(assets.classes_csv("verb"))
    noun_keys = actions.load_class_keys(assets.classes_csv("noun"))
    gt = ground_truth(train_csv(out), videos, verb_keys, noun_keys)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    asset_record = json.loads((assets.root / "ASSETS.json").read_text()) if (assets.root / "ASSETS.json").exists() else None
    vi = {k: i for i, k in enumerate(verb_keys)}
    ni = {k: i for i, k in enumerate(noun_keys)}

    summary = []
    for v in videos:
        import video_io
        path = find_video(args.videos, v)
        md5, sha = hashes(path)
        info = video_io.inspect_video(path)
        rec = {"video_id": v, "split": "tune" if v in TUNE else "held-out",
               "source": {"file": path.name, "md5": md5, "sha256": sha, "duration_sec": info.duration,
                          "fps": info.fps, "width": info.width, "height": info.height, "rotation": info.rotation},
               "timebase": "D3: seconds = decoded PTS - stream start_time (video_io); not trimmed or rebased",
               "gt_source": f"epic-kitchens-100-annotations@{ANNOTATIONS_COMMIT} EPIC_100_train.csv",
               "vocab_note": "class id -> key via EPIC_100_{verb,noun}_classes.csv for BOTH predictions and GT",
               "gt": gt[v]}
        if not args.gt_only:
            import torch
            from eval.metrics import action_map
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            acts = actions.detect_actions(info)
            secs = time.time() - t0
            preds = [{"segment": [a.start, a.end], "verb": a.verb, "noun": a.noun, "score": round(float(a.score), 6),
                      "verb_class": vi[a.verb], "noun_class": ni[a.noun]} for a in acts]
            m = action_map(gt[v], preds)
            eng = actions._ENGINE
            rec.update({
                "predictions": preds,
                "engine": {"activa_commit": commit, "assets": asset_record, "device": config.DEVICE,
                           "nms_impl": eng["verb"].nms_impl,
                           "frame_source": dataclasses.asdict(FrameSource()), "preprocess": dataclasses.asdict(Preprocess()),
                           "grid": dataclasses.asdict(GRID), "pairing": dataclasses.asdict(PAIRING)},
                "runtime": {"detect_actions_sec": round(secs, 1),
                            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None},
                "local_proxy_mAP": {k: m[k]["mAP"] for k in ("pair", "verb", "noun")}})
        (out / f"{v}.json").write_text(json.dumps(rec, indent=1))
        row = {"video_id": v, "split": rec["split"], "duration": round(info.duration, 1), "n_gt": len(gt[v]),
               "sha256": sha}
        if not args.gt_only:
            row.update({"n_pred": len(rec["predictions"]), "sec": rec["runtime"]["detect_actions_sec"],
                        **{f"mAP_{k}": round(x, 4) for k, x in rec["local_proxy_mAP"].items()}})
        summary.append(row)
        log(row)

    if not args.gt_only:
        from eval.metrics import action_map
        for split, vids in (("tune", TUNE), ("held-out", HELD_OUT)):
            vids = [x for x in vids if x in videos]
            if vids:
                recs = {x: json.loads((out / f"{x}.json").read_text()) for x in vids}
                m = action_map({x: r["gt"] for x, r in recs.items()}, {x: r["predictions"] for x, r in recs.items()})
                summary.append({"split_total": split, **{f"mAP_{k}": round(m[k]["mAP"], 4) for k in ("pair", "verb", "noun")}})
                log(summary[-1])
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    log(f"wrote {len(videos)} files + SUMMARY.json to {out}")


if __name__ == "__main__":
    main()
