"""Standalone M2 diagnostic output, not the official full ACTIVA JSON schema."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from contracts import Importance
from video_io import inspect_video, iter_frames
from importance import rank_normalize, smooth


def motion_importance(info):
    from PIL import Image
    times, changes, previous = [], [], None
    for timestamp, rgb in iter_frames(info.path, 2.0):
        image = np.asarray(Image.fromarray(rgb).resize((32, 32)), dtype=np.float32) / 255
        changes.append(0 if previous is None else float(np.mean(np.abs(image - previous))))
        times.append(timestamp)
        previous = image
    return Importance(np.asarray(times), rank_normalize(smooth(changes)))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--categories", help="JSON list or object with categories list")
    parser.add_argument("--weights")
    parser.add_argument("--mode", choices=["clip", "motion-only"], default="clip")
    parser.add_argument("--importance", choices=["constant", "novelty", "motion", "text"], default="novelty")
    parser.add_argument("--pooling", choices=["mean", "weighted", "top"], default="mean")
    parser.add_argument("--cache-dir", help="Optional DEVELOPMENT export; inference never reads caches")
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error("Input and output must differ.")
    if args.categories:
        os.environ["ACTIVA_CATEGORIES"] = str(Path(args.categories).resolve())
    if args.weights:
        os.environ["ACTIVA_CLIP_WEIGHTS"] = str(Path(args.weights).resolve())
    os.environ["ACTIVA_IMPORTANCE"] = args.importance
    os.environ["ACTIVA_POOLING"] = args.pooling
    started = time.perf_counter()
    try:
        info = inspect_video(args.input)
        clip, category = None, None
        if args.mode == "motion-only":
            importance = motion_importance(info)
        else:
            from clip_features import extract_clip_features
            from importance import predict_importance
            from category import classify_video
            from config import settings
            if not settings().categories:
                raise ValueError("CLIP mode requires --categories or ACTIVA_CATEGORIES.")
            clip = extract_clip_features(info)
            importance = predict_importance(info, clip)
            category = classify_video(info, clip, importance)
        result = {
            "format": "activa-m2-diagnostic-v1", "video": asdict(info),
            "mode": args.mode, "model": clip.model_name if clip else None,
            "category": asdict(category) if category else None,
            "importance_method": args.importance if clip else "motion",
            "times": importance.times.tolist(), "importance": importance.values.tolist(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(args.output, result)
        if args.cache_dir:
            directory = Path(args.cache_dir)
            (directory / "importance").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / "importance" / f"{info.video_id}.npz", times=importance.times, values=importance.values)
            if clip:
                (directory / "clip").mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory / "clip" / f"{info.video_id}.npz", times=clip.times, feats=clip.feats, model_name=clip.model_name)
        print(f"Wrote {args.output}: {len(importance.times)} samples in {result['elapsed_seconds']:.2f}s")
    except Exception as error:
        parser.exit(1, f"Perception failed: {error}\n")

if __name__ == "__main__":
    main()
