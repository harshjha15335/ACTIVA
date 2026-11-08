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
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--model")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require-canonical-vocabulary", action="store_true")
    parser.add_argument("--mode", choices=["clip", "motion-only"], default="clip")
    parser.add_argument("--importance", choices=["constant", "novelty", "motion", "text"], default=None)
    parser.add_argument("--pooling", choices=["mean", "weighted", "top"], default=None)
    parser.add_argument("--cache-dir", help="Optional DEVELOPMENT export; inference never reads caches")
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error("Input and output must differ.")
    if args.categories:
        os.environ["ACTIVA_CATEGORIES"] = str(Path(args.categories).resolve())
    if args.weights:
        os.environ["ACTIVA_CLIP_WEIGHTS"] = str(Path(args.weights).resolve())
    for value, name in [(args.device, "ACTIVA_DEVICE"), (args.batch_size, "ACTIVA_BATCH_SIZE"),
                        (args.model, "ACTIVA_CLIP_MODEL")]:
        if value is not None:
            os.environ[name] = str(value)
    if args.fp16 is not None:
        os.environ["ACTIVA_CLIP_FP16"] = "1" if args.fp16 else "0"
    if args.require_canonical_vocabulary:
        os.environ["ACTIVA_REQUIRE_CANONICAL_VOCAB"] = "1"
    if args.importance is not None:
        os.environ["ACTIVA_IMPORTANCE"] = args.importance
    if args.pooling is not None:
        os.environ["ACTIVA_POOLING"] = args.pooling
    started = time.perf_counter()
    try:
        from perception_options import settings
        cfg = settings()
        info = inspect_video(args.input)
        clip, category = None, None
        if args.mode == "motion-only":
            importance = motion_importance(info)
        else:
            from clip_features import extract_clip_features
            from importance import predict_importance
            from category import classify_video
            from category_vocab import load_categories
            vocabulary = load_categories(cfg)
            clip = extract_clip_features(info)
            importance = predict_importance(info, clip)
            category = classify_video(info, clip, importance)
        result = {
            "format": "activa-m2-diagnostic-v1", "video": asdict(info),
            "mode": args.mode, "model": clip.model_name if clip else None,
            "category": asdict(category) if category else None,
            "vocabulary_source": vocabulary.source if clip else None,
            "canonical_vocabulary": vocabulary.canonical if clip else False,
            "importance_method": cfg.baseline if clip else "motion",
            "pooling": cfg.pooling if clip else None,
            "times": importance.times.tolist(), "importance": importance.values.tolist(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if args.cache_dir:
            if clip is None:
                # Retain clearly labeled legacy motion diagnostics; these are not CLIP caches.
                directory = Path(args.cache_dir) / "importance"
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory / f"{info.video_id}.npz", times=importance.times, values=importance.values)
            else:
                from perception_cache import write_cache
                result["cache_manifest"] = str(write_cache(args.cache_dir, info, clip, importance,
                    {"importance": cfg.baseline, "pooling": cfg.pooling}))
        atomic_json(args.output, result)
        print(f"Wrote {args.output}: {len(importance.times)} samples in {result['elapsed_seconds']:.2f}s")
    except Exception as error:
        parser.exit(1, f"Perception failed: {error}\n")

if __name__ == "__main__":
    main()
