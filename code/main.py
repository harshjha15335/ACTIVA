"""ACTIVA pipeline entry point. Owned by M4.

    python code/main.py <input.mp4> <output.json>      (normally via ./run.sh)

Each stage runs inside _stage(): if a module raises or returns something that
breaks contracts.py, the full traceback goes to stderr and a safe fallback is
used, so a valid video always yields a valid JSON. An invalid video or missing
vocabulary fails with a non-zero exit code and writes no JSON.
Logs go to stderr only; stdout stays empty.

The demo app must call run() / predict() from here, never a separate path.
"""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataclasses import dataclass, field  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

import config  # noqa: E402
import fallbacks  # noqa: E402
import schema  # noqa: E402
import video_io  # noqa: E402
import vocab as vocab_module  # noqa: E402
from contracts import Action, CategoryResult, ClipFeats, Highlight, Importance, VideoInfo  # noqa: E402

log = logging.getLogger("activa")

EXIT_OK, EXIT_INTERNAL, EXIT_USAGE, EXIT_VIDEO, EXIT_CONFIG = 0, 1, 2, 4, 5


@dataclass
class PipelineResult:
    info: VideoInfo
    clip: ClipFeats
    actions: list
    importance: Importance
    category: CategoryResult
    highlights: list
    output: dict
    timings: dict = field(default_factory=dict)    # stage -> seconds
    fallbacks: list = field(default_factory=list)  # stages that used a fallback


# --- Stage output checks: return None if fine, else a short problem string ---

def _finite(x):
    return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool) and math.isfinite(x)


def check_clip(clip, info):
    if not isinstance(clip, ClipFeats):
        return f"expected ClipFeats, got {type(clip).__name__}"
    times, feats = np.asarray(clip.times), np.asarray(clip.feats)
    if times.ndim != 1 or feats.ndim != 2 or len(feats) != len(times):
        return f"shape mismatch: times {times.shape}, feats {feats.shape}"
    if len(times) and (not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)):
        return "times must be finite and strictly increasing"
    if len(times) and (times[0] < 0 or times[-1] > info.duration):
        return "times outside [0, duration]"
    if not np.all(np.isfinite(feats)):
        return "features contain NaN/inf"
    return None


def check_importance(imp, clip):
    if not isinstance(imp, Importance):
        return f"expected Importance, got {type(imp).__name__}"
    times, values = np.asarray(imp.times), np.asarray(imp.values)
    if times.shape != np.asarray(clip.times).shape or not np.allclose(times, clip.times):
        return "importance times must equal clip.times"
    if values.shape != times.shape:
        return f"values shape {values.shape} != times shape {times.shape}"
    if len(values) and (not np.all(np.isfinite(values)) or values.min() < 0 or values.max() > 1):
        return "values must be finite and in [0, 1]"
    return None


def check_category(cat, categories):
    if not isinstance(cat, CategoryResult):
        return f"expected CategoryResult, got {type(cat).__name__}"
    if cat.label not in categories:
        return f"label {cat.label!r} is not a supported category"
    return None


def check_highlights(highlights, info):
    if not isinstance(highlights, (list, tuple)):
        return f"expected a list, got {type(highlights).__name__}"
    total, previous_end = 0.0, -1.0
    for i, h in enumerate(highlights):
        if not isinstance(h, Highlight):
            return f"item {i}: expected Highlight, got {type(h).__name__}"
        if not (_finite(h.start) and _finite(h.end) and _finite(h.importance)):
            return f"item {i}: non-finite values"
        if not (0 <= h.start < h.end <= info.duration):
            return f"item {i}: need 0 <= start < end <= duration"
        if not 0 <= h.importance <= 1:
            return f"item {i}: importance outside [0, 1]"
        if h.start < previous_end:
            return f"item {i}: not sorted / overlaps the previous clip"
        previous_end = h.end
        total += h.end - h.start
    if total > config.BUDGET_RATIO * info.duration:
        return f"total {total:.2f}s is over the {config.BUDGET_RATIO:.0%} budget"
    return None


def sanitize_actions(actions, info, vocab):
    """Keep valid actions, clamp times into the video, log what was dropped."""
    if not isinstance(actions, (list, tuple)):
        raise TypeError(f"expected a list of Action, got {type(actions).__name__}")
    kept, dropped = [], {}

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    for a in actions:
        if not isinstance(a, Action):
            drop("not an Action")
        elif a.verb not in vocab.verbs or a.noun not in vocab.nouns or not vocab.is_valid_pair(a.verb, a.noun):
            drop("verb/noun not in vocabulary")
        elif not (_finite(a.start) and _finite(a.end) and _finite(a.score)):
            drop("non-finite values")
        else:
            start, end = max(0.0, float(a.start)), min(info.duration, float(a.end))
            if start >= end:
                drop("empty interval inside the video")
                continue
            kept.append(Action(a.verb, a.noun, start, end, min(1.0, max(0.0, float(a.score)))))
    if dropped:
        log.warning("actions: dropped %d of %d: %s", sum(dropped.values()), len(actions), dropped)
    return kept


# --- Pipeline -----------------------------------------------------------------

def _call(module_name, function_name, *args):
    """Import lazily so a module that fails to import also falls back."""
    module = importlib.import_module(module_name)
    return getattr(module, function_name)(*args)


def _stage(name, compute, check, fallback, result_timings, result_fallbacks):
    started = time.perf_counter()
    try:
        value = compute()
        problem = check(value)
        if problem:
            raise ValueError(f"{name} returned invalid output: {problem}")
        status = "ok"
    except Exception:
        log.error("stage %s failed, using fallback:\n%s", name, traceback.format_exc().rstrip())
        value = fallback()
        status = "FALLBACK"
        result_fallbacks.append(name)
    elapsed = time.perf_counter() - started
    result_timings[name] = elapsed
    log.info("stage %-11s %-8s %7.2fs", name, status, elapsed)
    return value


def _seed():
    random.seed(config.SEED)
    np.random.seed(config.SEED)


def run(path) -> PipelineResult:
    """Full pipeline with intermediate results (for the demo app and tests)."""
    _seed()
    started = time.perf_counter()
    info = video_io.inspect_video(path)
    low, high = config.EXPECTED_DURATION_RANGE
    if not low <= info.duration <= high:
        log.warning("video is %.1fs long; expected %.0f-%.0fs", info.duration, low, high)
    vocab = vocab_module.load()
    log.info("video %s: %.2fs, %.2f fps, %dx%d, rotation %d",
             info.video_id, info.duration, info.fps, info.width, info.height, info.rotation)

    timings, used_fallbacks = {}, []

    def stage(*args):
        return _stage(*args, timings, used_fallbacks)

    clip = stage("clip", lambda: _call("clip_features", "extract_clip_features", info),
                 lambda c: check_clip(c, info), lambda: fallbacks.empty_clip(info))
    acts = stage("actions", lambda: sanitize_actions(_call("actions", "detect_actions", info), info, vocab),
                 lambda a: None, lambda: [])
    imp = stage("importance", lambda: _call("importance", "predict_importance", info, clip),
                lambda i: check_importance(i, clip), lambda: fallbacks.flat_importance(clip))
    cat = stage("category", lambda: _call("category", "classify_video", info, clip, imp),
                lambda c: check_category(c, vocab.categories),
                lambda: fallbacks.default_category(vocab.categories))
    hls = stage("highlights", lambda: list(_call("optimizer", "select_highlights", info, acts, imp, clip)),
                lambda h: check_highlights(h, info), lambda: fallbacks.uniform_highlights(info))

    out = schema.to_output(info, cat, acts, hls)
    schema.validate_output(out, info.duration, vocab)
    timings["total"] = time.perf_counter() - started
    log.info("done: %d actions, %d highlights, label %r, %.2fs total%s",
             len(out["actions"]), len(out["highlights"]), out["label"], timings["total"],
             f", fallbacks: {used_fallbacks}" if used_fallbacks else "")
    return PipelineResult(info, clip, acts, imp, cat, hls, out, timings, used_fallbacks)


def predict(path) -> dict:
    return run(path).output


def write_json_atomic(obj, out_path):
    """Write to a temp file in the same folder, then rename over the target."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_path.parent, prefix=f".{out_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("Usage: python code/main.py <input.mp4> <output.json>", file=sys.stderr)
        return EXIT_USAGE
    logging.basicConfig(stream=sys.stderr, level=config.LOG_LEVEL,
                        format="[activa] %(levelname)s %(message)s")
    source, target = args
    try:
        output = predict(source)
    except video_io.VideoError as error:
        log.error("invalid input video: %s", error)
        return EXIT_VIDEO
    except vocab_module.VocabError as error:
        log.error("configuration error: %s", error)
        return EXIT_CONFIG
    except Exception:
        log.error("internal error:\n%s", traceback.format_exc().rstrip())
        return EXIT_INTERNAL
    try:
        write_json_atomic(output, target)
    except OSError as error:
        log.error("cannot write %s: %s", target, error)
        return EXIT_INTERNAL
    log.info("wrote %s", target)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
