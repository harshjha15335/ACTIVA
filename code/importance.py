"""Training-free importance baselines; human-importance quality is unmeasured."""
import os
import numpy as np
from contracts import Importance
from perception_options import settings

def validate_clip(info, clip):
    times, feats = np.asarray(clip.times), np.asarray(clip.feats)
    if times.ndim != 1 or feats.ndim != 2 or len(times) != len(feats) or not len(times):
        raise ValueError("Expected nonempty times [N] and features [N, D].")
    if not np.isfinite(info.duration) or info.duration <= 0:
        raise ValueError("Video duration must be finite and positive.")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(feats)):
        raise ValueError("Features and timestamps must be finite.")
    if np.any(np.diff(times) <= 0) or np.any(times < 0) or np.any(times >= info.duration):
        raise ValueError("Timestamps must increase strictly within the video.")
    norms = np.linalg.norm(feats, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("Embeddings must be nonzero.")
    return times, feats / norms[:, None]

def rank_normalize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Scores must be a nonempty finite vector.")
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - counts + (counts - 1) / 2
    return ((ranks[inverse] + 0.5) / len(values)).astype(np.float32)

def smooth(values, radius=2):
    radius = min(radius, len(values) - 1)
    if radius <= 0:
        return np.asarray(values)
    return np.convolve(np.pad(values, radius, mode="edge"), np.ones(2 * radius + 1) / (2 * radius + 1), mode="valid")

# Fixed before any non-manifest evaluation. Never tune these against manifest videos.
# Avoid "white screen": legitimate snowy action footage can be mostly white.
CARD_PROMPTS = (
    "a full screen opening title card with large printed words",
    "a video ending credits screen with typography",
    "a brand or production company logo on a plain background",
    "a nearly black video frame showing no scene",
)
FOOTAGE_PROMPTS = (
    "a real world scene filmed by a moving first person action camera",
    "a person participating in a real outdoor sport",
    "a camera recording a real environment with natural scenery",
    "a first person view of a real activity happening",
)


def card_frame_mask(feats, cfg):
    """Zero-shot argmax; only uninterrupted runs of at least two frames qualify.

    This is an opt-in experimental candidate until source-disjoint go/no-go
    checks pass. No ground-truth labels, video IDs or tuned thresholds are used.
    """
    from clip_backend import text_features
    prompts = CARD_PROMPTS + FOOTAGE_PROMPTS
    vectors = np.asarray(text_features(prompts, cfg))
    feats = np.asarray(feats)
    if (feats.ndim != 2 or vectors.shape != (len(prompts), feats.shape[1])
            or not np.all(np.isfinite(vectors))):
        raise ValueError("Card prompt vectors must be finite and match CLIP dimension.")
    proposed = np.argmax(feats @ vectors.T, axis=1) < len(CARD_PROMPTS)
    mask = np.zeros(len(proposed), dtype=bool)
    padded = np.concatenate(([False], proposed, [False])).astype(np.int8)
    edges = np.diff(padded)
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start >= 2:
            mask[start:end] = True
    return mask


def _novelty_raw(times, feats):
    """Original ±10 s novelty, evaluated only on supplied (non-card) rows."""
    left = np.searchsorted(times, times - 10, side="left")
    right = np.searchsorted(times, times + 10, side="right")
    sums = np.vstack([np.zeros((1, feats.shape[1])), np.cumsum(feats, axis=0)])
    means = (sums[right] - sums[left]) / (right - left)[:, None]
    means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
    return np.round(1 - np.sum(feats * means, axis=1), 6)


def predict_importance(info, clip):
    times, feats = validate_clip(info, clip)
    cfg = settings()
    mode = cfg.baseline
    if mode == "constant":
        values = np.full(len(times), 0.5, dtype=np.float32)
    elif mode == "novelty":
        # Default OFF until the issue #27 non-manifest go/no-go passes.
        # ACTIVA_CARD_MASK=0 is the explicit kill switch even after activation.
        flag = os.environ.get("ACTIVA_CARD_MASK", "0")
        if flag not in ("0", "1"):
            raise ValueError("ACTIVA_CARD_MASK must be 0 or 1.")
        mask = card_frame_mask(feats, cfg) if flag == "1" else np.zeros(len(times), bool)
        values = np.zeros(len(times), dtype=np.float32)
        keep = ~mask
        if np.any(keep):
            # Excluding cards from both local ±10s context and rank/smoothing
            # prevents the first real footage frame from being inflated.
            values[keep] = rank_normalize(smooth(_novelty_raw(times[keep], feats[keep])))
        # All-card input deliberately yields zeros, not misleading peaks.
    else:
        if mode == "motion":
            from video_io import iter_frames
            from PIL import Image
            frame_times, changes, previous = [], [], None
            for t, rgb in iter_frames(info.path, 2.0):
                image = np.asarray(Image.fromarray(rgb).resize((32, 32)), dtype=np.float32) / 255
                changes.append(0 if previous is None else np.mean(np.abs(image - previous)))
                frame_times.append(t)
                previous = image
            raw = np.interp(times, frame_times, changes)
        elif mode == "text":
            from clip_backend import text_features
            vectors = text_features(["an important moment", "a person interacting with an object", "a blurry boring view of the floor"], cfg)
            raw = feats @ ((vectors[0] + vectors[1]) / 2 - vectors[2])
        else:
            raise ValueError(f"Unknown importance baseline: {mode}")
        values = rank_normalize(smooth(raw))
    return Importance(times.copy(), values)
