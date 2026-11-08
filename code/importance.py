"""Training-free importance baselines; human-importance quality is unmeasured."""
import numpy as np
from contracts import Importance
from config import settings

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

def predict_importance(info, clip):
    times, feats = validate_clip(info, clip)
    mode = settings().baseline
    if mode == "constant":
        values = np.full(len(times), 0.5, dtype=np.float32)
    else:
        if mode == "novelty":
            left = np.searchsorted(times, times - 10, side="left")
            right = np.searchsorted(times, times + 10, side="right")
            sums = np.vstack([np.zeros((1, feats.shape[1])), np.cumsum(feats, axis=0)])
            means = (sums[right] - sums[left]) / (right - left)[:, None]
            means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
            raw = np.round(1 - np.sum(feats * means, axis=1), 6)
        elif mode == "motion":
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
            vectors = text_features(["an important moment", "a person interacting with an object", "a blurry boring view of the floor"], settings())
            raw = feats @ ((vectors[0] + vectors[1]) / 2 - vectors[2])
        else:
            raise ValueError(f"Unknown importance baseline: {mode}")
        values = rank_normalize(smooth(raw))
    return Importance(times.copy(), values)
