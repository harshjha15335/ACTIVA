"""Safe outputs used when a module crashes or returns something invalid. Owned by M4.

Each fallback is valid under contracts.py and the output schema, so a module
failure on a good video still produces a valid JSON (guide A8).
"""
import numpy as np

import config
from contracts import CategoryResult, ClipFeats, Highlight, Importance


def empty_clip(info):
    return ClipFeats(times=np.zeros(0, dtype=np.float64),
                     feats=np.zeros((0, 0), dtype=np.float32),
                     model_name="fallback/none")


def flat_importance(clip):
    times = np.asarray(clip.times, dtype=np.float64).copy()
    return Importance(times=times, values=np.full(times.shape, 0.5))


def default_category(categories):
    return CategoryResult(label=categories[0])


def uniform_highlights(info, ratio=None, clip_seconds=None):
    """Evenly spaced clips, each centred in an equal slice of the video.

    Total length stays within ratio * duration (default BUDGET_SAFETY).
    Returns [] when the budget cannot fit a MIN_CLIP_SECONDS clip.
    """
    ratio = config.BUDGET_SAFETY if ratio is None else ratio
    length = config.FALLBACK_CLIP_SECONDS if clip_seconds is None else clip_seconds
    budget = ratio * info.duration
    count = int(budget // length)
    if count == 0:
        if budget < config.MIN_CLIP_SECONDS:
            return []
        count, length = 1, budget
    slice_len = info.duration / count
    highlights = []
    for i in range(count):
        start = i * slice_len + (slice_len - length) / 2
        highlights.append(Highlight(start=start, end=start + length, importance=0.5))
    return highlights
