"""Safe outputs used when a module crashes or returns something invalid. Owned by M4.

Each fallback is valid under contracts.py and the output schema, so a module
failure on a good video still produces a valid JSON (guide A8).
"""
import math

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
    """'other' (the kit's "none of these" label) if supported, else the first label."""
    label = config.FALLBACK_LABEL if config.FALLBACK_LABEL in categories else categories[0]
    return CategoryResult(label=label, scores={label: 0.0})


def _ceil2(x):
    s = 10 ** config.OUTPUT_DECIMALS
    return math.ceil(x * s - 1e-7) / s


def _floor2(x):
    s = 10 ** config.OUTPUT_DECIMALS
    return math.floor(x * s + 1e-7) / s


def uniform_highlights(info, ratio=None, clip_seconds=None):
    """Evenly spaced clips of clip_seconds (default 6 s), each centred in an equal slice.

    Times sit on the output's 2-decimal grid, every clip is >= MIN_CLIP_SECONDS and
    the total stays within ratio * duration (default BUDGET_SAFETY). When that
    budget is under MIN_CLIP_SECONDS, one MIN_CLIP_SECONDS clip is returned
    (config.SHORT_VIDEO_SINGLE_CLIP), or [] if that policy is off.
    """
    ratio = config.BUDGET_SAFETY if ratio is None else ratio
    length = config.FALLBACK_CLIP_SECONDS if clip_seconds is None else clip_seconds
    duration = info.duration
    limit = _floor2(duration)
    budget = ratio * duration
    if budget >= config.MIN_CLIP_SECONDS:
        count = max(1, int(budget // length))
        if budget < length:
            length = _floor2(budget)
    elif config.SHORT_VIDEO_SINGLE_CLIP and limit > 0:
        count, length = 1, min(config.MIN_CLIP_SECONDS, limit)
    else:
        return []
    slice_len = duration / count
    highlights = []
    for i in range(count):
        start = _ceil2(i * slice_len + (slice_len - length) / 2)
        start = round(max(0.0, min(start, limit - length)), config.OUTPUT_DECIMALS)
        highlights.append(Highlight(start=start, end=round(start + length, config.OUTPUT_DECIMALS),
                                    importance=0.5))
    return highlights
