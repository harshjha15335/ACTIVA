"""G0 uniform selector stub. Combined optimization is a later milestone.

M4 supplies contracts.py; this module does not redefine the shared contract.
"""
from __future__ import annotations

from math import ceil, isfinite
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from contracts import Action, ClipFeats, Highlight, Importance, VideoInfo


def uniform_intervals(duration, ratio=0.15, clip_length=3.0, min_clip=1.0):
    """Evenly spread clips; floor no budget and never emit sub-minimum clips."""
    duration, ratio = float(duration), float(ratio)
    if not all(isfinite(x) for x in (duration, ratio, clip_length, min_clip)):
        raise ValueError("Selector settings must be finite")
    if duration < 0 or not 0 <= ratio <= 1 or not 0 < min_clip <= clip_length:
        raise ValueError("Invalid duration, budget ratio or clip lengths")
    budget = duration * ratio
    if budget < min_clip:
        return []
    count = max(1, min(ceil(budget / clip_length), int(budget / min_clip)))
    # Leave numerical headroom so interval subtraction cannot exceed budget.
    length = (budget - 1e-10 * max(1.0, duration)) / count
    if length < min_clip:
        return [(0.0, min_clip)]
    spacing = duration / count
    intervals = []
    for i in range(count):
        start = i * spacing + (spacing - length) / 2
        intervals.append((start, start + length))
    return intervals


def select_highlights(info: VideoInfo, actions: list[Action], imp: Importance,
                      clip: ClipFeats) -> list[Highlight]:
    """Return deterministic uniform clips. Actions/features are unused by this stub."""
    from contracts import Highlight  # M4-owned dependency, required only at invocation

    times, values = np.asarray(imp.times, dtype=float), np.asarray(imp.values, dtype=float)
    if (times.ndim != 1 or times.shape != values.shape or not np.isfinite(times).all()
            or not np.isfinite(values).all() or (np.diff(times) <= 0).any()
            or (times < 0).any() or (times > info.duration).any()
            or (values < 0).any() or (values > 1).any()):
        raise ValueError("Importance must use aligned increasing times and values in [0,1]")
    result = []
    for start, end in uniform_intervals(info.duration):
        sample_times = np.linspace(start, end, max(2, ceil((end - start) * 2)), endpoint=False)
        mean = float(np.interp(sample_times, times, values).mean()) if len(times) else 0.0
        result.append(Highlight(start=start, end=end, importance=float(np.clip(mean, 0, 1))))
    return result
