"""M3 candidate intervals and exact integration of M2's irregular time grid.

No video, labels, caches or models are loaded here. Candidates sit on M4's
output grid so inward rounding cannot turn a 4-second clip into a shorter one.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

import config


@dataclass(frozen=True)
class Candidate:
    start: float
    end: float
    source: str
    action_ids: tuple[int, ...]


class ImportanceCurve:
    """Piecewise-linear importance; endpoint values extend to video boundaries."""

    def __init__(self, duration, imp):
        self.duration = float(duration)
        times, values = np.asarray(imp.times, float), np.asarray(imp.values, float)
        if not math.isfinite(self.duration) or self.duration < 0:
            raise ValueError("Video duration must be finite and nonnegative")
        if (times.ndim != 1 or values.shape != times.shape
                or not np.isfinite(times).all() or not np.isfinite(values).all()
                or (np.diff(times) <= 0).any() or (times < 0).any()
                or (times > self.duration).any() or (values < 0).any() or (values > 1).any()):
            raise ValueError("Importance needs aligned increasing times and values in [0,1]")
        self.times, self.values = times, values
        inner = (times > 0) & (times < self.duration)
        self.knots = np.r_[0.0, times[inner], self.duration]
        self.scores = np.interp(self.knots, times, values) if len(times) else np.zeros(len(self.knots))
        widths = np.diff(self.knots)
        self.slopes = np.divide(np.diff(self.scores), widths,
                                out=np.zeros_like(widths), where=widths > 0)
        self.prefix = np.r_[0.0, np.cumsum(widths * (self.scores[:-1] + self.scores[1:]) / 2)]

    def primitive(self, x):
        x = np.clip(np.asarray(x, float), 0, self.duration)
        i = np.clip(np.searchsorted(self.knots, x, side="right") - 1, 0, len(self.slopes) - 1)
        delta = x - self.knots[i]
        return self.prefix[i] + self.scores[i] * delta + self.slopes[i] * delta**2 / 2

    def integral(self, start, end):
        return np.maximum(0, self.primitive(end) - self.primitive(start))

    def budget_mass(self, budget):
        """Optimistic mass on <=20k time bins, with a fractional last bin."""
        n = max(1, min(20000, math.ceil(self.duration / 0.1)))
        edges = np.linspace(0, self.duration, n + 1)
        widths = np.diff(edges)
        masses = self.integral(edges[:-1], edges[1:])
        density = np.divide(masses, widths, out=np.zeros_like(masses), where=widths > 0)
        order = np.argsort(-density, kind="stable")
        take = np.clip(budget - np.r_[0, np.cumsum(widths[order])[:-1]], 0, widths[order])
        return float(np.sum(take * density[order]))


def build_candidates(info, actions, imp, *, max_candidates=500, budget=None, min_conf=0.2):
    """Action/tight/padded/snapped clips, sliding windows and importance peaks.

    Exact interval deduplication preserves close alternatives needed for packing.
    Pool quotas preserve class diversity, salient windows and temporal coverage.
    """
    duration = float(info.duration)
    curve = ImportanceCurve(duration, imp)
    scale = 10**config.OUTPUT_DECIMALS
    limit = math.floor(duration * scale + 1e-7)
    minimum = math.ceil(config.MIN_CLIP_SECONDS * scale - 1e-7)
    if max_candidates < 4:
        raise ValueError("Candidate cap must be at least four")
    budget = max(config.MIN_CLIP_SECONDS, duration * config.BUDGET_SAFETY) if budget is None else budget
    if limit < minimum:
        return []
    pool = {}

    def add(start, end, source):
        start = max(0, min(round(start * scale), limit - minimum))
        end = min(limit, max(start + minimum, round(end * scale)))
        if (end - start) / scale > budget + 1e-9:
            return
        key = (start, end)
        if key not in pool:
            pool[key] = source

    def centered(centre, length, source):
        length = max(config.MIN_CLIP_SECONDS, min(length, duration))
        start = max(0.0, min(centre - length / 2, duration - length))
        add(start, start + length, source)

    for action in actions:
        if action.score < min_conf:
            continue
        middle = (action.start + action.end) / 2
        for pad in (0.0, 0.5):
            centered(middle, action.end - action.start + 2 * pad, "action")
        for length in (config.MIN_CLIP_SECONDS, 6.0, 12.0):
            centered(middle, length, "action")
            centered(action.start + length / 2, length, "action")
            centered(action.end - length / 2, length, "action")

    # Importance maxima include endpoints. Tied plateaus are sampled, not discarded.
    values = curve.values
    if len(values):
        peaks = np.flatnonzero((values >= np.r_[-np.inf, values[:-1]])
                               & (values >= np.r_[values[1:], -np.inf]))
        ranked = sorted(peaks, key=lambda i: (-values[i], curve.times[i]))
        for i in ranked[:max_candidates]:
            for length in (config.MIN_CLIP_SECONDS, 6.0, 12.0):
                centered(curve.times[i], length, "peak")

    for length in sorted({config.MIN_CLIP_SECONDS, 6.0, 12.0, 18.0}):
        if length > budget:
            continue
        for start in np.arange(0, max(0.0, duration - length) + 0.001, length / 2):
            add(start, start + length, "window")
        add(duration - length, duration, "window")
    # Always have one feasible option, including narrow 4-second budgets.
    centered(duration / 2, min(budget, 6.0), "window")
    add(0, config.MIN_CLIP_SECONDS, "window")
    add(duration - config.MIN_CLIP_SECONDS, duration, "window")

    # Snap a bounded set of salient windows to confident intersecting actions.
    windows = [key for key, source in pool.items() if source == "window"]
    windows.sort(key=lambda key: (-float(curve.integral(key[0] / scale, key[1] / scale)), key))
    confident = sorted((a for a in actions if a.score >= 0.5),
                       key=lambda a: (-a.score, a.start, a.end))[:100]
    for a, b in windows[:100]:
        start, end = a / scale, b / scale
        for action in confident:
            if action.start < end and action.end > start:
                left, right = min(start, action.start), max(end, action.end)
                if right - left <= 1.5 * (end - start):
                    add(left, right, "snap")

    keys = sorted(pool)
    starts = np.array([a / scale for a, _ in keys])
    ends = np.array([b / scale for _, b in keys])
    masses = curve.integral(starts, ends)
    # Cheap class-aware quality used only for pool pruning, not final selection.
    class_credit = [dict() for _ in keys]
    ids = [[] for _ in keys]
    for index, action in enumerate(actions):
        overlap = np.maximum(0, np.minimum(ends, action.end) - np.maximum(starts, action.start))
        pair = (action.verb, action.noun)
        for i in np.flatnonzero(overlap > 0):
            ids[i].append(index)
            credit = action.score * overlap[i] / (action.end - action.start)
            class_credit[i][pair] = max(class_credit[i].get(pair, 0), credit)
    credits = [sum(c.values()) for c in class_credit]
    chosen = set()
    # One best candidate per pair first, ordered by confidence, not input order.
    best = {}
    for i, counts in enumerate(class_credit):
        for pair, credit in counts.items():
            key = (-credit, ends[i] - starts[i], -masses[i], keys[i])
            if pair not in best or key < best[pair][0]:
                best[pair] = (key, i)
    for _, i in sorted(best.values())[:max_candidates // 2]:
        chosen.add(i)
    action_order = sorted(range(len(keys)), key=lambda i: (-credits[i], -masses[i], keys[i]))
    for i in action_order:
        if len(chosen) >= max_candidates // 2 or credits[i] == 0:
            break
        chosen.add(i)
    importance_order = sorted(range(len(keys)), key=lambda i: (-masses[i] / (ends[i] - starts[i]),
                                                              -masses[i], keys[i]))
    target = 4 * max_candidates // 5
    for i in importance_order:
        if len(chosen) >= target:
            break
        chosen.add(i)
    remaining = [i for i in range(len(keys)) if i not in chosen]
    slots = max_candidates - len(chosen)
    if slots and remaining:
        chosen.update(remaining[i] for i in np.linspace(0, len(remaining) - 1,
                                                       min(slots, len(remaining)), dtype=int))
    return [Candidate(starts[i], ends[i], pool[keys[i]], tuple(ids[i])) for i in sorted(chosen)]
