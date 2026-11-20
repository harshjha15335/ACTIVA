"""Deterministic action-coverage and importance selection, owned by M3.

Production uses the joint selector. U/I/A/J share the canonical four arguments.
All scoring uses predictions, never annotations or development caches.
"""
from dataclasses import dataclass
import logging
from math import ceil, floor, isfinite

import numpy as np

import config
from candidates import ImportanceCurve, build_candidates
from contracts import Action, ClipFeats, Highlight, Importance, VideoInfo

log = logging.getLogger("activa.selector")


@dataclass(frozen=True)
class SelectorSettings:
    """Development defaults, not tuned/frozen benchmark settings."""
    coverage_weight: float = 0.6
    importance_weight: float = 0.4
    tau_cov: float = 0.5
    min_conf: float = 0.2
    max_candidates: int = 500
    alphas: tuple[float, ...] = (0.0, 0.5, 1.0)

    def __post_init__(self):
        numbers = (self.coverage_weight, self.importance_weight, self.tau_cov, self.min_conf, *self.alphas)
        if (not all(isfinite(x) for x in numbers) or min(self.coverage_weight, self.importance_weight) < 0
                or self.coverage_weight + self.importance_weight <= 0
                or not 0 < self.tau_cov <= 1 or not 0 <= self.min_conf <= 1
                or not self.alphas or any(x < 0 for x in self.alphas)
                or not isinstance(self.max_candidates, int) or self.max_candidates < 4):
            raise ValueError("Invalid selector settings")


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


def _budget(duration):
    scale = 10**config.OUTPUT_DECIMALS
    if duration == 0:
        return 0
    if floor(duration * scale + 1e-7) < ceil(config.MIN_CLIP_SECONDS * scale):
        raise ValueError("No in-range 4-second highlight exists for this video")
    hard = floor(config.BUDGET_RATIO * duration * scale + 1e-9)
    minimum = ceil(config.MIN_CLIP_SECONDS * scale)
    if hard < minimum:
        if not config.SHORT_VIDEO_SINGLE_CLIP:
            raise ValueError("Minimum highlight length conflicts with the duration budget")
        log.warning("provisional short-video policy: one %.2fs clip exceeds the 15%% budget; "
                    "organizer clarification pending", config.MIN_CLIP_SECONDS)
        return minimum
    return max(minimum, min(hard, floor(config.BUDGET_SAFETY * duration * scale + 1e-9)))


def _validate(info, actions, imp, clip, settings):
    curve = ImportanceCurve(info.duration, imp)
    if clip is not None:
        times, feats = np.asarray(clip.times), np.asarray(clip.feats)
        if (times.shape != curve.times.shape or not np.array_equal(times, curve.times)
                or feats.ndim != 2 or len(feats) != len(times) or not np.isfinite(feats).all()):
            raise ValueError("CLIP features and importance must have the same finite time grid")
    # Deduplicate identical instances, keeping maximum confidence. Sorting makes
    # action order and duplicate copies irrelevant to the objective.
    unique = {}
    for action in actions:
        if (not all(isfinite(x) for x in (action.start, action.end, action.score))
                or not 0 <= action.start < action.end <= info.duration
                or not 0 <= action.score <= 1 or not isinstance(action.verb, str)
                or not isinstance(action.noun, str) or not action.verb or not action.noun):
            raise ValueError("Invalid action label, interval or confidence")
        if action.score < settings.min_conf:
            continue
        key = (action.verb, action.noun, action.start, action.end)
        unique[key] = max(unique.get(key, 0), action.score)
    records = [Action(v, n, a, b, score) for (v, n, a, b), score in sorted(unique.items())]
    return curve, records


def _highlights(intervals, curve):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((float(start), float(end)))
    return [Highlight(a, b, float(np.clip(curve.integral(a, b) / (b - a), 0, 1))) for a, b in merged]


def _uniform(duration, budget_ticks, curve):
    if not budget_ticks:
        return []
    scale = 10**config.OUTPUT_DECIMALS
    minimum = ceil(config.MIN_CLIP_SECONDS * scale)
    count = max(1, min(ceil(budget_ticks / (6 * scale)), budget_ticks // minimum))
    length = budget_ticks // count
    limit = floor(duration * scale + 1e-7)
    intervals = []
    for i in range(count):
        start = round((i + 0.5) * limit / count - length / 2)
        start = max(0, min(start, limit - length))
        intervals.append((start / scale, (start + length) / scale))
    return _highlights(intervals, curve)


def select_variant(info, actions, imp, clip, *, mode="J", settings=None):
    """Greedy nonoverlapping interval packing, best true objective across alphas.

    C = mean over DISTINCT pairs of max_instance confidence * g(union_fraction),
    where g(r)=r when r>=tau_cov else 0. Duplicate instances never add credit.
    I = integrated interpolated importance / optimistic budget-mass normalizer.
    Candidate clips cannot overlap; touching clips merge on return. This bounded
    heuristic has no global optimality or submodularity guarantee.
    """
    if mode not in ("U", "I", "A", "J"):
        raise ValueError("Selector mode must be U, I, A or J")
    settings = SelectorSettings() if settings is None else settings
    curve, records = _validate(info, actions, imp, clip, settings)
    budget_ticks = _budget(curve.duration)
    if not budget_ticks:
        return []
    coverage_only = mode == "A" or mode == "J" and settings.importance_weight == 0
    importance_only = mode == "I" or mode == "J" and settings.coverage_weight == 0
    if mode == "U" or coverage_only and not records:
        return _uniform(curve.duration, budget_ticks, curve)
    if importance_only:
        records = []
    scale = 10**config.OUTPUT_DECIMALS
    pool_imp = Importance(curve.times, np.zeros_like(curve.values)) if coverage_only else imp
    pool = build_candidates(info, records, pool_imp, max_candidates=settings.max_candidates,
                            budget=budget_ticks / scale, min_conf=settings.min_conf)
    if not pool:
        raise ValueError("No feasible highlight candidates")
    starts = np.rint([c.start * scale for c in pool]).astype(np.int64)
    ends = np.rint([c.end * scale for c in pool]).astype(np.int64)
    lengths = ends - starts
    masses = curve.integral(starts / scale, ends / scale)
    if not records and not np.any(masses > 1e-12):
        return _uniform(curve.duration, budget_ticks, curve)
    normalizer = max(curve.budget_mass(budget_ticks / scale), 1e-12)
    if records:
        action_starts = np.array([a.start for a in records])
        action_ends = np.array([a.end for a in records])
        confidence = np.array([a.score for a in records])
        action_lengths = action_ends - action_starts
        pairs = [(a.verb, a.noun) for a in records]
        group_starts = np.array([i for i, pair in enumerate(pairs) if i == 0 or pair != pairs[i - 1]])
        overlaps = np.maximum(0, np.minimum(ends[:, None] / scale, action_ends)
                              - np.maximum(starts[:, None] / scale, action_starts))
    else:
        overlaps = np.zeros((len(pool), 0))
    wc = settings.coverage_weight if mode in ("A", "J") and records else 0.0
    wi = settings.importance_weight if mode in ("I", "J") else 0.0
    if mode == "A":
        wc = 1.0
    if mode == "I":
        wi = 1.0
    if not records and mode in ("I", "J"):
        wi = 1.0
    weight = wc + wi
    wc, wi = wc / weight, wi / weight

    def credits(overlap):
        if not records:
            return np.zeros(len(overlap))
        fractions = np.minimum(1, overlap / action_lengths)
        instance = confidence * np.where(fractions >= settings.tau_cov, fractions, 0)
        return np.maximum.reduceat(instance, group_starts, axis=1).mean(axis=1)

    best_score, best_intervals = -1.0, []
    for alpha in settings.alphas:
        selected = []
        active = np.ones(len(pool), dtype=bool)
        used, selected_mass, current_cov = 0, 0.0, 0.0
        accumulated = np.zeros(len(records))
        while True:
            feasible = np.flatnonzero(active & (lengths <= budget_ticks - used))
            if not len(feasible):
                break
            new_cov = credits(accumulated[None, :] + overlaps[feasible])
            gain = wc * (new_cov - current_cov) + wi * masses[feasible] / normalizer
            density = gain / (lengths[feasible] / scale)**alpha
            # lexsort: score, true gain, shorter duration, earlier start/end.
            order = np.lexsort((ends[feasible], starts[feasible], lengths[feasible], -gain, -density))
            index = int(feasible[order[0]])
            if gain[order[0]] <= 1e-12 and selected:
                break
            selected.append(index)
            used += int(lengths[index])
            selected_mass += float(masses[index])
            accumulated += overlaps[index]
            current_cov = float(credits(accumulated[None, :])[0])
            active &= (ends <= starts[index]) | (starts >= ends[index])
            active[index] = False
        value = wc * current_cov + wi * selected_mass / normalizer
        intervals = sorted((int(starts[i]), int(ends[i])) for i in selected)
        if value > best_score + 1e-12 or abs(value - best_score) <= 1e-12 and intervals < best_intervals:
            best_score, best_intervals = value, intervals
    result = _highlights([(a / scale, b / scale) for a, b in best_intervals], curve)
    # Tick arithmetic is exact; float arithmetic is only used at the contract boundary.
    assert sum(b - a for a, b in best_intervals) <= budget_ticks
    return result


def select_uniform(info: VideoInfo, actions: list[Action], imp: Importance,
                   clip: ClipFeats) -> list[Highlight]:
    return select_variant(info, actions, imp, clip, mode="U")


def select_importance(info: VideoInfo, actions: list[Action], imp: Importance,
                      clip: ClipFeats) -> list[Highlight]:
    return select_variant(info, actions, imp, clip, mode="I")


def select_coverage(info: VideoInfo, actions: list[Action], imp: Importance,
                    clip: ClipFeats) -> list[Highlight]:
    return select_variant(info, actions, imp, clip, mode="A")


def select_highlights(info: VideoInfo, actions: list[Action], imp: Importance,
                      clip: ClipFeats) -> list[Highlight]:
    return select_variant(info, actions, imp, clip, mode="J")
