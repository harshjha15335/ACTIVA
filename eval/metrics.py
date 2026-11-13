"""Pure local scorers. All intervals are half-open [start, end), in seconds.

These are development proxies until checked against the official student kit.
Actions/highlights may be dictionaries or objects exposing the documented fields.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import ceil, isfinite

import numpy as np
import pandas as pd

from ._vendor.actionformer_metrics import compute_average_precision_detection


def _field(item, name):
    return item[name] if isinstance(item, Mapping) else getattr(item, name)


def _interval(item):
    # Support canonical internal objects and the actual kit's output boundary.
    if isinstance(item, Mapping) and "segment" in item:
        if "start" in item or "end" in item or len(item["segment"]) != 2:
            raise ValueError("Ambiguous or malformed segment")
        start, end = map(float, item["segment"])
    else:
        start, end = float(_field(item, "start")), float(_field(item, "end"))
    if not (isfinite(start) and isfinite(end) and 0 <= start < end):
        raise ValueError("Intervals require finite 0 <= start < end")
    return start, end


def _ratio(value, name):
    value = float(value)
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def merge_intervals(intervals):
    """Validate and union intervals; overlapping and adjacent pieces are merged."""
    merged = []
    for start, end in sorted(_interval(item) for item in intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def union_duration(highlights):
    return sum(end - start for start, end in merge_intervals(highlights))


def budget_ok(highlights, duration, ratio=0.15):
    """Require sorted, non-overlapping, in-range clips and union <= budget.

    Invalid intervals/configuration return False; no budget tolerance is assumed.
    Adjacent intervals are allowed. An empty selection is valid at duration zero.
    """
    try:
        duration, ratio = float(duration), _ratio(ratio, "ratio")
        if not isfinite(duration) or duration < 0:
            return False
        highlights = list(highlights)
        previous_end = 0.0
        for item in highlights:
            start, end = _interval(item)
            if start < previous_end or end > duration:
                return False
            previous_end = end
        return union_duration(highlights) <= duration * ratio
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return False


def coverage(gt_actions, highlights, rule):
    """Fraction of distinct GT pairs covered by at least one GT instance.

    rule is explicit: 'any', 'centre', or 'frac>=X' for X in (0, 1].
    Fractional coverage sums intersection with the highlight UNION. Empty GT = 0.
    """
    threshold = None
    if rule not in ("any", "centre"):
        if not isinstance(rule, str) or not rule.startswith("frac>="):
            raise ValueError("Unknown coverage rule")
        threshold = _ratio(float(rule[6:]), "coverage fraction")
        if threshold == 0:
            raise ValueError("Coverage fraction must be positive")
    intervals = merge_intervals(highlights)
    classes, covered = set(), set()
    for action in gt_actions:
        start, end = _interval(action)
        pair = (_field(action, "verb"), _field(action, "noun"))
        classes.add(pair)
        overlap = sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals)
        if rule == "any":
            hit = overlap > 0
        elif rule == "centre":
            hit = any(a <= (start + end) / 2 < b for a, b in intervals)
        else:
            hit = overlap / (end - start) >= threshold
        if hit:
            covered.add(pair)
    return len(covered) / len(classes) if classes else 0.0


def _action_rows(actions, predicted):
    # A sequence denotes ONE video; a mapping is video_id -> sequence of actions.
    groups = actions if isinstance(actions, Mapping) else {"__single_video__": actions}
    rows = []
    for video_id, items in groups.items():
        if not isinstance(video_id, str):
            raise ValueError("Video IDs must be strings")
        for item in items:
            start, end = _interval(item)
            verb, noun = _field(item, "verb"), _field(item, "noun")
            if not isinstance(verb, str) or not isinstance(noun, str):
                raise ValueError("Action labels must be strings")
            row = {"video-id": video_id, "t-start": start, "t-end": end,
                   "verb": verb, "noun": noun, "pair": (verb, noun)}
            if predicted:
                row["score"] = _ratio(_field(item, "score"), "action confidence")
            rows.append(row)
    return rows


def action_map(gt_actions, pred_actions, tious=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """ANet-style AP for pair, verb and noun classes, macro-averaged over GT classes.

    Supply sequences for one video, or mappings video_id -> actions for dataset AP.
    Dataset AP pools scored predictions across videos BEFORE computing AP.
    Missing confidence raises; predictions of GT-absent classes are excluded (ANet).
    No GT gives zero. Stable input order breaks equal-confidence ties.
    """
    if isinstance(gt_actions, Mapping) != isinstance(pred_actions, Mapping):
        raise ValueError("GT and predictions must both be sequences or video mappings")
    thresholds = np.asarray(list(tious), dtype=float)
    if (thresholds.ndim != 1 or not len(thresholds)
            or not np.isfinite(thresholds).all()
            or ((thresholds <= 0) | (thresholds > 1)).any()):
        raise ValueError("tIoUs must be a nonempty sequence in (0, 1]")
    gt, pred = _action_rows(gt_actions, False), _action_rows(pred_actions, True)
    result = {}
    for kind in ("pair", "verb", "noun"):
        classes = sorted({row[kind] for row in gt})
        aps = []
        per_class = []
        for label in classes:
            gt_frame = pd.DataFrame([row for row in gt if row[kind] == label])
            pred_frame = pd.DataFrame([row for row in pred if row[kind] == label])
            ap = compute_average_precision_detection(gt_frame, pred_frame, thresholds)
            aps.append(ap)
            per_class.append({"label": list(label) if kind == "pair" else label,
                              "AP": ap.tolist()})
        per_tiou = np.mean(aps, axis=0) if aps else np.zeros(len(thresholds))
        result[kind] = {"mAP": float(per_tiou.mean()), "tious": thresholds.tolist(),
                        "per_tiou": per_tiou.tolist(), "per_class": per_class}
    return result


def highlight_f1(human_importance, times, highlights, top_ratio=0.15):
    """Sample-level P/R/F1 on the supplied annotation grid.

    Top ceil(top_ratio*N) samples are positive; ties prefer earlier timestamps.
    This is rank-based (including all-zero curves), not time-weighted on uneven
    grids. Empty arrays and empty prediction yield zeros. No interpolation occurs.
    """
    values, times = np.asarray(human_importance, dtype=float), np.asarray(times, dtype=float)
    top_ratio = _ratio(top_ratio, "top_ratio")
    if (values.ndim != 1 or times.ndim != 1 or values.shape != times.shape
            or not np.isfinite(values).all() or not np.isfinite(times).all()
            or (values < 0).any() or (values > 1).any()
            or (times < 0).any() or (np.diff(times) <= 0).any()):
        raise ValueError("Need aligned finite 1D importance [0,1] and increasing times >= 0")
    intervals = merge_intervals(highlights)
    truth = np.zeros(len(times), dtype=bool)
    count = ceil(top_ratio * len(times))
    truth[np.argsort(-values, kind="stable")[:count]] = True
    predicted = np.zeros(len(times), dtype=bool)
    for start, end in intervals:
        predicted |= (times >= start) & (times < end)
    tp = int((truth & predicted).sum())
    precision = tp / int(predicted.sum()) if predicted.any() else 0.0
    recall = tp / count if count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def label_accuracy(gt_labels, pred_labels):
    """Sequence accuracy or keyed accuracy; missing keyed predictions are incorrect."""
    if isinstance(gt_labels, Mapping):
        if not isinstance(pred_labels, Mapping) or set(pred_labels) - set(gt_labels):
            raise ValueError("Predicted label IDs must be a subset of GT IDs")
        correct = sum(key in pred_labels and value == pred_labels[key]
                      for key, value in gt_labels.items())
        return correct / len(gt_labels) if gt_labels else 0.0
    if isinstance(pred_labels, Mapping) or isinstance(gt_labels, str) or isinstance(pred_labels, str):
        raise ValueError("Provide label sequences or video mappings")
    gt_labels, pred_labels = list(gt_labels), list(pred_labels)
    if len(gt_labels) != len(pred_labels):
        raise ValueError("Label sequences must have equal lengths")
    return sum(a == b for a, b in zip(gt_labels, pred_labels)) / len(gt_labels) if gt_labels else 0.0


def score_all(*, duration, gt_actions, pred_actions, highlights, coverage_rule,
              human_importance=None, times=None, gt_labels=None, pred_labels=None,
              ratio=0.15, top_ratio=0.15):
    """Single-video proxy report; total is unavailable if any metric is missing.

    The 0.40/0.25/0.25/0.10 combination is NOT an official Samsung score.
    Invalid output retains its diagnostics but cannot receive a weighted total.
    """
    highlights = list(highlights)
    gt_actions, pred_actions = list(gt_actions), list(pred_actions)
    if (human_importance is None) != (times is None):
        raise ValueError("Human importance and times must be supplied together")
    if (gt_labels is None) != (pred_labels is None):
        raise ValueError("GT and predicted labels must be supplied together")
    cov = coverage(gt_actions, highlights, coverage_rule)
    maps = action_map(gt_actions, pred_actions)
    f1 = None if times is None else highlight_f1(human_importance, times, highlights, top_ratio)
    acc = None if gt_labels is None else label_accuracy(gt_labels, pred_labels)
    valid = budget_ok(highlights, duration, ratio)
    total = None
    if valid and f1 is not None and acc is not None:
        total = 0.40 * cov + 0.25 * maps["pair"]["mAP"] + 0.25 * f1["f1"] + 0.10 * acc
    return {"scorer": "local_proxy_not_official", "budget_ok": valid,
            "highlight_seconds": union_duration(highlights), "coverage_rule": coverage_rule,
            "coverage": cov, "action_map": maps, "highlight": f1,
            "label_accuracy": acc, "weighted_total_proxy": total}
