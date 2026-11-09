"""Official output JSON: build and validate. Owned by M4.

Output layout (challenge slides; VERIFY rounding/ordering/extra keys in the kit):

    {"video_id": str, "label": str,
     "actions":    [{"verb", "noun", "start", "end"}, ...],
     "highlights": [{"start", "end", "importance"}, ...]}

Rounding is chosen so it can never break validity: highlight starts round up
and ends round down (clips only shrink, so the budget still holds), and every
end is capped at the rounded-down video duration. Intervals that collapse to
zero length after rounding are dropped.
"""
import math
from numbers import Real

import config

TOP_KEYS = ("video_id", "label", "actions", "highlights")
ACTION_KEYS = ("verb", "noun", "start", "end")
HIGHLIGHT_KEYS = ("start", "end", "importance")
_EPS = 1e-6  # one microsecond of float slack


class SchemaError(ValueError):
    """The output dict breaks the official format or our own checks."""


def _scale(decimals):
    return 10 ** decimals


def _round(x, decimals):
    return round(float(x), decimals)


def _floor(x, decimals):
    s = _scale(decimals)
    return round(math.floor(float(x) * s + 1e-7) / s, decimals)


def _ceil(x, decimals):
    s = _scale(decimals)
    return round(math.ceil(float(x) * s - 1e-7) / s, decimals)


def to_output(info, cat, actions, highlights, decimals=None):
    """Turn pipeline results into the official dict (not validated here)."""
    d = config.OUTPUT_DECIMALS if decimals is None else decimals
    limit = _floor(info.duration, d)

    out_actions = []
    for a in sorted(actions, key=lambda a: (a.start, a.end, a.verb, a.noun)):
        start = max(0.0, _round(a.start, d))
        end = min(limit, _round(a.end, d))
        if start >= end:
            continue
        item = {"verb": a.verb, "noun": a.noun, "start": start, "end": end}
        if config.INCLUDE_ACTION_SCORE:
            item["score"] = _round(a.score, 4)
        out_actions.append(item)

    out_highlights = []
    for h in sorted(highlights, key=lambda h: (h.start, h.end)):
        start = max(0.0, _ceil(h.start, d))
        end = min(limit, _floor(h.end, d))
        if start >= end:
            continue
        importance = min(1.0, max(0.0, _round(h.importance, 4)))
        out_highlights.append({"start": start, "end": end, "importance": importance})

    return {"video_id": info.video_id, "label": cat.label,
            "actions": out_actions, "highlights": out_highlights}


def _is_number(x):
    return isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x)


def _check_keys(obj, required, where, errors, optional=()):
    if not isinstance(obj, dict):
        errors.append(f"{where}: expected an object, got {type(obj).__name__}")
        return False
    missing = [k for k in required if k not in obj]
    if missing:
        errors.append(f"{where}: missing keys {missing}")
    extra = [k for k in obj if k not in required and k not in optional]
    if extra and not config.ALLOW_EXTRA_KEYS:
        errors.append(f"{where}: unexpected keys {extra}")
    return not missing


def _check_interval(obj, where, duration, errors):
    start, end = obj["start"], obj["end"]
    if not (_is_number(start) and _is_number(end)):
        errors.append(f"{where}: start/end must be finite numbers")
        return False
    if not (0 <= start < end <= duration + _EPS):
        errors.append(f"{where}: need 0 <= start < end <= duration ({start}, {end}, {duration})")
        return False
    return True


def validate_output(d, duration, vocab=None):
    """Raise SchemaError listing every problem found; return None if valid."""
    if vocab is None:
        import vocab as vocab_module
        vocab = vocab_module.load()
    verbs, nouns, categories, is_valid_pair = vocab
    verbs, nouns, categories = set(verbs), set(nouns), set(categories)
    errors = []

    if not _is_number(duration) or duration <= 0:
        raise SchemaError(f"duration must be a positive number, got {duration!r}")
    if not _check_keys(d, TOP_KEYS, "output", errors):
        raise SchemaError("; ".join(errors))

    if not isinstance(d["video_id"], str) or not d["video_id"]:
        errors.append("video_id: must be a non-empty string")
    if d["label"] not in categories:
        errors.append(f"label: {d['label']!r} is not a supported category")

    if not isinstance(d["actions"], list):
        errors.append("actions: must be a list")
    else:
        optional = ("score",) if config.INCLUDE_ACTION_SCORE else ()
        for i, a in enumerate(d["actions"]):
            where = f"actions[{i}]"
            if not _check_keys(a, ACTION_KEYS, where, errors, optional):
                continue
            if a["verb"] not in verbs:
                errors.append(f"{where}: unknown verb {a['verb']!r}")
            if a["noun"] not in nouns:
                errors.append(f"{where}: unknown noun {a['noun']!r}")
            if a["verb"] in verbs and a["noun"] in nouns and not is_valid_pair(a["verb"], a["noun"]):
                errors.append(f"{where}: invalid verb-noun pair ({a['verb']!r}, {a['noun']!r})")
            _check_interval(a, where, duration, errors)
            if "score" in a and not (_is_number(a["score"]) and 0 <= a["score"] <= 1):
                errors.append(f"{where}: score must be in [0, 1]")

    if not isinstance(d["highlights"], list):
        errors.append("highlights: must be a list")
    else:
        total, previous_end = 0.0, None
        for i, h in enumerate(d["highlights"]):
            where = f"highlights[{i}]"
            if not _check_keys(h, HIGHLIGHT_KEYS, where, errors):
                continue
            if not (_is_number(h["importance"]) and 0 <= h["importance"] <= 1):
                errors.append(f"{where}: importance must be in [0, 1]")
            if not _check_interval(h, where, duration, errors):
                continue
            if previous_end is not None and h["start"] < previous_end - _EPS:
                errors.append(f"{where}: highlights must be sorted and non-overlapping")
            previous_end = h["end"] if previous_end is None else max(previous_end, h["end"])
            total += h["end"] - h["start"]
        budget = config.BUDGET_RATIO * duration
        if total > budget + _EPS:
            errors.append(f"highlights: total {total:.3f}s exceeds the {config.BUDGET_RATIO:.0%} budget ({budget:.3f}s)")

    if errors:
        raise SchemaError("; ".join(errors))
