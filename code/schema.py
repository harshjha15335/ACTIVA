"""Official output JSON: build and validate. Owned by M4.

Layout from the student kit (TEAM_README.md s3, examples P02_109.json, skiing_03.json):

    {"team_id": str, "video_id": str, "duration_sec": number,
     "label": {"name": str, "score": number},
     "actions":    [{"segment": [start, end], "verb": str, "noun": str, "score": number}, ...],
     "highlights": [{"segment": [start, end]}, ...]}

validate_output() is the "ACTIVA local spec validator": our reading of the README
while the kit's validate.py is missing. It raises SchemaError for what the README
calls invalid and returns warnings for what it says validate.py only warns about
(segments under 4 s, total over the 15% budget, more than 2 decimals).

Rounding never breaks validity: timestamps go to 2 decimals, highlight starts
round up and ends round down (clips only shrink), and every end is capped at
the rounded-down duration.
"""
import math
from numbers import Real

import config

TOP_KEYS = ("team_id", "video_id", "duration_sec", "label", "actions", "highlights")
LABEL_KEYS = ("name", "score")
ACTION_KEYS = ("segment", "verb", "noun", "score")
HIGHLIGHT_KEYS = ("segment",)
_EPS = 1e-6  # float slack for comparisons


class SchemaError(ValueError):
    """The output dict breaks the official format."""


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


def round_highlight(start, end, duration, decimals=None):
    """Inward rounding used for highlights: (ceil(start), min(floor(end), floor(duration)))."""
    d = config.OUTPUT_DECIMALS if decimals is None else decimals
    return max(0.0, _ceil(start, d)), min(_floor(duration, d), _floor(end, d))


def _clip01(x):
    x = float(x)
    return min(1.0, max(0.0, x)) if math.isfinite(x) else 0.0


def label_score(cat):
    """Confidence of the chosen label: its entry in CategoryResult.scores, clipped to [0, 1]."""
    scores = cat.scores if isinstance(cat.scores, dict) else {}
    return _round(_clip01(scores.get(cat.label, 0.0)), config.SCORE_DECIMALS)


def to_output(info, cat, actions, highlights, team_id=None, decimals=None):
    """Turn pipeline results into the official dict (not validated here).

    Highlights are expected to be already normalised by main.py (sorted, >= 4 s,
    non-overlapping); rounding here is idempotent for them.
    """
    d = config.OUTPUT_DECIMALS if decimals is None else decimals
    team = config.team_id() if team_id is None else team_id
    limit = _floor(info.duration, d)

    out_actions = []
    for a in sorted(actions, key=lambda a: (a.start, a.end, a.verb, a.noun)):
        start = max(0.0, _round(a.start, d))
        end = min(limit, _round(a.end, d))
        if start >= end:
            continue
        out_actions.append({"segment": [start, end], "verb": a.verb, "noun": a.noun,
                            "score": _round(_clip01(a.score), config.SCORE_DECIMALS)})

    out_highlights = []
    for h in sorted(highlights, key=lambda h: (h.start, h.end)):
        start, end = round_highlight(h.start, h.end, info.duration, d)
        if start >= end:
            continue
        out_highlights.append({"segment": [start, end]})

    return {"team_id": team, "video_id": info.video_id,
            "duration_sec": _round(info.duration, d),
            "label": {"name": cat.label, "score": label_score(cat)},
            "actions": out_actions, "highlights": out_highlights}


def _is_number(x):
    return isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x)


def _decimals_over(x, decimals):
    return abs(x * _scale(decimals) - round(x * _scale(decimals))) > 1e-6


def _check_keys(obj, required, where, errors):
    if not isinstance(obj, dict):
        errors.append(f"{where}: expected an object, got {type(obj).__name__}")
        return False
    missing = [k for k in required if k not in obj]
    if missing:
        errors.append(f"{where}: missing keys {missing}")
    extra = [k for k in obj if k not in required]
    if extra and not config.ALLOW_EXTRA_KEYS:
        errors.append(f"{where}: unexpected keys {extra}")
    return not missing


def _segment(obj, where, limit, errors):
    """Return (start, end) if obj['segment'] is a valid [start, end], else None."""
    seg = obj["segment"]
    if not (isinstance(seg, list) and len(seg) == 2 and all(_is_number(x) for x in seg)):
        errors.append(f"{where}.segment: must be [start, end] with two finite numbers")
        return None
    start, end = seg
    if not (0 <= start < end <= limit + _EPS):
        errors.append(f"{where}.segment: need 0 <= start < end <= {limit:g}, got {seg}")
        return None
    return float(start), float(end)


def _score(x, where, errors):
    if not (_is_number(x) and 0 <= x <= 1):
        errors.append(f"{where}: must be a number in [0, 1], got {x!r}")


def validate_output(d, duration=None, vocab=None, team_id=None):
    """Check one output dict against the kit's rules.

    duration: the expected duration (manifest or probed); duration_sec must be
    within +-0.5 s of it. team_id: if given, must match exactly.
    Raises SchemaError listing every error; returns a list of warnings.
    """
    if vocab is None:
        import vocab as vocab_module
        vocab = vocab_module.load()
    verbs, nouns, labels, is_valid_pair = vocab
    verbs, nouns, labels = set(verbs), set(nouns), set(labels)
    errors, warnings = [], []
    dec = config.OUTPUT_DECIMALS

    if not _check_keys(d, TOP_KEYS, "output", errors):
        raise SchemaError("; ".join(errors))

    for key in ("team_id", "video_id"):
        if not isinstance(d[key], str) or not d[key].strip():
            errors.append(f"{key}: must be a non-empty string")
    if team_id is not None and d["team_id"] != team_id:
        errors.append(f"team_id: {d['team_id']!r} != expected {team_id!r}")

    dur = d["duration_sec"]
    if not (_is_number(dur) and dur > 0):
        errors.append(f"duration_sec: must be a positive number, got {dur!r}")
        raise SchemaError("; ".join(errors))
    if duration is not None and abs(dur - duration) > config.DURATION_TOLERANCE + _EPS:
        errors.append(f"duration_sec: {dur} is more than {config.DURATION_TOLERANCE}s from {duration}")
    limit = dur + config.DURATION_TOLERANCE

    if _check_keys(d["label"], LABEL_KEYS, "label", errors):
        if d["label"]["name"] not in labels:
            errors.append(f"label.name: {d['label']['name']!r} is not a supported label")
        _score(d["label"]["score"], "label.score", errors)

    if not isinstance(d["actions"], list):
        errors.append("actions: must be a list")
    else:
        if len(d["actions"]) > config.MAX_ACTIONS:
            errors.append(f"actions: {len(d['actions'])} items, at most {config.MAX_ACTIONS} allowed")
        for i, a in enumerate(d["actions"]):
            where = f"actions[{i}]"
            if not _check_keys(a, ACTION_KEYS, where, errors):
                continue
            if a["verb"] not in verbs:
                errors.append(f"{where}: unknown verb {a['verb']!r}")
            if a["noun"] not in nouns:
                errors.append(f"{where}: unknown noun {a['noun']!r}")
            if a["verb"] in verbs and a["noun"] in nouns and not is_valid_pair(a["verb"], a["noun"]):
                errors.append(f"{where}: invalid verb-noun pair ({a['verb']!r}, {a['noun']!r})")
            _score(a["score"], f"{where}.score", errors)
            seg = _segment(a, where, limit, errors)
            if seg and any(_decimals_over(x, dec) for x in seg):
                warnings.append(f"{where}.segment: more than {dec} decimals")

    hls = d["highlights"]
    if not isinstance(hls, list) or not hls:
        errors.append("highlights: must be a non-empty list")
    else:
        total, longest, previous_end = 0.0, 0.0, None
        for i, h in enumerate(hls):
            where = f"highlights[{i}]"
            if not _check_keys(h, HIGHLIGHT_KEYS, where, errors):
                continue
            seg = _segment(h, where, limit, errors)
            if seg is None:
                continue
            start, end = seg
            if previous_end is not None and start < previous_end - _EPS:
                errors.append(f"{where}: highlights must be sorted and non-overlapping")
            elif previous_end is not None and abs(start - previous_end) <= _EPS:
                warnings.append(f"{where}: touches the previous segment (strict non-overlap unclear)")
            previous_end = end if previous_end is None else max(previous_end, end)
            length = end - start
            total += length
            longest = max(longest, length)
            if length < config.MIN_CLIP_SECONDS - _EPS:
                warnings.append(f"{where}: {length:.2f}s is under {config.MIN_CLIP_SECONDS:g}s (ignored when scoring)")
            if _decimals_over(start, dec) or _decimals_over(end, dec):
                warnings.append(f"{where}.segment: more than {dec} decimals")
        if total and longest < config.MIN_CLIP_SECONDS - _EPS:
            errors.append(f"highlights: need at least one segment >= {config.MIN_CLIP_SECONDS:g}s")
        budget = config.BUDGET_RATIO * dur
        if total > budget + _EPS:
            warnings.append(f"highlights: total {total:.2f}s exceeds the {config.BUDGET_RATIO:.0%} budget ({budget:.2f}s)")

    if errors:
        raise SchemaError("; ".join(errors))
    return warnings
