import copy
import random

import pytest

import config
import schema
import vocab
from contracts import Action, CategoryResult, Highlight, VideoInfo
from schema import SchemaError


@pytest.fixture
def v(dev_vocab):
    return vocab.load()


def info(duration=120.0):
    return VideoInfo("vid", "/x/vid.mp4", duration, 30.0, 640, 360)


def good(duration=120.0):
    return {
        "video_id": "vid",
        "label": "dev-category-a",
        "actions": [{"verb": "dev-verb-a", "noun": "dev-noun-a", "start": 1.0, "end": 4.5}],
        "highlights": [{"start": 2.0, "end": 5.0, "importance": 0.7},
                       {"start": 10.0, "end": 12.0, "importance": 0.2}],
    }


def test_valid_output_passes(v):
    schema.validate_output(good(), 120.0, v)


def test_to_output_key_order_sorting_and_no_score_by_default():
    acts = [Action("dev-verb-b", "dev-noun-b", 9.0, 10.0, 0.9), Action("dev-verb-a", "dev-noun-a", 1.0, 2.0, 0.4)]
    hls = [Highlight(20.0, 22.0, 0.3), Highlight(5.0, 6.0, 0.9)]
    out = schema.to_output(info(), CategoryResult("dev-category-a"), acts, hls)
    assert list(out) == ["video_id", "label", "actions", "highlights"]
    assert [a["start"] for a in out["actions"]] == [1.0, 9.0]
    assert [h["start"] for h in out["highlights"]] == [5.0, 20.0]
    assert list(out["actions"][0]) == ["verb", "noun", "start", "end"]
    assert list(out["highlights"][0]) == ["start", "end", "importance"]


def test_rounding_only_shrinks_highlights_and_caps_at_duration():
    out = schema.to_output(info(10.009), CategoryResult("c"), [Action("v", "n", 9.004, 10.009, 1)],
                           [Highlight(1.001, 1.509, 0.123456), Highlight(9.0, 10.009, 0.5)])
    assert out["highlights"][0] == {"start": 1.01, "end": 1.5, "importance": 0.1235}
    assert out["highlights"][1]["end"] == 10.0
    assert out["actions"][0]["end"] == 10.0


def test_intervals_that_vanish_after_rounding_are_dropped():
    out = schema.to_output(info(), CategoryResult("c"), [Action("v", "n", 3.001, 3.004, 1)],
                           [Highlight(1.001, 1.004, 0.5)])
    assert out["actions"] == [] and out["highlights"] == []


def test_score_included_only_when_enabled(monkeypatch, v):
    monkeypatch.setattr(config, "INCLUDE_ACTION_SCORE", True)
    out = schema.to_output(info(), CategoryResult("dev-category-a"),
                           [Action("dev-verb-a", "dev-noun-a", 1, 2, 0.87654)], [])
    assert out["actions"][0]["score"] == 0.8765
    schema.validate_output(out, 120.0, v)


def mutate(fn):
    d = copy.deepcopy(good())
    fn(d)
    return d


BAD = {
    "missing key": lambda d: d.pop("label"),
    "extra top key": lambda d: d.update(extra=1),
    "unknown label": lambda d: d.update(label="cooking"),
    "empty video_id": lambda d: d.update(video_id=""),
    "unknown verb": lambda d: d["actions"][0].update(verb="cut"),
    "unknown noun": lambda d: d["actions"][0].update(noun="tomato"),
    "invalid pair": lambda d: d["actions"][0].update(verb="dev-verb-b", noun="dev-noun-a"),
    "score not allowed": lambda d: d["actions"][0].update(score=0.5),
    "start == end": lambda d: d["actions"][0].update(start=4.5),
    "negative start": lambda d: d["actions"][0].update(start=-0.5),
    "end after duration": lambda d: d["actions"][0].update(end=120.5),
    "NaN": lambda d: d["actions"][0].update(start=float("nan")),
    "bool as number": lambda d: d["actions"][0].update(start=True),
    "string time": lambda d: d["highlights"][0].update(start="2.0"),
    "importance > 1": lambda d: d["highlights"][0].update(importance=1.5),
    "overlapping": lambda d: d["highlights"][1].update(start=4.0),
    "unsorted": lambda d: d["highlights"].reverse(),
    "over budget": lambda d: d["highlights"].append({"start": 30.0, "end": 49.0, "importance": 0.5}),
    "actions not a list": lambda d: d.update(actions={}),
    "highlight missing key": lambda d: d["highlights"][0].pop("importance"),
}


@pytest.mark.parametrize("name", BAD)
def test_invalid_outputs_are_rejected(v, name):
    with pytest.raises(SchemaError):
        schema.validate_output(mutate(BAD[name]), 120.0, v)


def test_error_message_names_the_field(v):
    with pytest.raises(SchemaError, match=r"actions\[0\].*unknown verb"):
        schema.validate_output(mutate(BAD["unknown verb"]), 120.0, v)


def test_exactly_fifteen_percent_passes_and_just_over_fails(v):
    d = good()
    d["highlights"] = [{"start": 0.0, "end": 18.0, "importance": 0.5}]
    schema.validate_output(d, 120.0, v)
    d["highlights"][0]["end"] = 18.01
    with pytest.raises(SchemaError, match="budget"):
        schema.validate_output(d, 120.0, v)


def test_touching_highlights_are_allowed(v):
    d = good()
    d["highlights"] = [{"start": 0.0, "end": 5.0, "importance": 0.5},
                       {"start": 5.0, "end": 9.0, "importance": 0.5}]
    schema.validate_output(d, 120.0, v)


def test_random_within_budget_selections_stay_valid_after_rounding(v):
    rng = random.Random(0)
    for _ in range(300):
        duration = rng.uniform(5, 900)
        budget = config.BUDGET_RATIO * duration
        cuts = sorted(rng.uniform(0, duration) for _ in range(2 * rng.randint(1, 12)))
        spans = list(zip(cuts[::2], cuts[1::2]))
        total = sum(e - s for s, e in spans)
        scale = min(1.0, budget / total) if total else 1.0
        hls = [Highlight(s, s + (e - s) * scale, rng.random()) for s, e in spans]
        out = schema.to_output(info(duration), CategoryResult("dev-category-a"), [], hls)
        schema.validate_output(out, duration, v)
