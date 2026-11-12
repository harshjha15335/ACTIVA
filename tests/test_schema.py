"""Output format from the student kit (TEAM_README.md s3/s4). Our local spec validator."""
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


def good():
    return {
        "team_id": "TEST_TEAM",
        "video_id": "vid",
        "duration_sec": 120.0,
        "label": {"name": "dev-category-a", "score": 0.9},
        "actions": [{"segment": [1.0, 4.5], "verb": "dev-verb-a", "noun": "dev-noun-a", "score": 0.8}],
        "highlights": [{"segment": [2.0, 7.0]}, {"segment": [10.0, 16.0]}],
    }


def errors_of(d, v, duration=120.0, **kw):
    with pytest.raises(SchemaError) as e:
        schema.validate_output(d, duration, v, **kw)
    return str(e.value)


def test_valid_output_passes_without_warnings(v):
    assert schema.validate_output(good(), 120.0, v) == []


def test_empty_actions_are_allowed(v):
    d = good()
    d["actions"] = []          # "May be [] for FPVSum videos"
    assert schema.validate_output(d, 120.0, v) == []


def test_kit_examples_against_real_vocabulary(kit_vocab):
    """P02_109.json passes; skiing_03.json fails only because its label 'skiing' is not in the kit's labels."""
    import json
    real = vocab.load(kit_vocab)
    kit = kit_vocab.parent
    epic = json.loads((kit / "P02_109.json").read_text())
    assert schema.validate_output(epic, epic["duration_sec"], real) == []
    ski = json.loads((kit / "skiing_03.json").read_text())
    with pytest.raises(SchemaError, match=r"^label.name: 'skiing' is not a supported label$"):
        schema.validate_output(ski, ski["duration_sec"], real)


def test_to_output_matches_kit_layout(dev_vocab):
    acts = [Action("dev-verb-b", "dev-noun-b", 9.0, 10.0, 0.9), Action("dev-verb-a", "dev-noun-a", 1.0, 2.0, 0.4)]
    hls = [Highlight(5.0, 10.0, 0.9), Highlight(20.0, 26.0, 0.3)]
    cat = CategoryResult("dev-category-a", {"dev-category-a": 0.87654, "other": 0.1})
    out = schema.to_output(info(), cat, acts, hls)
    assert list(out) == ["team_id", "video_id", "duration_sec", "label", "actions", "highlights"]
    assert out["team_id"] == "TEST_TEAM" and out["duration_sec"] == 120.0
    assert out["label"] == {"name": "dev-category-a", "score": 0.8765}
    assert out["actions"][0] == {"segment": [1.0, 2.0], "verb": "dev-verb-a", "noun": "dev-noun-a", "score": 0.4}
    assert [list(a) for a in out["actions"]] == [["segment", "verb", "noun", "score"]] * 2
    assert out["highlights"] == [{"segment": [5.0, 10.0]}, {"segment": [20.0, 26.0]}]
    schema.validate_output(out, 120.0, vocab.load())


def test_label_score_is_clipped_and_defaults_to_zero(dev_vocab):
    assert schema.label_score(CategoryResult("a", {"a": 1.7})) == 1.0
    assert schema.label_score(CategoryResult("a", {"a": -0.2})) == 0.0
    assert schema.label_score(CategoryResult("a")) == 0.0
    assert schema.label_score(CategoryResult("a", {"a": float("nan")})) == 0.0


def test_missing_team_id_is_a_config_error(monkeypatch):
    monkeypatch.delenv("ACTIVA_TEAM_ID", raising=False)
    monkeypatch.setattr(config, "TEAM_ID", None)
    with pytest.raises(config.ConfigError):
        schema.to_output(info(), CategoryResult("x"), [], [])
    monkeypatch.setattr(config, "TEAM_ID", "  T07 ")
    assert config.team_id() == "T07"


def test_rounding_only_shrinks_highlights_and_caps_at_duration(dev_vocab):
    hls = [Highlight(1.001, 5.999, 0.5), Highlight(100.004, 119.996, 0.5)]
    out = schema.to_output(info(119.996), CategoryResult("dev-category-a"), [], hls)
    assert out["highlights"] == [{"segment": [1.01, 5.99]}, {"segment": [100.01, 119.99]}]
    assert out["duration_sec"] == 120.0


@pytest.mark.parametrize("mutate, expected", [
    (lambda d: d.pop("team_id"), "missing keys ['team_id']"),
    (lambda d: d.update(extra=1), "unexpected keys ['extra']"),
    (lambda d: d.update(team_id=""), "team_id: must be a non-empty string"),
    (lambda d: d.update(video_id=3), "video_id: must be a non-empty string"),
    (lambda d: d.update(label="dev-category-a"), "label: expected an object"),
    (lambda d: d["label"].update(name="skiing"), "not a supported label"),
    (lambda d: d["label"].update(score=1.5), "label.score"),
    (lambda d: d["label"].pop("score"), "label: missing keys ['score']"),
    (lambda d: d["actions"][0].update(verb="cut"), "unknown verb 'cut'"),
    (lambda d: d["actions"][0].update(noun="tomato"), "unknown noun 'tomato'"),
    (lambda d: d["actions"][0].update(verb="dev-verb-b", noun="dev-noun-a"), "invalid verb-noun pair"),
    (lambda d: d["actions"][0].update(score=True), "score: must be a number"),
    (lambda d: d["actions"][0].pop("score"), "missing keys ['score']"),
    (lambda d: d["actions"][0].update(segment=[4.5, 1.0]), "need 0 <= start < end"),
    (lambda d: d["actions"][0].update(segment=[1.0, 120.6]), "need 0 <= start < end"),
    (lambda d: d["actions"][0].update(segment=[1.0]), "two finite numbers"),
    (lambda d: d["actions"][0].update(segment=[1.0, float("nan")]), "two finite numbers"),
    (lambda d: d.update(actions={}), "actions: must be a list"),
    (lambda d: d.update(highlights=[]), "highlights: must be a non-empty list"),
    (lambda d: d["highlights"][1].update(segment=[6.0, 12.0]), "sorted and non-overlapping"),
    (lambda d: d["highlights"].reverse(), "sorted and non-overlapping"),
    (lambda d: d.update(highlights=[{"segment": [0.0, 3.0]}]), "at least one segment >= 4"),
    (lambda d: d["highlights"][0].update(importance=0.5), "unexpected keys ['importance']"),
    (lambda d: d.update(duration_sec=0), "duration_sec: must be a positive number"),
    (lambda d: d.update(duration_sec=121.0), "more than 0.5s from 120.0"),
])
def test_each_rule_is_enforced(v, mutate, expected):
    d = good()
    mutate(d)
    assert expected in errors_of(d, v)


def test_end_may_exceed_duration_by_half_a_second(v):
    d = good()
    d["actions"][0]["segment"] = [100.0, 120.5]
    assert schema.validate_output(d, 120.0, v) == []


def test_action_limit_is_2000(v):
    d = good()
    d["actions"] = d["actions"] * 2001
    assert "at most 2000" in errors_of(d, v)


def test_team_id_must_match_when_given(v):
    assert "team_id" in errors_of(good(), v, team_id="OTHER")


def test_warnings_for_short_segments_budget_and_decimals(v):
    d = good()
    d["highlights"] = [{"segment": [0.0, 2.0]}, {"segment": [3.0, 7.0]}, {"segment": [8.0, 30.123]}]
    warnings = schema.validate_output(d, 120.0, v)
    text = " ".join(warnings)
    assert "under 4s" in text and "exceeds the 15% budget" in text and "decimals" in text


def test_touching_highlights_warn_but_pass(v):
    d = good()
    d["highlights"] = [{"segment": [2.0, 7.0]}, {"segment": [7.0, 12.0]}]
    assert any("touches" in w for w in schema.validate_output(d, 120.0, v))


def test_exactly_fifteen_percent_gives_no_budget_warning(v):
    d = good()
    d["highlights"] = [{"segment": [0.0, 18.0]}]
    assert schema.validate_output(d, 120.0, v) == []
    d["highlights"] = [{"segment": [0.0, 18.01]}]
    assert any("budget" in w for w in schema.validate_output(d, 120.0, v))


def test_random_selections_stay_valid_after_rounding(v):
    rng = random.Random(0)
    for _ in range(300):
        duration = rng.uniform(30, 2000)
        t, hls = 0.0, []
        while True:
            start = t + rng.uniform(0.0, duration / 5)
            end = start + rng.uniform(4.02, 12.0)
            if end > duration:
                break
            hls.append(Highlight(start, end, rng.random()))
            t = end
        if not hls:
            continue
        out = schema.to_output(info(duration), CategoryResult("dev-category-a"), [], hls, team_id="T")
        for h in out["highlights"]:
            s, e = h["segment"]
            assert round(s, 2) == s and round(e, 2) == e
        warnings = schema.validate_output(copy.deepcopy(out), duration, v)
        assert not [w for w in warnings if "decimals" in w]
