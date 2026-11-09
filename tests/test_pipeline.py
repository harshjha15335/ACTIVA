import sys
import types

import numpy as np
import pytest

import config
import fallbacks
import main
import schema
import vocab
from contracts import Action, CategoryResult, Highlight, Importance, VideoInfo


def fake_module(monkeypatch, name, **functions):
    module = types.ModuleType(name)
    for key, fn in functions.items():
        setattr(module, key, fn)
    monkeypatch.setitem(sys.modules, name, module)


def boom(*args):
    raise RuntimeError("module crashed")


def assert_valid(result):
    schema.validate_output(result.output, result.info.duration, vocab.load())


def test_stub_pipeline_runs_without_fallbacks(media, dev_vocab):
    result = main.run(media["short"])
    assert result.fallbacks == []
    assert_valid(result)
    out = result.output
    assert out["video_id"] == "short" and out["label"] == "dev-category-a"
    assert out["actions"] == []
    total = sum(h["end"] - h["start"] for h in out["highlights"])
    assert 0 < total <= config.BUDGET_RATIO * 120
    assert {"clip", "actions", "importance", "category", "highlights", "total"} <= set(result.timings)


def test_same_input_same_output(media, dev_vocab):
    assert main.predict(media["short"]) == main.predict(media["short"])


@pytest.mark.parametrize("module, function", [
    ("clip_features", "extract_clip_features"),
    ("actions", "detect_actions"),
    ("importance", "predict_importance"),
    ("category", "classify_video"),
    ("optimizer", "select_highlights"),
])
def test_crashing_module_falls_back_to_valid_output(media, dev_vocab, monkeypatch, module, function):
    fake_module(monkeypatch, module, **{function: boom})
    result = main.run(media["short"])
    assert len(result.fallbacks) == 1
    assert_valid(result)


def test_module_that_fails_to_import_falls_back(media, dev_vocab, monkeypatch):
    monkeypatch.setitem(sys.modules, "actions", None)   # import raises ImportError
    result = main.run(media["short"])
    assert result.fallbacks == ["actions"] and result.output["actions"] == []


def test_over_budget_selector_is_replaced(media, dev_vocab, monkeypatch):
    fake_module(monkeypatch, "optimizer",
                select_highlights=lambda info, a, i, c: [Highlight(0.0, info.duration * 0.5, 0.9)])
    result = main.run(media["short"])
    assert result.fallbacks == ["highlights"]
    assert_valid(result)


def test_wrong_types_are_replaced(media, dev_vocab, monkeypatch):
    fake_module(monkeypatch, "optimizer", select_highlights=lambda *a: [{"start": 0, "end": 1, "importance": 1}])
    fake_module(monkeypatch, "category", classify_video=lambda *a: CategoryResult("not-a-category"))
    fake_module(monkeypatch, "importance",
                predict_importance=lambda info, clip: Importance(clip.times, np.full(len(clip.times), 2.0)))
    result = main.run(media["short"])
    assert sorted(result.fallbacks) == ["category", "highlights", "importance"]
    assert_valid(result)


def test_bad_actions_are_dropped_and_good_ones_clamped(media, dev_vocab, monkeypatch):
    acts = [
        Action("dev-verb-a", "dev-noun-a", -2.0, 3.0, 0.9),     # clamped to start 0
        Action("dev-verb-b", "dev-noun-a", 5.0, 6.0, 0.5),      # invalid pair
        Action("cut", "tomato", 5.0, 6.0, 0.5),                 # not in vocabulary
        Action("dev-verb-a", "dev-noun-b", 7.0, float("nan"), 0.5),
        Action("dev-verb-b", "dev-noun-b", 119.0, 999.0, 1.5),  # clamped to duration, score to 1
        "not an action",
    ]
    fake_module(monkeypatch, "actions", detect_actions=lambda info: acts)
    result = main.run(media["short"])
    assert result.fallbacks == []
    assert [(a.start, a.end, a.score) for a in result.actions] == [(0.0, 3.0, 0.9), (119.0, 120.0, 1.0)]
    assert_valid(result)


def test_black_video(media, dev_vocab):
    assert_valid(main.run(media["black"]))


@pytest.mark.parametrize("duration", [0.5, 1.0, 6.0, 6.76, 7.0, 20.3, 120.0, 333.3, 600.0, 3600.0])
def test_uniform_highlights_fit_budget(duration):
    info = VideoInfo("v", "v.mp4", duration, 30.0, 1, 1)
    hls = fallbacks.uniform_highlights(info)
    assert main.check_highlights(hls, info) is None
    total = sum(h.end - h.start for h in hls)
    assert total <= config.BUDGET_SAFETY * duration + 1e-9
    if config.BUDGET_SAFETY * duration < config.MIN_CLIP_SECONDS:
        assert hls == []
    else:
        assert hls and all(h.end - h.start >= config.MIN_CLIP_SECONDS for h in hls)


def test_check_highlights_rules():
    info = VideoInfo("v", "v.mp4", 100.0, 30.0, 1, 1)
    assert main.check_highlights([Highlight(0, 5, 0.5), Highlight(5, 10, 0.5)], info) is None
    assert "overlaps" in main.check_highlights([Highlight(0, 5, 0.5), Highlight(4, 6, 0.5)], info)
    assert "budget" in main.check_highlights([Highlight(0, 16, 0.5)], info)
    assert "duration" in main.check_highlights([Highlight(95, 101, 0.5)], info)
    assert "importance" in main.check_highlights([Highlight(0, 1, 1.5)], info)
