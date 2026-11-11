import sys
import types

import numpy as np
import pytest

import config
import fallbacks
import main
import schema
import vocab
from contracts import Action, CategoryResult, ClipFeats, Highlight, Importance, VideoInfo


def fake_module(monkeypatch, name, **functions):
    module = types.ModuleType(name)
    for key, fn in functions.items():
        setattr(module, key, fn)
    monkeypatch.setitem(sys.modules, name, module)


def boom(*args):
    raise RuntimeError("module crashed")


@pytest.fixture(autouse=True)
def stub_modules(monkeypatch):
    """Test main.py on its own, whatever real modules are on the branch."""
    def clip(info):
        times = np.arange(0.0, info.duration, 0.5)
        feats = np.zeros((len(times), 4), np.float32)
        feats[:, 0] = 1.0
        return ClipFeats(times, feats, "test")

    fake_module(monkeypatch, "clip_features", extract_clip_features=clip)
    fake_module(monkeypatch, "actions", detect_actions=lambda info: [])
    fake_module(monkeypatch, "importance",
                predict_importance=lambda info, c: Importance(c.times.copy(), np.full(len(c.times), 0.5)))
    fake_module(monkeypatch, "category",
                classify_video=lambda info, c, i: CategoryResult(vocab.load().categories[0],
                                                                 {vocab.load().categories[0]: 0.9}))
    fake_module(monkeypatch, "optimizer",
                select_highlights=lambda info, a, i, c: fallbacks.uniform_highlights(info))


def assert_valid(result):
    schema.validate_output(result.output, result.info.duration, vocab.load(), team_id="TEST_TEAM")


def test_stub_pipeline_runs_without_fallbacks(media, dev_vocab):
    result = main.run(media["short"])
    assert result.fallbacks == []
    assert_valid(result)
    out = result.output
    assert out["video_id"] == "short" and out["label"]["name"] == "dev-category-a"
    assert out["team_id"] == "TEST_TEAM" and abs(out["duration_sec"] - 120) <= 0.5
    assert out["actions"] == [] and result.warnings == []
    lengths = [h["segment"][1] - h["segment"][0] for h in out["highlights"]]
    assert 0 < sum(lengths) <= config.BUDGET_RATIO * 120
    assert min(lengths) >= config.MIN_CLIP_SECONDS
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


@pytest.mark.parametrize("duration", [5.0, 11.13, 15.28, 25.16, 26.0, 27.1, 40.0, 120.0, 333.3, 600.0, 1968.6])
def test_uniform_highlights_are_valid(duration):
    info = VideoInfo("v", "v.mp4", duration, 30.0, 1, 1)
    hls = fallbacks.uniform_highlights(info)
    assert main.check_highlights(hls, info) is None
    assert all(round(x, 2) == x for h in hls for x in (h.start, h.end))
    total = sum(h.end - h.start for h in hls)
    if config.BUDGET_SAFETY * duration >= config.MIN_CLIP_SECONDS:
        assert total <= config.BUDGET_SAFETY * duration + 1e-9
    else:   # 15% < 4 s: one 4 s clip (config.SHORT_VIDEO_SINGLE_CLIP)
        assert [round(h.end - h.start, 2) for h in hls] == [4.0]


def test_check_highlights_rules():
    info = VideoInfo("v", "v.mp4", 100.0, 30.0, 1, 1)
    assert main.check_highlights([Highlight(0, 5, 0.5), Highlight(5, 10, 0.5)], info) is None
    assert "overlaps" in main.check_highlights([Highlight(0, 5, 0.5), Highlight(4, 9, 0.5)], info)
    assert "budget" in main.check_highlights([Highlight(0, 16, 0.5)], info)
    assert "duration" in main.check_highlights([Highlight(95, 101, 0.5)], info)
    assert "importance" in main.check_highlights([Highlight(0, 5, 1.5)], info)
    assert "shorter than 4" in main.check_highlights([Highlight(0, 3.99, 0.5)], info)
    assert "no highlight" in main.check_highlights([], info)


def test_short_video_allows_one_four_second_clip_only():
    info = VideoInfo("v", "v.mp4", 11.13, 30.0, 1, 1)
    assert main.check_highlights([Highlight(3.0, 7.0, 0.5)], info) is None
    assert "budget" in main.check_highlights([Highlight(3.0, 7.5, 0.5)], info)
    assert "budget" in main.check_highlights([Highlight(0.0, 4.0, 0.5), Highlight(5.0, 9.0, 0.5)], info)


def test_normalize_drops_short_clips_merges_touching_and_rounds_inward():
    info = VideoInfo("v", "v.mp4", 100.0, 30.0, 1, 1)
    hls = main.normalize_highlights([Highlight(0.0, 3.0, 0.9), Highlight(10.004, 14.5, 0.2),
                                     Highlight(14.5, 18.5, 0.6), Highlight(50.0, 54.009, 0.5)], info)
    assert [(h.start, h.end) for h in hls] == [(10.01, 18.5), (50.0, 54.0)]
    assert hls[0].importance == pytest.approx((0.2 * 4.49 + 0.6 * 4.0) / 8.49)
    with pytest.raises(ValueError, match="overlaps"):
        main.normalize_highlights([Highlight(0, 5, 0.5), Highlight(4, 9, 0.5)], info)


def test_selector_with_only_short_clips_falls_back(media, dev_vocab, monkeypatch):
    fake_module(monkeypatch, "optimizer",
                select_highlights=lambda info, a, i, c: [Highlight(10.0, 13.0, 0.9), Highlight(50.0, 53.0, 0.9)])
    result = main.run(media["short"])
    assert result.fallbacks == ["highlights"]
    assert_valid(result)


def test_strict_mode_raises_instead_of_falling_back(media, dev_vocab, monkeypatch):
    monkeypatch.setenv("ACTIVA_STRICT", "1")
    fake_module(monkeypatch, "actions", detect_actions=boom)
    with pytest.raises(main.StageError, match="stage actions failed"):
        main.run(media["short"])


def test_strict_mode_passes_when_every_stage_works(media, dev_vocab, monkeypatch):
    monkeypatch.setenv("ACTIVA_STRICT", "1")
    assert main.run(media["short"]).fallbacks == []


def test_fallback_category_is_other(media, dev_vocab, monkeypatch):
    fake_module(monkeypatch, "category", classify_video=boom)
    result = main.run(media["short"])
    assert result.output["label"] == {"name": "other", "score": 0.0}


def test_actions_are_capped_at_2000_by_score(media, dev_vocab, monkeypatch):
    acts = [Action("dev-verb-a", "dev-noun-a", i * 0.05, i * 0.05 + 1, (i % 7) / 7) for i in range(2100)]
    fake_module(monkeypatch, "actions", detect_actions=lambda info: acts)
    result = main.run(media["short"])
    assert len(result.output["actions"]) == 2000
    assert min(a.score for a in result.actions) >= sorted(a.score for a in acts)[100]
    assert_valid(result)


def test_missing_team_id_fails_before_any_work(media, dev_vocab, monkeypatch):
    monkeypatch.delenv("ACTIVA_TEAM_ID")
    monkeypatch.setattr(config, "TEAM_ID", None)
    with pytest.raises(config.ConfigError):
        main.run(media["short"])
