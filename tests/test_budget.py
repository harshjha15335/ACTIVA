import random
from types import SimpleNamespace

import numpy as np
import pytest

from eval.metrics import budget_ok, union_duration
from contracts import Action, ClipFeats, Highlight, Importance, VideoInfo
from candidates import ImportanceCurve, build_candidates
from optimizer import (SelectorSettings, select_highlights, uniform_intervals,
                       select_uniform, select_importance, select_coverage, select_variant)


def clips(intervals):
    return [dict(start=a, end=b) for a, b in intervals]


@pytest.mark.parametrize("duration,intervals,expected", [
    (100, [(0, 15)], True), (100, [(0, 15.000001)], False),
    (100, [(0, 10), (5, 15)], False), (100, [(10, 12), (0, 1)], False),
    (100, [(0, 5), (5, 10)], True), (100, [(-1, 1)], False),
    (100, [(99, 101)], False), (100, [(1, 1)], False),
    (100, [(0, float("nan"))], False), (0, [], True), (-1, [], False),
    (float("inf"), [], False),
])
def test_budget_validation(duration, intervals, expected):
    assert budget_ok(clips(intervals), duration) is expected


def test_union_does_not_double_count_overlaps():
    assert union_duration(clips([(0, 10), (5, 15), (0, 10)])) == 15


def test_uniform_budget_properties_and_determinism():
    rng = random.Random(42)
    durations = [0, 0.01, 1, 6, 1 / 0.15, 10, 20, 30, 600]
    durations += [rng.uniform(30, 600) for _ in range(500)]
    for duration in durations:
        intervals = uniform_intervals(duration)
        assert intervals == uniform_intervals(duration)
        assert budget_ok(clips(intervals), duration)
        assert all(end - start >= 1 for start, end in intervals)
        if duration * 0.15 >= 1:
            assert intervals


def test_selector_returns_canonical_highlights():
    rng = np.random.default_rng(42)
    for duration in (0, 30, 120, 600):
        times = np.arange(0, duration, 0.5)
        for values in (np.zeros(len(times)), rng.random(len(times))):
            imp = SimpleNamespace(times=times, values=values)
            info = SimpleNamespace(duration=duration)
            result = select_highlights(info, [], imp, None)
            assert result == select_highlights(info, [], imp, None)
            assert budget_ok(result, duration)
            assert all(isinstance(h, Highlight) and 0 <= h.importance <= 1 for h in result)


def test_selector_validates_importance():
    with pytest.raises(ValueError):
        select_highlights(SimpleNamespace(duration=30), [],
                          SimpleNamespace(times=[0, 0], values=[0, 1]), None)


def fixture(duration=120, peak=None, irregular=False):
    times = np.arange(0, duration, 0.5)
    if irregular:
        times = times + np.where(np.arange(len(times)) % 2, 0.037, 0)
    values = np.zeros(len(times), np.float32)
    if peak is not None:
        values[np.abs(times - peak) < 2] = 1
    feats = np.zeros((len(times), 512), np.float32)
    feats[:, 0] = 1
    return (VideoInfo("synthetic", "unused.mp4", duration, 30, 64, 64),
            Importance(times, values), ClipFeats(times, feats, "synthetic-test-only"))


def intervals(highlights):
    return [(h.start, h.end) for h in highlights]


@pytest.mark.parametrize("selector", [select_highlights, select_importance])
def test_early_and_late_importance_relocate_actual_selection(selector):
    info, early, clip = fixture(120, 10, irregular=True)
    _, late, _ = fixture(120, 105, irregular=True)
    first, second = selector(info, [], early, clip), selector(info, [], late, clip)
    assert any(h.start <= 10 < h.end for h in first)
    assert any(h.start <= 105 < h.end for h in second)
    assert intervals(first) != intervals(second)
    assert sum(h.end-h.start for h in first if h.start < 30) > 0
    assert all(h.start > 80 for h in second)


def test_joint_relocates_when_actions_move():
    info, imp, clip = fixture(60)
    actions = [Action("cut", "tomato", 4, 8, 0.9)]
    first = select_highlights(info, actions, imp, clip)
    second = select_highlights(info, [Action("cut", "tomato", 44, 48, 0.9)], imp, clip)
    assert intervals(first) != intervals(second)
    from eval.metrics import coverage
    assert coverage(actions, first, "frac>=0.5") == 1


def test_distinct_pairs_beat_repeated_high_confidence_pair():
    info, imp, clip = fixture(60)
    actions = [Action("cut", "tomato", 4, 8, 0.99), Action("cut", "tomato", 24, 28, 0.98),
               Action("wash", "dish", 44, 48, 0.8)]
    from eval.metrics import coverage
    for selector in (select_coverage, select_highlights):
        result = selector(info, actions, imp, clip)
        assert coverage(actions, result, "frac>=0.5") == 1
        assert budget_ok(result, info.duration)


def test_duplicate_copies_and_input_order_do_not_inflate_coverage():
    info, imp, clip = fixture(60)
    a, b = Action("cut", "tomato", 4, 8, 0.9), Action("wash", "dish", 44, 48, 0.8)
    expected = select_highlights(info, [a, b], imp, clip)
    assert select_highlights(info, [b, a] + [a] * 50, imp, clip) == expected


def test_confidence_gates_and_weights_are_real_ablations():
    info, imp, clip = fixture(60, 45)
    actions = [Action("cut", "tomato", 4, 8, 0.9)]
    only_action = select_coverage(info, actions, imp, clip)
    only_importance = select_importance(info, actions, imp, clip)
    assert intervals(only_action) != intervals(only_importance)
    assert select_highlights(info, [Action("cut", "tomato", 4, 8, 0.01)], imp, clip) == select_highlights(info, [], imp, clip)
    assert intervals(select_variant(info, actions, imp, clip,
                                   settings=SelectorSettings(coverage_weight=1, importance_weight=0))) == intervals(only_action)
    _, other_importance, _ = fixture(60, 25)
    assert intervals(select_coverage(info, actions, other_importance, clip)) == intervals(only_action)
    moved = [Action("wash", "dish", 24, 28, .99)]
    assert select_importance(info, moved, imp, clip) == only_importance


@pytest.mark.parametrize("duration", [4, 11.13, 15.28, 25.16, 26.66])
def test_short_video_policy_is_explicit_and_not_a_budget_pass(duration, caplog):
    info, imp, clip = fixture(duration, duration/2)
    result = select_highlights(info, [], imp, clip)
    assert len(result) == 1
    assert result[0].end-result[0].start == pytest.approx(4)
    assert 0 <= result[0].start < result[0].end <= duration
    assert not budget_ok(result, duration)
    assert "provisional short-video policy" in caplog.text


def test_policy_disabled_or_impossible_duration_fail_clearly(monkeypatch):
    import config
    info, imp, clip = fixture(11.13)
    monkeypatch.setattr(config, "SHORT_VIDEO_SINGLE_CLIP", False)
    with pytest.raises(ValueError, match="conflicts"):
        select_highlights(info, [], imp, clip)
    info, imp, clip = fixture(2)
    with pytest.raises(ValueError, match="No in-range"):
        select_highlights(info, [], imp, clip)


@pytest.mark.parametrize("duration", [26.67, 27, 30, 37, 120, 600, 1968.6])
@pytest.mark.parametrize("selector", [select_uniform, select_importance, select_coverage, select_highlights])
def test_real_minimum_and_canonical_rounding_stay_safe(selector, duration):
    # Canonical integration, not a test-only replacement of shared types.
    from main import check_highlights, normalize_highlights
    info, imp, clip = fixture(duration, duration/3, irregular=True)
    actions = [Action("cut", "tomato", 1, min(duration, 5), .9)]
    result = selector(info, actions, imp, clip)
    normalized = normalize_highlights(result, info)
    assert result == normalized
    assert normalized and check_highlights(normalized, info) is None
    assert budget_ok(normalized, duration)
    assert all(h.end-h.start >= 4-1e-9 for h in normalized)


def test_seeded_random_actions_importance_overlap_and_determinism():
    rng = np.random.default_rng(1108)
    for _ in range(30):
        duration = float(rng.uniform(30, 600))
        info, imp, clip = fixture(duration, irregular=True)
        imp.values = rng.random(len(imp.times))
        actions = []
        for i in range(20):
            start = float(rng.uniform(0, duration-1))
            actions.append(Action(f"verb{i%4}", f"noun{i%3}", start,
                                  min(duration, start+float(rng.uniform(1, 20))), float(rng.random())))
        result = select_highlights(info, actions, imp, clip)
        assert result == select_highlights(info, actions[::-1], imp, clip)
        assert budget_ok(result, duration)
        assert all(h.end-h.start >= 4-1e-9 for h in result)


def test_empty_features_and_all_zero_signal_are_supported():
    info = VideoInfo("synthetic", "unused", 60, 30, 64, 64)
    imp, clip = Importance(np.array([]), np.array([])), ClipFeats(np.array([]), np.empty((0, 0)), "empty")
    result = select_highlights(info, [], imp, clip)
    assert result and budget_ok(result, 60)
    assert all(h.importance == 0 for h in result)


def test_bad_feature_grid_and_action_values_fail():
    info, imp, clip = fixture()
    clip.times = clip.times + .1
    with pytest.raises(ValueError, match="time grid"):
        select_highlights(info, [], imp, clip)
    with pytest.raises(ValueError, match="Invalid action"):
        select_highlights(info, [Action("cut", "tomato", 0, 4, float("nan"))], imp, None)


def test_candidate_pool_is_bounded_distinct_and_in_range():
    info, imp, clip = fixture(600, 300)
    actions = [Action("cut", "tomato", i, i+4, .9) for i in range(500)]
    pool = build_candidates(info, actions, imp, max_candidates=100)
    assert len(pool) <= 100 and len({(c.start, c.end) for c in pool}) == len(pool)
    assert all(0 <= c.start < c.end <= info.duration and c.end-c.start >= 4-1e-9 for c in pool)


def test_importance_integration_on_irregular_pts_has_known_answer():
    curve = ImportanceCurve(10, Importance(np.array([1., 3., 7.]), np.array([0., 1., 0.])))
    assert curve.integral(0, 10) == pytest.approx(3)
    assert curve.integral(1, 3) == pytest.approx(1)
    assert curve.integral(3, 7) == pytest.approx(2)


def test_existing_handoff_loader_can_import_actual_optimizer():
    import importlib.util
    from pathlib import Path
    # The existing M2 script executes the module without registering sys.modules.
    spec = importlib.util.spec_from_file_location("m3_dynamic_handoff", Path(__file__).parents[1]/"code"/"optimizer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    info, imp, clip = fixture(60, 10)
    assert module.select_highlights(info, [], imp, clip) == select_highlights(info, [], imp, clip)
