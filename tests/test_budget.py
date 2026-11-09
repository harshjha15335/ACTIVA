import random
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from eval.metrics import budget_ok, union_duration
from optimizer import select_highlights, uniform_intervals


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


def test_typed_stub_uses_m4_highlight_without_owning_contracts(monkeypatch):
    @dataclass
    class Highlight:
        start: float
        end: float
        importance: float
    # Test-only stand-in. Production must use M4's real contracts.py.
    monkeypatch.setitem(sys.modules, "contracts", SimpleNamespace(Highlight=Highlight))
    rng = np.random.default_rng(42)
    for duration in (0, 2, 30, 120, 600):
        times = np.arange(0, duration, 0.5)
        for values in (np.zeros(len(times)), rng.random(len(times))):
            imp = SimpleNamespace(times=times, values=values)
            info = SimpleNamespace(duration=duration)
            result = select_highlights(info, [], imp, None)
            assert result == select_highlights(info, [], imp, None)
            assert budget_ok(result, duration)
            assert all(isinstance(h, Highlight) and 0 <= h.importance <= 1 for h in result)


def test_stub_validates_importance(monkeypatch):
    monkeypatch.setitem(sys.modules, "contracts", SimpleNamespace(Highlight=object))
    with pytest.raises(ValueError):
        select_highlights(SimpleNamespace(duration=30), [],
                          SimpleNamespace(times=[0, 0], values=[0, 1]), None)
