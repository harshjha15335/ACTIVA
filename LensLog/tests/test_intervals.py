import random

import pytest

from eval.metrics import merge_intervals, union_duration


def test_merge_adjacent_nested_and_unsorted():
    intervals = [dict(start=5, end=10), dict(start=1, end=5), dict(start=2, end=3)]
    assert merge_intervals(intervals) == [(1, 10)]
    assert merge_intervals([]) == []


def test_union_matches_independent_integer_grid_oracle():
    rng = random.Random(18)
    for _ in range(300):
        intervals, selected = [], set()
        for _ in range(rng.randrange(20)):
            start = rng.randrange(100)
            end = rng.randrange(start + 1, 101)
            intervals.append(dict(start=start, end=end))
            selected.update(range(start, end))
        assert union_duration(intervals) == len(selected)


@pytest.mark.parametrize("start,end", [(2, 1), (0, 0), (-1, 1), (0, float("inf"))])
def test_invalid_intervals_raise(start, end):
    with pytest.raises(ValueError):
        merge_intervals([dict(start=start, end=end)])
