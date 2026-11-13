import math

import numpy as np
import pytest

from action_config import PairingSettings
from actions import (PairDetections, build_pair_prior, detect_actions, expected_num_windows,
                     finalize, grid_to_seconds, nms_pairs, pair_detections, seconds_to_grid,
                     tiou_matrix, window_centres)
from contracts import Action, VideoInfo

VERBS = [f"v{i}" for i in range(97)]
NOUNS = [f"n{i}" for i in range(300)]


# ---------------- time conversion ---------------- #
def test_window_zero_centre_matches_actionformer():
    # window 0 covers frames [0, 32) at 30 fps -> centre at 16/30 s
    assert grid_to_seconds(0) == pytest.approx(16 / 30)
    assert grid_to_seconds(1) == pytest.approx(32 / 30)


def test_grid_seconds_roundtrip():
    t = np.array([0.0, 0.5, 12.34, 599.9])
    assert np.allclose(grid_to_seconds(seconds_to_grid(t)), t)


def test_window_centres_inside_video():
    for duration in (120.0, 300.0, 600.0):
        n = expected_num_windows(duration)
        c = window_centres(n)
        assert len(c) == n
        assert np.all(np.diff(c) > 0)
        assert c[-1] <= duration
        # roughly one window every 16/30 s
        assert n == pytest.approx(duration * 30 / 16, abs=3)


def test_short_video_has_no_full_window():
    assert expected_num_windows(0.5) == 0


# ---------------- tIoU ---------------- #
def test_tiou_known_values():
    m = tiou_matrix([[10, 20]], [[12, 22], [20, 30], [10, 20]])
    assert m[0, 0] == pytest.approx(8 / 12)
    assert m[0, 1] == 0.0
    assert m[0, 2] == 1.0


def test_tiou_zero_length_safe():
    assert tiou_matrix([[5, 5]], [[5, 5]])[0, 0] == 0.0


# ---------------- prior ---------------- #
def test_prior_rows_are_distributions():
    p = build_pair_prior([0, 0, 0, 1], [3, 3, 4, 5])
    assert p.shape == (97, 300)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p[0, 3] > p[0, 4] > p[0, 6]


# ---------------- pairing ---------------- #
def _pair(**kw):
    s = PairingSettings(**{"exp_prior": 0.0, **kw})
    return pair_detections(
        v_segs=[[10, 20], [100, 110]], v_labels=[1, 2], v_scores=[0.9, 0.8],
        n_segs=[[11, 19], [50, 60], [101, 109]], n_labels=[7, 8, 9], n_scores=[0.5, 0.99, 0.4],
        s=s)


def test_pairs_only_from_overlapping_segments():
    p = _pair()
    got = sorted(zip(p.verbs.tolist(), p.nouns.tolist()))
    assert got == [(1, 7), (2, 9)]          # noun 8 at 50-60 s overlaps no verb
    assert np.allclose(sorted(p.scores), sorted([0.9 * 0.5, 0.8 * 0.4]))


def test_pairing_respects_valid_pair_filter():
    p = pair_detections([[10, 20]], [1], [0.9], [[11, 19], [12, 18]], [7, 8], [0.5, 0.6],
                        is_valid_pair=lambda v, n: n != 8, s=PairingSettings(exp_prior=0.0))
    assert p.nouns.tolist() == [7]


def test_top_k_nouns_per_verb():
    p = pair_detections([[0, 10]], [0], [1.0], [[0, 10]] * 5, [1, 2, 3, 4, 5],
                        [0.1, 0.5, 0.3, 0.9, 0.2], s=PairingSettings(exp_prior=0.0, top_k_nouns=2))
    assert sorted(p.nouns.tolist()) == [2, 4]


def test_prior_changes_ranking():
    prior = np.full((97, 300), 1e-3)
    prior[0, 1] = 1.0
    p = pair_detections([[0, 10]], [0], [1.0], [[0, 10], [0, 10]], [1, 2], [0.4, 0.5],
                        prior=prior, s=PairingSettings(exp_prior=1.0, top_k_nouns=1))
    assert p.nouns.tolist() == [1]


def test_empty_inputs_give_empty_pairs():
    assert len(pair_detections([], [], [], [[0, 1]], [0], [1.0])) == 0
    assert len(pair_detections([[0, 1]], [0], [1.0], [], [], [])) == 0


# ---------------- NMS ---------------- #
def test_nms_suppresses_same_pair_only():
    p = PairDetections(np.array([1, 1, 1, 2]), np.array([7, 7, 7, 7]),
                       np.array([[0, 10], [1, 11], [30, 40], [0, 10]], float),
                       np.array([0.9, 0.8, 0.7, 0.6]))
    k = nms_pairs(p, 0.5)
    kept = sorted(zip(k.verbs.tolist(), k.segs[:, 0].tolist()))
    assert kept == [(1, 0.0), (1, 30.0), (2, 0.0)]


# ---------------- finalize ---------------- #
def test_finalize_invariants_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        duration = float(rng.uniform(30, 600))
        n = int(rng.integers(0, 60))
        starts = rng.uniform(-20, duration + 20, n)
        segs = np.stack([starts, starts + rng.uniform(-1, 30, n)], axis=1)
        p = PairDetections(rng.integers(0, 97, n), rng.integers(0, 300, n), segs,
                           rng.uniform(-0.2, 1.2, n))
        out = finalize(p, duration, VERBS, NOUNS, PairingSettings(max_actions=40))
        assert len(out) <= 40
        prev = math.inf
        for a in out:
            assert isinstance(a, Action)
            assert all(math.isfinite(x) for x in (a.start, a.end, a.score))
            assert 0.0 <= a.start < a.end <= duration
            assert 0.0 <= a.score <= 1.0
            assert a.score <= prev
            prev = a.score
            assert a.verb in VERBS and a.noun in NOUNS


def test_finalize_rejects_bad_duration():
    with pytest.raises(ValueError):
        finalize(PairDetections.empty(), 0.0, VERBS, NOUNS)


def test_finalize_is_deterministic():
    p = PairDetections(np.array([1, 2, 3]), np.array([4, 5, 6]),
                       np.array([[0, 5], [3, 9], [10, 12]], float), np.array([0.5, 0.5, 0.7]))
    assert finalize(p, 60, VERBS, NOUNS) == finalize(p, 60, VERBS, NOUNS)


# ---------------- engine wiring ---------------- #
def test_detect_actions_fails_loudly_without_assets(tmp_path, monkeypatch):
    import actions
    monkeypatch.setenv("ACTIVA_M1_ASSETS", str(tmp_path / "none"))
    actions._ENGINE.clear()
    info = VideoInfo("x", "/tmp/x.mp4", 120.0, 30.0, 640, 360)
    with pytest.raises(actions.ActionEngineUnavailable):
        detect_actions(info)


def test_load_class_keys_uses_id_column(tmp_path):
    from actions import load_class_keys
    f = tmp_path / "c.csv"
    f.write_text("id,key,instances,category\n1,put,5,x\n0,take,9,x\n2,turn-on,1,x\n", encoding="utf-8")
    assert load_class_keys(f) == ["take", "put", "turn-on"]
    f.write_text("id,key\n0,take\n2,put\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_class_keys(f)


def test_map_to_vocab_rejects_unknown():
    from actions import map_to_vocab
    assert map_to_vocab(["take", "put"], ["take", "put", "other"], "verb") == ["take", "put"]
    with pytest.raises(ValueError):
        map_to_vocab(["take", "yeet"], ["take"], "verb")


def test_detections_to_actions_end_to_end():
    from actions import detections_to_actions
    verb = (np.array([[10.0, 20.0], [40.0, 45.0]]), np.array([1, 2]), np.array([0.9, 0.6]))
    noun = (np.array([[11.0, 19.0], [41.0, 44.0]]), np.array([7, 9]), np.array([0.5, 0.8]))
    out = detections_to_actions(verb, noun, 60.0, VERBS, NOUNS, s=PairingSettings(exp_prior=0.0))
    assert [(a.verb, a.noun) for a in out] == [("v2", "n9"), ("v1", "n7")]
    assert out[0].score == pytest.approx(0.6 * 0.8) and out[0].start == 40.0
    assert detections_to_actions(verb, noun, 60.0, VERBS, NOUNS, is_valid_pair=lambda v, n: False) == []
