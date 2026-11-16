"""Offline unit tests for an UNAPPROVED title-card mask candidate.

These fixtures test mechanics only; they are not FPVSum or EPIC quality evidence.
"""
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from contracts import ClipFeats, VideoInfo
import importance


@pytest.fixture(autouse=True)
def clean_flags(monkeypatch):
    monkeypatch.delenv("ACTIVA_CARD_MASK", raising=False)
    monkeypatch.delenv("ACTIVA_IMPORTANCE", raising=False)


def _sample(kinds):
    # Three-dimensional toy embeddings; no actual pretrained model is claimed.
    feats = np.array([[1., 0., 0.] if k == "card" else [0., 1., 0.]
                      for k in kinds], dtype=np.float32)
    times = np.arange(len(kinds), dtype=np.float64) * 0.5
    return (VideoInfo("synthetic", "", float(len(kinds)) / 2, 2, 32, 32),
            ClipFeats(times, feats, "toy-test-only"))


def _mock_text_features(monkeypatch):
    # First four prompts are cards, remaining four are footage.
    def vectors(prompts, cfg):
        assert tuple(prompts) == importance.CARD_PROMPTS + importance.FOOTAGE_PROMPTS
        result = np.zeros((len(prompts), 3), dtype=np.float32)
        result[:len(importance.CARD_PROMPTS), 0] = 1
        result[len(importance.CARD_PROMPTS):, 1] = 1
        return result

    monkeypatch.setitem(sys.modules, "clip_backend",
                        types.SimpleNamespace(text_features=vectors))


def test_card_prompts_are_frozen_without_snow_hazard():
    assert len(importance.CARD_PROMPTS) >= 3
    assert len(importance.FOOTAGE_PROMPTS) >= 3
    assert all("white screen" not in s.lower() for s in
               importance.CARD_PROMPTS + importance.FOOTAGE_PROMPTS)


def test_argmax_requires_two_consecutive_frames(monkeypatch):
    _mock_text_features(monkeypatch)
    _, clip = _sample(["card", "card", "footage", "card", "footage",
                       "card", "card", "card"])
    mask = importance.card_frame_mask(clip.feats, object())
    np.testing.assert_array_equal(
        mask, [True, True, False, False, False, True, True, True])


def test_masked_cards_zero_and_real_context_stays_flat(monkeypatch):
    _mock_text_features(monkeypatch)
    monkeypatch.setenv("ACTIVA_CARD_MASK", "1")
    info, clip = _sample(["card", "card", "footage", "footage",
                          "footage", "footage", "footage", "footage"])
    result = importance.predict_importance(info, clip)
    np.testing.assert_array_equal(result.times, clip.times)
    np.testing.assert_array_equal(result.values[:2], [0., 0.])
    np.testing.assert_array_equal(result.values[2:], np.full(6, 0.5))
    assert result.values.dtype == np.float32


def test_single_card_does_not_trigger_mask(monkeypatch):
    _mock_text_features(monkeypatch)
    monkeypatch.setenv("ACTIVA_CARD_MASK", "1")
    info, clip = _sample(["card", "footage", "footage", "footage"])
    mask = importance.card_frame_mask(clip.feats, object())
    assert not mask.any()
    enabled = importance.predict_importance(info, clip).values
    monkeypatch.setenv("ACTIVA_CARD_MASK", "0")
    disabled = importance.predict_importance(info, clip).values
    np.testing.assert_array_equal(enabled, disabled)


def test_all_card_video_has_no_artificial_novelty(monkeypatch):
    _mock_text_features(monkeypatch)
    monkeypatch.setenv("ACTIVA_CARD_MASK", "1")
    info, clip = _sample(["card", "card", "card", "card"])
    np.testing.assert_array_equal(
        importance.predict_importance(info, clip).values,
        np.zeros(4, dtype=np.float32))


def test_flag_off_does_not_request_text_vectors(monkeypatch):
    def fail(*args):
        raise AssertionError("zero-shot text should not be called with flag OFF")
    monkeypatch.setitem(sys.modules, "clip_backend",
                        types.SimpleNamespace(text_features=fail))
    info, clip = _sample(["card", "card", "footage", "footage"])
    result = importance.predict_importance(info, clip).values
    ref = importance.rank_normalize(
        importance.smooth(importance._novelty_raw(clip.times, clip.feats)))
    np.testing.assert_array_equal(result, ref)


def test_invalid_mask_flag_is_rejected(monkeypatch):
    monkeypatch.setenv("ACTIVA_CARD_MASK", "maybe")
    info, clip = _sample(["footage", "footage"])
    with pytest.raises(ValueError, match="ACTIVA_CARD_MASK"):
        importance.predict_importance(info, clip)


def test_other_baseline_is_unchanged(monkeypatch):
    monkeypatch.setenv("ACTIVA_IMPORTANCE", "constant")
    monkeypatch.setenv("ACTIVA_CARD_MASK", "1")
    info, clip = _sample(["card", "card", "footage", "footage"])
    np.testing.assert_array_equal(
        importance.predict_importance(info, clip).values,
        np.full(4, .5, dtype=np.float32))
