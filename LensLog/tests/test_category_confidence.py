"""Comprehensive unit tests for M2 category confidence scoring and softmax calibration."""
import numpy as np
import pytest
from category import softmax, classify_video
from contracts import ClipFeats, Importance, VideoInfo


def test_softmax_basic_properties():
    logits = np.array([1.0, 2.0, 3.0])
    probs = softmax(logits, temperature=1.0)
    assert probs.shape == (3,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert sum(probs) == pytest.approx(1.0)
    # Monotonicity / rank preservation
    assert probs[2] > probs[1] > probs[0]


def test_softmax_numerical_stability_extremes():
    # Extremely large numbers should not overflow to NaN or Inf
    large_logits = np.array([1000.0, 1001.0, 999.0])
    probs = softmax(large_logits, temperature=1.0)
    assert np.all(np.isfinite(probs))
    assert sum(probs) == pytest.approx(1.0)
    assert probs[1] > probs[0] > probs[2]

    # Extremely negative numbers
    neg_logits = np.array([-1000.0, -1001.0, -999.0])
    probs_neg = softmax(neg_logits, temperature=1.0)
    assert np.all(np.isfinite(probs_neg))
    assert sum(probs_neg) == pytest.approx(1.0)
    assert probs_neg[2] > probs_neg[0] > probs_neg[1]


def test_softmax_temperature_scaling():
    logits = np.array([0.5, 0.2, 0.1])
    # Lower temperature -> sharper distribution
    sharp = softmax(logits, temperature=0.01)
    assert sharp[0] > 0.99
    # Higher temperature -> flatter distribution
    flat = softmax(logits, temperature=10.0)
    assert flat[0] == pytest.approx(flat[1], abs=0.1)


def test_softmax_ties_and_uniformity():
    logits = np.array([0.4, 0.4, 0.4, 0.4])
    probs = softmax(logits)
    assert np.allclose(probs, 0.25)
    assert sum(probs) == pytest.approx(1.0)


def test_classify_video_confidence_bounds(monkeypatch, tmp_path):
    import category
    path = tmp_path / 'absent_vocab.json'
    monkeypatch.setenv('ACTIVA_VOCAB', str(path))
    labels_path = tmp_path / 'test_labels.json'
    labels = [f"activity_{i}" for i in range(16)]
    labels_path.write_text(str(labels).replace("'", '"'))
    monkeypatch.setenv('ACTIVA_CATEGORIES', str(labels_path))

    def fake_text_features(prompts, cfg):
        vectors = np.zeros((len(prompts), 512), np.float32)
        for i in range(len(prompts)):
            vectors[i, i % 512] = 1.0
        return vectors

    monkeypatch.setattr(category, 'text_features', fake_text_features)

    info = VideoInfo(video_id='test', path='', duration=10.0, fps=10, width=64, height=48, rotation=0)
    times = np.linspace(0.1, 9.9, 20)
    feats = np.zeros((20, 512), dtype=np.float32)
    feats[:, 0] = 1.0  # matches first prompt/category
    clip = ClipFeats(times=times, feats=feats, model_name='test-model')
    imp = Importance(times=times, values=np.ones(20, dtype=np.float32) * 0.5)

    res = classify_video(info, clip, imp)
    assert res.label in labels
    assert len(res.scores) == 16
    assert sum(res.scores.values()) == pytest.approx(1.0, rel=1e-5)
    for k, v in res.scores.items():
        assert 0.0 <= v <= 1.0
        assert np.isfinite(v)
    # The matching label should have highest probability
    assert res.scores[res.label] == max(res.scores.values())
