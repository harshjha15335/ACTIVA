import numpy as np
import pytest

import slowfast_features as sf
from slowfast_features import Preprocess, iter_windows, resize_frame, to_pathways, window_starts

torch = pytest.importorskip("torch")


def frames(n, h=4, w=6):
    # frame k is filled with value k so windows can be checked by content
    return [np.full((h, w, 3), k % 256, np.uint8) for k in range(n)]


@pytest.mark.parametrize("n", [0, 1, 31, 32, 33, 48, 100, 16848])
def test_window_counts(n):
    assert len(window_starts(n, pad_last=True)) == (-(-n // 16) if n else 0)
    assert len(window_starts(n, pad_last=False)) == (max(0, (n - 32) // 16 + 1) if n >= 32 else 0)


def test_published_p01_11_count_matches_padded_rule():
    # P01_11: 561.6 s -> 16848 frames at 30 fps; published features have T = 1053.
    assert len(window_starts(16848, pad_last=True)) == 1053


@pytest.mark.parametrize("pad", [True, False])
@pytest.mark.parametrize("n", [5, 32, 47, 50, 100])
def test_streaming_windows_match_naive(n, pad):
    fr = frames(n)
    got = list(iter_windows(fr, pad_last=pad))
    assert [s for s, _ in got] == window_starts(n, pad_last=pad)
    for start, clip in got:
        assert len(clip) == 32
        want = [min(start + k, n - 1) for k in range(32)]   # padding repeats the last frame
        assert [int(c[0, 0, 0]) for c in clip] == want


def test_resize_keeps_epic_frame_size_and_bgr():
    rgb = np.zeros((1080, 1920, 3), np.uint8)
    rgb[..., 0] = 200  # red
    out = resize_frame(rgb, Preprocess())
    assert out.shape == (256, 456, 3)
    assert out[0, 0, 2] == 200 and out[0, 0, 0] == 0          # BGR: red in last channel
    out_rgb = resize_frame(rgb, Preprocess(color="rgb"))
    assert out_rgb[0, 0, 0] == 200


def test_crop_and_bad_color():
    rgb = np.zeros((1080, 1920, 3), np.uint8)
    assert resize_frame(rgb, Preprocess(crop=256)).shape == (256, 256, 3)
    with pytest.raises(ValueError):
        resize_frame(rgb, Preprocess(color="hsv"))


def test_to_pathways_shapes_and_normalisation():
    clips = np.full((2, 32, 8, 10, 3), 255, np.uint8)
    slow, fast = to_pathways(clips, alpha=4)
    assert tuple(fast.shape) == (2, 3, 32, 8, 10)
    assert tuple(slow.shape) == (2, 3, 8, 8, 10)
    assert torch.allclose(fast, torch.full_like(fast, (1.0 - 0.45) / 0.225))


def test_extract_features_with_fake_model(monkeypatch):
    calls = []

    def fake_run(bundle, clips):
        calls.append(clips.shape)
        b = clips.shape[0]
        return (np.ones((b, 2304), np.float32), np.full((b, 97), 1 / 97, np.float32),
                np.full((b, 300), 1 / 300, np.float32))

    monkeypatch.setattr(sf, "run_clips", fake_run)
    out = sf.extract_features(None, bundle=None, p=Preprocess(height=8, width=12),
                              batch_size=3, frames=[np.zeros((16, 24, 3), np.uint8)] * 70)
    assert out["n_frames"] == 70
    assert out["feats"].shape == (5, 2304)                     # ceil(70 / 16)
    assert out["starts"].tolist() == [0, 16, 32, 48, 64]
    assert np.allclose(out["centres"], (np.arange(5) * 16 + 16) / 30)
    assert [c[0] for c in calls] == [3, 2]
    assert calls[0][1:] == (32, 8, 12, 3)


def test_extract_features_empty_video(monkeypatch):
    monkeypatch.setattr(sf, "run_clips", lambda *a: pytest.fail("no windows expected"))
    out = sf.extract_features(None, bundle=None, frames=[])
    assert out["feats"].shape == (0, 2304) and out["n_frames"] == 0


def test_load_slowfast_missing_assets_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        sf.load_slowfast(tmp_path / "nope", tmp_path / "cfg.yaml", tmp_path / "SlowFast.pyth")
