import numpy as np
import pytest

import slowfast_features as sf
from slowfast_features import Preprocess, iter_windows, resize_frame, to_pathways, window_starts

torch = pytest.importorskip("torch")


def frames(n, h=4, w=6):
    # frame k is filled with value k so windows can be checked by content
    return [np.full((h, w, 3), k % 256, np.uint8) for k in range(n)]


@pytest.mark.parametrize("n", [0, 1, 15, 16, 31, 32, 33, 48, 100, 16848])
def test_window_counts(n):
    assert sf.num_windows(n, rule="floor") == n // 16
    assert sf.num_windows(n, rule="ceil") == -(-n // 16)
    assert sf.num_windows(n, rule="complete") == (max(0, (n - 32) // 16 + 1) if n >= 32 else 0)


@pytest.mark.parametrize("n_frames,published_T", [(885, 55), (957, 59), (458, 28), (755, 47), (16848, 1053)])
def test_floor_rule_matches_published_feature_counts(n_frames, published_T):
    # measured in the Step 4 gate: P02_13, P03_21, P26_30, P28_23, P01_11
    assert len(window_starts(n_frames)) == published_T


def test_unknown_rule_rejected():
    with pytest.raises(ValueError):
        sf.num_windows(10, rule="round")


@pytest.mark.parametrize("rule", ["floor", "ceil", "complete"])
@pytest.mark.parametrize("n", [5, 16, 32, 47, 50, 100])
def test_streaming_windows_match_naive(n, rule):
    fr = frames(n)
    got = list(iter_windows(fr, rule=rule))
    assert [s for s, _ in got] == window_starts(n, rule=rule)
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
    assert out["feats"].shape == (4, 2304)                     # floor(70 / 16)
    assert out["starts"].tolist() == [0, 16, 32, 48]
    assert np.allclose(out["centres"], (np.arange(4) * 16 + 16) / 30)
    assert [c[0] for c in calls] == [3, 1]
    assert calls[0][1:] == (32, 8, 12, 3)


def test_extract_features_empty_video(monkeypatch):
    monkeypatch.setattr(sf, "run_clips", lambda *a: pytest.fail("no windows expected"))
    out = sf.extract_features(None, bundle=None, frames=[])
    assert out["feats"].shape == (0, 2304) and out["n_frames"] == 0


def test_load_slowfast_missing_assets_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        sf.load_slowfast(tmp_path / "nope", tmp_path / "cfg.yaml", tmp_path / "SlowFast.pyth")


def test_ffmpeg_select_expression():
    assert sf._ffmpeg_select(sf.FrameSource(offset=1, step=2)) == r"select='gte(n\,1)*not(mod(n-1\,2))'"


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        sf.extract_features("x.mp4", bundle=None, source=sf.FrameSource(backend="opencv"))


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize("jpeg_q", [None, 4])
def test_ffmpeg_frames_rate_size_and_order(tmp_path, jpeg_q):
    import subprocess
    vid = tmp_path / "t.mp4"
    # 2 s of 60 fps test pattern whose frame counter changes every frame
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=60",
                    "-t", "2", "-pix_fmt", "yuv420p", str(vid)], check=True)
    fr = list(sf.ffmpeg_frames(vid, Preprocess(), sf.FrameSource(jpeg_q=jpeg_q), size=(456, 256)))
    assert len(fr) == 60                      # 120 frames at 60 fps, every 2nd kept
    assert fr[0].shape == (256, 456, 3) and fr[0].dtype == np.uint8
    assert not np.array_equal(fr[0], fr[1])


@pytest.mark.parametrize("shift", [-2, -1, 0, 1, 2])
def test_shift_moves_window_content_but_not_window_count(monkeypatch, shift):
    seen = []

    def fake_run(bundle, clips):
        seen.extend(int(c[0, 0, 0, 0]) for c in clips)   # first frame value of each window
        b = clips.shape[0]
        return (np.zeros((b, 2304), np.float32), np.zeros((b, 97), np.float32), np.zeros((b, 300), np.float32))

    monkeypatch.setattr(sf, "run_clips", fake_run)
    fr = [np.full((8, 12, 3), k, np.uint8) for k in range(70)]
    out = sf.extract_features(None, bundle=None, p=Preprocess(height=8, width=12, color="rgb"),
                              frames=iter(fr), source=sf.FrameSource(shift=shift))
    assert len(out["feats"]) == 70 // 16                  # count follows the real video
    want_first = [max(0, 16 * i + shift) for i in range(4)]
    assert seen == want_first


def test_default_shift_is_minus_one():
    assert sf.FrameSource().shift == -1
