"""M1: NVDEC option and filter order in slowfast_features.ffmpeg_frames (no GPU needed)."""
import shutil

import numpy as np
import pytest

import slowfast_features as sf


def test_select_runs_before_scale():
    vf = sf._ffmpeg_vf(sf.FrameSource(), 456, 256)
    assert vf.index("select=") < vf.index("scale=")
    assert vf.startswith("fps=60.0,")


@pytest.mark.parametrize("mode,device,accels,want", [
    ("none", "cuda", {"cuda"}, False),
    ("cuda", "cpu", set(), True),
    ("auto", "cpu", {"cuda"}, False),
    ("auto", "cuda", {"cuda"}, True),
    ("auto", "cuda", {"vaapi"}, False),
])
def test_use_nvdec(monkeypatch, mode, device, accels, want):
    import config
    monkeypatch.setattr(config, "DEVICE", device)
    monkeypatch.setattr(sf, "_HWACCELS", frozenset(accels))
    assert sf.use_nvdec(sf.FrameSource(hwaccel=mode)) is want


def test_bad_hwaccel_rejected():
    with pytest.raises(ValueError):
        sf.use_nvdec(sf.FrameSource(hwaccel="vulkan"))


def _fake_raw(fail_hw_after):
    calls = []

    def raw(cmd, path, w, h, counter):
        calls.append(cmd)
        for i in range(3):
            if "-hwaccel" in cmd and i == fail_hw_after:
                raise RuntimeError("cuda init failed")
            counter[0] += 1
            yield np.full((h, w, 3), i, np.uint8)
    return raw, calls


def test_nvdec_failure_before_first_frame_retries_on_cpu(monkeypatch):
    raw, calls = _fake_raw(fail_hw_after=0)
    monkeypatch.setattr(sf, "_raw_frames", raw)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")
    fr = list(sf.ffmpeg_frames("x.mp4", src=sf.FrameSource(hwaccel="cuda"), size=(8, 4)))
    assert len(fr) == 3
    assert "-hwaccel" in calls[0] and "-hwaccel" not in calls[1]


def test_nvdec_failure_mid_stream_is_not_masked(monkeypatch):
    raw, calls = _fake_raw(fail_hw_after=2)
    monkeypatch.setattr(sf, "_raw_frames", raw)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")
    with pytest.raises(RuntimeError, match="cuda init failed"):
        list(sf.ffmpeg_frames("x.mp4", src=sf.FrameSource(hwaccel="cuda"), size=(8, 4)))
    assert len(calls) == 1


def test_cpu_path_has_no_hwaccel(monkeypatch):
    raw, calls = _fake_raw(fail_hw_after=99)
    monkeypatch.setattr(sf, "_raw_frames", raw)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert len(list(sf.ffmpeg_frames("x.mp4", src=sf.FrameSource(hwaccel="none"), size=(8, 4)))) == 3
    assert "-hwaccel" not in calls[0]
