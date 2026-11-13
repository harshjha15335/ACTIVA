import subprocess

import numpy as np
import pytest

import video_io
from contracts import CLIP_FPS
from video_io import VideoError


def test_inspect_short(media):
    info = video_io.inspect_video(media["short"])
    assert info.video_id == "short"
    assert info.duration == pytest.approx(120.0, abs=0.1)
    assert info.fps == pytest.approx(15.0)
    assert (info.width, info.height, info.rotation) == (320, 180, 0)


def test_frames_on_two_fps_grid(media):
    frames = list(video_io.iter_frames(media["short"], CLIP_FPS))
    times = np.array([t for t, _ in frames])
    assert times[0] == 0.0
    assert np.all(np.diff(times) > 0)
    assert len(times) == pytest.approx(120 * CLIP_FPS, abs=2)
    # each time is the first real frame at/after its slot: within one source frame
    slots = np.floor(times * CLIP_FPS + 1e-8) / CLIP_FPS
    assert np.all(times - slots < 1 / 15 + 1e-6)
    rgb = frames[0][1]
    assert rgb.dtype == np.uint8 and rgb.shape == (180, 320, 3)


def test_vfr_uses_real_timestamps(media):
    times = [t for t, _ in video_io.iter_frames(media["vfr"], CLIP_FPS)]
    assert all(b > a for a, b in zip(times, times[1:]))
    assert not any(4.0 < t < 8.0 for t in times), "frames inside the removed gap were invented"
    assert any(t >= 8.0 for t in times)


def test_rotation_matches_ffmpeg_display_orientation(media):
    info = video_io.inspect_video(media["rotated"])
    assert info.rotation == 90
    _, ours = next(iter(video_io.iter_frames(media["rotated"], CLIP_FPS)))
    assert ours.shape == (320, 180, 3)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(media["rotated"]), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    reference = np.frombuffer(raw, np.uint8).reshape(320, 180, 3).astype(int)   # ffmpeg auto-rotates
    same = np.abs(ours.astype(int) - reference).mean()
    flipped = np.abs(np.rot90(ours, 2).astype(int) - reference).mean()
    assert same < 3 < flipped


def test_black_video_decodes(media):
    frames = list(video_io.iter_frames(media["black"], CLIP_FPS))
    assert len(frames) > 200 and max(rgb.max() for _, rgb in frames) < 30


@pytest.mark.parametrize("name", ["corrupt", "empty", "audio_only", "not_video"])
def test_bad_inputs_raise_video_error(media, name):
    with pytest.raises(VideoError):
        video_io.inspect_video(media[name])


def test_missing_input_raises_video_error(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        video_io.inspect_video(tmp_path / "missing.mp4")


def test_bad_target_fps(media):
    with pytest.raises(ValueError):
        next(video_io.iter_frames(media["short"], 0))


def _damage_packets(src, dst, bad):
    """Remux src's video, overwriting the 4-byte NAL length of packets in bad (as in two EPIC files)."""
    import av
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        stream = inp.streams.video[0]
        target = out.add_stream_from_template(stream)
        for n, packet in enumerate(inp.demux(stream)):
            if packet.dts is None:
                continue
            if n in bad:
                broken = av.Packet(b"\xff\xff\xff\xff" + bytes(packet)[4:])
                broken.pts, broken.dts, broken.time_base = packet.pts, packet.dts, packet.time_base
                broken.is_keyframe = packet.is_keyframe
                packet = broken
            packet.stream = target
            out.mux(packet)


def test_damaged_packets_are_skipped_not_fatal(tmp_path, caplog):
    import av
    src, damaged = tmp_path / "src.mp4", tmp_path / "damaged.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15",
                    "-t", "20", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src)],
                   check=True)
    _damage_packets(src, damaged, {40, 41, 150})
    with pytest.raises(av.error.InvalidDataError):   # a plain decode loop stops at the first bad packet
        with av.open(str(damaged)) as container:
            for _ in container.decode(video=0):
                pass
    clean = [t for t, _ in video_io.iter_frames(src, CLIP_FPS)]
    with caplog.at_level("WARNING", logger="activa.video_io"):
        times = [t for t, _ in video_io.iter_frames(damaged, CLIP_FPS)]
    assert len(times) == len(clean) == 40 and times[-1] > 19
    assert np.all(np.diff(times) > 0) and np.max(np.diff(times)) < 1.0
    assert "skipped 3 damaged packet" in caplog.text