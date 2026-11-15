"""scripts/make_demo_reel.py: overlay timing, action picking, and an end-to-end render."""
import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_demo_reel", ROOT / "scripts" / "make_demo_reel.py")
reel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reel)

PRED = {
    "team_id": "T", "video_id": "synthetic", "duration_sec": 12.0,
    "label": {"name": "cooking", "score": 0.5},
    "actions": [
        {"segment": [1.0, 2.0], "verb": "open", "noun": "door", "score": 0.9},     # outside both highlights
        {"segment": [2.5, 3.5], "verb": "take", "noun": "cup", "score": 0.2},
        {"segment": [3.0, 6.0], "verb": "wash", "noun": "pan", "score": 0.6},
        {"segment": [4.0, 5.0], "verb": "cut", "noun": "onion", "score": 0.4},
        {"segment": [8.0, 10.0], "verb": "put", "noun": "plate", "score": 0.7},
    ],
    "highlights": [{"segment": [3.0, 7.0]}, {"segment": [7.5, 11.52]}],
}


def test_plan_uses_whole_frames_after_the_title():
    clips = reel.plan(PRED, title_sec=2.0)
    assert [(b, e) for b, e, _, _ in clips] == [(3.0, 7.0), (7.5, 11.52)]
    assert [frames for *_, frames in clips] == [120, 121]          # 4.02 s rounds to 121 frames
    assert clips[0][2] == 2.0 and clips[1][2] == pytest.approx(2.0 + 120 / 30)


def test_overlay_picks_top_scores_inside_the_clip_in_time_order():
    picked = reel.overlay_actions(PRED, 3.0, 7.0, k=2)
    assert [(a["verb"], a0, a1) for a0, a1, a in picked] == [("wash", 3.0, 6.0), ("cut", 4.0, 5.0)]
    assert reel.overlay_actions(PRED, 3.0, 7.0, k=0) == []


def test_ass_events_land_at_reel_times():
    ass, total = reel.build_ass(PRED, 1280, 720, title_sec=2.0, k_actions=3)
    assert total == pytest.approx(2.0 + 241 / 30)
    # "wash pan" runs 3.0-6.0 s in the source = 2.0-5.0 s in the reel
    assert any(line.startswith("Dialogue: 3,0:00:02.00,0:00:05.00,Action") and "wash pan" in line
               for line in ass.splitlines())
    # "put plate" 8.0-10.0 s in the source; clip 2 starts at 7.5 s in the source and 6.0 s in the reel
    assert any(line.startswith("Dialogue: 3,0:00:06.50,0:00:08.50,Action") and "put plate" in line
               for line in ass.splitlines())
    assert "open door" not in ass.split("[Events]")[1].split("HIGHLIGHT")[1]
    assert "HIGHLIGHT 2/2" in ass and "PlayResX: 1280" in ass


def _ffmpeg_can_render():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return False
    filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return " subtitles " in filters and "libx264" in encoders


@pytest.mark.skipif(not _ffmpeg_can_render(), reason="needs ffmpeg with libass and libx264")
def test_end_to_end_reel_has_the_planned_length(tmp_path):
    video = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=12",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    out = tmp_path / "reel" / "out.mp4"
    total = reel.make_reel(video, PRED, out, width=320, height=180, title_sec=1.0)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                            "-show_entries", "stream=width,height,nb_read_frames", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True, check=True).stdout.strip()
    width, height, frames = (int(x) for x in probe.split(","))
    assert (width, height) == (320, 180)
    assert frames == 30 + 120 + 121
    assert total == pytest.approx(frames / 30)


def test_no_highlights_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no highlights"):
        reel.make_reel(tmp_path / "x.mp4", {**PRED, "highlights": []}, tmp_path / "o.mp4")
