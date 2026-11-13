"""M4 regression test: the M1 ffmpeg frame reader must not hang on a video with many damaged packets.

P30_08 (EPIC-KITCHENS, official MD5) hung the full run: ffmpeg wrote its decode errors to an
unread stderr pipe, filled it and blocked, while Python blocked reading stdout.
"""
import subprocess
import threading

from test_video_io import _damage_packets

import slowfast_features as sf


def test_ffmpeg_frames_survive_lots_of_decoder_errors(tmp_path):
    src, damaged = tmp_path / "src.mp4", tmp_path / "damaged.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
                    "-t", "30", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src)],
                   check=True)
    _damage_packets(src, damaged, set(range(1, 900, 2)))   # ~180 KB of ffmpeg errors, more than any pipe buffer
    result = {}

    def read():
        try:
            result["frames"] = sum(1 for _ in sf.ffmpeg_frames(damaged, sf.Preprocess(), sf.FrameSource(),
                                                               size=(64, 36)))
        except Exception as error:   # surfaced below
            result["error"] = error

    worker = threading.Thread(target=read, daemon=True)
    worker.start()
    worker.join(timeout=120)
    assert not worker.is_alive(), "ffmpeg_frames deadlocked on a video with many damaged packets"
    assert "error" not in result, result.get("error")
    assert abs(result["frames"] - 30 * 30) <= 2   # 60 fps grid, every 2nd frame: 30 per second
