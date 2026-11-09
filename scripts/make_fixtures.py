"""Generate synthetic test videos with FFmpeg (no real data needed).

    python scripts/make_fixtures.py <out_dir>

Used by the test suite (conftest.py) and handy for manual run.sh checks.
"""
from pathlib import Path
import shutil
import subprocess
import sys

H264 = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def make_fixtures(out_dir):
    """Create every fixture once; return {name: path}."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = {name: out / f"{name}.mp4" for name in
         ("short", "long", "black", "vfr", "rotated", "corrupt", "empty", "audio_only", "not_video")}

    if not p["short"].exists():   # 2 minutes, with audio
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=22050",
                "-t", "120", *H264, "-c:a", "aac", str(p["short"]))
    if not p["long"].exists():    # 10 minutes
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10", "-t", "600", *H264, str(p["long"]))
    if not p["black"].exists():   # "no action" / importance collapse
        _ffmpeg("-f", "lavfi", "-i", "color=c=black:size=320x180:rate=15", "-t", "130", *H264, str(p["black"]))
    if not p["vfr"].exists():     # frames between 4 s and 8 s removed
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15", "-t", "20",
                "-vf", "select='not(between(t,4,8))'", "-fps_mode", "vfr", *H264, str(p["vfr"]))
    if not p["rotated"].exists():  # 90 degree display-matrix rotation
        src = out / "_rotated_src.mp4"
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15", "-t", "10", *H264, str(src))
        _ffmpeg("-display_rotation:v:0", "90", "-i", str(src), "-c", "copy", str(p["rotated"]))
    if not p["corrupt"].exists():  # first half of a real MP4 (index atom lost)
        data = p["short"].read_bytes()
        p["corrupt"].write_bytes(data[: len(data) // 2])
    if not p["empty"].exists():
        p["empty"].write_bytes(b"")
    if not p["audio_only"].exists():
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=22050", "-t", "5",
                "-c:a", "aac", str(p["audio_only"]))
    if not p["not_video"].exists():
        p["not_video"].write_text("this is not a video\n")
    return p


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/make_fixtures.py <out_dir>")
    for name, path in make_fixtures(sys.argv[1]).items():
        print(f"{name:11s} {path}")
