"""M1 (#36 F2): check that the FFmpeg < 5.1-compatible decode gives the same frames and actions.

    ACTIVA_DEVICE=cuda python kaggle/m1_ffmpeg_compat_check.py --video <path to P30_06.MP4>

old: fps -> scale -> select, `-fps_mode passthrough`   (main before this change; FFmpeg >= 5.1 only)
new: fps -> scale -> select -> setpts=N/(30*TB), `-r 30` (this branch; any FFmpeg)
Compares the frame count and max abs pixel difference over the whole video, then runs
`actions.detect_actions` with each command and compares the action lists (segments, labels,
scores). Prints a JSON report; with --out also writes it. Dev tooling, needs FFmpeg >= 5.1
for the `old` side.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out")
    ap.add_argument("--skip-actions", action="store_true")
    args = ap.parse_args()

    import slowfast_features as sf
    import video_io
    from slowfast_features import FrameSource, Preprocess

    new_args = sf._ffmpeg_rate_args

    def old_args(src, out_w, out_h):
        vf = f"fps={src.fps},scale={out_w}:{out_h}:flags=bicubic,{sf._ffmpeg_select(src)}"
        return ["-vf", vf, "-fps_mode", "passthrough"]

    version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    info = video_io.inspect_video(args.video)
    p = Preprocess()
    w, h = (info.height, info.width) if info.rotation in (90, 270) else (info.width, info.height)
    size = p.out_size(w, h)
    report = {"video": Path(args.video).name, "duration_sec": info.duration, "ffmpeg": version}

    def frames(rate_args):
        sf._ffmpeg_rate_args = rate_args
        try:
            yield from sf.ffmpeg_frames(args.video, p, FrameSource(), size=size)
        finally:
            sf._ffmpeg_rate_args = new_args

    n_old = n_new = 0
    diff = 0
    old_it, new_it = frames(old_args), frames(new_args)
    while True:
        a, b = next(old_it, None), next(new_it, None)
        if a is None or b is None:
            n_old += a is not None; n_new += b is not None
            n_old += sum(1 for _ in old_it); n_new += sum(1 for _ in new_it)
            break
        n_old += 1; n_new += 1
        diff = max(diff, int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()))
    report["frames"] = {"old": n_old, "new": n_new, "max_abs_pixel_diff": diff}
    print("frames", report["frames"], file=sys.stderr, flush=True)

    if not args.skip_actions:
        import actions
        out = {}
        for name, ra in (("old", old_args), ("new", new_args)):
            sf._ffmpeg_rate_args = ra
            try:
                out[name] = [(a.start, a.end, a.verb, a.noun, float(a.score)) for a in actions.detect_actions(info)]
            finally:
                sf._ffmpeg_rate_args = new_args
        a, b = out["old"], out["new"]
        same_len = len(a) == len(b)
        report["actions"] = {
            "old": len(a), "new": len(b),
            "identical": a == b,
            "same_segments_and_labels": same_len and all(x[:4] == y[:4] for x, y in zip(a, b)),
            "max_abs_score_diff": max((abs(x[4] - y[4]) for x, y in zip(a, b)), default=0.0) if same_len else None,
        }

    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
