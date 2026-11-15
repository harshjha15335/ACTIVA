"""M1 (#15 speed item): measure decode speed-ups and prove they don't change the frames/features.

    ACTIVA_DEVICE=cuda python kaggle/m1_decode_benchmark.py --video data/videos/.../P01_11.MP4

Compares three ffmpeg pipelines on one video:
  old_cpu   fps -> scale -> select   (main before this change), CPU decode
  new_cpu   fps -> select -> scale   (resize only the kept frames), CPU decode
  new_nvdec fps -> select -> scale   with -hwaccel cuda (NVDEC)
Reports decode-only wall time and frames/s for each, the max abs pixel difference of
new_cpu and new_nvdec against old_cpu (frame by frame, whole video), and then full
`extract_features` time and max abs feature / v_prob / n_prob difference for CPU vs NVDEC
decoding. Writes a JSON report (default dev_cache/m1_decode_benchmark.json). Dev tooling.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=str(ROOT / "dev_cache" / "m1_decode_benchmark.json"))
    ap.add_argument("--skip-features", action="store_true")
    args = ap.parse_args()

    import config
    import slowfast_features as sf
    import video_io
    from slowfast_features import FrameSource, Preprocess

    p = Preprocess()
    info = video_io.inspect_video(args.video)
    w, h = (info.height, info.width) if info.rotation in (90, 270) else (info.width, info.height)
    size = p.out_size(w, h)
    have_cuda = "cuda" in sf.ffmpeg_hwaccels()
    report = {"video": Path(args.video).name, "duration_sec": info.duration, "size_in": [w, h], "size_out": size,
              "ffmpeg_hwaccels": sorted(sf.ffmpeg_hwaccels()), "device": config.DEVICE, "decode": {}}

    def old_cpu():
        src = FrameSource(hwaccel="none")
        vf = f"fps={src.fps},scale={size[0]}:{size[1]}:flags=bicubic,{sf._ffmpeg_select(src)}"
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(args.video), "-vf", vf, "-fps_mode", "passthrough",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        return sf._raw_frames(cmd, args.video, size[0], size[1], [0])

    variants = {"old_cpu": old_cpu,
                "new_cpu": lambda: sf.ffmpeg_frames(args.video, p, FrameSource(hwaccel="none"), size=size)}
    if have_cuda:
        variants["new_nvdec"] = lambda: sf.ffmpeg_frames(args.video, p, FrameSource(hwaccel="cuda"), size=size)

    for name, make in variants.items():          # timing pass, one at a time
        t0 = time.time()
        n = sum(1 for _ in make())
        dt = time.time() - t0
        report["decode"][name] = {"frames": n, "sec": round(dt, 2), "fps": round(n / dt, 1),
                                  "x_realtime": round(info.duration / dt, 2)}
        print(name, report["decode"][name], file=sys.stderr, flush=True)

    gens = {k: make() for k, make in variants.items()}   # equality pass, frame by frame
    diff = {k: 0 for k in variants if k != "old_cpu"}
    frames = 0
    for frs in zip(*gens.values()):
        ref = frs[0].astype(np.int16)
        for k, f in zip(list(gens)[1:], frs[1:]):
            diff[k] = max(diff[k], int(np.abs(ref - f.astype(np.int16)).max()))
        frames += 1
    report["max_abs_pixel_diff_vs_old_cpu"] = diff
    report["frames_compared"] = frames
    print("pixel diff", diff, "over", frames, "frames", file=sys.stderr, flush=True)

    if not args.skip_features:
        from action_config import engine_assets
        a = engine_assets()
        bundle = sf.load_slowfast(a.slowfast_code, a.slowfast_cfg, a.slowfast_ckpt, config.DEVICE)
        feats = {}
        for name, hw in (("cpu_decode", "none"), ("nvdec_decode", "cuda")):
            if hw == "cuda" and not have_cuda:
                continue
            t0 = time.time()
            feats[name] = sf.extract_features(args.video, bundle, p, source=FrameSource(hwaccel=hw))
            report.setdefault("extract_features_sec", {})[name] = round(time.time() - t0, 1)
            print(name, report["extract_features_sec"][name], "s", file=sys.stderr, flush=True)
        if len(feats) == 2:
            a_, b_ = feats["cpu_decode"], feats["nvdec_decode"]
            report["feature_max_abs_diff"] = {k: float(np.abs(a_[k] - b_[k]).max()) for k in ("feats", "v_prob", "n_prob")}
            report["windows"] = [len(a_["feats"]), len(b_["feats"])]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
