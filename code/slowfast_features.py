"""Member 1: raw MP4 -> EPIC-100 SlowFast features (ActionFormer input).

ActionFormer's published EPIC features were made with the SlowFast 8x8 R50 model
fine-tuned on EPIC-KITCHENS-100 from github.com/epic-kitchens/epic-kitchens-slowfast
(ActionFormer issue #116), one 2304-D vector per 32-frame window at 30 fps with a
16-frame stride, after 3D global average pooling (issue #70), feeding whole 16:9
frames rather than a centre crop (issue #91). The published .npz files also hold
`v_prob`/`n_prob`: that model's own verb/noun softmax for each window.

What the EPIC training code does (epic-kitchens-slowfast @ bf50519), which our
preprocessing must match:
  * frames are EPIC `rgb_frames` JPEGs (456x256) read with cv2.imread -> **BGR**
    channel order, never converted to RGB;
  * /255, minus mean 0.45, divided by std 0.225 (all channels);
  * 32 frames spaced 2 apart at 60 fps (= consecutive frames at 30 fps);
  * slow pathway = torch.linspace(0, 31, 32 // ALPHA=4) indices of the fast pathway.

Which exact variant reproduces the published features is decided by measurement
(guide Step 4), so colour order, frame size and last-window padding are settings.

Time base: frames come from `video_io.iter_frames(path, 30)`, the team-wide decoder
(seconds from the first decoded frame). Window i starts at sampled frame 16 i, so its
centre is (16 i + 16) / 30 s, the same formula ActionFormer uses (`actions.grid_to_seconds`).

The SlowFast model code itself (Apache-2.0) is loaded from `code_root`, a local copy
of epic-kitchens-slowfast. No network access happens here.
"""
from __future__ import annotations

import math
import sys
import types
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from action_config import GRID, FeatureGrid

MEAN = 0.45
STD = 0.225


@dataclass(frozen=True)
class Preprocess:
    color: str = "bgr"          # "bgr" (as the EPIC training code) or "rgb"
    height: int = 256           # EPIC rgb_frames are 456x256
    width: Optional[int] = None  # None: keep aspect ratio (1920x1080 -> 456x256)
    crop: Optional[int] = None  # None: whole frame (ActionFormer issue #91)
    window_rule: str = "floor"  # T = n // 16, as the published features (see num_windows)

    def out_size(self, src_w: int, src_h: int) -> tuple[int, int]:
        """(width, height) after resizing, before any crop."""
        if self.width is not None:
            return int(self.width), int(self.height)
        w = int(round(self.height * src_w / src_h / 2.0) * 2)  # even width
        return max(w, 2), int(self.height)


def resize_frame(rgb: np.ndarray, p: Preprocess) -> np.ndarray:
    """RGB uint8 HxWx3 -> resized (and optionally centre-cropped) uint8 in p.color order."""
    from PIL import Image

    h, w = rgb.shape[:2]
    out_w, out_h = p.out_size(w, h)
    img = Image.fromarray(rgb)
    if (w, h) != (out_w, out_h):
        img = img.resize((out_w, out_h), Image.BICUBIC)  # ffmpeg's default scaler is bicubic
    arr = np.asarray(img)
    if p.crop is not None:
        c = int(p.crop)
        y0, x0 = max((arr.shape[0] - c) // 2, 0), max((arr.shape[1] - c) // 2, 0)
        arr = arr[y0:y0 + c, x0:x0 + c]
    if p.color == "bgr":
        arr = arr[:, :, ::-1]
    elif p.color != "rgb":
        raise ValueError(f"color must be 'bgr' or 'rgb', got {p.color!r}")
    return np.ascontiguousarray(arr)


@dataclass(frozen=True)
class FrameSource:
    """How frames are decoded. Default reproduces EPIC's frame pipeline with ffmpeg:
    resample to 60 fps, scale to 456x256, keep every 2nd frame (SlowFast sampling rate 2
    at 60 fps = 30 fps). `jpeg_q` additionally round-trips frames through JPEG like the
    EPIC `rgb_frames` dumps the model was trained on (ffmpeg `-q:v`)."""
    backend: str = "ffmpeg"          # "ffmpeg" or "pyav" (video_io.iter_frames at 30 fps)
    fps: float = 60.0
    step: int = 2
    offset: int = 0                  # first kept frame index at `fps`
    jpeg_q: Optional[int] = None     # None: raw frames
    shift: int = -1                  # frames (at 30 fps) to start windows earlier (<0) or later (>0);
                                     # gate v3: -1/-2 best match the published features (0 -> 0.938, -1 -> 0.943)


def _ffmpeg_select(src: FrameSource) -> str:
    return f"select='gte(n\\,{src.offset})*not(mod(n-{src.offset}\\,{src.step}))'"


def ffmpeg_frames(path, p: "Preprocess" = None, src: FrameSource = FrameSource(),
                  size: Optional[tuple[int, int]] = None) -> Iterator[np.ndarray]:
    """Decode with ffmpeg -> RGB uint8 frames at 1/step of src.fps, already resized.

    Time origin: ffmpeg starts at the stream's first frame, so frame k of the output is at
    k * step / fps seconds, the same origin as video_io (PTS - start_time). ffmpeg applies
    the display-matrix rotation by default, like video_io.iter_frames.
    """
    import shutil
    import subprocess
    import tempfile

    from PIL import Image

    p = p or Preprocess()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install FFmpeg.")
    if size is None:
        import video_io
        info = video_io.inspect_video(path)
        w, h = (info.height, info.width) if info.rotation in (90, 270) else (info.width, info.height)
        size = p.out_size(w, h)
    out_w, out_h = size
    vf = f"fps={src.fps},scale={out_w}:{out_h}:flags=bicubic,{_ffmpeg_select(src)}"
    base = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vf", vf, "-fps_mode", "passthrough"]  # -vsync was removed in ffmpeg 7+

    if src.jpeg_q is None:
        cmd = base + ["-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        # stderr goes to a temp file, not a pipe: on a video with many damaged packets ffmpeg
        # writes enough errors to fill an unread stderr pipe and blocks, while we block on stdout.
        with tempfile.TemporaryFile() as errfile:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfile)
            nbytes = out_w * out_h * 3
            try:
                while True:
                    buf = proc.stdout.read(nbytes)
                    if len(buf) < nbytes:
                        break
                    yield np.frombuffer(buf, np.uint8).reshape(out_h, out_w, 3)
            finally:
                proc.stdout.close()
                code = proc.wait()
                errfile.seek(0)
                err = errfile.read().decode(errors="replace")
            if code != 0:
                raise RuntimeError(f"ffmpeg failed on {path}: {err.strip()[-500:]}")
        return

    with tempfile.TemporaryDirectory(prefix="activa_frames_") as tmp:
        cmd = base + ["-q:v", str(int(src.jpeg_q)), f"{tmp}/%010d.jpg"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on {path}: {res.stderr.strip()[-500:]}")
        for f in sorted(Path(tmp).glob("*.jpg")):
            with Image.open(f) as im:
                yield np.asarray(im.convert("RGB"))
            f.unlink()


WINDOW_RULES = ("floor", "ceil", "complete")


def num_windows(n_frames: int, grid: FeatureGrid = GRID, rule: str = "floor") -> int:
    """How many feature windows a video of n sampled (30 fps) frames gets.

    floor   : n // 16. Matches every published ActionFormer EPIC feature file we checked
              (P02_13 885 -> 55, P03_21 957 -> 59, P26_30 458 -> 28, P28_23 755 -> 47,
              P01_11 16848 -> 1053); trailing windows are padded with the last frame.
    ceil    : ceil(n / 16), padded.
    complete: only windows with all 32 frames, (n - 32) // 16 + 1.
    """
    n = int(n_frames)
    if rule not in WINDOW_RULES:
        raise ValueError(f"window rule must be one of {WINDOW_RULES}, got {rule!r}")
    if n <= 0:
        return 0
    if rule == "floor":
        return n // grid.stride
    if rule == "ceil":
        return -(-n // grid.stride)
    return max(0, (n - grid.num_frames) // grid.stride + 1) if n >= grid.num_frames else 0


def window_starts(n_frames: int, grid: FeatureGrid = GRID, rule: str = "floor") -> list[int]:
    """Start index (in 30 fps sampled frames) of every feature window."""
    return [i * grid.stride for i in range(num_windows(n_frames, grid, rule))]


def iter_windows(frames: Iterable[np.ndarray], grid: FeatureGrid = GRID,
                 rule: str = "floor", count_adjust: int = 0) -> Iterator[tuple[int, list]]:
    """Stream (start_index, 32 frames) windows without holding the whole video in memory.

    Complete windows are emitted as soon as their frames arrive; once the stream ends the
    remaining windows required by `rule` are emitted, padded by repeating the last frame.
    `count_adjust` is added to the frame count before applying `rule` (used when frames were
    prepended or dropped by a shift, so the window count still follows the real video).
    """
    if rule not in WINDOW_RULES:
        raise ValueError(f"window rule must be one of {WINDOW_RULES}, got {rule!r}")
    buf: deque = deque()
    first = 0        # sampled-frame index of buf[0]
    next_start = 0
    n = 0
    for frame in frames:
        buf.append(frame)
        n += 1
        while next_start + grid.num_frames <= n:
            lo = next_start - first
            yield next_start, list(buf)[lo:lo + grid.num_frames]
            next_start += grid.stride
            # drop frames no later window needs
            while first < next_start and buf:
                buf.popleft()
                first += 1
    total = num_windows(n + count_adjust, grid, rule)
    if n == 0 or not buf:
        return
    last = buf[-1]
    while next_start < total * grid.stride:
        lo = next_start - first
        clip = list(buf)[lo:]
        clip = clip + [last] * (grid.num_frames - len(clip))
        yield next_start, clip
        next_start += grid.stride


def to_pathways(clips: np.ndarray, alpha: int = 4):
    """uint8 [B, T, H, W, 3] -> [slow, fast] float tensors [B, 3, T', H, W], normalised."""
    import torch

    x = torch.from_numpy(clips).float().div_(255.0).sub_(MEAN).div_(STD)
    x = x.permute(0, 4, 1, 2, 3).contiguous()           # B C T H W
    t = x.shape[2]
    idx = torch.linspace(0, t - 1, t // alpha).long()   # pack_pathway_output
    return [x.index_select(2, idx), x]


# --------------------------------------------------------------------------- #
# Model loading (epic-kitchens-slowfast code, local only)
# --------------------------------------------------------------------------- #
@dataclass
class SlowFastBundle:
    model: object
    device: str
    alpha: int
    _feat: list = field(default_factory=list)


def _stub_detectron2():
    """head_helper imports detectron2's ROIAlign (only used by the AVA head)."""
    if "detectron2.layers" in sys.modules:
        return
    d2 = types.ModuleType("detectron2")
    layers = types.ModuleType("detectron2.layers")

    class ROIAlign:  # pragma: no cover - never instantiated for EPIC
        def __init__(self, *a, **k):
            raise RuntimeError("ROIAlign is not available (detection heads are unused).")

    layers.ROIAlign = ROIAlign
    d2.layers = layers
    sys.modules.setdefault("detectron2", d2)
    sys.modules["detectron2.layers"] = layers


def load_slowfast(code_root, cfg_yaml, checkpoint, device: str = "cpu") -> SlowFastBundle:
    """Build SlowFast 8x8 R50 (EPIC) and load SlowFast.pyth. Raises if anything is missing."""
    import torch

    code_root, cfg_yaml, checkpoint = Path(code_root), Path(cfg_yaml), Path(checkpoint)
    for p in (code_root, cfg_yaml, checkpoint):
        if not p.exists():
            raise FileNotFoundError(f"SlowFast asset not found: {p}")
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    _stub_detectron2()
    from slowfast.config.defaults import get_cfg
    from slowfast.models.video_model_builder import SlowFast

    cfg = get_cfg()
    cfg.merge_from_file(str(cfg_yaml))
    cfg.NUM_GPUS = 1 if device.startswith("cuda") else 0
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    model = SlowFast(cfg)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = state.get("model_state", state)
    model.load_state_dict(state, strict=True)  # raises on any missing/unexpected key
    model.eval().to(device)

    bundle = SlowFastBundle(model=model, device=device, alpha=int(cfg.SLOWFAST.ALPHA))

    def hook(_module, inputs, _output):
        paths = inputs[0]
        bundle._feat.append(torch.cat([p.mean(dim=(2, 3, 4)) for p in paths], dim=1))

    model.head.register_forward_hook(hook)
    return bundle


def run_clips(bundle: SlowFastBundle, clips: np.ndarray):
    """[B, 32, H, W, 3] uint8 -> (feats [B, 2304], v_prob [B, 97], n_prob [B, 300]) numpy."""
    import torch

    bundle._feat.clear()
    with torch.no_grad():
        inputs = [t.to(bundle.device, non_blocking=True) for t in to_pathways(clips, bundle.alpha)]
        v, n = bundle.model(inputs)
    feats = bundle._feat.pop()
    return (feats.float().cpu().numpy(), v.float().cpu().numpy(), n.float().cpu().numpy())


def prefetch(iterable, maxsize: int = 512):
    """Run `iterable` in a background thread so decoding overlaps GPU inference.

    Order is preserved; an exception raised by the producer is re-raised in the consumer.
    """
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=max(1, int(maxsize)))
    done = object()

    def worker():
        try:
            for item in iterable:
                q.put(item)
            q.put(done)
        except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
            q.put(("__error__", exc))

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is done:
            return
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
            raise item[1]
        yield item


def extract_features(path, bundle: SlowFastBundle, p: Preprocess = Preprocess(),
                     batch_size: int = 8, grid: FeatureGrid = GRID, frames=None,
                     source: FrameSource = FrameSource(), prefetch_frames: int = 512) -> dict:
    """Raw video -> {'feats','v_prob','n_prob','starts','centres','n_frames'}.

    `frames` may be supplied (an iterable of RGB uint8 frames at grid.fps) for testing;
    otherwise they are decoded per `source` (ffmpeg by default, or video_io/PyAV).
    """
    if frames is None:
        if source.backend == "pyav":
            import video_io
            frames = (rgb for _t, rgb in video_io.iter_frames(path, grid.fps))
        elif source.backend == "ffmpeg":
            frames = ffmpeg_frames(path, p, source)
        else:
            raise ValueError(f"unknown frame backend {source.backend!r}")

    count = [0]
    shift = int(source.shift)

    def prepped():
        first = True
        for i, rgb in enumerate(frames):
            count[0] += 1
            if i < shift:                      # shift > 0: drop the first frames
                continue
            img = resize_frame(rgb, p)
            if first and shift < 0:            # shift < 0: repeat the first frame
                for _ in range(-shift):
                    yield img
            first = False
            yield img

    feats, vps, nps, starts, batch, batch_starts = [], [], [], [], [], []

    def flush():
        if not batch:
            return
        f, v, n = run_clips(bundle, np.stack(batch))
        feats.append(f); vps.append(v); nps.append(n); starts.extend(batch_starts)
        batch.clear(); batch_starts.clear()

    stream = prefetch(prepped(), prefetch_frames) if prefetch_frames > 0 else prepped()
    for start, clip in iter_windows(stream, grid, p.window_rule, count_adjust=shift):
        batch.append(np.stack(clip)); batch_starts.append(start)
        if len(batch) >= batch_size:
            flush()
    flush()

    if not feats:
        return {"feats": np.zeros((0, grid.feat_dim), np.float32), "v_prob": np.zeros((0, 97), np.float32),
                "n_prob": np.zeros((0, 300), np.float32), "starts": np.zeros(0, int),
                "centres": np.zeros(0), "n_frames": count[0]}
    starts = np.asarray(starts, int)
    return {"feats": np.concatenate(feats).astype(np.float32), "v_prob": np.concatenate(vps),
            "n_prob": np.concatenate(nps), "starts": starts,
            "centres": (starts + 0.5 * grid.num_frames) / grid.fps, "n_frames": count[0]}
