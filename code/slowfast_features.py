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
    pad_last: bool = True       # emit a final padded window (published T = ceil(n/16)?)

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


def window_starts(n_frames: int, grid: FeatureGrid = GRID, pad_last: bool = True) -> list[int]:
    """Start index (in 30 fps sampled frames) of every feature window.

    pad_last=False: only complete windows, (n - 32) // 16 + 1 of them.
    pad_last=True : one window per stride step, ceil(n / 16), later windows padded by
                    repeating the last frame.
    """
    n = int(n_frames)
    if n <= 0:
        return []
    if pad_last:
        return list(range(0, n, grid.stride))
    if n < grid.num_frames:
        return []
    return list(range(0, n - grid.num_frames + 1, grid.stride))


def iter_windows(frames: Iterable[np.ndarray], grid: FeatureGrid = GRID,
                 pad_last: bool = True) -> Iterator[tuple[int, list]]:
    """Stream (start_index, 32 frames) windows without holding the whole video in memory."""
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
    if not pad_last or n == 0:
        return
    last = buf[-1] if buf else None
    while next_start < n:
        lo = next_start - first
        clip = list(buf)[lo:]
        if last is None:
            break
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


def extract_features(path, bundle: SlowFastBundle, p: Preprocess = Preprocess(),
                     batch_size: int = 8, grid: FeatureGrid = GRID, frames=None) -> dict:
    """Raw video -> {'feats','v_prob','n_prob','starts','centres','n_frames'}.

    `frames` may be supplied (an iterable of RGB uint8 frames at grid.fps) for testing;
    by default they come from video_io.iter_frames(path, grid.fps).
    """
    if frames is None:
        import video_io
        frames = (rgb for _t, rgb in video_io.iter_frames(path, grid.fps))

    count = [0]

    def prepped():
        for rgb in frames:
            count[0] += 1
            yield resize_frame(rgb, p)

    feats, vps, nps, starts, batch, batch_starts = [], [], [], [], [], []

    def flush():
        if not batch:
            return
        f, v, n = run_clips(bundle, np.stack(batch))
        feats.append(f); vps.append(v); nps.append(n); starts.extend(batch_starts)
        batch.clear(); batch_starts.clear()

    for start, clip in iter_windows(prepped(), grid, p.pad_last):
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
