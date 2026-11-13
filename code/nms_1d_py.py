"""Pure-Python stand-in for ActionFormer's compiled `nms_1d_cpu` extension.

ActionFormer (libs/utils/nms.py) calls `nms_1d_cpu.nms` and `nms_1d_cpu.softnms`,
a C++ extension that must be compiled per machine. The evaluation machine may not
have a compiler, so ACTIVA installs this module under that name instead
(`install()`), mirroring libs/utils/csrc/nms_cpu.cpp @ 61ea7eb:

  * areas = x2 - x1 + 1e-6, overlap = inter / (area_i + area_j - inter), float32;
  * hard NMS keeps segments in descending score order, dropping overlap >= threshold;
  * soft-NMS repeatedly takes the current highest score (first on ties), writes
    (x1, x2, score) to `dets`, decays every remaining score by the method's weight
    (0 vanilla, 1 linear, 2 gaussian exp(-ovr^2 / sigma)) and discards those below
    min_score; it returns the original indices in selection order.

The C++ loop swaps discarded entries with the last one, which only changes the order in
which exactly-equal scores are visited; the kept set and scores are otherwise identical.
"""
from __future__ import annotations

import sys
import types

import numpy as np


def _arrays(segs, scores):
    s = np.asarray(segs.detach().cpu().numpy() if hasattr(segs, "detach") else segs, np.float32).reshape(-1, 2)
    c = np.asarray(scores.detach().cpu().numpy() if hasattr(scores, "detach") else scores, np.float32).reshape(-1)
    return s, c


def nms_indices(segs, scores, iou_threshold: float) -> np.ndarray:
    s, c = _arrays(segs, scores)
    if len(s) == 0:
        return np.zeros(0, np.int64)
    x1, x2 = s[:, 0], s[:, 1]
    areas = x2 - x1 + np.float32(1e-6)
    order = np.argsort(-c, kind="stable")
    keep = np.ones(len(order), bool)
    for a in range(len(order)):
        if not keep[a]:
            continue
        i = order[a]
        rest = order[a + 1:]
        inter = np.maximum(np.float32(0), np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        ovr = inter / (areas[i] + areas[rest] - inter)
        keep[a + 1:] &= ~(ovr >= np.float32(iou_threshold))
    return order[keep].astype(np.int64)


def softnms_arrays(segs, scores, iou_threshold: float, sigma: float, min_score: float, method: int):
    """-> (indices in selection order, dets [K, 3] = x1, x2, decayed score)."""
    s, c = _arrays(segs, scores)
    x1, x2, sc = s[:, 0].copy(), s[:, 1].copy(), c.copy()
    areas = x2 - x1 + np.float32(1e-6)
    idx = np.arange(len(s))
    out_idx, dets = [], []
    while len(idx):
        k = int(np.argmax(sc))                 # first maximum, as the C++ scan
        out_idx.append(idx[k]); dets.append((x1[k], x2[k], sc[k]))
        keep = np.ones(len(idx), bool); keep[k] = False
        ix1, ix2, iarea = x1[k], x2[k], areas[k]
        r = np.flatnonzero(keep)
        inter = np.maximum(np.float32(0), np.minimum(ix2, x2[r]) - np.maximum(ix1, x1[r]))
        ovr = inter / (iarea + areas[r] - inter)
        if method == 0:
            w = np.where(ovr >= iou_threshold, 0.0, 1.0)
        elif method == 1:
            w = np.where(ovr >= iou_threshold, 1.0 - ovr, 1.0)
        elif method == 2:
            w = np.exp(-(ovr * ovr) / np.float32(sigma))
        else:
            raise ValueError(f"unknown soft-NMS method {method}")
        sc[r] = (sc[r] * w).astype(np.float32)
        keep[r] = sc[r] >= np.float32(min_score)
        idx, x1, x2, sc, areas = idx[keep], x1[keep], x2[keep], sc[keep], areas[keep]
    return np.asarray(out_idx, np.int64), np.asarray(dets, np.float32).reshape(-1, 3)


# ---- the nms_1d_cpu API (torch tensors) ------------------------------------- #
def nms(segs, scores, iou_threshold):
    import torch
    return torch.from_numpy(nms_indices(segs, scores, float(iou_threshold)))


def softnms(segs, scores, dets, iou_threshold, sigma, min_score, method):
    import torch
    inds, d = softnms_arrays(segs, scores, float(iou_threshold), float(sigma), float(min_score), int(method))
    if len(d):
        dets[: len(d)] = torch.from_numpy(d).to(dets.dtype)
    return torch.from_numpy(inds)


def install(force: bool = False) -> str:
    """Register this module as `nms_1d_cpu` unless the compiled one is importable.

    Returns "compiled" or "python" so callers can log which implementation runs.
    """
    if not force:
        try:
            import nms_1d_cpu  # noqa: F401
            if getattr(sys.modules["nms_1d_cpu"], "__activa_python__", False):
                return "python"
            return "compiled"
        except ImportError:
            pass
    mod = types.ModuleType("nms_1d_cpu")
    mod.nms, mod.softnms, mod.__activa_python__ = nms, softnms, True
    sys.modules["nms_1d_cpu"] = mod
    return "python"
