"""PLACEHOLDER - owned by M2. Replace with the m2-perception version.

Returns a valid dummy ClipFeats on the shared CLIP_FPS grid without decoding
the video. Keep the signature: extract_clip_features(info) -> ClipFeats.
"""
import numpy as np

from contracts import CLIP_FPS, ClipFeats

STUB_DIM = 512


def extract_clip_features(info) -> ClipFeats:
    times = np.arange(0.0, info.duration, 1.0 / CLIP_FPS, dtype=np.float64)
    feats = np.zeros((len(times), STUB_DIM), dtype=np.float32)
    feats[:, 0] = 1.0  # unit length, as the contract requires
    return ClipFeats(times=times, feats=feats, model_name="stub/none")
