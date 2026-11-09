"""PLACEHOLDER - owned by M2. Replace with the m2-perception version.

Keep the signature: predict_importance(info, clip) -> Importance on clip.times.
"""
import numpy as np

from contracts import Importance


def predict_importance(info, clip) -> Importance:
    times = np.asarray(clip.times, dtype=np.float64).copy()
    return Importance(times=times, values=np.full(times.shape, 0.5))
