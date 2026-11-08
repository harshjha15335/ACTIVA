"""M2 validation adapter around M4-owned config; shared config stays untouched."""
from dataclasses import replace
import os
import re
from pathlib import Path
from config import settings as shared_settings

IMPORTANCE_METHODS = ("constant", "novelty", "motion", "text")
POOLING_METHODS = ("mean", "weighted", "top")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def settings():
    cfg = shared_settings()
    if cfg.baseline not in IMPORTANCE_METHODS:
        raise ValueError(f"Unknown importance baseline: {cfg.baseline}")
    if cfg.pooling not in POOLING_METHODS:
        raise ValueError(f"Unknown category pooling: {cfg.pooling}")
    if not isinstance(cfg.batch_size, int) or cfg.batch_size < 1:
        raise ValueError("CLIP batch size must be a positive integer.")
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", cfg.device):
        raise ValueError("Supported inference devices are cpu or cuda[:index].")
    for flag in ["ACTIVA_CLIP_FP16", "ACTIVA_REQUIRE_CANONICAL_VOCAB"]:
        if os.environ.get(flag, "0") not in ("0", "1"):
            raise ValueError(f"{flag} must be 0 or 1.")
    weights = Path(cfg.weights).expanduser()
    # Explicit user paths are caller-relative; only implicit defaults use the repo.
    if not os.environ.get("ACTIVA_CLIP_WEIGHTS") and not weights.is_absolute():
        weights = PROJECT_ROOT / weights
    return replace(cfg, weights=str(weights.resolve()))
