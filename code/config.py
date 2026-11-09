"""Shared configuration. Owned by M4.

Every tunable number and path lives here or in a module's own *_config.py.
Anything marked VERIFY must be checked against the official student kit
(see docs/kit_checklist.md) before submission.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path

from contracts import CLIP_FPS  # noqa: F401  (re-exported for convenience)

ROOT = Path(__file__).resolve().parents[1]

# --- Runtime -----------------------------------------------------------------
SEED = 0
DEVICE = os.environ.get("ACTIVA_DEVICE", "cpu")      # "cpu" or "cuda"; same variable M2 reads
LOG_LEVEL = os.environ.get("ACTIVA_LOG_LEVEL", "INFO")

# --- Vocabulary ----------------------------------------------------------------
# VERIFY: file name and layout of the kit's supported_classes.json.
DEFAULT_VOCAB_PATH = ROOT / "kit" / "supported_classes.json"


def vocab_path():
    """ACTIVA_VOCAB overrides the default kit location."""
    return Path(os.environ.get("ACTIVA_VOCAB", DEFAULT_VOCAB_PATH))


# --- Highlight budget ------------------------------------------------------------
BUDGET_RATIO = 0.15      # official: total highlight duration <= 15% of video
BUDGET_SAFETY = 0.148    # what selectors should aim for (headroom for rounding)
                         # VERIFY: whether the organisers allow any tolerance
MIN_CLIP_SECONDS = 1.0   # shortest highlight the fallback selector will emit
FALLBACK_CLIP_SECONDS = 3.0

# --- Output JSON -------------------------------------------------------------------
OUTPUT_DECIMALS = 2          # VERIFY: rounding rule
INCLUDE_ACTION_SCORE = False # VERIFY: whether actions may carry a "score" field (kit A3 #5)
ALLOW_EXTRA_KEYS = False     # VERIFY: whether extra keys are tolerated

# Videos are expected to be 2-10 minutes; outside this we only warn,
# because full EPIC videos can be longer (VERIFY, kit A3 #7).
EXPECTED_DURATION_RANGE = (120.0, 600.0)


# --- M2 perception settings (moved here unchanged from m2-perception) ----------
# code/perception_options.py imports `settings` from this module.
@dataclass(frozen=True)
class Settings:
    weights: str
    categories: tuple[str, ...]
    model: str = "ViT-B-32"
    device: str = "cpu"
    batch_size: int = 16
    baseline: str = "novelty"
    pooling: str = "mean"


def settings():
    categories = []
    if os.environ.get("ACTIVA_CATEGORIES"):
        data = json.loads(Path(os.environ["ACTIVA_CATEGORIES"]).read_text())
        categories = data.get("categories", []) if isinstance(data, dict) else data
        if not isinstance(categories, list) or not categories or any(not isinstance(x, str) or not x.strip() for x in categories) or len(set(categories)) != len(categories):
            raise ValueError("Category JSON requires unique nonempty strings.")
    batch = int(os.environ.get("ACTIVA_BATCH_SIZE", "16"))
    if batch < 1:
        raise ValueError("Batch size must be positive.")
    return Settings(os.environ.get("ACTIVA_CLIP_WEIGHTS", "weights/ViT-B-32.safetensors"), tuple(categories), os.environ.get("ACTIVA_CLIP_MODEL", "ViT-B-32"), os.environ.get("ACTIVA_DEVICE", "cpu"), batch, os.environ.get("ACTIVA_IMPORTANCE", "novelty"), os.environ.get("ACTIVA_POOLING", "mean"))
