"""Shared configuration. Owned by M4.

Every tunable number and path lives here or in a module's own *_config.py.
Rules quoted from the student kit's TEAM_README.md are marked with its section.
Open questions for the organisers are tracked in docs/kit_checklist.md.
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

# --- Team -------------------------------------------------------------------------
# Every output JSON carries team_id, identical across all 187 files (TEAM_README s3).
# Registered team name. Provisional: no official team ID has been issued yet;
# change here if the organisers assign one. ACTIVA_TEAM_ID overrides it.
TEAM_ID = "LensLog"


class ConfigError(RuntimeError):
    """A required setting is missing (run.sh exit code 5)."""


def team_id():
    value = os.environ.get("ACTIVA_TEAM_ID") or TEAM_ID
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("team_id is not set: set config.TEAM_ID or the ACTIVA_TEAM_ID environment variable.")
    return value.strip()


def strict():
    """ACTIVA_STRICT=1: a stage that fails is an error (exit 6), not a fallback."""
    return os.environ.get("ACTIVA_STRICT", "") not in ("", "0")


# --- Vocabulary ----------------------------------------------------------------
# Student kit: supported_classes.json with "labels" (16), "verbs" (98), "nouns" (301).
DEFAULT_VOCAB_PATH = ROOT / "kit" / "supported_classes.json"
DEFAULT_MANIFEST_PATH = ROOT / "kit" / "manifest.csv"
FALLBACK_LABEL = "other"   # official label for "none of the categories"


def vocab_path():
    """ACTIVA_VOCAB overrides the default kit location."""
    return Path(os.environ.get("ACTIVA_VOCAB", DEFAULT_VOCAB_PATH))


# --- Highlights (TEAM_README s1, s3, s4) -----------------------------------------
BUDGET_RATIO = 0.15      # total highlight length should be <= 15% of duration_sec;
                         # over budget is a validator warning plus a soft scoring penalty
BUDGET_SAFETY = 0.148    # what selectors should aim for (headroom for rounding)
MIN_CLIP_SECONDS = 4.0   # segments shorter than this are ignored when scoring, and the
                         # validator needs at least one segment of 4.0 s or more
FALLBACK_CLIP_SECONDS = 6.0
# When 15% of the video is under 4 s (3 manifest videos: 11.13, 15.28, 25.16 s), the
# two rules conflict. We emit one 4.0 s segment: the validator rejects a file without
# one (score 0), while the budget is only a warning. Pending organiser confirmation.
SHORT_VIDEO_SINGLE_CLIP = True

# --- Output JSON (TEAM_README s3) ---------------------------------------------------
OUTPUT_DECIMALS = 2          # "Timestamps should be in seconds (up to 2 decimal places)"
SCORE_DECIMALS = 4
MAX_ACTIONS = 2000           # actions: 0-2000 items
DURATION_TOLERANCE = 0.5     # duration_sec within +-0.5 s of manifest; end <= duration_sec + 0.5
ALLOW_EXTRA_KEYS = False     # the kit lists exactly 6 top-level fields; extras untested

# Manifest durations run from 11 s to 1969 s (EPIC) and 37-377 s (FPVSum).
# Outside this range we only warn.
EXPECTED_DURATION_RANGE = (10.0, 2000.0)


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
    return Settings(os.environ.get("ACTIVA_CLIP_WEIGHTS", "weights/ViT-B-32-laion400m_e32.safetensors"), tuple(categories), os.environ.get("ACTIVA_CLIP_MODEL", "ViT-B-32"), os.environ.get("ACTIVA_DEVICE", "cpu"), batch, os.environ.get("ACTIVA_IMPORTANCE", "novelty"), os.environ.get("ACTIVA_POOLING", "mean"))
