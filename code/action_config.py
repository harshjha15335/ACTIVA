"""Member 1 (Action Engine) settings.

Every tunable number for the action engine lives here so it can be tuned and frozen
in one place. M4 owns the shared `config.py`; these are M1-only settings and can be
folded into it once the shared foundation exists.

Values marked "ActionFormer" are copied from the public EPIC configs
(configs/epic_slowfast_{verb,noun}.yaml) and must not be changed: they define how
the published features and checkpoints were produced. Values marked "tune" are
starting points to tune on development data only (never on evaluation videos).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureGrid:
    # ActionFormer EPIC SlowFast settings (fixed).
    fps: float = 30.0          # features were computed on video resampled to 30 fps
    num_frames: int = 32       # frames per SlowFast window
    stride: int = 16           # frames between consecutive windows
    feat_dim: int = 2304       # input_dim in the ActionFormer EPIC configs


@dataclass(frozen=True)
class PairingSettings:
    # Verb + noun -> verb-noun pair (tune).
    tau_pair: float = 0.3          # min tIoU between a verb and a noun segment
    allow_centre_inside: bool = True   # also pair if the noun centre lies inside the verb
    exp_verb: float = 1.0          # a in  score = p_v^a * p_n^b * prior^c
    exp_noun: float = 1.0          # b
    exp_prior: float = 0.5         # c
    top_k_nouns: int = 2           # nouns kept per verb segment
    segment_mode: str = "verb"     # "verb" | "union" | "intersection"
    nms_tiou: float = 0.5          # class-wise NMS threshold on pairs
    min_score: float = 0.0         # drop pairs below this score
    max_actions: int = 2000        # official cap: 0-2000 actions per video (TEAM_README)
    min_length: float = 0.1        # seconds; shorter segments are dropped
    round_digits: int = 3          # agree final rounding with M4


GRID = FeatureGrid()
PAIRING = PairingSettings()


# --------------------------------------------------------------------------- #
# Local model assets (never downloaded at runtime)
# --------------------------------------------------------------------------- #
import os  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = REPO_ROOT / "weights" / "m1"


@dataclass(frozen=True)
class EngineAssets:
    """Where the action engine finds its code and weights. Layout under ACTIVA_M1_ASSETS:

        epic-kitchens-slowfast/                 code @ bf50519 (Apache-2.0)
        SlowFast.pyth                           EPIC-100 SlowFast 8x8 R50 checkpoint
        actionformer_release/                   code @ 61ea7eb (MIT)
        epic_slowfast_verb_reproduce/*.pth.tar  ActionFormer EPIC verb checkpoint
        epic_slowfast_noun_reproduce/*.pth.tar  ActionFormer EPIC noun checkpoint
        EPIC_100_verb_classes.csv               class id -> key (EPIC annotations repo)
        EPIC_100_noun_classes.csv
    """
    root: Path

    @property
    def slowfast_code(self): return self.root / "epic-kitchens-slowfast"
    @property
    def slowfast_cfg(self): return self.slowfast_code / "configs" / "EPIC-KITCHENS" / "SLOWFAST_8x8_R50.yaml"
    @property
    def slowfast_ckpt(self): return self.root / "SlowFast.pyth"
    @property
    def af_code(self): return self.root / "actionformer_release"
    def af_cfg(self, kind): return self.af_code / "configs" / f"epic_slowfast_{kind}.yaml"
    def af_ckpt_dir(self, kind): return self.root / f"epic_slowfast_{kind}_reproduce"
    def classes_csv(self, kind): return self.root / f"EPIC_100_{kind}_classes.csv"

    def required(self):
        return [self.slowfast_code, self.slowfast_cfg, self.slowfast_ckpt, self.af_code,
                self.af_cfg("verb"), self.af_cfg("noun"), self.af_ckpt_dir("verb"),
                self.af_ckpt_dir("noun"), self.classes_csv("verb"), self.classes_csv("noun")]

    def missing(self):
        return [str(p) for p in self.required() if not p.exists()]


def engine_assets() -> EngineAssets:
    return EngineAssets(Path(os.environ.get("ACTIVA_M1_ASSETS", DEFAULT_ASSETS)))
