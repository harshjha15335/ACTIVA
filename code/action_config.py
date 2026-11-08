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
    max_actions: int = 500         # cap on actions written per video
    min_length: float = 0.1        # seconds; shorter segments are dropped
    round_digits: int = 3          # agree final rounding with M4


GRID = FeatureGrid()
PAIRING = PairingSettings()
