"""Guide A5 contracts bootstrapped locally; M4 review pending."""
from dataclasses import dataclass, field
import numpy as np
CLIP_FPS = 2.0
@dataclass
class VideoInfo:
    video_id: str
    path: str
    duration: float
    fps: float
    width: int
    height: int
    rotation: int = 0
@dataclass
class ClipFeats:
    times: np.ndarray
    feats: np.ndarray
    model_name: str
@dataclass
class Action:
    verb: str
    noun: str
    start: float
    end: float
    score: float
@dataclass
class Importance:
    times: np.ndarray
    values: np.ndarray
@dataclass
class CategoryResult:
    label: str
    scores: dict = field(default_factory=dict)
@dataclass
class Highlight:
    start: float
    end: float
    importance: float
