"""Shared interface types (guide A5). Owned by M4.

Every module talks to the others only through these types. Field names, types
and meanings are fixed; a change needs a board request approved by M4.
Times are seconds from the first decoded frame (see video_io.py).
"""
from dataclasses import dataclass, field

import numpy as np

CLIP_FPS = 2.0  # shared time grid for CLIP features and importance (frames/second)


@dataclass
class VideoInfo:
    video_id: str        # file name without extension, e.g. "P01_11"
    path: str            # absolute path to the MP4
    duration: float      # seconds (from ffprobe)
    fps: float           # average source frame rate
    width: int
    height: int
    rotation: int = 0    # degrees, from metadata


@dataclass
class ClipFeats:
    times: np.ndarray    # shape [N], seconds, strictly increasing
    feats: np.ndarray    # shape [N, D], float32, L2-normalised CLIP image embeddings
    model_name: str      # e.g. "ViT-B-32/openai"


@dataclass
class Action:
    verb: str            # exact string from supported_classes.json
    noun: str            # exact string from supported_classes.json
    start: float         # seconds, 0 <= start < end <= duration
    end: float
    score: float         # internal confidence in [0, 1]


@dataclass
class Importance:
    times: np.ndarray    # same grid as ClipFeats.times
    values: np.ndarray   # shape [N], each in [0, 1]


@dataclass
class CategoryResult:
    label: str                                   # exact string from supported_classes.json
    scores: dict = field(default_factory=dict)   # label -> score (internal only)


@dataclass
class Highlight:
    start: float
    end: float
    importance: float    # in [0, 1]
