"""Member 1: Action Engine.

Public entry point (fixed signature, guide A5):
    detect_actions(info: VideoInfo) -> list[Action]

Current state (G0): `detect_actions` is a STUB that returns no actions, so the rest of
the pipeline can run end-to-end. The helpers below are the real, tested building
blocks that the full engine will use once the raw-MP4 feature gate (guide Step 4)
is decided:

    raw MP4 -> SlowFast features [T, 2304] -> ActionFormer verb model + noun model
            -> pair_detections -> nms_pairs -> finalize -> list[Action]

Detections arrays used here (all numpy, seconds):
    segs   [N, 2] float  (start, end)
    labels [N]    int    class id (EPIC-100 verb or noun id)
    scores [N]    float  confidence in [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np

from action_config import GRID, PAIRING, FeatureGrid, PairingSettings

if TYPE_CHECKING:  # M4 owns contracts.py; imported only when building real Actions
    from contracts import Action, VideoInfo


# --------------------------------------------------------------------------- #
# Feature grid <-> seconds (copied from ActionFormer, do not reinvent)
# --------------------------------------------------------------------------- #
def grid_to_seconds(x, grid: FeatureGrid = GRID):
    """Feature-grid coordinate -> seconds.

    Same formula as ActionFormer's postprocessing:
        segs = (segs * stride + 0.5 * nframes) / fps
    Window i covers frames [16i, 16i+32) at 30 fps, so its centre is (16i + 16) / 30 s.
    """
    return (np.asarray(x, dtype=float) * grid.stride + 0.5 * grid.num_frames) / grid.fps


def seconds_to_grid(t, grid: FeatureGrid = GRID):
    """Seconds -> feature-grid coordinate (inverse of grid_to_seconds).

    Matches ActionFormer's dataset code:
        segments * fps / feat_stride - feat_offset,  feat_offset = 0.5 * num_frames / feat_stride
    """
    return np.asarray(t, dtype=float) * grid.fps / grid.stride - 0.5 * grid.num_frames / grid.stride


def window_centres(num_windows: int, grid: FeatureGrid = GRID) -> np.ndarray:
    """Centre time (seconds) of each SlowFast feature window."""
    return grid_to_seconds(np.arange(int(num_windows)), grid)


def expected_num_windows(duration: float, grid: FeatureGrid = GRID) -> int:
    """Approximate window count for a video after resampling to grid.fps.

    Planning/sanity helper only. The exact count of the published features (including
    how the last partial window is handled) must be checked against the published
    .npz files in the Step 4 comparison.
    """
    n_frames = int(np.floor(float(duration) * grid.fps + 1e-9))
    if n_frames < grid.num_frames:
        return 0
    return (n_frames - grid.num_frames) // grid.stride + 1


# --------------------------------------------------------------------------- #
# Temporal overlap
# --------------------------------------------------------------------------- #
def tiou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise temporal IoU between segment sets a [N,2] and b [M,2] -> [N,M]."""
    a = np.asarray(a, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1, 2)
    inter = np.clip(
        np.minimum(a[:, None, 1], b[None, :, 1]) - np.maximum(a[:, None, 0], b[None, :, 0]),
        0.0, None)
    union = (a[:, 1] - a[:, 0])[:, None] + (b[:, 1] - b[:, 0])[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


# --------------------------------------------------------------------------- #
# Verb-noun prior (from EPIC TRAIN annotations only)
# --------------------------------------------------------------------------- #
def build_pair_prior(verb_ids: Sequence[int], noun_ids: Sequence[int],
                     num_verbs: int = 97, num_nouns: int = 300,
                     smoothing: float = 1.0) -> np.ndarray:
    """P(noun | verb) with add-`smoothing` counts, shape [num_verbs, num_nouns].

    Pass the verb_class / noun_class columns of EPIC_100_train.csv. Never build this
    from validation or evaluation annotations (that is test leakage).
    """
    counts = np.full((num_verbs, num_nouns), float(smoothing))
    np.add.at(counts, (np.asarray(verb_ids, dtype=int), np.asarray(noun_ids, dtype=int)), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Pairing, NMS, finalisation
# --------------------------------------------------------------------------- #
@dataclass
class PairDetections:
    verbs: np.ndarray    # [K] int
    nouns: np.ndarray    # [K] int
    segs: np.ndarray     # [K, 2] seconds
    scores: np.ndarray   # [K]

    def __len__(self) -> int:
        return len(self.scores)

    @staticmethod
    def empty() -> "PairDetections":
        return PairDetections(np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2)), np.zeros(0))

    def take(self, idx) -> "PairDetections":
        idx = np.asarray(idx, dtype=int)
        return PairDetections(self.verbs[idx], self.nouns[idx], self.segs[idx], self.scores[idx])


def pair_detections(v_segs, v_labels, v_scores, n_segs, n_labels, n_scores,
                    prior: Optional[np.ndarray] = None,
                    is_valid_pair: Optional[Callable[[int, int], bool]] = None,
                    s: PairingSettings = PAIRING) -> PairDetections:
    """Combine separate verb and noun detections into verb-noun pair detections.

    For each verb segment, candidate nouns are those whose tIoU >= tau_pair, or (if
    allowed) whose centre lies inside the verb segment. Pair score:
        p_v^a * p_n^b * prior[v, n]^c
    The top_k_nouns highest-scoring nouns are kept per verb segment.
    """
    v_segs = np.asarray(v_segs, dtype=float).reshape(-1, 2)
    n_segs = np.asarray(n_segs, dtype=float).reshape(-1, 2)
    v_labels, n_labels = np.asarray(v_labels, dtype=int), np.asarray(n_labels, dtype=int)
    v_scores, n_scores = np.asarray(v_scores, dtype=float), np.asarray(n_scores, dtype=float)
    if len(v_segs) == 0 or len(n_segs) == 0:
        return PairDetections.empty()

    tiou = tiou_matrix(v_segs, n_segs)
    ok = tiou >= s.tau_pair
    if s.allow_centre_inside:
        centres = n_segs.mean(axis=1)
        ok |= (centres[None, :] >= v_segs[:, 0:1]) & (centres[None, :] <= v_segs[:, 1:2])

    out_v, out_n, out_s, out_score = [], [], [], []
    for i in range(len(v_segs)):
        cand = np.flatnonzero(ok[i])
        if is_valid_pair is not None:
            cand = np.array([j for j in cand if is_valid_pair(int(v_labels[i]), int(n_labels[j]))], dtype=int)
        if len(cand) == 0:
            continue
        sc = (v_scores[i] ** s.exp_verb) * (n_scores[cand] ** s.exp_noun)
        if prior is not None and s.exp_prior:
            sc = sc * prior[v_labels[i], n_labels[cand]] ** s.exp_prior
        # stable: highest score first, ties broken by noun index order
        order = np.argsort(-sc, kind="stable")[: s.top_k_nouns]
        for j_local in order:
            j = cand[j_local]
            if s.segment_mode == "verb":
                seg = v_segs[i]
            elif s.segment_mode == "union":
                seg = (min(v_segs[i, 0], n_segs[j, 0]), max(v_segs[i, 1], n_segs[j, 1]))
            elif s.segment_mode == "intersection":
                seg = (max(v_segs[i, 0], n_segs[j, 0]), min(v_segs[i, 1], n_segs[j, 1]))
                if seg[1] <= seg[0]:
                    seg = v_segs[i]
            else:
                raise ValueError(f"Unknown segment_mode {s.segment_mode!r}")
            out_v.append(v_labels[i]); out_n.append(n_labels[j])
            out_s.append(seg); out_score.append(sc[j_local])

    if not out_score:
        return PairDetections.empty()
    return PairDetections(np.array(out_v, int), np.array(out_n, int),
                          np.array(out_s, float).reshape(-1, 2), np.array(out_score, float))


def nms_pairs(p: PairDetections, tiou_thresh: float = PAIRING.nms_tiou) -> PairDetections:
    """Class-wise (per verb-noun pair) greedy temporal NMS. Deterministic."""
    if len(p) == 0:
        return p
    keep = []
    keys = p.verbs.astype(np.int64) * 100000 + p.nouns.astype(np.int64)
    for key in np.unique(keys):
        idx = np.flatnonzero(keys == key)
        idx = idx[np.argsort(-p.scores[idx], kind="stable")]
        while len(idx):
            best, rest = idx[0], idx[1:]
            keep.append(best)
            if len(rest) == 0:
                break
            ious = tiou_matrix(p.segs[best:best + 1], p.segs[rest])[0]
            idx = rest[ious < tiou_thresh]
    keep = np.array(sorted(keep), dtype=int)
    return p.take(keep)


def finalize(p: PairDetections, duration: float,
             verb_names: Sequence[str], noun_names: Sequence[str],
             s: PairingSettings = PAIRING, action_cls=None) -> list:
    """Clamp, filter, sort by score, cap and convert to Action objects.

    Guarantees for every output: finite numbers, 0 <= start < end <= duration,
    end - start >= min_length, score in [0, 1], sorted by score (highest first).
    `action_cls` defaults to contracts.Action (M4-owned).
    """
    if action_cls is None:
        from contracts import Action as action_cls  # noqa: N813
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    if len(p) == 0:
        return []

    segs = np.clip(p.segs, 0.0, duration)
    scores = np.clip(np.nan_to_num(p.scores, nan=0.0), 0.0, 1.0)
    good = (np.isfinite(segs).all(axis=1) & (segs[:, 1] - segs[:, 0] >= s.min_length)
            & (scores >= s.min_score))
    idx = np.flatnonzero(good)
    idx = idx[np.argsort(-scores[idx], kind="stable")][: s.max_actions]

    out = []
    for i in idx:
        start = round(float(segs[i, 0]), s.round_digits)
        end = round(float(segs[i, 1]), s.round_digits)
        if not 0.0 <= start < end <= duration:  # rounding can collapse tiny segments
            continue
        out.append(action_cls(verb=str(verb_names[p.verbs[i]]), noun=str(noun_names[p.nouns[i]]),
                              start=start, end=end, score=float(scores[i])))
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def detect_actions(info: "VideoInfo") -> list:
    """Return every action in the full video, highest confidence first.

    G0 STUB: returns an empty list so `run.sh` works end-to-end before the action
    models are integrated. Replaced by the real engine after the Step 4 gate.
    """
    return []
