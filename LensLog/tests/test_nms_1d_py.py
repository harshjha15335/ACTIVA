import numpy as np
import pytest

import nms_1d_py as n

torch = pytest.importorskip("torch")


def test_hard_nms_keeps_best_and_non_overlapping():
    segs = torch.tensor([[0, 10], [1, 11], [20, 30], [0, 10]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7, 0.95])
    assert n.nms(segs, scores, 0.5).tolist() == [3, 2]


def test_softnms_gaussian_matches_reference():
    segs = torch.tensor([[0, 10], [2, 12], [50, 60]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.5])
    dets = segs.new_empty((3, 3))
    inds = n.softnms(segs, scores, dets, 0.1, 0.4, 0.001, 2)
    ovr = 8 / 12
    decayed = 0.8 * np.exp(-(ovr ** 2) / 0.4)
    assert decayed < 0.5
    assert inds.tolist() == [0, 2, 1]              # selection order follows decayed scores
    got = dict(zip(inds.tolist(), dets[:, 2].tolist()))
    assert got[0] == pytest.approx(0.9) and got[2] == pytest.approx(0.5) and got[1] == pytest.approx(decayed, rel=1e-5)


def test_softnms_selection_order_and_min_score():
    segs = torch.tensor([[0, 10], [0, 10], [30, 40]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.85, 0.6])
    dets = segs.new_empty((3, 3))
    inds = n.softnms(segs, scores, dets, 0.1, 0.4, 0.5, 2)   # identical seg decays to 0.85*e^-2.5 < 0.5
    assert inds.tolist() == [0, 2]
    assert dets[:2, 2].tolist() == pytest.approx([0.9, 0.6])


def test_softnms_empty():
    e = torch.zeros((0, 2))
    assert n.softnms(e, torch.zeros(0), e.new_empty((0, 3)), 0.1, 0.5, 0.0, 2).numel() == 0


def test_install_registers_python_fallback():
    import sys
    assert n.install(force=True) == "python"
    assert sys.modules["nms_1d_cpu"].softnms is n.softnms
