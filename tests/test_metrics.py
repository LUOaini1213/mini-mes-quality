"""Anomaly-detection metrics."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mesq.vision.metrics import aupro, auroc, pixel_auroc, rates_at_threshold


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500) + y, 1)  # rounding creates ties
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_auroc_edge_cases():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert auroc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == 0.0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5
    assert np.isnan(auroc([1, 1], [0.1, 0.2]))


def test_pixel_auroc_and_subsampling():
    rng = np.random.default_rng(1)
    m = (rng.random((4, 16, 16)) < 0.1).astype(np.uint8)
    s = m + rng.normal(scale=0.5, size=m.shape)
    full = pixel_auroc(m, s)
    assert full == pytest.approx(roc_auc_score(m.ravel(), s.ravel()))
    assert pixel_auroc(m, s, max_pixels=500) == pytest.approx(full, abs=0.08)


def test_aupro_perfect_and_inverted():
    m = np.zeros((2, 32, 32), np.uint8)
    m[0, 4:10, 4:10] = 1
    m[1, 20:30, 2:5] = 1
    assert aupro(m, m.astype(float)) == pytest.approx(1.0, abs=1e-6)
    assert aupro(m, 1.0 - m) == pytest.approx(0.0, abs=1e-6)


def test_aupro_weights_regions_equally():
    # one big region fully detected, one small region missed -> PRO = 0.5 at every low-FPR threshold
    m = np.zeros((1, 64, 64), np.uint8)
    m[0, 0:30, 0:30] = 1
    m[0, 50:52, 50:52] = 1
    s = np.zeros((1, 64, 64))
    s[0, 0:30, 0:30] = 1.0
    assert aupro(m, s) == pytest.approx(0.5, abs=0.02)
    # pixel AUROC would instead reward the big region by area
    assert pixel_auroc(m, s) > 0.95


def test_rates_at_threshold():
    r = rates_at_threshold([0, 0, 1, 1, 1], [0.1, 0.9, 0.8, 0.2, 0.95], 0.5)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (2, 1, 1, 1)
    assert r["tpr"] == pytest.approx(2 / 3) and r["fpr"] == pytest.approx(0.5)
