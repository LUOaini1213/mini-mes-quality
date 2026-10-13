"""Anomaly-detection metrics: image AUROC, pixel AUROC and AUPRO (per-region overlap integrated up to FPR 0.3).

AUROC uses the rank (Mann-Whitney) formulation with tie handling, so it needs only numpy/scipy; it is checked
against scikit-learn in the tests. AUPRO follows the definition of Bergmann et al. (MVTec AD, IJCV 2021): for each
threshold, PRO is the mean over connected ground-truth regions of the fraction of the region's pixels predicted
anomalous; FPR is computed on normal pixels; the PRO curve is integrated up to FPR = 0.3 and divided by 0.3.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata


def auroc(labels, scores) -> float:
    y = np.asarray(labels).astype(bool).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    n1 = int(y.sum()); n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)  # average ranks for ties
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pixel_auroc(masks: np.ndarray, maps: np.ndarray, max_pixels: int = 20_000_000, seed: int = 0) -> float:
    m = masks.reshape(-1)
    s = maps.reshape(-1)
    if m.size > max_pixels:  # deterministic subsample to bound memory; reported in results when it happens
        idx = np.random.default_rng(seed).choice(m.size, max_pixels, replace=False)
        m, s = m[idx], s[idx]
    return auroc(m, s)


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3, n_thresholds: int = 400) -> float:
    """masks: (N, H, W) binary; maps: (N, H, W) anomaly scores."""
    masks = masks.astype(bool)
    normal_scores = maps[~masks]
    if normal_scores.size == 0 or not masks.any():
        return float("nan")
    # thresholds spanning FPR from ~0 to a bit beyond the limit (quantiles of the normal-pixel scores)
    qs = np.linspace(1.0, max(0.0, 1.0 - fpr_limit * 1.2), n_thresholds)
    ths = np.quantile(normal_scores, qs)
    regions = []
    for i in range(masks.shape[0]):
        lab, n = ndimage.label(masks[i])
        for r in range(1, n + 1):
            regions.append(np.sort(maps[i][lab == r]))
    normal_sorted = np.sort(normal_scores)
    fprs, pros = [], []
    for t in ths:
        fpr = 1.0 - np.searchsorted(normal_sorted, t, side="right") / normal_sorted.size
        pro = np.mean([1.0 - np.searchsorted(reg, t, side="right") / reg.size for reg in regions])
        fprs.append(fpr); pros.append(pro)
    fprs = np.array(fprs); pros = np.array(pros)
    order = np.argsort(fprs, kind="stable")
    fprs, pros = fprs[order], pros[order]
    if fprs[0] > 0:
        fprs = np.concatenate([[0.0], fprs]); pros = np.concatenate([[0.0], pros])
    pro_at_limit = np.interp(fpr_limit, fprs, pros)
    keep = fprs <= fpr_limit
    x = np.concatenate([fprs[keep], [fpr_limit]])
    y = np.concatenate([pros[keep], [pro_at_limit]])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(y, x) / fpr_limit)


def rates_at_threshold(labels, scores, threshold) -> dict:
    y = np.asarray(labels).astype(bool)
    pred = np.asarray(scores) > threshold
    tp = int((pred & y).sum()); fn = int((~pred & y).sum())
    fp = int((pred & ~y).sum()); tn = int((~pred & ~y).sum())
    return dict(threshold=float(threshold), tp=tp, fn=fn, fp=fp, tn=tn,
                tpr=tp / max(1, tp + fn), fpr=fp / max(1, fp + tn))
