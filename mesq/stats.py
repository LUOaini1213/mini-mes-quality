"""Small, dependency-light statistics used by the traceability analysis (all hand-checked in tests/test_stats.py)."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def risk_ratio_ci(a: int, n1: int, c: int, n0: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Risk ratio (a/n1)/(c/n0) with a Katz log interval; 0.5 is added to every cell if any event count is zero."""
    if a == 0 or c == 0 or a == n1 or c == n0:
        a, c, n1, n0 = a + 0.5, c + 0.5, n1 + 1, n0 + 1
    rr = (a / n1) / (c / n0)
    se = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    return rr, rr * math.exp(-z * se), rr * math.exp(z * se)


def fisher_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p-value for odds ratio > 1 in [[a, b], [c, d]] (rows: exposed / unexposed,
    columns: fail / pass)."""
    return float(stats.fisher_exact([[a, b], [c, d]], alternative="greater")[1])


def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), returned in the input order."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def holm_adjust(p) -> np.ndarray:
    """Holm step-down adjusted p-values, input order."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.maximum.accumulate((n - np.arange(n)) * p[order])
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def cmh_greater(tables) -> tuple[float, float]:
    """Cochran-Mantel-Haenszel test across strata of 2x2 tables [[a, b], [c, d]] (exposed/unexposed x fail/pass).

    Returns (one-sided p-value for a common odds ratio > 1, with continuity correction; Mantel-Haenszel common
    odds ratio). Strata with fewer than 2 units or no variation in a margin contribute nothing."""
    sa = se = sv = num = den = 0.0
    for t in tables:
        (a, b), (c, d) = t
        n = a + b + c + d
        if n < 2:
            continue
        r1, r0, c1, c0 = a + b, c + d, a + c, b + d
        if min(r1, r0, c1, c0) == 0:
            continue
        sa += a
        se += r1 * c1 / n
        sv += r1 * r0 * c1 * c0 / (n * n * (n - 1))
        num += a * d / n
        den += b * c / n
    if sv == 0:
        return 1.0, float("nan")
    z = (sa - se - 0.5) / math.sqrt(sv)
    p = float(stats.norm.sf(z))
    orr = num / den if den > 0 else float("inf")
    return p, orr
