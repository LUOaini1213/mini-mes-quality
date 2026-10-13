"""Defect -> lot -> equipment / material traceability with explicit statistics.

Input: the per-unit factor table from mesq.db.unit_factor_table (as-processed path through the genealogy plus the
inspection decision). Ground-truth tables are never read here; scoring against them happens in score_against_truth.

Procedure
1. Candidates. For every step up to and including the inspection step: each equipment, operator and shift level,
   each equipment x shift combination, and each material batch. A candidate is a 0/1 exposure per unit.
2. Screening. One-sided Fisher exact test (defect odds higher when exposed) per candidate; Benjamini-Hochberg
   across all candidates. This is the "naive" one-factor-at-a-time view a dashboard would give.
3. Explain-away selection. Forward: repeatedly add the screened candidate with the smallest Cochran-Mantel-Haenszel
   p-value, stratified by the candidates already selected, while it beats a Bonferroni threshold over the
   candidates still in play. Backward: drop any selected candidate that is no longer significant given all the
   others. What is left are the reported root-cause candidates; everything else that screened positive is reported
   as explained (a probable confounder or a coarser/finer description of a selected cause).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .stats import bh_adjust, cmh_greater, fisher_greater, risk_ratio_ci, wilson_ci

PRE_INSPECTION_STEPS = ["PRINT", "PLACE", "REFLOW", "AOI"]  # ASSY happens after inspection and cannot cause AOI fails


def candidates(units: pd.DataFrame, steps=PRE_INSPECTION_STEPS, min_exposed: int = 20) -> dict[str, np.ndarray]:
    cands: dict[str, np.ndarray] = {}
    for s in steps:
        for f in ("equipment", "operator", "shift"):
            col = f"{s}.{f}"
            if col not in units:
                continue
            for lvl in sorted(units[col].dropna().unique()):
                cands[f"{col}={lvl}"] = (units[col] == lvl).to_numpy()
        if f"{s}.equipment" in units and f"{s}.shift" in units:
            for (e, sh), _ in units.groupby([f"{s}.equipment", f"{s}.shift"]):
                cands[f"{s}.equipment={e} & {s}.shift={sh}"] = ((units[f"{s}.equipment"] == e) &
                                                               (units[f"{s}.shift"] == sh)).to_numpy()
    for col in units.columns:
        if ".material." in col:
            step = col.split(".")[0]
            for lvl in sorted(units[col].dropna().unique()):
                cands[f"{step}.material={lvl}"] = (units[col] == lvl).to_numpy()
    n = len(units)
    return {k: v for k, v in cands.items() if min_exposed <= v.sum() <= n - min_exposed}


def screen(units: pd.DataFrame, cands: dict[str, np.ndarray]) -> pd.DataFrame:
    y = units["fail"].to_numpy().astype(bool)
    rows = []
    for key, x in cands.items():
        a = int((x & y).sum()); n1 = int(x.sum())
        c = int((~x & y).sum()); n0 = int((~x).sum())
        rr, lo, hi = risk_ratio_ci(a, n1, c, n0)
        ci1 = wilson_ci(a, n1)
        rows.append(dict(candidate=key, n_exposed=n1, fail_exposed=a, rate_exposed=a / n1,
                         rate_exposed_lo=ci1[0], rate_exposed_hi=ci1[1], n_unexposed=n0, fail_unexposed=c,
                         rate_unexposed=c / n0 if n0 else float("nan"), risk_ratio=rr, rr_lo=lo, rr_hi=hi,
                         p_fisher=fisher_greater(a, n1 - a, c, n0 - c)))
    df = pd.DataFrame(rows)
    df["q_bh"] = bh_adjust(df["p_fisher"].to_numpy())
    return df.sort_values(["p_fisher", "candidate"]).reset_index(drop=True)


def _strata_tables(x: np.ndarray, y: np.ndarray, strata: np.ndarray):
    tables = []
    for s in np.unique(strata):
        m = strata == s
        xs, ys = x[m], y[m]
        a = int((xs & ys).sum()); b = int((xs & ~ys).sum())
        c = int((~xs & ys).sum()); d = int((~xs & ~ys).sum())
        tables.append([[a, b], [c, d]])
    return tables


def _strata(cands, keys, n, base=None):
    """Stratum code per unit: the nuisance strata `base` (e.g. calendar week) crossed with the 0/1 values of the
    candidates in `keys`."""
    code = np.zeros(n, dtype=np.int64) if base is None else np.asarray(base, dtype=np.int64).copy()
    for k in keys:
        code = code * 2 + cands[k].astype(np.int64)
    return code


def conditional_p(cands, y, key, given, base=None) -> tuple[float, float]:
    return cmh_greater(_strata_tables(cands[key], y, _strata(cands, given, len(y), base)))


@dataclass
class TraceResult:
    screen: pd.DataFrame
    selected: list[str]
    steps: list[dict] = field(default_factory=list)
    explained: dict = field(default_factory=dict)   # screened candidate -> (cmh p given selected, odds ratio)


def analyse(units: pd.DataFrame, alpha: float = 0.05, max_causes: int = 6, time_strata: bool = True) -> TraceResult:
    """time_strata: also stratify every conditional test by calendar week, so that a candidate that merely
    coincides in time with a time-varying cause (e.g. a material batch in use while a machine drifts) is not
    credited with that cause's effect."""
    cands = candidates(units)
    scr = screen(units, cands)
    y = units["fail"].to_numpy().astype(bool)
    base = (units["day"].to_numpy() // 7) if (time_strata and "day" in units) else None
    pool = list(scr.loc[scr.q_bh < alpha, "candidate"])
    selected: list[str] = []
    log = []
    while pool and len(selected) < max_causes:
        tests = [(k, *conditional_p(cands, y, k, selected, base)) for k in pool]
        tests.sort(key=lambda t: (t[1], t[0]))
        k, p, orr = tests[0]
        thr = alpha / len(pool)
        log.append(dict(action="forward", candidate=k, p_cmh=p, odds_ratio=orr, threshold=thr, accepted=p < thr))
        if p >= thr:
            break
        selected.append(k)
        pool.remove(k)
    changed = True
    while changed and len(selected) > 1:
        changed = False
        worst = None
        for k in selected:
            p, orr = conditional_p(cands, y, k, [s for s in selected if s != k], base)
            if worst is None or p > worst[1]:
                worst = (k, p, orr)
        thr = alpha / len(selected)
        if worst[1] >= thr:
            log.append(dict(action="backward_drop", candidate=worst[0], p_cmh=worst[1], odds_ratio=worst[2],
                            threshold=thr, accepted=False))
            selected.remove(worst[0])
            changed = True
    explained = {}
    for k in scr.loc[scr.q_bh < alpha, "candidate"]:
        if k not in selected:
            explained[k] = conditional_p(cands, y, k, selected, base)
    return TraceResult(scr, selected, log, explained)


# ---- scoring (reads ground truth; only used for evaluation) ---------------------------------------------------------

def _parts(key: str) -> set[str]:
    return {p.strip() for p in key.split("&")}


def classify(key: str, truth: list[str], traps: dict[str, str]) -> str:
    if key in truth:
        return "true_cause"
    if key in traps:
        return "planted_confounder"
    for t in truth:
        if _parts(key) < _parts(t) or _parts(t) < _parts(key):
            return "coarser_or_finer_than_true_cause"
    return "other_false_lead"


def score_against_truth(res: TraceResult, truth: list[str], traps: dict[str, str], alpha: float = 0.05) -> dict:
    screened = list(res.screen.loc[res.screen.q_bh < alpha, "candidate"])
    sel_cls = {k: classify(k, truth, traps) for k in res.selected}
    scr_cls = {k: classify(k, truth, traps) for k in screened}
    return dict(
        truth=truth,
        recovered=[t for t in truth if t in res.selected],
        missed=[t for t in truth if t not in res.selected],
        selected=res.selected,
        selected_false_leads={k: v for k, v in sel_cls.items() if v != "true_cause"},
        screened=screened,
        screened_false_leads={k: v for k, v in scr_cls.items() if v != "true_cause"},
        recall=sum(t in res.selected for t in truth) / len(truth),
        precision=(sum(v == "true_cause" for v in sel_cls.values()) / len(sel_cls)) if sel_cls else float("nan"),
    )
