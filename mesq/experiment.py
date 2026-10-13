"""End-to-end MES traceability experiment: generate synthetic MES -> attach inspection results -> analyse -> score.

Inspection modes
* oracle      decision = synthetic ground truth (perfect inspection; upper bound)
* detector    decision = PatchCore-lite decision on a real VisA test image matched to the unit's true state
* no_reuse    as detector, but only as many units as the image pool covers without reusing an image
* simulated   detector with fixed TPR/FPR (used where the real images are not available, e.g. CI)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import db, link, trace
from .generator import BAD_BATCH_KEY, DRIFT_KEY, TRAP_KEYS, GenConfig, generate

TRUTH = [DRIFT_KEY, BAD_BATCH_KEY]


def inspection(tables, mode, pool=None, seed=0, tpr=0.6, fpr=0.05):
    aoi = link.aoi_equipment_of_units(tables)
    gt = tables["gt_unit"]
    if mode == "oracle":
        return link.oracle_results(gt, aoi)
    if mode == "detector":
        return link.assign_images(gt, aoi, pool, seed=seed)
    if mode == "no_reuse":
        return link.assign_images(gt, aoi, pool, seed=seed, no_reuse=True)
    if mode == "simulated":
        return link.simulated_results(gt, aoi, tpr, fpr, seed=seed)
    raise ValueError(mode)


def run_one(seed: int, mode: str, pool=None, db_path=":memory:", cfg: GenConfig | None = None, **kw):
    cfg = cfg or GenConfig(seed=seed)
    tables = generate(cfg)
    ins = inspection(tables, mode, pool, seed=seed, **kw)
    tables["inspection_result"] = ins
    con = db.build(tables, db_path)
    units = db.unit_factor_table(con)
    res = trace.analyse(units)
    score = trace.score_against_truth(res, TRUTH, TRAP_KEYS)
    gt = tables["gt_unit"].set_index("unit_id").loc[units.index]
    obs = dict(n_units=int(len(units)), true_defect_rate=float(gt.true_defect.mean()),
               observed_fail_rate=float(units.fail.mean()),
               inspection_tpr=float(units.fail[gt.true_defect == 1].mean()) if gt.true_defect.sum() else float("nan"),
               inspection_fpr=float(units.fail[gt.true_defect == 0].mean()),
               reuse_factor=ins.attrs.get("reuse_factor"))
    return dict(seed=seed, mode=mode, obs=obs, score=score), res, units, con, tables


def weekly_rates(units: pd.DataFrame, key: str) -> pd.DataFrame:
    cands = trace.candidates(units, min_exposed=0)
    x = cands[key]
    wk = units["day"] // 7
    rows = []
    for w in sorted(wk.unique()):
        m = (wk == w).to_numpy()
        for exp in (1, 0):
            sel = m & (x == bool(exp))
            n = int(sel.sum()); k = int(units.fail.to_numpy()[sel].sum())
            from .stats import wilson_ci
            lo, hi = wilson_ci(k, n)
            rows.append(dict(cause=key, week=int(w) + 1, exposed=exp, n=n, fails=k, rate=k / n if n else float("nan"),
                             lo=lo, hi=hi))
    return pd.DataFrame(rows)


def summarise(runs: list[dict]) -> dict:
    out = {}
    for mode in sorted({r["mode"] for r in runs}):
        rs = [r for r in runs if r["mode"] == mode]
        per_cause = {t: float(np.mean([t in r["score"]["recovered"] for r in rs])) for t in TRUTH}
        sel_fl = [len(r["score"]["selected_false_leads"]) for r in rs]
        scr_fl = [len(r["score"]["screened_false_leads"]) for r in rs]
        traps_screened = {k: float(np.mean([k in r["score"]["screened"] for r in rs])) for k in TRAP_KEYS}
        traps_selected = {k: float(np.mean([k in r["score"]["selected"] for r in rs])) for k in TRAP_KEYS}
        out[mode] = dict(
            n_seeds=len(rs),
            both_causes_recovered=float(np.mean([len(r["score"]["missed"]) == 0 for r in rs])),
            recovery_rate_by_cause=per_cause,
            mean_recall=float(np.mean([r["score"]["recall"] for r in rs])),
            seeds_with_false_lead_selected=float(np.mean([n > 0 for n in sel_fl])),
            mean_false_leads_selected=float(np.mean(sel_fl)),
            mean_false_leads_screened_naive=float(np.mean(scr_fl)),
            trap_flagged_by_naive_screen=traps_screened,
            trap_selected_after_stratification=traps_selected,
            mean_units=float(np.mean([r["obs"]["n_units"] for r in rs])),
            mean_true_defect_rate=float(np.mean([r["obs"]["true_defect_rate"] for r in rs])),
            mean_observed_fail_rate=float(np.mean([r["obs"]["observed_fail_rate"] for r in rs])),
            mean_inspection_tpr=float(np.nanmean([r["obs"]["inspection_tpr"] for r in rs])),
            mean_inspection_fpr=float(np.mean([r["obs"]["inspection_fpr"] for r in rs])),
            false_lead_examples=sorted({k for r in rs for k in r["score"]["selected_false_leads"]}),
        )
    return out


def run(out_dir: Path, pool: pd.DataFrame | None, seeds=range(1, 21), primary_seed: int = 7,
        build_dir: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["oracle", "detector", "no_reuse"] if pool is not None else ["oracle", "simulated"]
    # primary seed: full detail
    detail = {}
    for mode in modes:
        path = (build_dir / f"mes_seed{primary_seed}_{mode}.sqlite") if build_dir else ":memory:"
        if build_dir:
            build_dir.mkdir(parents=True, exist_ok=True)
        r, res, units, con, tables = run_one(primary_seed, mode, pool, db_path=path)
        res.screen.to_csv(out_dir / f"screen_seed{primary_seed}_{mode}.csv", index=False, float_format="%.6g")
        detail[mode] = dict(r, selection_log=res.steps,
                            explained={k: dict(p_cmh=v[0], odds_ratio=v[1]) for k, v in res.explained.items()})
        if mode in ("detector", "simulated"):
            wr = pd.concat([weekly_rates(units, k) for k in TRUTH])
            wr.to_csv(out_dir / f"weekly_rates_seed{primary_seed}_{mode}.csv", index=False, float_format="%.4f")
            counts = {k: int(len(v)) for k, v in tables.items()}
            detail["table_row_counts"] = counts
        con.close()
    runs = []
    for s in seeds:
        for mode in modes:
            r, *_ = run_one(s, mode, pool)
            runs.append(r)
    summary = dict(truth=TRUTH, traps=TRAP_KEYS, generator_defaults=asdict(GenConfig()), seeds=list(seeds),
                   primary_seed=primary_seed, primary=detail, multi_seed=summarise(runs))
    (out_dir / "trace_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([dict(seed=r["seed"], mode=r["mode"], **r["obs"], recall=r["score"]["recall"],
                       recovered=";".join(r["score"]["recovered"]),
                       selected_false_leads=";".join(r["score"]["selected_false_leads"]),
                       n_screened=len(r["score"]["screened"])) for r in runs]).drop(columns=["reuse_factor"]) \
        .to_csv(out_dir / "trace_runs.csv", index=False, float_format="%.5f")
    return summary
