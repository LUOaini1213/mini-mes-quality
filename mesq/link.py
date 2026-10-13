"""Link the visual-inspection results to the synthetic MES (documented, clearly synthetic mapping).

The MES units are synthetic; the images are real public-dataset test images (VisA). Each synthetic unit is given
one test image whose ground-truth label matches the unit's synthetic ground-truth defect state:

* defective units draw from the anomalous test images, good units from the normal test images;
* each pool is shuffled with a fixed seed and used in round-robin order, so every image is used about equally
  often. The pool is much smaller than the number of units, so images ARE reused (the reuse factor is reported);
* the inspection decision written to the MES is the detector's decision on that image at the threshold fixed on
  held-out normal training images. Detector misses and false alarms therefore propagate into the traceability
  analysis at the detector's real error rates on this dataset.

Other modes used for comparison:
* ``oracle``: the decision equals the synthetic ground truth (perfect inspection, upper bound).
* ``no_reuse``: a random sample of units, cut off as soon as the next unit would need an image a second time
  (so the defect rate is not distorted; in practice the normal-image pool runs out first).
* ``simulated``: a detector with given TPR / FPR, used by the CI tests that must not depend on real images.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_VERSION = "patchcore_lite-wrn50_2-visa_1cls"


def image_pool(score_csvs: list[pd.DataFrame]) -> pd.DataFrame:
    pool = pd.concat(score_csvs, ignore_index=True)
    return pool[["image", "label", "patchcore_score", "patchcore_decision"]].copy()


def assign_images(gt_unit: pd.DataFrame, aoi_equipment: pd.Series, pool: pd.DataFrame, seed: int = 0,
                  no_reuse: bool = False) -> pd.DataFrame:
    """Return inspection_result rows for the units in gt_unit (ordered by unit_id)."""
    rng = np.random.default_rng(seed)
    pools = {lbl: pool[pool.label == lbl].sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)
             for lbl in (0, 1)}
    cursor = {0: 0, 1: 0}
    gt = gt_unit.sort_values("unit_id")
    if no_reuse:
        gt = gt.sample(frac=1.0, random_state=int(rng.integers(2**31)))
    rows = []
    for u, d in zip(gt.unit_id, gt.true_defect):
        d = int(d)
        if no_reuse and cursor[d] >= len(pools[d]):
            break  # stop at the first unit that would need a reused image: keeps the subset a random sample
        img = pools[d].iloc[cursor[d] % len(pools[d])]
        cursor[d] += 1
        rows.append(dict(unit_id=u, step_seq=40, equipment_id=aoi_equipment[u], image_ref=f"visa/{img.image}",
                         anomaly_score=float(img.patchcore_score), decision=img.patchcore_decision,
                         model_version=MODEL_VERSION))
    out = pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)
    out.attrs["reuse_factor"] = {int(k): cursor[k] / len(pools[k]) for k in (0, 1)}
    return out


def oracle_results(gt_unit: pd.DataFrame, aoi_equipment: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(dict(unit_id=gt_unit.unit_id, step_seq=40, equipment_id=gt_unit.unit_id.map(aoi_equipment),
                             image_ref=None, anomaly_score=gt_unit.true_defect.astype(float),
                             decision=np.where(gt_unit.true_defect == 1, "FAIL", "PASS"), model_version="oracle"))


def simulated_results(gt_unit: pd.DataFrame, aoi_equipment: pd.Series, tpr: float, fpr: float,
                      seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    u = rng.random(len(gt_unit))
    fail = np.where(gt_unit.true_defect == 1, u < tpr, u < fpr)
    return pd.DataFrame(dict(unit_id=gt_unit.unit_id, step_seq=40, equipment_id=gt_unit.unit_id.map(aoi_equipment),
                             image_ref=None, anomaly_score=u, decision=np.where(fail, "FAIL", "PASS"),
                             model_version=f"simulated_tpr{tpr}_fpr{fpr}"))


def aoi_equipment_of_units(tables: dict[str, pd.DataFrame]) -> pd.Series:
    """unit_id -> AOI equipment, through unit_lot (the AOI step may run on a split child lot)."""
    ls = tables["lot_step"]
    aoi = ls[ls.step_seq == 40][["lot_id", "equipment_id"]]
    m = tables["unit_lot"].merge(aoi, on="lot_id")
    return m.set_index("unit_id")["equipment_id"]
