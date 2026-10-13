"""Generator invariants: determinism, pseudonymity, injected causes present and recoverable."""
import re

import numpy as np
import pandas as pd
import pytest

from mesq import db, link, trace
from mesq.generator import BAD_BATCH_KEY, DRIFT_KEY, TRAP_KEYS, GenConfig, generate


@pytest.fixture(scope="module")
def T():
    return generate(GenConfig(seed=11))


def test_deterministic_given_seed():
    a, b = generate(GenConfig(seed=5, days=14)), generate(GenConfig(seed=5, days=14))
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])
    c = generate(GenConfig(seed=6, days=14))
    assert not a["gt_unit"].equals(c["gt_unit"])


def test_pseudonymous_ids_only(T):
    assert all(re.fullmatch(r"OP-\d{3}", o) for o in T["operator"].operator_id)
    assert set(T["operator"].columns) == {"operator_id", "home_shift"}
    assert all(re.fullmatch(r"SUP-[A-Z]", s) for s in T["material_batch"].supplier_code)


def test_ground_truth_declared(T):
    assert set(T["gt_cause"].cause_key) == {DRIFT_KEY, BAD_BATCH_KEY}


def _unit_paths(T):
    con = db.build({k: v for k, v in T.items()} | {"inspection_result": link.oracle_results(
        T["gt_unit"], link.aoi_equipment_of_units(T))})
    U = db.unit_factor_table(con)
    return U.join(T["gt_unit"].set_index("unit_id"))


def test_injected_effects_are_in_the_probabilities(T):
    U = _unit_paths(T)
    cfg = GenConfig()
    bad = U["PRINT.material.SOLDER_PASTE"] == cfg.bad_batch
    assert bad.sum() >= 100
    assert (U.loc[bad, "defect_prob"] >= cfg.bad_batch_add).all()
    drift = (U["PLACE.equipment"] == "PNP-02") & (U["PLACE.shift"] == "C")
    late = U["day"] >= cfg.drift_start_day + 7
    assert U.loc[drift & late & ~bad, "defect_prob"].mean() > U.loc[~drift & ~bad, "defect_prob"].mean() + 0.05
    # outside the causes, the probability is the base rate plus small lot noise
    clean = U.loc[~drift & ~bad, "defect_prob"]
    assert abs(clean.mean() - cfg.base_defect) < 0.003 and clean.max() < 0.05


def test_traps_are_confounded_but_not_causal(T):
    U = _unit_paths(T)
    drift = (U["PLACE.equipment"] == "PNP-02") & (U["PLACE.shift"] == "C")
    op = U["PLACE.operator"] == "OP-009"
    assert (op & drift).sum() / op.sum() > 0.45         # OP-009 is over-represented on the drifting machine (1/3 if random) ...
    assert (op & ~drift).sum() > 0                       # ... but not only, so the two are separable
    bad = U["PRINT.material.SOLDER_PASTE"] == "SP-B07"
    rfl1 = U["REFLOW.equipment"] == "RFL-01"   # oven trap: RFL-02 is down while SP-B07 is loaded
    assert rfl1[bad].mean() > rfl1[~bad].mean() + 0.15
    # no effect of the trap operator once the drift machine is excluded
    clean = ~drift & ~bad
    assert abs(U.loc[clean & op, "defect_prob"].mean() - U.loc[clean & ~op, "defect_prob"].mean()) < 0.005


@pytest.mark.parametrize("seed", range(1, 21))
def test_bad_batch_exposure_every_results_seed(seed):
    T = generate(GenConfig(seed=seed))
    lots = T["material_consumption"].query("batch_id == 'SP-B07'").lot_id
    n = T["unit_lot"][T["unit_lot"].lot_id.isin(lots)].unit_id.nunique()
    assert n >= 100


@pytest.mark.parametrize("seed", [2, 4])
def test_ground_truth_recoverable_with_perfect_inspection(seed):
    T = generate(GenConfig(seed=seed))
    T["inspection_result"] = link.oracle_results(T["gt_unit"], link.aoi_equipment_of_units(T))
    U = db.unit_factor_table(db.build(T))
    res = trace.analyse(U)
    sc = trace.score_against_truth(res, [DRIFT_KEY, BAD_BATCH_KEY], TRAP_KEYS)
    assert sc["missed"] == [] and sc["selected_false_leads"] == {}
    assert "PLACE.operator=OP-009" in sc["screened"]          # the trap fools the naive screen ...
    assert "PLACE.operator=OP-009" not in sc["selected"]      # ... but not the stratified selection
