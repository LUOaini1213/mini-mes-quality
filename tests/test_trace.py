"""Traceability analysis on small hand-built unit tables and on a simulated-detector end-to-end run."""
import numpy as np
import pandas as pd
import pytest

from mesq import trace
from mesq.experiment import TRUTH, run_one
from mesq.generator import TRAP_KEYS


def toy_units(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    eq = rng.choice(["M1", "M2", "M3"], n)
    shift = rng.choice(["A", "B", "C"], n)
    # operator X works mostly on M2 (confounder), M2 is the real cause
    op = np.where(eq == "M2", np.where(rng.random(n) < 0.8, "X", "Y"), rng.choice(["X", "Y", "Z"], n, p=[0.1, 0.45, 0.45]))
    p = 0.03 + 0.10 * (eq == "M2")
    fail = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"PLACE.equipment": eq, "PLACE.shift": shift, "PLACE.operator": op,
                         "ASSY.equipment": rng.choice(["Q1", "Q2"], n), "fail": fail, "day": rng.integers(0, 28, n)})


def test_candidates_exclude_post_inspection_steps_and_rare_levels():
    U = toy_units()
    c = trace.candidates(U)
    assert "PLACE.equipment=M2" in c and "PLACE.equipment=M2 & PLACE.shift=C" in c
    assert not any(k.startswith("ASSY") for k in c)
    U.loc[:5, "PLACE.operator"] = "RARE"
    assert "PLACE.operator=RARE" not in trace.candidates(U)


def test_screen_rates_and_counts():
    U = toy_units()
    s = trace.screen(U, trace.candidates(U)).set_index("candidate")
    row = s.loc["PLACE.equipment=M2"]
    m2 = U["PLACE.equipment"] == "M2"
    assert row.n_exposed == m2.sum() and row.fail_exposed == U.fail[m2].sum()
    assert row.rate_exposed_lo <= row.rate_exposed <= row.rate_exposed_hi
    assert s.index[0] == "PLACE.equipment=M2"
    assert (s.q_bh >= s.p_fisher - 1e-12).all()


def test_analyse_selects_cause_not_confounder():
    res = trace.analyse(toy_units())
    assert "PLACE.operator=X" in list(res.screen.loc[res.screen.q_bh < 0.05, "candidate"])
    assert res.selected == ["PLACE.equipment=M2"]
    assert res.explained["PLACE.operator=X"][0] > 0.05
    # forward steps use a Bonferroni threshold over the candidates still in play
    n_screened = int((res.screen.q_bh < 0.05).sum())
    assert res.steps[0]["threshold"] == pytest.approx(0.05 / n_screened)
    assert res.steps[1]["threshold"] == pytest.approx(0.05 / (n_screened - 1))


def test_no_selection_under_the_null():
    U = toy_units(seed=3)
    U["fail"] = (np.random.default_rng(9).random(len(U)) < 0.05).astype(int)
    assert trace.analyse(U).selected == []


def test_classification_of_leads():
    truth = ["PLACE.equipment=PNP-02 & PLACE.shift=C", "PRINT.material=SP-B07"]
    assert trace.classify("PRINT.material=SP-B07", truth, TRAP_KEYS) == "true_cause"
    assert trace.classify("PLACE.operator=OP-009", truth, TRAP_KEYS) == "planted_confounder"
    assert trace.classify("PLACE.equipment=PNP-02", truth, TRAP_KEYS) == "coarser_or_finer_than_true_cause"
    assert trace.classify("AOI.equipment=AOI-01", truth, TRAP_KEYS) == "other_false_lead"


def test_end_to_end_with_simulated_detector_is_pinned():
    r, res, units, con, tables = run_one(4, "simulated", tpr=0.6, fpr=0.05)
    con.close()
    assert 0.5 < r["obs"]["inspection_tpr"] < 0.7 and 0.03 < r["obs"]["inspection_fpr"] < 0.07
    assert TRUTH[0] in r["score"]["recovered"]
    assert r["score"]["recall"] == len(r["score"]["recovered"]) / 2
