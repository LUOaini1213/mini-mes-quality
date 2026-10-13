"""Statistics helpers against hand-computed / reference values."""
import math

import numpy as np
import pytest
from scipy import stats

from mesq.stats import bh_adjust, cmh_greater, fisher_greater, holm_adjust, risk_ratio_ci, wilson_ci


def test_wilson_reference_values():
    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0 and hi == pytest.approx(0.1611, abs=1e-4)
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_risk_ratio_katz():
    rr, lo, hi = risk_ratio_ci(20, 100, 10, 100)
    assert rr == pytest.approx(2.0)
    se = math.sqrt(1 / 20 - 1 / 100 + 1 / 10 - 1 / 100)
    assert lo == pytest.approx(2 * math.exp(-1.959964 * se)) and hi == pytest.approx(2 * math.exp(1.959964 * se))
    rr0, _, _ = risk_ratio_ci(0, 50, 5, 50)   # zero cell -> continuity correction, finite
    assert 0 < rr0 < 1


def test_fisher_one_sided_direction():
    p_more = fisher_greater(10, 90, 2, 98)
    p_less = fisher_greater(2, 98, 10, 90)
    assert p_more < 0.05 < p_less
    assert p_more == pytest.approx(stats.fisher_exact([[10, 90], [2, 98]], alternative="greater")[1])


def test_bh_hand_example():
    p = [0.01, 0.04, 0.03, 0.20]
    # sorted: .01,.03,.04,.20 -> .04,.06,.0533,.20 -> monotone: .04,.0533,.0533,.20
    assert np.allclose(bh_adjust(p), [0.04, 0.05333333, 0.05333333, 0.20])
    assert len(bh_adjust([])) == 0


def test_holm_hand_example():
    p = [0.01, 0.04, 0.03, 0.20]
    # sorted .01*4=.04, .03*3=.09, .04*2=.08->.09 (monotone), .20*1=.20
    assert np.allclose(holm_adjust(p), [0.04, 0.09, 0.09, 0.20])


def test_cmh_hand_example():
    tables = [[[10, 40], [5, 45]], [[8, 12], [4, 16]]]
    # stratum 1: n=100, E=50*15/100=7.5, V=50*50*15*85/(100^2*99)
    # stratum 2: n=40,  E=20*12/40=6,    V=20*20*12*28/(40^2*39)
    E = 7.5 + 6
    V = 50 * 50 * 15 * 85 / (100 ** 2 * 99) + 20 * 20 * 12 * 28 / (40 ** 2 * 39)
    z = (18 - E - 0.5) / math.sqrt(V)
    p, orr = cmh_greater(tables)
    assert p == pytest.approx(stats.norm.sf(z))
    assert orr == pytest.approx((10 * 45 / 100 + 8 * 16 / 40) / (40 * 5 / 100 + 12 * 4 / 40))


def test_cmh_explains_away_confounder():
    # exposure only looks harmful because it is concentrated in the high-risk stratum
    tables = [[[18, 72], [2, 8]], [[1, 9], [9, 81]]]   # within each stratum the rates are equal
    p, orr = cmh_greater(tables)
    assert orr == pytest.approx(1.0) and p > 0.3
    pooled = fisher_greater(19, 81, 11, 89)
    assert pooled < 0.1
