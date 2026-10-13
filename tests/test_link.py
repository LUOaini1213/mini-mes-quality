"""Vision -> MES mapping (uses a fake score table, no images)."""
import numpy as np
import pandas as pd

from mesq import link


def fake_pool(n0=10, n1=6):
    return pd.DataFrame(dict(image=[f"n{i}.JPG" for i in range(n0)] + [f"a{i}.JPG" for i in range(n1)],
                             label=[0] * n0 + [1] * n1,
                             patchcore_score=np.r_[np.linspace(0, 1, n0), np.linspace(1, 2, n1)],
                             patchcore_decision=["PASS"] * (n0 - 1) + ["FAIL"] + ["FAIL"] * (n1 - 2) + ["PASS"] * 2))


def gt(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(dict(unit_id=[f"U{i:04d}" for i in range(n)], true_defect=(rng.random(n) < 0.2).astype(int),
                             defect_prob=0.2))


def test_images_match_unit_truth_and_decisions_come_from_detector():
    g, pool = gt(), fake_pool()
    aoi = pd.Series("AOI-01", index=g.unit_id)
    r = link.assign_images(g, aoi, pool, seed=1).merge(g, on="unit_id")
    lab = dict(zip("visa/" + pool.image, pool.label))
    dec = dict(zip("visa/" + pool.image, pool.patchcore_decision))
    assert (r.image_ref.map(lab) == r.true_defect).all()
    assert (r.decision == r.image_ref.map(dec)).all()
    assert len(r) == len(g)


def test_round_robin_reuse_is_balanced():
    g, pool = gt(300), fake_pool()
    r = link.assign_images(g, pd.Series("AOI-01", index=g.unit_id), pool, seed=2)
    counts = r.image_ref.value_counts()
    normals = counts[counts.index.str.contains("/n")]
    assert normals.max() - normals.min() <= 1


def test_no_reuse_mode_never_repeats_an_image():
    g, pool = gt(300), fake_pool()
    r = link.assign_images(g, pd.Series("AOI-01", index=g.unit_id), pool, seed=3, no_reuse=True)
    assert r.image_ref.is_unique
    assert len(r) <= len(pool)


def test_simulated_detector_rates():
    g = gt(20000, seed=4)
    r = link.simulated_results(g, pd.Series("AOI-01", index=g.unit_id), tpr=0.7, fpr=0.05, seed=0).merge(g, on="unit_id")
    fail = r.decision == "FAIL"
    assert abs(fail[r.true_defect == 1].mean() - 0.7) < 0.03
    assert abs(fail[r.true_defect == 0].mean() - 0.05) < 0.01


def test_no_reuse_subset_is_not_enriched_for_defects():
    g, pool = gt(5000, seed=5), fake_pool(200, 200)
    r = link.assign_images(g, pd.Series("AOI-01", index=g.unit_id), pool, seed=6, no_reuse=True).merge(g, on="unit_id")
    assert 150 <= len(r) <= 300                      # stops when the normal pool (binding) is exhausted
    assert abs(r.true_defect.mean() - g.true_defect.mean()) < 0.08
