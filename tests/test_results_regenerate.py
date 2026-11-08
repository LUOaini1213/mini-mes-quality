"""Recompute the results that are cheap and deterministic, and compare them with the committed files.

* 20-seed traceability summary: experiment.run on the committed per-image scores (results/vision/*_scores.csv),
  compared with results/mes/trace_summary.json (multi_seed) and results/mes/trace_runs.csv.
* SOP assistant, BM25-only system: sop_eval.run with MESQ_NO_DENSE=1, compared with the bm25_only entry of
  results/sop/sop_eval.json.

The vision metrics (they need the VisA images) and the hybrid SOP system (it needs the embedding model) are not
recomputed here. Takes about a minute.
"""
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from mesq import experiment, sop_eval
from mesq.link import image_pool

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def assert_same(new, old, path="$"):
    """Exact for strings/ints/structure, 1e-9 relative for floats (platform libm differences)."""
    if isinstance(old, dict):
        assert isinstance(new, dict) and set(new) == set(old), f"{path}: keys {sorted(new)} != {sorted(old)}"
        for k in old:
            assert_same(new[k], old[k], f"{path}.{k}")
    elif isinstance(old, list):
        assert isinstance(new, list) and len(new) == len(old), f"{path}: length differs"
        for i, (a, b) in enumerate(zip(new, old)):
            assert_same(a, b, f"{path}[{i}]")
    elif isinstance(old, float) and math.isnan(old):
        assert isinstance(new, float) and math.isnan(new), f"{path}: {new} != nan"
    elif isinstance(old, float) or isinstance(new, float):
        assert new == pytest.approx(old, rel=1e-9, abs=1e-12), f"{path}: {new} != {old}"
    else:
        assert new == old, f"{path}: {new!r} != {old!r}"


def _json(obj):
    # the committed files went through json.dumps(default=str); compare like with like
    return json.loads(json.dumps(obj, default=str))


@pytest.fixture(scope="module")
def mes_rerun(tmp_path_factory):
    out = tmp_path_factory.mktemp("mes")
    pool = image_pool([pd.read_csv(p) for p in sorted((RESULTS / "vision").glob("*_scores.csv"))])
    summary = experiment.run(out, pool, seeds=range(1, 21))
    return out, _json(summary)


def test_mes_multi_seed_summary_matches_committed(mes_rerun):
    _, summary = mes_rerun
    committed = json.loads((RESULTS / "mes" / "trace_summary.json").read_text(encoding="utf-8"))
    assert summary["seeds"] == committed["seeds"]
    assert summary["generator_defaults"] == committed["generator_defaults"]
    assert_same(summary["multi_seed"], committed["multi_seed"], "multi_seed")


def test_mes_per_seed_runs_match_committed(mes_rerun):
    out, _ = mes_rerun
    new = pd.read_csv(out / "trace_runs.csv", dtype=str, keep_default_na=False)
    old = pd.read_csv(RESULTS / "mes" / "trace_runs.csv", dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(new, old)


def test_sop_bm25_only_matches_committed(tmp_path, monkeypatch):
    monkeypatch.setenv("MESQ_NO_DENSE", "1")
    rep = _json(sop_eval.run(tmp_path))
    committed = json.loads((RESULTS / "sop" / "sop_eval.json").read_text(encoding="utf-8"))
    assert rep["matches_frozen_hash"] and rep["question_file_sha256"] == committed["question_file_sha256"]
    assert_same(rep["systems"]["bm25_only"], committed["systems"]["bm25_only"], "systems.bm25_only")
    new_rows = (tmp_path / "sop_answers_bm25_only.jsonl").read_text(encoding="utf-8").splitlines()
    old_rows = (RESULTS / "sop" / "sop_answers_bm25_only.jsonl").read_text(encoding="utf-8").splitlines()
    assert_same([json.loads(x) for x in new_rows if x], [json.loads(x) for x in old_rows if x], "answers")
