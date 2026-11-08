"""run_all.py must not silently drop the hybrid SOP row from results/ and the README when the model is missing."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run_all():
    spec = importlib.util.spec_from_file_location("run_all", ROOT / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _results(tmp_path, systems):
    (tmp_path / "sop").mkdir(parents=True)
    (tmp_path / "sop" / "sop_eval.json").write_text(json.dumps({"systems": {s: {} for s in systems}}),
                                                    encoding="utf-8")
    return tmp_path


def test_committed_results_contain_hybrid():
    committed = json.loads((ROOT / "results" / "sop" / "sop_eval.json").read_text(encoding="utf-8"))
    assert "hybrid" in committed["systems"]


def test_stops_before_writing_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MESQ_NO_DENSE", "1")          # embedding model unavailable
    mod = _run_all()
    # point run_all at a scratch copy so that a broken guard cannot touch the real results/ or README
    results = _results(tmp_path / "results", ["hybrid", "bm25_only"])
    readme = tmp_path / "README.md"
    readme.write_bytes((ROOT / "README.md").read_bytes())
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "RESULTS", results)
    monkeypatch.setattr(mod, "BUILD", tmp_path / "build")
    from mesq import experiment, sop_eval

    def started(*a, **k):
        raise AssertionError("pipeline started although the hybrid row cannot be reproduced")
    monkeypatch.setattr(experiment, "run", started)
    monkeypatch.setattr(sop_eval, "run", started)
    with pytest.raises(SystemExit) as e:
        mod.main([])
    assert "hybrid" in str(e.value) and "Nothing was written" in str(e.value)
    assert readme.read_bytes() == (ROOT / "README.md").read_bytes()
    assert sorted(p.name for p in results.rglob("*")) == ["sop", "sop_eval.json"]


def test_guard_allows_explicit_bm25_only_and_bm25_results(tmp_path, monkeypatch):
    monkeypatch.setenv("MESQ_NO_DENSE", "1")
    mod = _run_all()
    mod.check_sop_systems(_results(tmp_path / "a", ["hybrid", "bm25_only"]), allow_drop=True)
    mod.check_sop_systems(_results(tmp_path / "b", ["bm25_only"]))
    with pytest.raises(SystemExit):
        mod.check_sop_systems(_results(tmp_path / "c", ["hybrid", "bm25_only"]))
