"""README numbers are generated from results/ and must match them."""
import json
import re
from pathlib import Path

import pytest

from mesq import report

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


@pytest.fixture(scope="module")
def readme():
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _block(text):
    m = re.search(re.escape(report.BEGIN) + r".*?" + re.escape(report.END), text, re.S)
    assert m, "README results markers missing"
    return m.group(0)


def test_readme_block_equals_rendering_of_results(readme):
    assert _block(readme) == report.readme_block(RESULTS)


def test_headline_json_matches_results():
    stored = json.loads((RESULTS / "headline.json").read_text(encoding="utf-8"))
    assert stored == pytest.approx(report.headline(RESULTS))


def test_numbers_in_readme_tables_match_headline(readme):
    h = json.loads((RESULTS / "headline.json").read_text(encoding="utf-8"))
    rows = {}
    for line in _block(readme).splitlines():
        if line.startswith("| ") and not line.startswith("| Category") and not line.startswith("| Inspection") \
                and not line.startswith("| System"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows[(cells[0], cells[1])] = cells
    checked = 0
    for (cat, method), cells in rows.items():
        k = f"vision.{cat}.{method}.image_auroc"
        if k in h:
            assert float(cells[2]) == pytest.approx(100 * h[k], abs=0.051)
            checked += 1
    for (mode, rate), cells in rows.items():
        k = f"mes.{mode}.both_causes_recovered"
        if k in h:
            assert float(rate.rstrip("%")) == pytest.approx(100 * h[k], abs=0.051)
            checked += 1
    for (name, acc), cells in rows.items():
        k = f"sop.{name}.test_accuracy"
        if k in h:
            assert float(acc.split("%")[0]) == pytest.approx(100 * h[k], abs=0.051)
            checked += 1
    assert checked >= 5
    # README prose must not carry stray headline numbers outside the generated block
    prose = readme.replace(_block(readme), "")
    for key in ("vision.pcb1.patchcore_lite.image_auroc", "sop.hybrid.test_accuracy"):
        if key in h:
            assert f"{100 * h[key]:.1f}" not in prose
