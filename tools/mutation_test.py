"""Hand-written mutation testing: break the code on purpose and check that the test suite notices.

Each mutation replaces one exact snippet in one source file, runs pytest, and restores the file (also on error).
A mutation is "caught" when pytest fails. Results go to results/mutation.json.

    python tools/mutation_test.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MUTATIONS = [
    # (file, original snippet, mutated snippet, what it breaks)
    ("mesq/stats.py", "centre = (p + z * z / (2 * n)) / den", "centre = (p + z * z / n) / den", "Wilson centre"),
    ("mesq/stats.py", "q = np.minimum.accumulate(ranked[::-1])[::-1]", "q = ranked", "BH monotonicity"),
    ("mesq/stats.py", "adj = np.maximum.accumulate((n - np.arange(n)) * p[order])",
     "adj = np.maximum.accumulate((n - np.arange(n) - 1) * p[order])", "Holm multipliers off by one"),
    ("mesq/stats.py", "z = (sa - se - 0.5) / math.sqrt(sv)", "z = (sa - se + 0.5) / math.sqrt(sv)",
     "CMH continuity correction sign"),
    ("mesq/stats.py", 'alternative="greater"', 'alternative="less"', "Fisher test direction"),
    ("mesq/stats.py", "se = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)", "se = math.sqrt(1 / a + 1 / c)",
     "Katz RR standard error"),
    ("mesq/db.py", "SELECT g.parent_lot_id, a.depth + 1 FROM lot_genealogy g JOIN anc a ON g.child_lot_id = a.lot_id",
     "SELECT g.parent_lot_id, a.depth + 1 FROM lot_genealogy g JOIN anc a ON g.parent_lot_id = a.lot_id",
     "ancestor recursion joins the wrong side"),
    ("mesq/db.py", "SELECT g.child_lot_id, d.depth + 1 FROM lot_genealogy g JOIN des d ON g.parent_lot_id = d.lot_id",
     "SELECT g.child_lot_id, d.depth + 1 FROM lot_genealogy g JOIN des d ON g.child_lot_id = d.lot_id",
     "descendant recursion joins the wrong side"),
    ("mesq/db.py", 'out["fail"] = (out.pop("decision") == "FAIL").astype(int)',
     'out["fail"] = (out.pop("decision") == "PASS").astype(int)', "inspection decision inverted"),
    ("mesq/db.py", "FROM unit_lot ul JOIN lot_step ls ON ls.lot_id = ul.lot_id\n",
     "FROM unit u JOIN lot_step ls ON ls.lot_id = u.final_lot_id JOIN unit_lot ul ON ul.unit_id = u.unit_id AND ul.lot_id = u.final_lot_id\n",
     "factor table ignores genealogy (final lot only)"),
    ("mesq/generator.py", "            if paste == cfg.bad_batch:\n", "            if paste != cfg.bad_batch:\n",
     "bad batch effect applied to every other batch"),
    ("mesq/generator.py", 'if code == "PLACE" and eq == "PNP-02" and shift == "C" and day >= cfg.drift_start_day:',
     'if code == "PLACE" and eq == "PNP-02" and shift == "B" and day >= cfg.drift_start_day:',
     "drift injected on the wrong shift"),
    ("mesq/generator.py", "parts = [units[:cut], units[cut:]]", "parts = [units[:cut], units[cut - 1:]]",
     "split puts one unit in both children"),
    ("mesq/trace.py", 'PRE_INSPECTION_STEPS = ["PRINT", "PLACE", "REFLOW", "AOI"]',
     'PRE_INSPECTION_STEPS = ["PRINT", "PLACE", "REFLOW", "AOI", "ASSY"]', "post-inspection step treated as cause"),
    ("mesq/trace.py", "tests = [(k, *conditional_p(cands, y, k, selected, base)) for k in pool]",
     "tests = [(k, *conditional_p(cands, y, k, [], base)) for k in pool]", "forward step not stratified"),
    ("mesq/trace.py", "thr = alpha / len(pool)", "thr = alpha", "no multiplicity correction in forward step"),
    ("mesq/vision/metrics.py", "return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))",
     "return float((r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))", "AUROC rank offset"),
    ("mesq/vision/metrics.py", "return float(trapezoid(y, x) / fpr_limit)", "return float(trapezoid(y, x))",
     "AUPRO not normalised"),
    ("mesq/vision/models.py", "i = int(torch.argmax(mind))", "i = int(torch.argmin(mind))", "coreset picks nearest"),
    ("mesq/vision/models.py", "out.append(torch.cdist(queries[i:i + chunk], bank).min(1).values)",
     "out.append(torch.cdist(queries[i:i + chunk], bank).max(1).values)", "kNN uses farthest neighbour"),
    ("mesq/link.py", "pools = {lbl: pool[pool.label == lbl]", "pools = {lbl: pool[pool.label != lbl]",
     "image label does not match unit truth"),
    ("mesq/link.py", "            break  # stop at the first unit", "            continue  # stop at the first unit",
     "no-reuse subset enriched for defects (selection bias)"),
    ("mesq/sop.py", 'refuse = (not order or feats["oov_fraction"] > self.refuse_oov or',
     'refuse = (not order or feats["oov_fraction"] < self.refuse_oov or', "OOV refusal inverted"),
    ("mesq/sop.py", "return Answer(question, \"answered\", f\"{clause} {self.chunks[ci].cite}\", self.chunks[ci].cite, keys, feats)",
     "return Answer(question, \"answered\", f\"{clause} {self.chunks[order[-1]].cite}\", self.chunks[order[-1]].cite, keys, feats)",
     "citation points to the wrong chunk"),
    ("mesq/sop_eval.py", 'return cite_ok and all(k.lower() in ans.text.lower() for k in q["answer_keys"])',
     'return cite_ok and any(k.lower() in ans.text.lower() for k in q["answer_keys"])', "answer keys: any instead of all"),
    ("mesq/report.py", 'return "n/a" if x is None else f"{100 * x:.1f}"', 'return "n/a" if x is None else f"{x:.1f}"',
     "README numbers not scaled to percent"),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    py = sys.executable
    env = dict(os.environ, OMP_NUM_THREADS="2", MESQ_NO_DENSE="1")
    before = {f: sha(ROOT / f) for f in {m[0] for m in MUTATIONS}}
    base = subprocess.run([py, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    if base.returncode != 0:
        sys.exit("baseline test suite fails - fix it before mutation testing\n" + base.stdout[-2000:])
    out = []
    for f, old, new, what in MUTATIONS:
        p = ROOT / f
        raw = p.read_bytes()                      # restore byte-for-byte (Path.write_text would rewrite newlines)
        src = raw.decode("utf-8")
        if "\r\n" in src:
            old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        n = src.count(old)
        if n != 1:
            out.append(dict(file=f, mutation=what, status=f"NOT APPLIED (snippet found {n} times)"))
            continue
        t = time.time()
        try:
            p.write_bytes(src.replace(old, new).encode("utf-8"))
            r = subprocess.run([py, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=900)
            caught = r.returncode != 0
            failing = [ln.split(" ")[1] for ln in r.stdout.splitlines() if ln.startswith("FAILED") or ln.startswith("ERROR")]
        finally:
            p.write_bytes(raw)
        out.append(dict(file=f, mutation=what, status="caught" if caught else "SURVIVED", first_failing=failing[:1],
                        seconds=round(time.time() - t, 1)))
        print(f"{'caught  ' if caught else 'SURVIVED'} {f}: {what}", flush=True)
    after = {f: sha(ROOT / f) for f in before}
    assert before == after, "a source file was not restored!"
    applied = [o for o in out if o["status"] in ("caught", "SURVIVED")]
    summary = dict(total=len(applied), caught=sum(o["status"] == "caught" for o in applied),
                   not_applied=len(out) - len(applied), mutations=out, sources_restored=True)
    (ROOT / "results" / "mutation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"caught {summary['caught']}/{summary['total']}")


if __name__ == "__main__":
    main()
