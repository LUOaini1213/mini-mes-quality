"""Evaluate the SOP assistant on the frozen question set (sop/questions.jsonl, hash in sop/questions.sha256).

* Refusal thresholds are chosen on the 10 dev questions only; the 32 test questions are scored once.
* Retrieval: hit@1 / hit@3 / MRR of the gold section, for BM25, dense and hybrid, on answerable test questions.
* Answers: correct = every answer key appears in the answer AND the citation is the gold section.
  Unanswerable questions are correct only when refused.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .sop import SOPAssistant

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "sop" / "questions.jsonl"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gold_key(q):
    return f"{q['gold_doc']}#{q['gold_section']}"


def is_correct(q, ans) -> bool:
    if not q["answerable"]:
        return ans.status == "refused"
    if ans.status != "answered":
        return False
    cite_ok = ans.citation == f"[{q['gold_doc']} section {q['gold_section']}]"
    return cite_ok and all(k.lower() in ans.text.lower() for k in q["answer_keys"])


def retrieval_metrics(asst: SOPAssistant, qs, mode) -> dict:
    hits1 = hits3 = 0
    rr = []
    for q in qs:
        order, _ = asst.rank(q["question"], mode)
        keys = [asst.key(i) for i in order]
        g = gold_key(q)
        hits1 += keys[:1] == [g]
        hits3 += g in keys[:3]
        rr.append(1.0 / (keys.index(g) + 1) if g in keys else 0.0)
    n = len(qs)
    return dict(n=n, hit_at_1=hits1 / n, hit_at_3=hits3 / n, mrr=float(np.mean(rr)))


def choose_thresholds(asst: SOPAssistant, dev) -> dict:
    feats = []
    for q in dev:
        _, f = asst.rank(q["question"])
        f["oov_fraction"] = asst.oov_fraction(q["question"])
        feats.append(f)
    dense_vals = sorted({round(f.get("dense_top", 0.0), 6) for f in feats})
    d_grid = [0.0] + [(a + b) / 2 for a, b in zip(dense_vals, dense_vals[1:])] if "dense_top" in feats[0] else [0.0]
    o_grid = [1.0, 0.5, 0.34, 0.25, 0.0]
    best = None
    results = []
    for td in d_grid:
        for to in o_grid:
            asst.refuse_dense, asst.refuse_oov = td, to
            acc = float(np.mean([is_correct(q, asst.ask(q["question"])) for q in dev]))
            results.append((acc, td, to))
    top = max(r[0] for r in results)
    tied = [r for r in results if r[0] == top]
    # tie-break: least aggressive OOV rule, then the middle of the tied dense-threshold interval
    to = max(r[2] for r in tied)
    tds = sorted(r[1] for r in tied if r[2] == to)
    td = tds[len(tds) // 2]
    asst.refuse_dense, asst.refuse_oov = td, to
    return dict(refuse_dense=td, refuse_oov=to, dev_accuracy=top, n_dev=len(dev), grid_points=len(results))


def evaluate(asst: SOPAssistant, qs) -> tuple[dict, list[dict]]:
    rows = []
    for q in qs:
        a = asst.ask(q["question"])
        chunk_of_cite = next((c for c in asst.chunks if c.cite == a.citation), None)
        verbatim = (a.status != "answered") or (chunk_of_cite is not None and
                                                 a.text.replace(" " + a.citation, "") in chunk_of_cite.clauses)
        rows.append(dict(id=q["id"], question=q["question"], answerable=q["answerable"], status=a.status,
                         answer=a.text, citation=a.citation, gold=gold_key(q) if q["answerable"] else None,
                         correct=is_correct(q, a), verbatim_support=verbatim,
                         **{k: round(v, 4) for k, v in a.features.items()}))
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    m = dict(n=len(rows), n_answerable=len(ans), n_unanswerable=len(una),
             accuracy=float(np.mean([r["correct"] for r in rows])),
             answerable_correct=float(np.mean([r["correct"] for r in ans])),
             answerable_refused=float(np.mean([r["status"] == "refused" for r in ans])),
             answerable_answered_wrong=float(np.mean([r["status"] == "answered" and not r["correct"] for r in ans])),
             unanswerable_refused=float(np.mean([r["status"] == "refused" for r in una])),
             unanswerable_answered=float(np.mean([r["status"] == "answered" for r in una])),
             verbatim_support_of_answers=float(np.mean([r["verbatim_support"] for r in rows if r["status"] == "answered"]))
             if any(r["status"] == "answered" for r in rows) else float("nan"))
    return m, rows


def run(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    qs = load_questions()
    sha = hashlib.sha256(QUESTIONS.read_bytes()).hexdigest()
    frozen = (ROOT / "sop" / "questions.sha256").read_text(encoding="utf-8").split()[0]
    dev = [q for q in qs if q["split"] == "dev"]
    test = [q for q in qs if q["split"] == "test"]
    report = dict(question_file_sha256=sha, matches_frozen_hash=sha == frozen, n_dev=len(dev), n_test=len(test))
    systems = {}
    for name, use_dense in [("hybrid", True), ("bm25_only", False)]:
        asst = SOPAssistant(use_dense=use_dense)
        if use_dense and asst.mode != "hybrid":
            report["dense_unavailable"] = True
            continue
        thr = choose_thresholds(asst, dev)
        ret = {mode: retrieval_metrics(asst, [q for q in test if q["answerable"]], mode)
               for mode in (["bm25", "dense", "hybrid"] if use_dense else ["bm25"])}
        m, rows = evaluate(asst, test)
        systems[name] = dict(mode=asst.mode, thresholds=thr, retrieval_test=ret, answers_test=m)
        (out_dir / f"sop_answers_{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    report["systems"] = systems
    (out_dir / "sop_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
