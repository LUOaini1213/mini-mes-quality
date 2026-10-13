"""SOP assistant: corpus, retrieval, citations, refusal, frozen question set (BM25 mode, as in CI)."""
import hashlib
import json
from pathlib import Path

import pytest

from mesq.sop import SOPAssistant, load_chunks, tokenize
from mesq.sop_eval import QUESTIONS, is_correct, load_questions

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def asst():
    a = SOPAssistant(use_dense=False, refuse_oov=0.34)
    assert a.mode == "bm25"
    return a


def test_every_document_is_marked_synthetic():
    for p in (ROOT / "sop" / "docs").glob("*.md"):
        assert "SYNTHETIC DOCUMENT" in p.read_text(encoding="utf-8")[:400], p.name


def test_chunking_by_section():
    chunks = load_chunks()
    assert len({c.doc_id for c in chunks}) == 9
    prn = [c for c in chunks if c.doc_id == "SOP-PRN-001"]
    assert [c.section for c in prn] == ["1", "2", "3", "4", "5"]
    sec2 = prn[1]
    assert sec2.clauses[0].startswith("2.1 ") and len(sec2.clauses) == 5
    assert all("\n" not in cl for cl in sec2.clauses)


def test_retrieval_finds_the_right_section(asst):
    order, _ = asst.rank("What oxygen level is allowed in the nitrogen reflow tunnel?")
    assert asst.key(order[0]) == "SOP-RFL-003#4"


def test_answer_is_a_verbatim_clause_with_citation(asst):
    a = asst.ask("How long must solder paste warm up before opening the jar?")
    assert a.status == "answered" and a.citation == "[SOP-PRN-001 section 2]"
    body = a.text.replace(" " + a.citation, "")
    chunk = next(c for c in asst.chunks if c.cite == a.citation)
    assert body in chunk.clauses and "4 hours" in body


def test_refuses_out_of_scope_question(asst):
    a = asst.ask("What is the solder pot temperature of the wave soldering machine?")
    assert a.status == "refused" and a.citation is None
    assert asst.ask("Which plating chemistry is used for gold fingers?").status == "refused"


def test_oov_fraction():
    a = SOPAssistant(use_dense=False)
    assert a.oov_fraction("How long must paste warm up?") == 0.0
    assert a.oov_fraction("conformal coating cure") == 1.0


def test_is_correct_requires_keys_and_gold_citation(asst):
    q = dict(answerable=True, gold_doc="SOP-PRN-001", gold_section="2", answer_keys=["4 hours"])
    a = asst.ask("How long must solder paste warm up before opening the jar?")
    assert is_correct(q, a)
    assert not is_correct(dict(q, gold_section="3"), a)
    assert not is_correct(dict(q, answer_keys=["5 hours"]), a)
    assert not is_correct(dict(answerable=False), a)
    # every key is required, not just one of them
    assert not is_correct(dict(q, answer_keys=["4 hours", "8 hours"]), a)


def test_question_set_is_frozen_and_split():
    frozen = (ROOT / "sop" / "questions.sha256").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(QUESTIONS.read_bytes()).hexdigest() == frozen
    qs = load_questions()
    assert len({q["id"] for q in qs}) == len(qs)
    dev = [q for q in qs if q["split"] == "dev"]
    test = [q for q in qs if q["split"] == "test"]
    assert len(dev) == 10 and len(test) == 32
    assert sum(not q["answerable"] for q in test) == 8


def test_tokenizer_keeps_numbers():
    assert tokenize("Peak 235 to 245 degC, 500,000 picks") == ["peak", "235", "245", "degc", "500,000", "picks"]
