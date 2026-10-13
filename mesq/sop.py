"""Grounded SOP / work-instruction assistant over the SYNTHETIC documents in sop/docs.

Retrieval: BM25 + dense embeddings (BAAI/bge-small-en-v1.5 via sentence-transformers, loaded from the local
Hugging Face cache only - nothing is downloaded at run time) fused with reciprocal-rank fusion. When the model is
not available (e.g. in CI) the assistant runs BM25-only and says so in its output.

Answers are extractive: the assistant returns the numbered SOP clause that best answers the question, verbatim,
with a citation "[DOC-ID section N]". It refuses when the question does not look covered by the documents. No LLM is
used (see README: a local Ollama server was not already running, and the project does not start or pull models).

BM25, tokenisation and RRF are adapted from the owner's earlier project fm-knowledge-assistant
(src/fmka/retrieval.py, same author); the dense part there used DuckDB-VSS, here plain numpy is enough.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DOCS_DIR = Path(__file__).resolve().parent.parent / "sop" / "docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---- BM25 (adapted from fm-knowledge-assistant) -----------------------------------------------------------------
TOKEN = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")
STOP = set("a an and are as at be by for from has have in is it its of on or shall that the this to was were which "
           "with not no any all been being can may must should do does how what when where who whom why which "
           "there their they them i we you he she our your my me".split())
QUESTION_WORDS = set("many much long often allowed required used need needed happens happen kept set".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75)."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = self.len.mean() if len(docs) else 0.0
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        q = tokenize(query)
        s = np.zeros(len(self.tf))
        for i, c in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            s[i] = sum(self.idf.get(t, 0.0) * c[t] * (self.k1 + 1) / (c[t] + norm) for t in q if t in c)
        return s


def rrf(rankings, k=60):
    """Reciprocal-rank fusion of several rankings (lists of item ids, best first)."""
    agg: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            agg[i] = agg.get(i, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(agg.items(), key=lambda x: (-x[1], x[0]))]


# ---- corpus -----------------------------------------------------------------------------------------------------
@dataclass
class Chunk:
    doc_id: str
    section: str
    title: str
    text: str                      # section body
    clauses: list[str] = field(default_factory=list)

    @property
    def cite(self) -> str:
        return f"[{self.doc_id} section {self.section}]"


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks = []
    for p in sorted(Path(docs_dir).glob("*.md")):
        text = p.read_text(encoding="utf-8")
        doc_id = re.match(r"#\s+(\S+)", text).group(1)
        for m in re.finditer(r"^##\s+(\d+)\s+(.+?)\n(.*?)(?=^##\s|\Z)", text, flags=re.S | re.M):
            sec, title, body = m.group(1), m.group(2).strip(), m.group(3).strip()
            clauses = [re.sub(r"\s+", " ", c).strip()
                       for c in re.split(r"\n(?=\d+\.\d+\s)", body) if c.strip()]
            flat = re.sub(r"\s+", " ", body)
            chunks.append(Chunk(doc_id, sec, title, f"{title}. {flat}", clauses))
    return chunks


def _load_embedder():
    if os.environ.get("MESQ_NO_DENSE"):
        return None
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download at run time
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBED_MODEL, device="cpu")
    except Exception:  # model not in the local cache, or library missing
        return None


@dataclass
class Answer:
    question: str
    status: str                    # answered | refused
    text: str
    citation: str | None
    retrieved: list[str]           # chunk keys "DOC#section", best first
    features: dict


class SOPAssistant:
    def __init__(self, docs_dir: Path = DOCS_DIR, use_dense: bool = True, refuse_dense: float = 0.0,
                 refuse_oov: float = 1.0):
        self.chunks = load_chunks(docs_dir)
        self.bm25 = BM25([f"{c.doc_id} {c.text}" for c in self.chunks])
        self.vocab = set(self.bm25.idf)
        self.model = _load_embedder() if use_dense else None
        self.mode = "hybrid" if self.model is not None else "bm25"
        if self.model is not None:
            self.emb = self.model.encode([c.text for c in self.chunks], normalize_embeddings=True)
            self.clause_texts = [(ci, j, cl) for ci, c in enumerate(self.chunks) for j, cl in enumerate(c.clauses)]
            self.clause_emb = self.model.encode([cl for _, _, cl in self.clause_texts], normalize_embeddings=True)
        self.refuse_dense, self.refuse_oov = refuse_dense, refuse_oov

    def key(self, i: int) -> str:
        return f"{self.chunks[i].doc_id}#{self.chunks[i].section}"

    def _qvec(self, q):
        return self.model.encode([BGE_QUERY_PREFIX + q], normalize_embeddings=True)[0]

    def rank(self, question: str, mode: str | None = None) -> tuple[list[int], dict]:
        mode = mode or self.mode
        bm = self.bm25.scores(question)
        bm_rank = [int(i) for i in np.argsort(-bm, kind="stable") if bm[i] > 0]
        feats = dict(bm25_top=float(bm.max()) if len(bm) else 0.0)
        if mode == "bm25" or self.model is None:
            return bm_rank, feats
        qv = self._qvec(question)
        sims = self.emb @ qv
        dense_rank = [int(i) for i in np.argsort(-sims, kind="stable")]
        feats["dense_top"] = float(sims.max())
        if mode == "dense":
            return dense_rank, feats
        return rrf([bm_rank, dense_rank]), feats

    def oov_fraction(self, question: str) -> float:
        terms = [t for t in tokenize(question) if t not in QUESTION_WORDS and not t.isdigit()]
        if not terms:
            return 0.0
        return sum(t not in self.vocab for t in terms) / len(terms)

    def ask(self, question: str, k: int = 3) -> Answer:
        order, feats = self.rank(question)
        feats["oov_fraction"] = self.oov_fraction(question)
        keys = [self.key(i) for i in order]
        refuse = (not order or feats["oov_fraction"] > self.refuse_oov or
                  ("dense_top" in feats and feats["dense_top"] < self.refuse_dense))
        if refuse:
            return Answer(question, "refused", "The SOP documents do not appear to cover this question.", None,
                          keys, feats)
        top = order[:k]
        qtok = set(tokenize(question))
        best, best_s = None, -1e9
        qv = self._qvec(question) if self.model is not None else None
        for rank, ci in enumerate(top):
            for j, cl in enumerate(self.chunks[ci].clauses):
                ctok = set(tokenize(cl))
                lex = len(qtok & ctok) / max(1, len(qtok))
                s = lex - 0.05 * rank
                if qv is not None:
                    idx = next(n for n, (a, b, _) in enumerate(self.clause_texts) if a == ci and b == j)
                    s += float(self.clause_emb[idx] @ qv)
                if s > best_s:
                    best, best_s = (ci, cl), s
        ci, clause = best
        feats["clause_score"] = best_s
        return Answer(question, "answered", f"{clause} {self.chunks[ci].cite}", self.chunks[ci].cite, keys, feats)
