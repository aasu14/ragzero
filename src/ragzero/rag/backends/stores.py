"""Vector store and keyword index backends.

- InMemoryVectorStore: cosine-similarity brute-force search.
- FaissVectorStore: HNSW-backed; lazily imports faiss.
- BM25KeywordIndex: dependency-free BM25 implementation suitable for
  test corpora. Production should use Elasticsearch / OpenSearch /
  Tantivy for a real corpus at 10M scale.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from ..interfaces import Chunk, KeywordIndex, VectorStore


# ---------- Vector stores ----------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    # Vectors from embedders are L2-normalized, so dot = cosine.
    return dot


class InMemoryVectorStore(VectorStore):
    """Brute-force cosine search. O(N) per query; fine up to ~100k chunks."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def search(self, query_vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        scored = [(c, _cosine(query_vector, v)) for c, v in zip(self._chunks, self._vectors)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def size(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()


class FaissVectorStore(VectorStore):  # pragma: no cover
    """HNSW index via FAISS. Suitable for millions of vectors.

    Lazily imports faiss so the rest of the library doesn't require it.
    """

    def __init__(self, dim: int, M: int = 32) -> None:
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as e:
            raise ImportError("faiss and numpy required for FaissVectorStore") from e
        self._faiss = faiss
        self._np = np
        self._index = faiss.IndexHNSWFlat(dim, M)
        self._index.metric_type = faiss.METRIC_INNER_PRODUCT
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        arr = self._np.array(vectors, dtype="float32")
        self._index.add(arr)
        self._chunks.extend(chunks)

    def search(self, query_vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        qv = self._np.array([query_vector], dtype="float32")
        scores, ids = self._index.search(qv, k)
        out: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            out.append((self._chunks[int(idx)], float(score)))
        return out

    def size(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:  # pragma: no cover
        # FAISS HNSW indexes can't be efficiently reset in place; rebuild a
        # fresh empty index with the same parameters.
        dim = self._index.d
        # IndexHNSWFlat doesn't expose M directly; default back to 32 since
        # the rebuilt index is empty and M only affects new inserts.
        self._index = self._faiss.IndexHNSWFlat(dim, 32)
        self._index.metric_type = self._faiss.METRIC_INNER_PRODUCT
        self._chunks.clear()


# ---------- Keyword index ----------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25KeywordIndex(KeywordIndex):
    """In-memory BM25. Reference: Robertson & Zaragoza (2009)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._token_lists: list[list[str]] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            tokens = _tokenize(c.text)
            tf = Counter(tokens)
            self._chunks.append(c)
            self._token_lists.append(tokens)
            self._term_freqs.append(tf)
            self._doc_lens.append(len(tokens))
            for term in tf:
                self._df[term] += 1
        total_len = sum(self._doc_lens)
        self._avgdl = total_len / len(self._doc_lens) if self._doc_lens else 0.0

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q_terms = _tokenize(query)
        if not q_terms or not self._chunks:
            return []
        N = len(self._chunks)
        scored: list[tuple[Chunk, float]] = []
        for i, chunk in enumerate(self._chunks):
            score = 0.0
            tf = self._term_freqs[i]
            dl = self._doc_lens[i]
            for term in q_terms:
                if term not in tf:
                    continue
                df = self._df[term]
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                num = tf[term] * (self.k1 + 1)
                denom = tf[term] + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
                score += idf * num / denom
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
