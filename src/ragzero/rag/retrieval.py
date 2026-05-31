"""Steps 2-3: Hybrid retrieval.

Combines:
- BM25 keyword search (sparse) — catches exact terms, rare entities
- Dense vector search via ANN (semantic) — catches paraphrases

Fusion strategy: Reciprocal Rank Fusion (RRF). It's a simple, parameter-light
fusion that doesn't require score calibration between retrievers — you only
need the rank position from each list.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
and individual rank learning methods" (SIGIR 2009).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .interfaces import Chunk, Embedder, KeywordIndex, ScoredChunk, VectorStore


@dataclass(frozen=True)
class RetrievalResult:
    """Output of hybrid retrieval with provenance for observability."""
    chunks: list[ScoredChunk]
    bm25_ranks: dict[str, int]       # chunk_id -> rank in BM25 list
    dense_ranks: dict[str, int]      # chunk_id -> rank in dense list
    top_bm25_score: float = 0.0      # raw score of the BM25 top hit (for relevance gating)
    top_dense_score: float = 0.0     # raw score of the dense top hit


class HybridRetriever:
    """Hybrid retriever: BM25 + dense ANN, fused with RRF.

    `k` is the per-retriever fetch size. The final returned list is
    capped at `final_k`. Setting k > final_k gives RRF more candidates
    to fuse and typically improves recall.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        k: int = 50,
        final_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.k = k
        self.final_k = final_k
        self.rrf_k = rrf_k

    def index(self, chunks: list[Chunk]) -> None:
        """Add chunks to both indexes. Single-batch path for small inputs."""
        if not chunks:
            return
        vectors = self.embedder.embed([c.text for c in chunks])
        self.vector_store.add(chunks, vectors)
        self.keyword_index.add(chunks)

    def index_batched(
        self,
        chunks: list[Chunk],
        batch_size: int = 32,
        on_batch: "Callable[[int, int], None] | None" = None,
    ) -> None:
        """Embed chunks in batches, invoking on_batch(done, total) after each batch.

        Used by the job system to report progress on large ingestion runs.
        Keyword index is built in one shot (fast); only the embedding loop
        actually streams progress.
        """
        if not chunks:
            return
        total = len(chunks)
        all_vectors: list[list[float]] = []
        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            batch_vectors = self.embedder.embed([c.text for c in batch])
            all_vectors.extend(batch_vectors)
            if on_batch is not None:
                on_batch(min(start + batch_size, total), total)
        self.vector_store.add(chunks, all_vectors)
        self.keyword_index.add(chunks)

    def search(self, query: str, filters: list | None = None) -> RetrievalResult:
        # Dense — pass filters if the backend's search accepts them; otherwise
        # we'll post-filter below.
        qv = self.embedder.embed([query])[0]
        # Over-fetch when post-filtering so we don't run out of candidates.
        k_dense = self.k * 3 if filters else self.k
        try:
            dense_hits = self.vector_store.search(qv, k_dense, filters=filters)
        except TypeError:
            dense_hits = self.vector_store.search(qv, k_dense)
        # Sparse — keyword index is in-process only; always post-filter.
        bm25_hits = self.keyword_index.search(query, k_dense)

        if filters:
            from .filters import matches
            def _passes(chunk):
                extras = {"source": chunk.source, "doc_id": chunk.doc_id, "doc_type": chunk.metadata.get("doc_type")}
                return matches(filters, chunk.metadata, extras)
            dense_hits = [(c, s) for (c, s) in dense_hits if _passes(c)][: self.k]
            bm25_hits = [(c, s) for (c, s) in bm25_hits if _passes(c)][: self.k]

        # Build rank maps
        dense_ranks = {c.chunk_id: rank for rank, (c, _) in enumerate(dense_hits)}
        bm25_ranks = {c.chunk_id: rank for rank, (c, _) in enumerate(bm25_hits)}

        # Collect all unique chunks
        chunks_by_id: dict[str, Chunk] = {}
        scores_by_id: dict[str, float] = {}
        for c, s in dense_hits:
            chunks_by_id[c.chunk_id] = c
            scores_by_id.setdefault(c.chunk_id, s)
        for c, s in bm25_hits:
            chunks_by_id.setdefault(c.chunk_id, c)
            scores_by_id.setdefault(c.chunk_id, s)

        # RRF fusion
        fused: list[ScoredChunk] = []
        for chunk_id, chunk in chunks_by_id.items():
            rrf_score = 0.0
            if chunk_id in dense_ranks:
                rrf_score += 1.0 / (self.rrf_k + dense_ranks[chunk_id] + 1)
            if chunk_id in bm25_ranks:
                rrf_score += 1.0 / (self.rrf_k + bm25_ranks[chunk_id] + 1)
            fused.append(ScoredChunk(chunk=chunk, retrieval_score=rrf_score))

        fused.sort(key=lambda sc: sc.retrieval_score, reverse=True)
        return RetrievalResult(
            chunks=fused[: self.final_k],
            bm25_ranks=bm25_ranks,
            dense_ranks=dense_ranks,
            top_bm25_score=bm25_hits[0][1] if bm25_hits else 0.0,
            top_dense_score=dense_hits[0][1] if dense_hits else 0.0,
        )
