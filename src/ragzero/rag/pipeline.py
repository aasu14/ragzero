"""Pipeline orchestrator.

Wires all 10 steps together. Depends only on the abstract interfaces +
the step modules — no backend code is imported here, so the pipeline
remains backend-agnostic.

Flow:
  ingest -> chunk -> index
  query -> [cache hit?] -> retrieve -> score -> generate -> gate -> answer
                                                                |
                                                                +-> trace
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .cache import QueryCache
from .confidence import ConfidenceConfig, ConfidenceScorer
from .fallback import FallbackConfig, FallbackGate, refusal_answer
from .generation import ConstrainedGenerator, to_answer
from .ingest import Chunker, Ingestor
from .interfaces import Answer, Chunk, Document, Embedder, KeywordIndex, LLM, Tracer, VectorStore
from .observability import new_trace_id
from .retrieval import HybridRetriever, RetrievalResult


@dataclass
class PipelineConfig:
    retriever_k: int = 50
    final_k: int = 10
    chunk_size: int = 800
    chunk_overlap: int = 100
    confidence: ConfidenceConfig = None  # type: ignore[assignment]
    fallback: FallbackConfig = None      # type: ignore[assignment]
    max_context_chunks: int = 8
    max_tokens: int = 512
    # Pluggable chunker. Defaults preserve the original sliding-window
    # behavior (chunk_size + chunk_overlap above). Set to e.g. "sentence",
    # "paragraph", "markdown_header" to switch strategies.
    chunker_strategy: str = "fixed_size"
    chunker_settings: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.confidence is None:
            self.confidence = ConfidenceConfig()
        if self.fallback is None:
            self.fallback = FallbackConfig()
        if self.chunker_settings is None:
            # Default settings carry chunk_size/chunk_overlap so legacy
            # configs that only set those still work with strategy="fixed_size".
            self.chunker_settings = {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }


class RAGPipeline:
    """End-to-end RAG pipeline.

    Construction is verbose because every dependency is injected — this
    is what makes it testable and pluggable. Use `build_dev_pipeline`
    or `build_prod_pipeline` factories for common configurations.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        llm: LLM,
        tracer: Tracer,
        query_cache: QueryCache | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.ingestor = Ingestor()
        from .chunkers import build_chunker
        try:
            self.chunker = build_chunker(self.config.chunker_strategy, self.config.chunker_settings)
        except (ValueError, KeyError):
            # Unknown strategy in config — fall back to the original chunker
            # so a broken UI selection doesn't break the whole pipeline.
            self.chunker = Chunker(self.config.chunk_size, self.config.chunk_overlap)
        self.retriever = HybridRetriever(
            embedder=embedder,
            vector_store=vector_store,
            keyword_index=keyword_index,
            k=self.config.retriever_k,
            final_k=self.config.final_k,
        )
        self.scorer = ConfidenceScorer(self.config.confidence)
        self.generator = ConstrainedGenerator(
            llm=llm,
            max_context_chunks=self.config.max_context_chunks,
            max_tokens=self.config.max_tokens,
        )
        self.gate = FallbackGate(self.config.fallback)
        self.tracer = tracer
        self.query_cache = query_cache

    # ---------- Indexing ----------

    def ingest_documents(self, docs: Iterable[Document]) -> int:
        """Chunk and index a stream of pre-loaded documents."""
        all_chunks: list[Chunk] = []
        n = 0
        for doc in docs:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
            n += 1
        self.retriever.index(all_chunks)
        return n

    def ingest_documents_progress(
        self,
        docs: list[Document],
        on_chunk_done: "Callable[[int, int, str], None] | None" = None,
        on_embed_batch: "Callable[[int, int], None] | None" = None,
        batch_size: int = 32,
    ) -> tuple[int, int]:
        """Same as ingest_documents but reports progress.

        - on_chunk_done(docs_done, docs_total, doc_source) fires after each doc is chunked
        - on_embed_batch(chunks_embedded, chunks_total) fires after each embedding batch

        Returns (n_docs, n_chunks).
        """
        all_chunks: list[Chunk] = []
        total_docs = len(docs)
        for i, doc in enumerate(docs, start=1):
            all_chunks.extend(self.chunker.chunk(doc))
            if on_chunk_done is not None:
                on_chunk_done(i, total_docs, doc.source)
        self.retriever.index_batched(all_chunks, batch_size=batch_size, on_batch=on_embed_batch)
        return total_docs, len(all_chunks)

    def ingest_directory(self, root: Path) -> int:
        return self.ingest_documents(self.ingestor.ingest_directory(root))

    # ---------- Querying ----------

    def answer(self, query: str, filters: list | None = None) -> Answer:
        trace_id = new_trace_id()

        # Step 9: cache lookup — disabled when filters are present since the
        # cache key would need to include them to avoid wrong-result returns.
        if self.query_cache is not None and not filters:
            cached = self.query_cache.get(query)
            if cached is not None:
                self.tracer.emit(trace_id, "cache.hit", {"query": query})
                # Return a copy with the new trace_id for observability
                return Answer(
                    text=cached.text,
                    citations=cached.citations,
                    confidence=cached.confidence,
                    refused=cached.refused,
                    refusal_reason=cached.refusal_reason,
                    trace_id=trace_id,
                )

        self.tracer.emit(trace_id, "query.start", {"query": query, "filters": [f.__dict__ for f in (filters or [])]})

        # Steps 2-3: retrieve
        retrieval = self.retriever.search(query, filters=filters)
        self.tracer.emit(
            trace_id,
            "retrieval",
            {
                "n_results": len(retrieval.chunks),
                "top_chunk_ids": [sc.chunk.chunk_id for sc in retrieval.chunks[:5]],
                "bm25_count": len(retrieval.bm25_ranks),
                "dense_count": len(retrieval.dense_ranks),
            },
        )

        # Step 4: confidence
        scored_chunks, aggregate_conf = self.scorer.score(retrieval)
        self.tracer.emit(
            trace_id,
            "confidence",
            {
                "aggregate": aggregate_conf,
                "per_chunk": [
                    {"chunk_id": sc.chunk.chunk_id, "confidence": sc.confidence}
                    for sc in scored_chunks[:5]
                ],
            },
        )

        # Early-refuse before paying for generation if confidence is already too low.
        # We only check pre-generation criteria here (chunks + retrieval floor +
        # aggregate confidence). Citation and insufficient-evidence checks run after
        # generation.
        early_refuse_reason: str | None = None
        top_retrieval = max(
            (sc.retrieval_score for sc in scored_chunks), default=0.0
        )
        if len(scored_chunks) < self.config.fallback.min_chunks:
            early_refuse_reason = (
                f"Retrieved {len(scored_chunks)} chunks, "
                f"need >= {self.config.fallback.min_chunks}"
            )
        elif top_retrieval < self.config.fallback.min_top_retrieval_score:
            early_refuse_reason = (
                f"Top retrieval score {top_retrieval:.4f} "
                f"< floor {self.config.fallback.min_top_retrieval_score}"
            )
        elif aggregate_conf < self.config.fallback.min_aggregate_confidence:
            early_refuse_reason = (
                f"Aggregate confidence {aggregate_conf:.3f} "
                f"< threshold {self.config.fallback.min_aggregate_confidence}"
            )

        if early_refuse_reason is not None:
            self.tracer.emit(trace_id, "fallback.early", {"reason": early_refuse_reason})
            answer = refusal_answer(early_refuse_reason, aggregate_conf, trace_id)
            self.tracer.emit(
                trace_id, "finalize", {"refused": True, "confidence": aggregate_conf}
            )
            if self.query_cache is not None:
                self.query_cache.set(query, answer)
            return answer

        # Steps 5-6: generate with citations
        gen_output = self.generator.generate(query, scored_chunks)
        self.tracer.emit(
            trace_id,
            "generation",
            {
                "insufficient_evidence": gen_output.insufficient_evidence,
                "n_citations": len(gen_output.citations),
                "text_len": len(gen_output.text),
            },
        )

        # Step 7: final fallback gate
        decision = self.gate.evaluate(scored_chunks, aggregate_conf, gen_output)
        if decision.refuse:
            self.tracer.emit(trace_id, "fallback.final", {"reason": decision.reason})
            answer = refusal_answer(decision.reason or "Refused", aggregate_conf, trace_id)
        else:
            answer = to_answer(gen_output, aggregate_conf, trace_id)
        self.tracer.emit(
            trace_id,
            "finalize",
            {"refused": answer.refused, "confidence": answer.confidence},
        )

        if self.query_cache is not None:
            self.query_cache.set(query, answer)
        return answer
