"""Graph-augmented RAG strategy.

Combines vector retrieval (semantic similarity) with graph traversal
(structured connections).

Flow:
1. Vector retrieval surfaces semantically similar chunks → seed chunks
2. Extract candidate entities from the query
3. Match query entities to graph entities (substring + fuzzy)
4. Traverse the graph from those entities (BFS, max_depth=2)
5. Collect chunks linked to traversed entities → graph chunks
6. Merge seed + graph chunks, run constrained generation with citations
7. Refuse if combined evidence is below threshold

This catches questions like "What companies did X's co-founders also work at?"
that pure vector retrieval misses.
"""
from __future__ import annotations

import re
from typing import Iterator

from ..generation import to_answer
from ..fallback import refusal_answer
from ..interfaces import ScoredChunk
from .base import Strategy, StrategyContext, StrategyEvent, make_event


_QUERY_TERM_RE = re.compile(r"[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3}|[a-zA-Z0-9]{3,}")


class GraphStrategy(Strategy):
    id = "graph"
    label = "Graph RAG (vector + entity graph)"

    def __init__(self, max_graph_depth: int = 2, max_graph_chunks: int = 5) -> None:
        self.max_graph_depth = max_graph_depth
        self.max_graph_chunks = max_graph_chunks

    def run(self, query: str, ctx: StrategyContext) -> Iterator[StrategyEvent]:
        filters = ctx.options.get("filters") if ctx.options else None
        graph = ctx.options.get("graph_store")
        if graph is None:
            yield make_event("error", "Graph store not configured", reason="no graph_store in options")
            # Fallback to simple
            answer = ctx.pipeline.answer(query, filters=filters)
            yield make_event("final", "Fell back to simple RAG", answer=answer)
            return

        yield make_event("info", "Starting Graph RAG", query=query)

        # 1. Vector retrieval (steps 2-3 of base pipeline)
        retrieval = ctx.pipeline.retriever.search(query, filters=filters)
        yield make_event(
            "retrieval",
            f"Vector search returned {len(retrieval.chunks)} chunks",
            top_chunks=[sc.chunk.chunk_id for sc in retrieval.chunks[:5]],
            top_bm25=retrieval.top_bm25_score,
            top_dense=retrieval.top_dense_score,
        )

        # 2. Extract entity-like terms from the query
        candidate_terms = self._candidate_entity_terms(query)
        yield make_event(
            "thought",
            f"Extracted entity candidates from query: {candidate_terms[:8]}",
            candidates=candidate_terms,
        )

        # 3. Match query terms to entities in the graph
        matched_entities = []
        if candidate_terms:
            matched_entities = graph.find_entities(candidate_terms, limit=5)
        yield make_event(
            "tool_call",
            "graph.find_entities",
            input=candidate_terms,
            matched=[e.name for e in matched_entities],
        )

        # 4. Traverse the graph
        graph_chunk_ids: list[str] = []
        paths_found = []
        for ent in matched_entities:
            paths = graph.neighbors(ent.name, max_depth=self.max_graph_depth)
            for p in paths:
                for node in p.nodes:
                    graph_chunk_ids.extend(node.chunk_ids)
                for edge in p.edges:
                    graph_chunk_ids.extend(edge.chunk_ids)
                paths_found.append({
                    "nodes": [n.name for n in p.nodes],
                    "edges": [{"src": e.source, "tgt": e.target, "type": e.rel_type} for e in p.edges],
                })
        yield make_event(
            "graph_hop",
            f"Traversed {len(paths_found)} path(s) from {len(matched_entities)} entity(ies)",
            paths=paths_found[:20],
            graph_chunks_collected=len(set(graph_chunk_ids)),
        )

        # 5. Merge seed + graph chunks
        seed_chunks = retrieval.chunks
        seed_ids = {sc.chunk.chunk_id for sc in seed_chunks}
        # Look up Chunk objects for graph_chunk_ids from the vector store
        merged: list[ScoredChunk] = list(seed_chunks)
        extra_added = 0
        unique_graph_ids = list(dict.fromkeys(graph_chunk_ids))  # preserve order, dedup
        for cid in unique_graph_ids[: self.max_graph_chunks]:
            if cid in seed_ids:
                continue
            chunk = self._lookup_chunk(ctx, cid)
            if chunk is None:
                continue
            # Synthetic high score for graph-confirmed chunks
            merged.append(ScoredChunk(chunk=chunk, retrieval_score=0.5, confidence=0.0))
            extra_added += 1
        yield make_event(
            "info",
            f"Merged {len(seed_chunks)} vector chunks + {extra_added} graph-linked chunks",
        )

        # 6. Score, generate, gate (same as base pipeline)
        # Build a synthetic RetrievalResult for the scorer
        from ..retrieval import RetrievalResult
        merged_result = RetrievalResult(
            chunks=merged,
            bm25_ranks=retrieval.bm25_ranks,
            dense_ranks=retrieval.dense_ranks,
            top_bm25_score=retrieval.top_bm25_score,
            top_dense_score=retrieval.top_dense_score,
        )
        scored, aggregate = ctx.pipeline.scorer.score(merged_result)
        yield make_event(
            "info",
            f"Aggregate confidence: {aggregate:.3f}",
            aggregate=aggregate,
        )

        # Generation
        gen_out = ctx.pipeline.generator.generate(query, scored)
        yield make_event(
            "generation",
            "Constrained generation complete",
            n_citations=len(gen_out.citations),
            insufficient_evidence=gen_out.insufficient_evidence,
        )

        # Fallback gate
        decision = ctx.pipeline.gate.evaluate(scored, aggregate, gen_out)
        if decision.refuse:
            answer = refusal_answer(decision.reason or "Refused", aggregate, ctx.trace_id)
            yield make_event("final", "Refused by fallback gate", answer=answer)
            return

        answer = to_answer(gen_out, aggregate, ctx.trace_id)
        yield make_event("final", "Done", answer=answer)

    @staticmethod
    def _candidate_entity_terms(query: str) -> list[str]:
        # Capitalized phrases first
        caps = [m.group() for m in re.finditer(
            r"[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3}", query
        )]
        # Plus any 4+ letter content word
        words = [w for w in re.findall(r"[a-zA-Z]{4,}", query.lower())]
        # Dedup preserving order
        seen = set()
        out = []
        for term in caps + words:
            t = term.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out[:10]

    @staticmethod
    def _lookup_chunk(ctx: StrategyContext, chunk_id: str):
        """Find a Chunk by ID in the vector store. O(N), fine for in-memory."""
        store = ctx.pipeline.retriever.vector_store
        # Most stores expose internal chunks via _chunks; both InMemory and Faiss do
        chunks = getattr(store, "_chunks", None)
        if chunks is None:
            return None
        for c in chunks:
            if c.chunk_id == chunk_id:
                return c
        return None
