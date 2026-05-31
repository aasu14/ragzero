"""Step 4: Confidence scoring.

Each retrieved chunk gets a trust score in [0, 1] computed from:
- Freshness: how recent the source is (exponential decay)
- Source quality: configurable per source (e.g., gov.uk > random_blog.com)
- Retrieval consistency: did both BM25 and dense retrievers agree?

The aggregate confidence (used by the fallback gate in step 7) is the
mean of the top-N chunk confidences. We don't use the max alone because
one high-confidence chunk can mask a context that's otherwise weak.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .interfaces import ScoredChunk
from .retrieval import RetrievalResult


@dataclass(frozen=True)
class ConfidenceConfig:
    # Freshness: confidence at age=half_life_days is 0.5
    half_life_days: float = 365.0
    # Source quality lookup; default if source not listed
    source_quality: dict[str, float] = None  # type: ignore[assignment]
    default_source_quality: float = 0.6
    # Weights for the linear combination (must sum to 1.0)
    w_freshness: float = 0.25
    w_source: float = 0.35
    w_consistency: float = 0.40
    # Aggregate over top-N chunks for the pipeline-level confidence
    aggregate_top_n: int = 5

    def quality_for(self, source: str) -> float:
        table = self.source_quality or {}
        for prefix, score in table.items():
            if source.startswith(prefix):
                return score
        return self.default_source_quality


class ConfidenceScorer:
    """Computes per-chunk confidence and an aggregate score."""

    def __init__(self, config: ConfidenceConfig | None = None) -> None:
        self.config = config or ConfidenceConfig()
        total_w = self.config.w_freshness + self.config.w_source + self.config.w_consistency
        if not math.isclose(total_w, 1.0, abs_tol=1e-6):
            raise ValueError(f"weights must sum to 1.0, got {total_w}")

    def score(self, result: RetrievalResult) -> tuple[list[ScoredChunk], float]:
        """Return (chunks_with_confidence, aggregate_confidence)."""
        scored: list[ScoredChunk] = []
        now = datetime.now(timezone.utc)

        for sc in result.chunks:
            freshness = self._freshness(sc.chunk.created_at, now)
            source_q = self.config.quality_for(sc.chunk.source)
            consistency = self._consistency(
                sc.chunk.chunk_id, result.bm25_ranks, result.dense_ranks
            )
            confidence = (
                self.config.w_freshness * freshness
                + self.config.w_source * source_q
                + self.config.w_consistency * consistency
            )
            confidence = max(0.0, min(1.0, confidence))
            scored.append(
                ScoredChunk(
                    chunk=sc.chunk,
                    retrieval_score=sc.retrieval_score,
                    confidence=confidence,
                )
            )

        top = scored[: self.config.aggregate_top_n]
        aggregate = sum(s.confidence for s in top) / len(top) if top else 0.0
        return scored, aggregate

    def _freshness(self, created_at: datetime, now: datetime) -> float:
        age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
        # Exponential decay with the configured half-life
        return 0.5 ** (age_days / self.config.half_life_days)

    @staticmethod
    def _consistency(
        chunk_id: str,
        bm25_ranks: dict[str, int],
        dense_ranks: dict[str, int],
    ) -> float:
        """1.0 if both retrievers agreed and ranked it high; lower otherwise."""
        in_bm25 = chunk_id in bm25_ranks
        in_dense = chunk_id in dense_ranks
        if in_bm25 and in_dense:
            # Both found it — discount by average rank (lower = better)
            avg_rank = (bm25_ranks[chunk_id] + dense_ranks[chunk_id]) / 2
            return max(0.5, 1.0 - avg_rank / 50.0)
        if in_bm25 or in_dense:
            return 0.4  # one retriever only — moderate trust
        return 0.0
