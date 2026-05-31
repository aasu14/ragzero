"""Step 7: Hallucination fallback gate.

This is the single most important hallucination control in the pipeline.
The rule: if aggregate confidence < threshold, OR retrieval returned no
chunks, OR the generator reported insufficient evidence, the pipeline
returns a refusal instead of an answer.

We expose three sub-thresholds because the right cutoff differs per
deployment:
- min_aggregate_confidence: gate on step-4 score
- min_chunks: must have at least this many retrieved chunks
- require_min_citations: generated answer must include >=N distinct citations
"""
from __future__ import annotations

from dataclasses import dataclass

from .generation import GenerationOutput
from .interfaces import Answer, ScoredChunk


@dataclass(frozen=True)
class FallbackConfig:
    min_aggregate_confidence: float = 0.55
    min_chunks: int = 1
    require_min_citations: int = 1
    # Minimum raw retrieval score the top chunk must achieve. Confidence
    # scoring blends freshness + source quality + consistency, so a corpus
    # of fresh, high-quality, mutually-confirming docs can clear the
    # confidence threshold even when no chunk is topically relevant.
    # Default of 0.0 disables the check; production should tune per-corpus
    # after measuring score distributions on known-irrelevant queries.
    min_top_retrieval_score: float = 0.0


@dataclass(frozen=True)
class FallbackDecision:
    refuse: bool
    reason: str | None


class FallbackGate:
    """Decides whether to return the generated answer or refuse."""

    def __init__(self, config: FallbackConfig | None = None) -> None:
        self.config = config or FallbackConfig()

    def evaluate(
        self,
        scored_chunks: list[ScoredChunk],
        aggregate_confidence: float,
        output: GenerationOutput,
    ) -> FallbackDecision:
        if len(scored_chunks) < self.config.min_chunks:
            return FallbackDecision(
                True, f"Retrieved {len(scored_chunks)} chunks, need >= {self.config.min_chunks}"
            )
        top_retrieval = max((sc.retrieval_score for sc in scored_chunks), default=0.0)
        if top_retrieval < self.config.min_top_retrieval_score:
            return FallbackDecision(
                True,
                f"Top retrieval score {top_retrieval:.4f} "
                f"< floor {self.config.min_top_retrieval_score}",
            )
        if aggregate_confidence < self.config.min_aggregate_confidence:
            return FallbackDecision(
                True,
                f"Aggregate confidence {aggregate_confidence:.3f} "
                f"< threshold {self.config.min_aggregate_confidence}",
            )
        if output.insufficient_evidence:
            return FallbackDecision(True, "Model reported insufficient evidence")
        distinct_citations = {c.chunk_id for c in output.citations}
        if len(distinct_citations) < self.config.require_min_citations:
            return FallbackDecision(
                True,
                f"Only {len(distinct_citations)} distinct citation(s), "
                f"need >= {self.config.require_min_citations}",
            )
        return FallbackDecision(False, None)


def refusal_answer(reason: str, confidence: float, trace_id: str | None = None) -> Answer:
    """Build an Answer object representing a refusal."""
    return Answer(
        text="",
        citations=[],
        confidence=confidence,
        refused=True,
        refusal_reason=reason,
        trace_id=trace_id,
    )
