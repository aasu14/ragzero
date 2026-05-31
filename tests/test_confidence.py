"""Tests for Step 4: confidence scoring."""
from datetime import datetime, timedelta, timezone

import pytest

from ragzero.rag.confidence import ConfidenceConfig, ConfidenceScorer
from ragzero.rag.interfaces import Chunk, ScoredChunk
from ragzero.rag.retrieval import RetrievalResult


def _chunk(chunk_id: str, source: str, age_days: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split(":")[0],
        text="some text",
        page=None,
        position=0,
        source=source,
        created_at=datetime.now(timezone.utc) - timedelta(days=age_days),
    )


def test_freshness_high_for_recent_doc():
    scorer = ConfidenceScorer(ConfidenceConfig(half_life_days=100.0))
    fresh = _chunk("a:0", "internal://docs/a", age_days=0)
    stale = _chunk("b:0", "internal://docs/b", age_days=1000)
    result = RetrievalResult(
        chunks=[
            ScoredChunk(chunk=fresh, retrieval_score=0.9),
            ScoredChunk(chunk=stale, retrieval_score=0.9),
        ],
        bm25_ranks={"a:0": 0, "b:0": 0},
        dense_ranks={"a:0": 0, "b:0": 0},
    )
    scored, _ = scorer.score(result)
    fresh_conf = next(s.confidence for s in scored if s.chunk.chunk_id == "a:0")
    stale_conf = next(s.confidence for s in scored if s.chunk.chunk_id == "b:0")
    assert fresh_conf > stale_conf


def test_source_quality_lookup():
    config = ConfidenceConfig(
        source_quality={"https://gov.": 0.95, "internal://": 0.8},
        default_source_quality=0.4,
    )
    scorer = ConfidenceScorer(config)
    assert config.quality_for("https://gov.uk/page") == 0.95
    assert config.quality_for("internal://doc.txt") == 0.8
    assert config.quality_for("random://blog.com") == 0.4


def test_consistency_higher_when_both_retrievers_agree():
    scorer = ConfidenceScorer()
    c1 = _chunk("agree:0", "internal://x")
    c2 = _chunk("only_dense:0", "internal://x")
    result = RetrievalResult(
        chunks=[
            ScoredChunk(chunk=c1, retrieval_score=0.9),
            ScoredChunk(chunk=c2, retrieval_score=0.9),
        ],
        bm25_ranks={"agree:0": 0},
        dense_ranks={"agree:0": 0, "only_dense:0": 0},
    )
    scored, _ = scorer.score(result)
    agree_conf = next(s.confidence for s in scored if s.chunk.chunk_id == "agree:0")
    one_only_conf = next(s.confidence for s in scored if s.chunk.chunk_id == "only_dense:0")
    assert agree_conf > one_only_conf


def test_aggregate_is_mean_of_top_n():
    scorer = ConfidenceScorer(ConfidenceConfig(aggregate_top_n=2))
    chunks = [
        ScoredChunk(chunk=_chunk(f"c{i}:0", "internal://x"), retrieval_score=1.0)
        for i in range(5)
    ]
    result = RetrievalResult(
        chunks=chunks,
        bm25_ranks={f"c{i}:0": i for i in range(5)},
        dense_ranks={f"c{i}:0": i for i in range(5)},
    )
    scored, aggregate = scorer.score(result)
    expected = sum(s.confidence for s in scored[:2]) / 2
    assert abs(aggregate - expected) < 1e-9


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ConfidenceScorer(ConfidenceConfig(w_freshness=0.5, w_source=0.5, w_consistency=0.5))


def test_confidence_in_unit_interval():
    scorer = ConfidenceScorer()
    chunks = [
        ScoredChunk(chunk=_chunk(f"c{i}:0", "weird://source"), retrieval_score=0.5)
        for i in range(3)
    ]
    result = RetrievalResult(
        chunks=chunks,
        bm25_ranks={},
        dense_ranks={f"c0:0": 0},
    )
    scored, agg = scorer.score(result)
    for s in scored:
        assert 0.0 <= s.confidence <= 1.0
    assert 0.0 <= agg <= 1.0
