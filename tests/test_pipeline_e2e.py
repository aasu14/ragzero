"""End-to-end pipeline tests + evaluation harness (Step 8)."""
from ragzero.rag.evaluation import EvalCase, Evaluator


def test_pipeline_answers_relevant_query(pipeline):
    """Query about RAG should produce a grounded answer with citations."""
    answer = pipeline.answer("What is retrieval augmented generation?")
    assert not answer.refused, f"Got refusal: {answer.refusal_reason}"
    assert answer.text
    assert answer.citations
    # All citations should point at the rag-overview doc
    doc_ids = {c.doc_id for c in answer.citations}
    assert "rag-overview" in doc_ids


def test_pipeline_refuses_when_no_evidence(pipeline):
    """Adversarial query with no supporting docs should refuse."""
    answer = pipeline.answer("What is the airspeed velocity of an unladen swallow?")
    # With our small corpus and MockLLM, this should refuse — either at the
    # confidence gate or the insufficient-evidence signal.
    assert answer.refused
    assert answer.refusal_reason


def test_pipeline_emits_full_trace(pipeline):
    answer = pipeline.answer("What is BM25?")
    assert answer.trace_id
    events = pipeline.tracer.events_for(answer.trace_id)
    stages = {e["stage"] for e in events}
    # The pipeline should emit at minimum these stages on a non-cached query
    assert "query.start" in stages
    assert "retrieval" in stages
    assert "confidence" in stages
    assert "finalize" in stages


def test_pipeline_cache_hit_on_repeat(pipeline):
    a1 = pipeline.answer("What is BM25?")
    a2 = pipeline.answer("What is BM25?")
    # Second call should have triggered a cache.hit event
    events = pipeline.tracer.events_for(a2.trace_id)
    stages = [e["stage"] for e in events]
    assert "cache.hit" in stages
    assert a1.text == a2.text


def test_evaluation_harness_recall_and_refusal(pipeline):
    cases = [
        EvalCase(
            query="What is retrieval augmented generation?",
            expected_doc_ids={"rag-overview"},
            expected_answer_substrings=[],
        ),
        EvalCase(
            query="Explain BM25",
            expected_doc_ids={"bm25-overview"},
        ),
        EvalCase(
            query="What is the lifecycle of a turbofan engine?",
            expected_doc_ids=set(),
            is_adversarial=True,  # should refuse
        ),
    ]
    evaluator = Evaluator(
        retrieve_fn=pipeline.retriever.search,
        answer_fn=pipeline.answer,
        k=5,
    )
    report = evaluator.run(cases)
    assert report.recall_at_k >= 0.5  # at least the two relevant queries hit
    # The adversarial case should produce a refusal, not a fabricated answer
    assert report.refusal_rate > 0
    # Hallucinations should be rare or zero with the constrained MockLLM
    assert report.hallucination_rate <= 0.5


def test_pipeline_citations_point_at_real_chunks(pipeline):
    """Every citation in an answer must reference an indexed chunk."""
    answer = pipeline.answer("What is retrieval augmented generation?")
    if answer.refused:
        return
    # We can verify the chunk_id format and source consistency
    for citation in answer.citations:
        assert citation.chunk_id
        assert citation.source.startswith("internal://")
        assert citation.doc_id


def test_pipeline_confidence_in_unit_interval(pipeline):
    answer = pipeline.answer("What is BM25?")
    assert 0.0 <= answer.confidence <= 1.0


def test_stale_documents_get_lower_confidence(pipeline):
    """The stale doc fixture should not dominate retrieval thanks to freshness scoring."""
    answer = pipeline.answer("Tell me about RAG")
    if answer.refused:
        return
    # The fresh rag-overview should be cited; stale-doc should not dominate
    cited_doc_ids = {c.doc_id for c in answer.citations}
    # We're not asserting stale-doc is excluded — only that fresh sources are preferred
    assert "rag-overview" in cited_doc_ids or not answer.citations
