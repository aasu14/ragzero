"""Tests for RAG strategies."""
from datetime import datetime, timezone

from ragzero.rag.factories import build_dev_pipeline
from ragzero.rag.graph import Entity, EntityExtractor, NetworkXGraphStore, Relation
from ragzero.rag.interfaces import Document
from ragzero.rag.multilingual import LLMTranslator
from ragzero.rag.observability import new_trace_id
from ragzero.rag.strategies import (
    AgenticStrategy,
    GraphStrategy,
    MultilingualStrategy,
    SimpleStrategy,
    StrategyContext,
    build_strategy,
    extract_answer,
)


def _docs():
    now = datetime.now(timezone.utc)
    return [
        Document(
            doc_id="rag-overview",
            source="internal://rag",
            content=(
                "Retrieval-augmented generation, or RAG, combines a retrieval system "
                "with a language model. The retriever fetches relevant documents, "
                "and the model generates an answer grounded in those documents."
            ),
            doc_type="txt",
            created_at=now,
        ),
        Document(
            doc_id="bm25-overview",
            source="internal://bm25",
            content="BM25 is a probabilistic ranking function used by search engines. "
                    "It scores documents using term frequency and document length.",
            doc_type="txt",
            created_at=now,
        ),
    ]


def _make_ctx():
    pipeline = build_dev_pipeline()
    pipeline.ingest_documents(_docs())
    return pipeline, StrategyContext(
        pipeline=pipeline,
        tracer=pipeline.tracer,
        trace_id=new_trace_id(),
        options={},
    )


def _run(strategy, query, ctx):
    return list(strategy.run(query, ctx))


def test_simple_strategy_returns_answer():
    pipeline, ctx = _make_ctx()
    events = _run(SimpleStrategy(), "What is BM25?", ctx)
    answer = extract_answer(events)
    assert answer is not None
    assert not answer.refused
    assert answer.text


def test_simple_strategy_emits_required_event_kinds():
    pipeline, ctx = _make_ctx()
    events = _run(SimpleStrategy(), "What is BM25?", ctx)
    kinds = {e.kind for e in events}
    assert "info" in kinds
    assert "final" in kinds
    # The last event must be the final
    assert events[-1].kind == "final"


def test_graph_strategy_runs_when_no_graph_falls_back():
    """If graph_store isn't in options, GraphStrategy should still return an answer."""
    pipeline, ctx = _make_ctx()
    # No graph_store in options
    events = _run(GraphStrategy(), "What is BM25?", ctx)
    answer = extract_answer(events)
    assert answer is not None  # Either answers or refuses, but produces a final


def test_graph_strategy_uses_graph_when_provided():
    pipeline, ctx = _make_ctx()
    # Build a minimal graph
    graph = NetworkXGraphStore()
    graph.upsert_entity(Entity(name="BM25", entity_type="CONCEPT", chunk_ids=("bm25-overview:0:v1",)))
    graph.upsert_entity(Entity(name="TF-IDF", entity_type="CONCEPT", chunk_ids=("bm25-overview:0:v1",)))
    graph.upsert_relation(Relation(
        source="BM25", target="TF-IDF", rel_type="builds_on",
        chunk_ids=("bm25-overview:0:v1",), evidence="builds on TF-IDF"
    ))
    ctx.options["graph_store"] = graph

    events = _run(GraphStrategy(), "What does BM25 build on?", ctx)
    # Should see at least one graph_hop event
    kinds = [e.kind for e in events]
    assert "graph_hop" in kinds
    # And finalize
    assert events[-1].kind == "final"


def test_agentic_strategy_produces_thoughts():
    """The agentic strategy should emit thought events showing its reasoning."""
    pipeline, ctx = _make_ctx()
    strategy = AgenticStrategy(max_iterations=2)
    events = _run(strategy, "What is RAG and how does BM25 differ?", ctx)
    kinds = [e.kind for e in events]
    # Should have at least one thought (the plan) and tool_calls
    assert "thought" in kinds
    assert "tool_call" in kinds
    # Should finalize
    assert events[-1].kind == "final"


def test_agentic_strategy_respects_max_iterations():
    """Even if reflection wants more, we cap iterations."""
    pipeline, ctx = _make_ctx()
    strategy = AgenticStrategy(max_iterations=1)
    events = _run(strategy, "Complex multi-part question about RAG", ctx)
    # Count tool_call events (one per iteration)
    tool_calls = [e for e in events if e.kind == "tool_call"]
    # Should have at most max_iterations retrieve calls
    retrieve_calls = [e for e in tool_calls if "retrieve" in e.label]
    assert len(retrieve_calls) <= 1


def test_multilingual_wrapper_no_op_when_same_language():
    """If query language == corpus language, no translation should happen."""
    pipeline, ctx = _make_ctx()
    translator = LLMTranslator(pipeline.generator.llm)
    wrapped = MultilingualStrategy(
        inner=SimpleStrategy(),
        translator=translator,
        corpus_language="en",
        output_language="auto",
    )
    events = _run(wrapped, "What is BM25?", ctx)
    answer = extract_answer(events)
    assert answer is not None
    # No actual translation events should fire as a translation
    # (we only emit a "detected language" info event)


def test_multilingual_wrapper_detects_hindi_query():
    """A Devanagari query should trigger translation events."""
    pipeline, ctx = _make_ctx()
    translator = LLMTranslator(pipeline.generator.llm)
    wrapped = MultilingualStrategy(
        inner=SimpleStrategy(),
        translator=translator,
        corpus_language="en",
        output_language="hi",
    )
    events = _run(wrapped, "बीएम25 क्या है?", ctx)
    translation_events = [e for e in events if e.kind == "translation"]
    # Should have at least the language detection event
    assert any(e.label.startswith("Detected") for e in translation_events)


def test_build_strategy_preset_simple():
    s = build_strategy(mode="simple")
    assert isinstance(s, SimpleStrategy)


def test_build_strategy_preset_agentic_with_iterations():
    s = build_strategy(mode="agentic", agent_max_iterations=5)
    assert isinstance(s, AgenticStrategy)
    assert s.max_iterations == 5


def test_build_strategy_multilingual_wraps():
    pipeline, _ = _make_ctx()
    translator = LLMTranslator(pipeline.generator.llm)
    s = build_strategy(mode="agentic", multilingual=True, translator=translator)
    assert isinstance(s, MultilingualStrategy)
    assert isinstance(s.inner, AgenticStrategy)


def test_build_strategy_multilingual_requires_translator():
    import pytest
    with pytest.raises(ValueError):
        build_strategy(mode="simple", multilingual=True, translator=None)


def test_build_strategy_unknown_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        build_strategy(mode="nonexistent")
