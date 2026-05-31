"""Tests for Steps 5-7: constrained generation, citations, and fallback gate."""
from datetime import datetime, timezone

from ragzero.rag.backends.llms import MockLLM
from ragzero.rag.fallback import FallbackConfig, FallbackGate, refusal_answer
from ragzero.rag.generation import ConstrainedGenerator, GenerationOutput, to_answer
from ragzero.rag.interfaces import Chunk, ScoredChunk


def _scored(chunk_id: str, text: str, confidence: float = 0.8) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=chunk_id.split(":")[0],
            text=text,
            page=1,
            position=0,
            source=f"internal://{chunk_id}",
            created_at=datetime.now(timezone.utc),
        ),
        retrieval_score=0.9,
        confidence=confidence,
    )


def test_generator_returns_insufficient_when_no_chunks():
    gen = ConstrainedGenerator(MockLLM())
    out = gen.generate("anything", [])
    assert out.insufficient_evidence
    assert out.citations == []


def test_generator_emits_insufficient_evidence_when_context_unrelated():
    gen = ConstrainedGenerator(MockLLM())
    chunks = [_scored("a:0", "Penguins are flightless birds.")]
    out = gen.generate("What is quantum chromodynamics?", chunks)
    assert out.insufficient_evidence


def test_generator_strips_uncited_claims():
    gen = ConstrainedGenerator(MockLLM(), strip_uncited=True)
    chunks = [_scored("a:0", "BM25 is a keyword ranking function.")]
    out = gen.generate("What is BM25?", chunks)
    assert not out.insufficient_evidence
    assert out.citations
    # Every claim in the output references a valid chunk id
    for citation in out.citations:
        assert citation.chunk_id == "a:0"


def test_generator_does_not_invent_chunk_ids():
    """If the model emits a chunk ID we didn't give it, we drop the claim."""
    class FakeLLM(MockLLM):
        def generate(self, prompt, context, max_tokens=512, temperature=0.0):
            return "Fabricated claim. [made_up_chunk_id]"
    gen = ConstrainedGenerator(FakeLLM())
    chunks = [_scored("real:0", "real content about RAG.")]
    out = gen.generate("Tell me about RAG", chunks)
    # The fabricated citation should be rejected, leaving nothing → insufficient
    assert out.insufficient_evidence or out.text == ""


def test_gate_refuses_below_confidence_threshold():
    gate = FallbackGate(FallbackConfig(min_aggregate_confidence=0.7))
    decision = gate.evaluate(
        scored_chunks=[_scored("a:0", "text", confidence=0.5)],
        aggregate_confidence=0.5,
        output=GenerationOutput(text="answer", citations=[], insufficient_evidence=False),
    )
    assert decision.refuse
    assert "confidence" in (decision.reason or "").lower()


def test_gate_refuses_with_no_chunks():
    gate = FallbackGate(FallbackConfig(min_chunks=1))
    decision = gate.evaluate(
        scored_chunks=[],
        aggregate_confidence=0.9,
        output=GenerationOutput(text="answer", citations=[], insufficient_evidence=False),
    )
    assert decision.refuse


def test_gate_refuses_when_insufficient_evidence():
    gate = FallbackGate()
    decision = gate.evaluate(
        scored_chunks=[_scored("a:0", "text", confidence=0.9)],
        aggregate_confidence=0.9,
        output=GenerationOutput(text="", citations=[], insufficient_evidence=True),
    )
    assert decision.refuse


def test_gate_requires_min_citations():
    gate = FallbackGate(FallbackConfig(require_min_citations=2, min_aggregate_confidence=0.0))
    from ragzero.rag.interfaces import Citation
    output = GenerationOutput(
        text="answer",
        citations=[Citation("d", "s", None, "a:0", datetime.now(timezone.utc))],
        insufficient_evidence=False,
    )
    decision = gate.evaluate(
        scored_chunks=[_scored("a:0", "text", confidence=0.9)],
        aggregate_confidence=0.9,
        output=output,
    )
    assert decision.refuse
    assert "citation" in (decision.reason or "").lower()


def test_gate_passes_with_good_inputs():
    gate = FallbackGate()
    from ragzero.rag.interfaces import Citation
    output = GenerationOutput(
        text="answer",
        citations=[Citation("d", "s", None, "a:0", datetime.now(timezone.utc))],
        insufficient_evidence=False,
    )
    decision = gate.evaluate(
        scored_chunks=[_scored("a:0", "text", confidence=0.9)],
        aggregate_confidence=0.9,
        output=output,
    )
    assert not decision.refuse


def test_refusal_answer_shape():
    a = refusal_answer("nope", 0.3, trace_id="t1")
    assert a.refused
    assert a.text == ""
    assert a.citations == []
    assert a.trace_id == "t1"


def test_to_answer_passes_through_citations():
    from ragzero.rag.interfaces import Citation
    output = GenerationOutput(
        text="hello",
        citations=[Citation("d", "s", 1, "a:0", datetime.now(timezone.utc))],
        insufficient_evidence=False,
    )
    a = to_answer(output, confidence=0.8, trace_id="t1")
    assert not a.refused
    assert a.text == "hello"
    assert len(a.citations) == 1
    assert a.confidence == 0.8
