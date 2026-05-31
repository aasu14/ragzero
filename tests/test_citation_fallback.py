"""Tests for citation handling — the 'always refuses with insufficient evidence' bug.

The original code stripped any sentence without a `[chunk_id]` tag, which made
real LLMs trigger false refusals when they returned a good answer but skipped
the citation tag. These tests pin down the fixed behavior:

1. Numeric citations [1], [2] are the new contract — easier for models to use
2. Old alphanumeric citations still work (back-compat with MockLLM)
3. If the model forgets to cite but the answer overlaps with retrieved chunks,
   we attribute citations via grounding inference instead of false-refusing
4. Off-topic ungrounded answers still get refused
"""
from datetime import datetime, timezone

from ragzero.rag.generation import ConstrainedGenerator
from ragzero.rag.interfaces import Chunk, ScoredChunk


def _chunk(label: str, text: str, source: str | None = None) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=label,
        doc_id=label.split(":")[0],
        text=text,
        page=None,
        position=0,
        source=source or f"internal://{label}",
        created_at=datetime.now(timezone.utc),
    )
    return ScoredChunk(chunk=chunk, retrieval_score=0.9, confidence=0.9)


class _FakeLLM:
    """LLM stub that always returns a fixed string."""
    def __init__(self, response):
        self.response = response
        self.last_prompt = None
    def generate(self, prompt, context, max_tokens=512, temperature=0.0):
        self.last_prompt = prompt
        return self.response


def test_numeric_citations_accepted():
    """The new prompt asks for [1], [2] — those should be parsed and mapped."""
    llm = _FakeLLM("The sky is blue [1]. Water boils at 100°C [2].")
    gen = ConstrainedGenerator(llm)
    chunks = [
        _chunk("text-abc:0:v1", "The sky is blue due to scattering."),
        _chunk("text-xyz:0:v1", "Water boils at 100 degrees Celsius."),
    ]
    out = gen.generate("Tell me about water and sky", chunks)
    assert not out.insufficient_evidence
    assert "sky is blue" in out.text
    # Two distinct citations should resolve back to the right chunk_ids
    chunk_ids = {c.chunk_id for c in out.citations}
    assert chunk_ids == {"text-abc:0:v1", "text-xyz:0:v1"}


def test_uncited_grounded_answer_is_kept():
    """The real bug: model returns a good answer but no citation tags.

    Old behavior: refusal with "insufficient evidence" because every sentence
    got stripped for lacking a citation.
    New behavior: detect grounding via term overlap and accept the answer,
    attributing citations to the matching chunk.
    """
    llm = _FakeLLM("The sky appears blue because of Rayleigh scattering of sunlight.")
    gen = ConstrainedGenerator(llm)
    chunks = [
        _chunk("doc1:0:v1", "Rayleigh scattering makes the sky appear blue. "
                            "Shorter wavelengths scatter more than longer ones."),
    ]
    out = gen.generate("Why is the sky blue?", chunks)
    assert not out.insufficient_evidence, (
        f"Expected grounded answer to be accepted, got refusal. text={out.text!r}"
    )
    assert "Rayleigh" in out.text
    # We inferred the citation
    assert len(out.citations) == 1
    assert out.citations[0].chunk_id == "doc1:0:v1"


def test_uncited_offtopic_answer_is_refused():
    """If the model answers from outside the corpus, term overlap fails → refusal."""
    llm = _FakeLLM("Quantum chromodynamics describes the strong interaction between quarks.")
    gen = ConstrainedGenerator(llm)
    chunks = [
        _chunk("doc1:0:v1", "The sky is blue. Water boils at 100 Celsius."),
    ]
    out = gen.generate("What is QCD?", chunks)
    # The answer has zero overlap with the chunk → should refuse
    assert out.insufficient_evidence


def test_explicit_insufficient_evidence_signal_honored():
    """If the model itself says [INSUFFICIENT_EVIDENCE], we trust it."""
    llm = _FakeLLM("[INSUFFICIENT_EVIDENCE]")
    gen = ConstrainedGenerator(llm)
    chunks = [_chunk("doc1:0:v1", "Some unrelated text about cooking.")]
    out = gen.generate("What is the speed of light?", chunks)
    assert out.insufficient_evidence


def test_partial_citation_mix_kept_with_inferred_extras():
    """If the model cites some sentences but not others, kept-cited sentences
    are preserved; the rest are dropped (existing behavior — we don't add
    inferred citations on top of partial coverage, that would be silently
    misleading)."""
    llm = _FakeLLM("The sky is blue [1]. Water freezes at zero Celsius. Trees produce oxygen.")
    gen = ConstrainedGenerator(llm)
    chunks = [_chunk("doc1:0:v1", "The sky is blue.")]
    out = gen.generate("Tell me facts", chunks)
    assert not out.insufficient_evidence
    # Only the cited sentence survives
    assert "sky is blue" in out.text
    assert "freezes" not in out.text
    assert "Trees" not in out.text


def test_invented_numeric_citation_rejected():
    """If the model cites [99] but only chunks [1] and [2] exist, [99] is invalid."""
    llm = _FakeLLM("Some claim [99].")
    gen = ConstrainedGenerator(llm)
    chunks = [_chunk("doc1:0:v1", "Real content about real things.")]
    out = gen.generate("question", chunks)
    # [99] is invalid → sentence stripped → grounding check on bare "Some claim"
    # which has zero overlap → refusal
    assert out.insufficient_evidence


def test_grounding_inference_requires_minimum_terms():
    """Very short answers ("Yes." "Blue.") shouldn't be auto-accepted via grounding."""
    llm = _FakeLLM("Blue.")
    gen = ConstrainedGenerator(llm)
    chunks = [_chunk("doc1:0:v1", "The sky appears blue due to Rayleigh scattering.")]
    out = gen.generate("Color?", chunks)
    # Too few terms in the answer to infer grounding confidently → refusal
    assert out.insufficient_evidence


def test_context_block_uses_numeric_labels():
    """The prompt sent to the model should label chunks [1], [2], not chunk_ids."""
    llm = _FakeLLM("[1] is correct.")
    gen = ConstrainedGenerator(llm)
    chunks = [
        _chunk("doc1:0:v1", "First chunk content."),
        _chunk("doc2:0:v1", "Second chunk content."),
    ]
    gen.generate("anything", chunks)
    prompt = llm.last_prompt
    # The context block must use [1], [2] not the raw chunk IDs
    assert "[1]" in prompt
    assert "[2]" in prompt
    # The raw chunk_id should NOT appear as a citation label
    # (it may still appear in the source URL — that's fine)
    assert prompt.count("doc1:0:v1") <= 1  # at most in the source line


def test_grounding_attributes_to_best_matching_chunk():
    """When multiple chunks exist and the answer matches one strongly, citations
    point to that chunk, not all of them."""
    llm = _FakeLLM("Photosynthesis converts sunlight into chemical energy via chlorophyll.")
    gen = ConstrainedGenerator(llm)
    chunks = [
        _chunk("about-photosynthesis:0:v1",
               "Photosynthesis is the process by which plants convert sunlight "
               "into chemical energy using chlorophyll molecules."),
        _chunk("about-cooking:0:v1", "Boil water before adding pasta."),
    ]
    out = gen.generate("How does photosynthesis work?", chunks)
    assert not out.insufficient_evidence
    chunk_ids = {c.chunk_id for c in out.citations}
    # The photosynthesis chunk should be cited; the cooking chunk should not
    assert "about-photosynthesis:0:v1" in chunk_ids
    assert "about-cooking:0:v1" not in chunk_ids
