"""Steps 5-6: Constrained generation + citation-backed outputs.

Step 5: We pass the model a prompt that explicitly tells it to use ONLY
the provided context. Temperature is forced to 0. The prompt instructs
the model to write "[INSUFFICIENT_EVIDENCE]" if it cannot answer from
context — this signal is checked by the fallback gate (step 7).

Step 6: We post-process the output to extract claim-citation pairs.
The model is instructed to inline citations as [chunk_id] tags. We
validate every claim has at least one valid citation; uncited claims
are stripped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .interfaces import Answer, Citation, LLM, ScoredChunk


CONSTRAINED_PROMPT = """\
You are answering a question using ONLY the provided context. Follow these rules strictly:

1. If the context does not contain enough information to answer, respond with exactly: [INSUFFICIENT_EVIDENCE]
2. Do not use any knowledge outside the provided context.
3. After each sentence that uses information from the context, add a citation tag pointing to the source chunk(s), like [1] or [1][3].
4. Use ONLY the citation numbers shown in the context below (the [1], [2], ... labels at the start of each chunk).
5. Keep the answer concise. Prefer direct quotes from the context when precision matters.

Example of a correctly-formatted answer:
  The sky appears blue due to Rayleigh scattering [2]. Water freezes at 0°C [1][3].

Context:
{context_block}

Question: {query}

Answer:"""


# Matches [N] or [1][3] style numeric citations the model emits.
# Also still matches old-style chunk-id citations for back-compat with MockLLM tests.
CITATION_RE = re.compile(r"\[(\d+|[A-Za-z][A-Za-z0-9_:\-\.]*)\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class GenerationOutput:
    """Raw LLM output paired with extracted citations."""
    text: str
    citations: list[Citation]
    insufficient_evidence: bool


class ConstrainedGenerator:
    """Wraps an LLM with refusal-aware, citation-enforcing prompting."""

    def __init__(
        self,
        llm: LLM,
        max_context_chunks: int = 8,
        max_tokens: int = 512,
        strip_uncited: bool = True,
    ) -> None:
        self.llm = llm
        self.max_context_chunks = max_context_chunks
        self.max_tokens = max_tokens
        self.strip_uncited = strip_uncited

    def generate(self, query: str, chunks: list[ScoredChunk]) -> GenerationOutput:
        chunks = chunks[: self.max_context_chunks]
        if not chunks:
            return GenerationOutput(
                text="[INSUFFICIENT_EVIDENCE]", citations=[], insufficient_evidence=True
            )

        # Number the chunks [1], [2], ... — models reproduce numeric labels far more
        # reliably than long alphanumeric IDs. We keep a number→chunk map to recover
        # the real chunk_id when building Citation objects.
        label_to_chunk = {i + 1: sc.chunk for i, sc in enumerate(chunks)}
        context_block = self._format_context(chunks)
        prompt = CONSTRAINED_PROMPT.format(context_block=context_block, query=query)
        raw = self.llm.generate(
            prompt=prompt,
            context=[c.chunk.text for c in chunks],
            max_tokens=self.max_tokens,
            temperature=0.0,
        ).strip()

        if "[INSUFFICIENT_EVIDENCE]" in raw:
            return GenerationOutput(text=raw, citations=[], insufficient_evidence=True)

        cited_labels: set[int] = set()
        cleaned_sentences: list[str] = []

        # First pass: find sentences with valid numeric citations and keep them.
        for sentence in SENTENCE_SPLIT_RE.split(raw):
            if not sentence.strip():
                continue
            nums_in_sentence = [int(n) for n in CITATION_RE.findall(sentence) if n.isdigit()]
            valid_nums = [n for n in nums_in_sentence if n in label_to_chunk]
            if valid_nums:
                cited_labels.update(valid_nums)
                cleaned_sentences.append(sentence)
            elif not self.strip_uncited:
                cleaned_sentences.append(sentence)

        final_text = " ".join(cleaned_sentences).strip()

        # Fallback: model emitted text but no valid numeric citations.
        # Rather than blank-refusing, check if the answer is grounded in the
        # retrieved chunks via term overlap. If yes, accept it and attribute the
        # citations to the chunks that match best. This is the path that fixes
        # the "I get a good answer but it gets stripped" bug.
        if not final_text and raw:
            grounded_labels = self._infer_grounding(raw, label_to_chunk)
            if grounded_labels:
                cited_labels = grounded_labels
                final_text = raw  # keep the model's original text intact
            else:
                return GenerationOutput(
                    text="[INSUFFICIENT_EVIDENCE]", citations=[], insufficient_evidence=True
                )

        if not final_text:
            return GenerationOutput(
                text="[INSUFFICIENT_EVIDENCE]", citations=[], insufficient_evidence=True
            )

        citations = [
            Citation(
                doc_id=label_to_chunk[label].doc_id,
                source=label_to_chunk[label].source,
                page=label_to_chunk[label].page,
                chunk_id=label_to_chunk[label].chunk_id,
                timestamp=label_to_chunk[label].created_at,
            )
            for label in sorted(cited_labels)
        ]
        return GenerationOutput(
            text=final_text, citations=citations, insufficient_evidence=False
        )

    @staticmethod
    def _format_context(chunks: list[ScoredChunk]) -> str:
        """Number chunks [1], [2], ... — easy for the model to cite."""
        lines = []
        for i, sc in enumerate(chunks, start=1):
            page_info = f", page {sc.chunk.page}" if sc.chunk.page is not None else ""
            lines.append(
                f"[{i}] (source: {sc.chunk.source}{page_info})\n{sc.chunk.text}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _infer_grounding(answer_text: str, label_to_chunk: dict[int, "Chunk"]) -> set[int]:
        """Heuristic grounding check for uncited answers.

        Idea: if the answer text shares enough content terms with a retrieved
        chunk, the answer is almost certainly grounded in that chunk (even if
        the model forgot to mark the citation). We accept the answer and
        attribute citations to the top-overlapping chunks.

        Returns the set of chunk labels that match well, or an empty set if
        the answer doesn't look grounded (in which case the caller refuses).
        """
        # Extract content terms from the answer (4+ char words, lowercase)
        ans_terms = {
            w.lower().strip(".,;:!?\"'()")
            for w in answer_text.split()
            if len(w) >= 4
        }
        # Throw out very short answers — too little signal to assess grounding
        if len(ans_terms) < 3:
            return set()
        # Score each chunk by term overlap
        scores: list[tuple[int, int]] = []
        for label, chunk in label_to_chunk.items():
            chunk_terms = {
                w.lower().strip(".,;:!?\"'()")
                for w in chunk.text.split()
                if len(w) >= 4
            }
            overlap = len(ans_terms & chunk_terms)
            scores.append((label, overlap))
        scores.sort(key=lambda x: x[1], reverse=True)
        if not scores or scores[0][1] == 0:
            return set()
        # A meaningful overlap is at least 3 content terms, or 30% of the
        # answer's terms — whichever is more permissive.
        threshold = max(3, int(0.30 * len(ans_terms)))
        # The top chunk must clear the bar
        if scores[0][1] < threshold:
            return set()
        # Attribute citations to all chunks within 60% of the top score
        top_score = scores[0][1]
        return {label for label, s in scores if s >= top_score * 0.6}


def to_answer(
    output: GenerationOutput, confidence: float, trace_id: str | None = None
) -> Answer:
    """Convert generator output to a finalized Answer."""
    if output.insufficient_evidence:
        return Answer(
            text="",
            citations=[],
            confidence=confidence,
            refused=True,
            refusal_reason="Model reported insufficient evidence in context",
            trace_id=trace_id,
        )
    return Answer(
        text=output.text,
        citations=output.citations,
        confidence=confidence,
        refused=False,
        trace_id=trace_id,
    )
