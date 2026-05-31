"""Sentence-based chunker.

Splits on sentence boundaries (using a simple regex; production code would
use spaCy / nltk / pysbd). Groups N sentences per chunk with optional overlap.
"""
from __future__ import annotations

import re

from ..interfaces import Chunk, Document


# Match end-of-sentence punctuation followed by whitespace. Avoids splitting
# on common abbreviations by requiring a capital letter after the boundary.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d])")


class SentenceChunker:
    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> None:
        if sentences_per_chunk < 1:
            raise ValueError("sentences_per_chunk must be >= 1")
        if overlap_sentences >= sentences_per_chunk:
            raise ValueError("overlap_sentences must be < sentences_per_chunk")
        self.n = sentences_per_chunk
        self.overlap = overlap_sentences

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content.strip()
        if not text:
            return []
        sentences = _SENT_BOUNDARY.split(text)
        # Collapse very-short fragments into the previous sentence — avoids
        # one-word "chunks" from punctuation noise.
        merged: list[str] = []
        for s in sentences:
            if merged and len(s) < 20:
                merged[-1] += " " + s
            else:
                merged.append(s)

        chunks: list[Chunk] = []
        step = max(1, self.n - self.overlap)
        position = 0
        for start in range(0, len(merged), step):
            window = merged[start : start + self.n]
            if not window:
                break
            text_block = " ".join(window).strip()
            if text_block:
                chunks.append(_make_chunk(doc, text_block, position, doc.version))
                position += 1
            if start + self.n >= len(merged):
                break
        return chunks


def _make_chunk(doc: Document, text: str, position: int, version: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}:{position}:v{version}",
        doc_id=doc.doc_id,
        text=text,
        page=None,
        position=position,
        source=doc.source,
        created_at=doc.created_at,
        metadata={"doc_type": doc.doc_type, "doc_version": version, "chunker": "sentence"},
    )
