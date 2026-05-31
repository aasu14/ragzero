"""Paragraph-based chunker.

Splits on blank-line paragraph boundaries and groups N paragraphs per chunk.
Hard char cap prevents one giant paragraph from producing an unbounded chunk.
"""
from __future__ import annotations

import re

from ..interfaces import Chunk, Document


_PARA_SPLIT = re.compile(r"\n\s*\n")


class ParagraphChunker:
    def __init__(self, paragraphs_per_chunk: int = 2, max_chars: int = 1500) -> None:
        if paragraphs_per_chunk < 1:
            raise ValueError("paragraphs_per_chunk must be >= 1")
        if max_chars < 100:
            raise ValueError("max_chars must be >= 100")
        self.n = paragraphs_per_chunk
        self.max_chars = max_chars

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content.strip()
        if not text:
            return []
        paras = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
        if not paras:
            return []

        chunks: list[Chunk] = []
        position = 0
        i = 0
        while i < len(paras):
            window = paras[i : i + self.n]
            block = "\n\n".join(window)
            # If the block exceeds the hard cap, slice down at char boundary
            # rather than trying anything clever — predictable + cheap.
            if len(block) > self.max_chars:
                block = block[: self.max_chars]
            chunks.append(_make_chunk(doc, block, position, doc.version))
            position += 1
            i += self.n
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
        metadata={"doc_type": doc.doc_type, "doc_version": version, "chunker": "paragraph"},
    )
