"""Recursive chunker (LangChain-style).

Tries successively finer separators until each piece fits within `chunk_size`:
    paragraph -> sentence -> word -> char
This usually preserves the most semantic structure available without
exceeding the size budget.
"""
from __future__ import annotations

import re

from ..interfaces import Chunk, Document


_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        if chunk_size < 100:
            raise ValueError("chunk_size must be >= 100")
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content.strip()
        if not text:
            return []
        raw_pieces = self._split(text, _SEPARATORS)
        # Merge adjacent pieces while the running size stays under the cap.
        merged: list[str] = []
        current: list[str] = []
        current_len = 0
        for piece in raw_pieces:
            piece = piece.strip()
            if not piece:
                continue
            if current_len + len(piece) + 1 > self.chunk_size and current:
                merged.append(" ".join(current).strip())
                # Carry the tail of the previous chunk forward as overlap.
                if self.overlap > 0:
                    tail = merged[-1][-self.overlap:]
                    current = [tail]
                    current_len = len(tail)
                else:
                    current = []
                    current_len = 0
            current.append(piece)
            current_len += len(piece) + 1
        if current:
            merged.append(" ".join(current).strip())

        chunks: list[Chunk] = []
        for position, block in enumerate(merged):
            if not block:
                continue
            chunks.append(_make_chunk(doc, block, position, doc.version))
        return chunks

    def _split(self, text: str, seps: list[str]) -> list[str]:
        if not seps or not text:
            return [text]
        sep = seps[0]
        if sep == "":
            # Char-level split as last resort
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        parts = text.split(sep)
        # If any single part is still too big, recurse with the next separator
        out: list[str] = []
        for p in parts:
            if len(p) <= self.chunk_size:
                out.append(p)
            else:
                out.extend(self._split(p, seps[1:]))
        return out


def _make_chunk(doc: Document, text: str, position: int, version: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}:{position}:v{version}",
        doc_id=doc.doc_id,
        text=text,
        page=None,
        position=position,
        source=doc.source,
        created_at=doc.created_at,
        metadata={"doc_type": doc.doc_type, "doc_version": version, "chunker": "recursive"},
    )
