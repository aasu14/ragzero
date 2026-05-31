"""Markdown / header-aware chunker.

Splits a document on Markdown ATX headers (#, ##, ###...). Each chunk is one
section, optionally truncated to `max_chars` to keep individual chunks small.
The header text is preserved at the top of each chunk so it shows up in
retrieved context.
"""
from __future__ import annotations

import re

from ..interfaces import Chunk, Document


# Match a Markdown header at the start of a line. Capture level + title.
_HEADER_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$", re.MULTILINE)


class MarkdownHeaderChunker:
    def __init__(self, max_chars: int = 1500, min_header_level: int = 1) -> None:
        if max_chars < 100:
            raise ValueError("max_chars must be >= 100")
        if not 1 <= min_header_level <= 6:
            raise ValueError("min_header_level must be in 1..6")
        self.max_chars = max_chars
        self.min_level = min_header_level

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content
        if not text.strip():
            return []
        # Find all header positions
        headers = [
            (m.start(), len(m.group(1)), m.group(2).strip())
            for m in _HEADER_RE.finditer(text)
        ]
        # Drop headers below min_level (e.g. min_level=2 ignores ### and deeper
        # as section boundaries — they ride inside the parent ## chunk).
        headers = [(pos, lvl, title) for (pos, lvl, title) in headers if lvl <= self.min_level + 5]
        # If no headers at all, fall back to a single chunk truncated to cap.
        if not headers:
            block = text.strip()[: self.max_chars]
            return [_make_chunk(doc, block, 0, doc.version, header=None)]

        # If the doc starts before the first header, capture that preamble.
        chunks: list[Chunk] = []
        position = 0
        if headers[0][0] > 0:
            preamble = text[: headers[0][0]].strip()
            if preamble:
                chunks.append(_make_chunk(doc, preamble[: self.max_chars], position, doc.version, header=None))
                position += 1

        # Each header spans from its start to the next header start (or EOF).
        bounds = [h[0] for h in headers] + [len(text)]
        for i, (start, lvl, title) in enumerate(headers):
            end = bounds[i + 1]
            section = text[start:end].strip()
            if not section:
                continue
            if len(section) > self.max_chars:
                section = section[: self.max_chars]
            chunks.append(_make_chunk(doc, section, position, doc.version, header=title))
            position += 1
        return chunks


def _make_chunk(doc: Document, text: str, position: int, version: int, header: str | None) -> Chunk:
    meta = {"doc_type": doc.doc_type, "doc_version": version, "chunker": "markdown_header"}
    if header:
        meta["section_header"] = header
    return Chunk(
        chunk_id=f"{doc.doc_id}:{position}:v{version}",
        doc_id=doc.doc_id,
        text=text,
        page=None,
        position=position,
        source=doc.source,
        created_at=doc.created_at,
        metadata=meta,
    )
