"""Per-row chunker: one chunk per Document, no splitting.

Used by structured-file ingest (CSV / JSON / JSONL), where each row of the
source file is already turned into its own Document. The document's content
field is the searchable text for that row (built from the user-selected
content_columns); we don't split it further.
"""
from __future__ import annotations

from ..interfaces import Chunk, Document


class PerRowChunker:
    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content.strip()
        if not text:
            return []
        return [Chunk(
            chunk_id=f"{doc.doc_id}:0:v{doc.version}",
            doc_id=doc.doc_id,
            text=text,
            page=None,
            position=0,
            source=doc.source,
            created_at=doc.created_at,
            metadata={**doc.metadata, "chunker": "per_row"},
        )]
