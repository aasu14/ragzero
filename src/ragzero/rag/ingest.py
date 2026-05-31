"""Step 1: Ingestion and normalization.

Responsibilities:
- Parse heterogeneous inputs (txt, pdf, docx, eml) into Document
- Deduplicate by content hash
- Chunk into overlapping windows
- Extract metadata
- Track versions

This module is intentionally I/O-light — file parsing is delegated to
adapter functions that can be swapped per format.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .interfaces import Chunk, Document


# ---------- Parsers ----------
# Each parser maps raw file bytes to plain text + format-specific metadata.
# Real impls would use pypdf, python-docx, mailparser, etc.

Parser = Callable[[bytes], tuple[str, dict]]


def parse_text(raw: bytes) -> tuple[str, dict]:
    return raw.decode("utf-8", errors="replace"), {}


def parse_pdf_stub(raw: bytes) -> tuple[str, dict]:
    # In production: use pypdf. Stub returns decoded bytes for testability.
    text = raw.decode("utf-8", errors="replace")
    pages = text.count("\f") + 1  # form-feed as page break heuristic
    return text, {"pages": pages}


PARSERS: dict[str, Parser] = {
    ".txt": parse_text,
    ".md": parse_text,
    ".pdf": parse_pdf_stub,
}


def detect_type(path: Path) -> str:
    return path.suffix.lstrip(".") or "txt"


# ---------- Ingestion ----------

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace; keep paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class Ingestor:
    """Loads documents from disk, deduplicates, and tracks versions.

    Versioning model: if a new file produces the same content hash as an
    existing doc_id, version stays the same. If the doc_id matches but
    hash differs, version is incremented. This lets you re-ingest updated
    files without orphaning citations.
    """

    def __init__(self) -> None:
        self._seen_hashes: dict[str, str] = {}        # hash -> doc_id
        self._versions: dict[str, int] = {}           # doc_id -> latest version

    def ingest_path(self, path: Path) -> Document | None:
        """Load a single file. Returns None if duplicate of existing doc."""
        ext = path.suffix.lower()
        parser = PARSERS.get(ext, parse_text)
        raw = path.read_bytes()
        text, fmt_meta = parser(raw)
        text = _normalize_whitespace(text)
        if not text:
            return None

        content_hash = _content_hash(text)
        if content_hash in self._seen_hashes:
            # Exact duplicate — skip
            return None

        doc_id = self._stable_doc_id(path)
        version = self._versions.get(doc_id, 0) + 1
        self._versions[doc_id] = version
        self._seen_hashes[content_hash] = doc_id

        return Document(
            doc_id=doc_id,
            source=str(path),
            content=text,
            doc_type=detect_type(path),
            created_at=datetime.now(timezone.utc),
            metadata={"content_hash": content_hash, **fmt_meta},
            version=version,
        )

    def ingest_directory(self, root: Path) -> Iterable[Document]:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in PARSERS:
                doc = self.ingest_path(path)
                if doc is not None:
                    yield doc

    @staticmethod
    def _stable_doc_id(path: Path) -> str:
        """Path-based ID so re-ingesting the same file keeps its identity."""
        return hashlib.md5(str(path.resolve()).encode()).hexdigest()[:12]


# ---------- Chunking ----------

class Chunker:
    """Sliding-window chunker with sentence-aware boundaries.

    `chunk_size` and `overlap` are measured in characters; for production
    you'd use a tokenizer. The character heuristic is sufficient for tests
    and keeps this dependency-free.
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content
        if len(text) <= self.chunk_size:
            return [self._make_chunk(doc, text, 0, 1)]

        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        position = 0
        for start in range(0, len(text), step):
            end = min(start + self.chunk_size, len(text))
            # Try to end on a sentence boundary
            window = text[start:end]
            if end < len(text):
                last_period = window.rfind(". ")
                if last_period > self.chunk_size // 2:
                    window = window[: last_period + 1]
            chunks.append(self._make_chunk(doc, window.strip(), position, doc.version))
            position += 1
            if end == len(text):
                break
        return chunks

    @staticmethod
    def _make_chunk(doc: Document, text: str, position: int, version: int) -> Chunk:
        # Approximate page: pages are sequential, ~3000 chars/page heuristic.
        page = (position * 800) // 3000 + 1 if doc.doc_type == "pdf" else None
        return Chunk(
            chunk_id=f"{doc.doc_id}:{position}:v{version}",
            doc_id=doc.doc_id,
            text=text,
            page=page,
            position=position,
            source=doc.source,
            created_at=doc.created_at,
            metadata={"doc_type": doc.doc_type, "doc_version": version},
        )
