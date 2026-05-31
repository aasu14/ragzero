"""Chunker registry — pluggable strategies for splitting Documents into Chunks.

Each strategy implements the same interface as `ingest.Chunker.chunk(doc) -> list[Chunk]`,
so it can be dropped into `pipeline.chunker` without changes elsewhere.

The CHUNKERS dict maps a string id (used in YAML config and the UI) to a
constructor callable `(settings: dict) -> ChunkerLike`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..interfaces import Chunk, Document
from ..ingest import Chunker as FixedSizeChunker  # current sliding-window impl


class ChunkerLike(Protocol):
    """Anything with a .chunk(doc) -> list[Chunk] method."""
    def chunk(self, doc: Document) -> list[Chunk]: ...


@dataclass
class ChunkerSpec:
    """Public metadata for the UI."""
    id: str
    label: str
    description: str
    fields: list[dict]  # field specs in the same shape as ProviderField


# ---------- Strategies ----------

def _fixed_size(settings: dict[str, Any]) -> ChunkerLike:
    return FixedSizeChunker(
        chunk_size=int(settings.get("chunk_size", 800) or 800),
        overlap=int(settings.get("chunk_overlap", 100) or 100),
    )


def _sentence(settings: dict[str, Any]) -> ChunkerLike:
    from .sentence import SentenceChunker
    return SentenceChunker(
        sentences_per_chunk=int(settings.get("sentences_per_chunk", 5) or 5),
        overlap_sentences=int(settings.get("overlap_sentences", 1) or 1),
    )


def _paragraph(settings: dict[str, Any]) -> ChunkerLike:
    from .paragraph import ParagraphChunker
    return ParagraphChunker(
        paragraphs_per_chunk=int(settings.get("paragraphs_per_chunk", 2) or 2),
        max_chars=int(settings.get("max_chars", 1500) or 1500),
    )


def _markdown(settings: dict[str, Any]) -> ChunkerLike:
    from .markdown import MarkdownHeaderChunker
    return MarkdownHeaderChunker(
        max_chars=int(settings.get("max_chars", 1500) or 1500),
        min_header_level=int(settings.get("min_header_level", 1) or 1),
    )


def _token(settings: dict[str, Any]) -> ChunkerLike:
    from .token import TokenChunker
    return TokenChunker(
        tokens_per_chunk=int(settings.get("tokens_per_chunk", 400) or 400),
        overlap_tokens=int(settings.get("overlap_tokens", 50) or 50),
        model=str(settings.get("model", "cl100k_base") or "cl100k_base"),
    )


def _recursive(settings: dict[str, Any]) -> ChunkerLike:
    from .recursive import RecursiveChunker
    return RecursiveChunker(
        chunk_size=int(settings.get("chunk_size", 800) or 800),
        overlap=int(settings.get("chunk_overlap", 100) or 100),
    )


def _per_row(settings: dict[str, Any]) -> ChunkerLike:
    from .per_row import PerRowChunker
    return PerRowChunker()


CHUNKERS: dict[str, Callable[[dict[str, Any]], ChunkerLike]] = {
    "fixed_size": _fixed_size,
    "sentence": _sentence,
    "paragraph": _paragraph,
    "markdown_header": _markdown,
    "token": _token,
    "recursive": _recursive,
    "per_row": _per_row,
}


def chunker_catalog() -> list[dict]:
    """Public catalog for the UI — labels, descriptions, and per-chunker
    settings so the chunker picker can render type-correct inputs."""
    return [
        {
            "id": "fixed_size",
            "label": "Fixed-size sliding window",
            "description": "Char-based windows with a configurable overlap. Sentence-boundary aware. Default and most reliable.",
            "fields": [
                {"key": "chunk_size", "label": "Chunk size (chars)", "type": "int", "default": 800},
                {"key": "chunk_overlap", "label": "Overlap (chars)", "type": "int", "default": 100},
            ],
        },
        {
            "id": "sentence",
            "label": "Sentence-based",
            "description": "Split on sentence boundaries; group N sentences per chunk. Good for prose / FAQs.",
            "fields": [
                {"key": "sentences_per_chunk", "label": "Sentences per chunk", "type": "int", "default": 5},
                {"key": "overlap_sentences", "label": "Overlap (sentences)", "type": "int", "default": 1},
            ],
        },
        {
            "id": "paragraph",
            "label": "Paragraph-based",
            "description": "One or N paragraphs per chunk. Falls back to a hard char cap to avoid mega-chunks.",
            "fields": [
                {"key": "paragraphs_per_chunk", "label": "Paragraphs per chunk", "type": "int", "default": 2},
                {"key": "max_chars", "label": "Hard cap (chars)", "type": "int", "default": 1500},
            ],
        },
        {
            "id": "markdown_header",
            "label": "Markdown / Header-aware",
            "description": "Splits on Markdown headers (#, ##, ###). Each chunk is one section. Works on .md and structured .txt.",
            "fields": [
                {"key": "max_chars", "label": "Max chars per chunk", "type": "int", "default": 1500},
                {"key": "min_header_level", "label": "Min header level (1=#, 2=##, ...)", "type": "int", "default": 1},
            ],
        },
        {
            "id": "token",
            "label": "Token-based (tiktoken)",
            "description": "True token counting via OpenAI's tiktoken. Best when targeting a token budget for a specific model. Requires `pip install tiktoken`.",
            "fields": [
                {"key": "tokens_per_chunk", "label": "Tokens per chunk", "type": "int", "default": 400},
                {"key": "overlap_tokens", "label": "Overlap (tokens)", "type": "int", "default": 50},
                {"key": "model", "label": "Encoding (e.g. cl100k_base, o200k_base)", "type": "text", "default": "cl100k_base"},
            ],
        },
        {
            "id": "recursive",
            "label": "Recursive (paragraph → sentence → char)",
            "description": "Tries paragraph splits first; if a piece is still too big, splits on sentences, then on chars. LangChain-style.",
            "fields": [
                {"key": "chunk_size", "label": "Max chunk size (chars)", "type": "int", "default": 800},
                {"key": "chunk_overlap", "label": "Overlap (chars)", "type": "int", "default": 100},
            ],
        },
        {
            "id": "per_row",
            "label": "Per-row (structured data)",
            "description": "One chunk per Document, no splitting. Used by structured-file ingest where each row is already one indexable unit.",
            "fields": [],
        },
    ]


def build_chunker(strategy: str, settings: dict[str, Any]) -> ChunkerLike:
    """Resolve a chunker by id. Raises ValueError for unknown ids."""
    if strategy not in CHUNKERS:
        raise ValueError(
            f"Unknown chunker strategy: {strategy!r}. Available: {sorted(CHUNKERS)}"
        )
    return CHUNKERS[strategy](settings or {})
