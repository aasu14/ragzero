"""Abstract interfaces for pluggable RAG backends.

Every backend in `rag.backends.*` implements one of these protocols.
The pipeline orchestrator depends only on these interfaces, never on
concrete implementations — this is what makes the system pluggable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


# ---------- Domain models ----------

@dataclass(frozen=True)
class Document:
    """A normalized document ready for chunking."""
    doc_id: str
    source: str           # filepath or URL
    content: str
    doc_type: str         # pdf, docx, email, etc.
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with provenance."""
    chunk_id: str
    doc_id: str
    text: str
    page: int | None       # for citations
    position: int          # ordinal within doc
    source: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with retrieval and confidence scores."""
    chunk: Chunk
    retrieval_score: float    # raw score from retriever
    confidence: float = 0.0   # 0.0–1.0 trust score (set by step 4)


@dataclass(frozen=True)
class Citation:
    """A traceable reference attached to a generated claim."""
    doc_id: str
    source: str
    page: int | None
    chunk_id: str
    timestamp: datetime


@dataclass(frozen=True)
class Answer:
    """Final pipeline output. Either a grounded answer or a refusal."""
    text: str
    citations: list[Citation]
    confidence: float
    refused: bool = False
    refusal_reason: str | None = None
    trace_id: str | None = None


# ---------- Backend interfaces ----------

class Embedder(ABC):
    """Produces dense vector representations of text."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text. Must be deterministic."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimension."""


class VectorStore(ABC):
    """ANN-backed vector index. Concrete impls: FAISS, Qdrant, in-memory."""

    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    @abstractmethod
    def search(self, query_vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        """Return (chunk, similarity_score) pairs. Score in [0, 1], higher = closer."""

    @abstractmethod
    def size(self) -> int: ...


class KeywordIndex(ABC):
    """BM25-style sparse retriever."""

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]: ...


class LLM(ABC):
    """Generation backend. Must support a `constrain` mode that uses only provided context."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: list[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate constrained to `context`. Implementation must instruct
        the model to refuse if the answer isn't supported by context."""


class Cache(ABC):
    """Generic key-value cache with TTL."""

    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


# ---------- Observability ----------

class TraceEvent(Protocol):
    """Anything that can be logged in a trace."""
    trace_id: str
    stage: str
    timestamp: datetime
    data: dict[str, Any]


class Tracer(ABC):
    """Records structured events for every pipeline stage."""

    @abstractmethod
    def emit(self, trace_id: str, stage: str, data: dict[str, Any]) -> None: ...

    @abstractmethod
    def events_for(self, trace_id: str) -> list[dict[str, Any]]: ...
