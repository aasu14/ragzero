"""Step 9: Multi-layer caching.

Three distinct caches with different invalidation rules:

- query_cache: full (query -> Answer). TTL = short (minutes-hours) because
  underlying data can change. Invalidated wholesale on corpus updates.
- retrieval_cache: (query -> RetrievalResult). Same TTL as query_cache.
- embedding_cache: (text_hash -> vector). TTL = effectively infinite if
  the embedder model is pinned, because the same text always maps to
  the same vector.

Each cache is a thin wrapper around the abstract Cache interface, so
swapping in-memory for Redis is a one-line config change.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from .interfaces import Answer, Cache


def _hash_key(*parts: str) -> str:
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


class QueryCache:
    """Full query -> Answer cache."""

    NAMESPACE = "q:"

    def __init__(self, backend: Cache, ttl_seconds: int = 3600) -> None:
        self.backend = backend
        self.ttl = ttl_seconds

    def get(self, query: str) -> Answer | None:
        cached = self.backend.get(self.NAMESPACE + _hash_key(query))
        return cached if isinstance(cached, Answer) else None

    def set(self, query: str, answer: Answer) -> None:
        self.backend.set(self.NAMESPACE + _hash_key(query), answer, self.ttl)


class EmbeddingCache:
    """Text -> vector cache. Key includes the embedder model id so that
    swapping models doesn't reuse stale vectors."""

    NAMESPACE = "e:"

    def __init__(self, backend: Cache, model_id: str) -> None:
        self.backend = backend
        self.model_id = model_id

    def get(self, text: str) -> list[float] | None:
        cached = self.backend.get(self.NAMESPACE + _hash_key(self.model_id, text))
        return cached if isinstance(cached, list) else None

    def set(self, text: str, vector: list[float]) -> None:
        # No TTL — embeddings are deterministic for a pinned model.
        self.backend.set(self.NAMESPACE + _hash_key(self.model_id, text), vector, None)


class CachedEmbedder:
    """Wraps any Embedder with an EmbeddingCache."""

    def __init__(self, inner, cache: EmbeddingCache) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [self.cache.get(t) for t in texts]
        missing_indices = [i for i, v in enumerate(results) if v is None]
        if missing_indices:
            new_vectors = self.inner.embed([texts[i] for i in missing_indices])
            for i, v in zip(missing_indices, new_vectors):
                self.cache.set(texts[i], v)
                results[i] = v
        return [r for r in results if r is not None]  # type: ignore[misc]


def serialize_answer(a: Answer) -> str:
    """JSON-encode an Answer (for Redis-backed caches)."""
    def default(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        if hasattr(o, "isoformat"):
            return o.isoformat()
        raise TypeError(f"Not serializable: {type(o)}")
    return json.dumps(asdict(a), default=default)
