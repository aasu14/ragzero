"""Session manager.

Each user gets a session_id (a cookie) with:
- credentials (LLM + embedder, encrypted at rest in memory only)
- a built pipeline (lazy — created when credentials change or query runs)
- ingestion state (chunks indexed, sources loaded)
- recent traces
- per-session graph store (for Graph RAG)
- RAG mode + strategy options

Sessions are stored in-memory and expire after `ttl_seconds` of inactivity.
For multi-process deployments swap this for Redis.
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .ingest import Ingestor
from .pipeline import RAGPipeline
from .schema import MetadataSchema, default_schema


@dataclass
class SessionState:
    session_id: str
    created_at: float
    last_seen: float
    # Provider configs
    llm_provider: str = "mock"
    llm_settings: dict[str, Any] = field(default_factory=dict)
    embedder_provider: str = "hash"
    embedder_settings: dict[str, Any] = field(default_factory=dict)
    vector_store_provider: str = "in_memory"
    vector_store_settings: dict[str, Any] = field(default_factory=dict)
    # Pipeline tunings (loaded from YAML + UI overrides)
    pipeline_overrides: dict[str, Any] = field(default_factory=dict)
    # Built pipeline (None until first valid config; rebuilt when config changes)
    pipeline: RAGPipeline | None = None
    pipeline_dirty: bool = True
    # Per-session ingestor so dedup hashes don't leak across sessions and can
    # be reset by `Clear all` (replaced with a fresh Ingestor).
    ingestor: Ingestor = field(default_factory=Ingestor)
    # Ingested sources (display only — actual data lives in the pipeline)
    sources: list[dict[str, Any]] = field(default_factory=list)
    # Recent queries for display
    history: list[dict[str, Any]] = field(default_factory=list)
    # Strategy & RAG mode
    rag_mode: str = "simple"                   # simple, graph, agentic, multilingual
    multilingual_enabled: bool = False
    output_language: str = "auto"
    agent_max_iterations: int = 3
    graph_max_depth: int = 2
    # Graph store for Graph RAG
    graph_store: Any | None = None
    graph_built: bool = False
    # Metadata schema (built-in fields + user-defined custom fields). The UI
    # uses this to render the per-ingest metadata form and the query filter bar.
    metadata_schema: MetadataSchema = field(default_factory=default_schema)
    # Active chunker (set via /api/session/chunker). Empty dict = use defaults.
    chunker_strategy: str = "fixed_size"
    chunker_settings: dict[str, Any] = field(default_factory=dict)
    # In-flight structured-upload previews awaiting commit. Keyed by preview_id.
    # Each entry: {"filename": str, "bytes": bytes, "preview": StructuredPreview}
    structured_previews: dict[str, dict[str, Any]] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_seen = time.time()


class SessionManager:
    """Thread-safe, TTL-bounded session store."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl = ttl_seconds
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def create(self) -> SessionState:
        sid = secrets.token_urlsafe(24)
        now = time.time()
        state = SessionState(session_id=sid, created_at=now, last_seen=now)
        with self._lock:
            self._sessions[sid] = state
        return state

    def get(self, sid: str) -> SessionState | None:
        with self._lock:
            state = self._sessions.get(sid)
            if state is None:
                return None
            if time.time() - state.last_seen > self._ttl:
                self._sessions.pop(sid, None)
                return None
            state.touch()
            return state

    def get_or_create(self, sid: str | None) -> SessionState:
        if sid:
            state = self.get(sid)
            if state is not None:
                return state
        return self.create()

    def delete(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)

    def gc(self) -> int:
        """Remove expired sessions. Returns count removed."""
        cutoff = time.time() - self._ttl
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.last_seen < cutoff]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)
