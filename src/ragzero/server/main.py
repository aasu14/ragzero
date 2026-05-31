"""FastAPI server.

Endpoints:
- GET  /api/providers              List available LLM and embedder providers
- GET  /api/session                Current session info (creates one if missing)
- POST /api/session/llm            Set LLM provider + credentials
- POST /api/session/embedder       Set embedder provider + credentials
- POST /api/session/test           Test the configured connections
- POST /api/session/pipeline       Update pipeline tuning (chunk size, thresholds, etc.)
- GET  /api/session/config         Get current config snapshot
- POST /api/ingest/upload          Upload one or more files
- POST /api/ingest/path            Ingest from a server-side filesystem path
- POST /api/ingest/url             Ingest from a URL
- POST /api/ingest/text            Ingest pasted text
- GET  /api/sources                List ingested sources for this session
- DELETE /api/sources              Clear all ingested data
- POST /api/query                  Ask a question
- GET  /api/trace/{trace_id}       Get full trace for a query
- GET  /api/health                 Health check

Published Space (public ask-only assistant):
- GET  /api/space                  Admin: current space config + readiness
- POST /api/space                  Admin: save display/access settings
- POST /api/space/publish          Admin: snapshot session + publish
- POST /api/space/unpublish        Admin: take the assistant offline
- GET  /api/space/hosting          Admin: LAN IP + hosting hints
- GET  /public/meta                Public: display config (no secrets)
- POST /public/query               Public: ask the published assistant

When RAGZERO_ADMIN_TOKEN is set, /api/* and the console require an X-Admin-Token
header; the public assistant (/a/<slug>, /public/*) stays open.

Static UI is served from /ui/dist if present.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import (
    Cookie, Depends, FastAPI, File, Form, HTTPException, Request, Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ragzero.rag.config import EMBEDDERS, LLMS, load_pipeline
from ragzero.rag.graph import EntityExtractor, NetworkXGraphStore
from ragzero.rag.ingest import Chunker, Ingestor
from ragzero.rag.interfaces import Document
from ragzero.rag.multilingual import LANGUAGES, LLMTranslator
from ragzero.rag.jobs import Job, JobManager  # noqa: E402
from ragzero.rag.observability import InMemoryTracer  # noqa: E402
from ragzero.rag.pipeline import RAGPipeline  # noqa: E402
from ragzero.rag.providers import (  # noqa: E402
    EMBEDDER_PROVIDERS, LLM_PROVIDERS, provider_catalog,
)
from ragzero.rag.sessions import SessionManager, SessionState  # noqa: E402
from ragzero.rag.strategies import (  # noqa: E402
    PRESETS, build_strategy, extract_answer,
)
from ragzero.server.spaces import SpaceConfig, SpaceManager  # noqa: E402

logger = logging.getLogger("rag.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


# ---------------- App setup ----------------

_PKG_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(
    os.environ.get("RAGZERO_CONFIG", str(_PKG_ROOT / "config" / "dev.yaml"))
)
UI_DIST = Path(
    os.environ.get("RAGZERO_UI_DIST", str(_PKG_ROOT / "ui" / "dist"))
)
SESSION_COOKIE = "rag_session"
SESSION_TTL = int(os.environ.get("RAG_SESSION_TTL", "3600"))

sessions = SessionManager(ttl_seconds=SESSION_TTL)
jobs = JobManager(retention_seconds=600)

# Admin token (gates the admin console + /api/*). Unset = open (local dev).
ADMIN_TOKEN = os.environ.get("RAGZERO_ADMIN_TOKEN") or None

# Single published Space (the public ask-only assistant) + its live runtime.
space_manager = SpaceManager()
_space_runtime: SessionState | None = None
_space_lock = threading.Lock()
# Per-session Ingestor instances live on SessionState — see sessions.py.
# A module-level singleton would leak content-hash dedup across sessions
# and survive `Clear all`, blocking re-ingestion of the same file.

app = FastAPI(title="RAG Console", version="0.1.0")

# CORS for the dev React server (vite at :5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def admin_guard(request: Request, call_next):
    """When an admin token is configured, gate the admin API + console. The
    public assistant (/public/*, /a/*) and static assets stay open so visitors
    never need the token."""
    if ADMIN_TOKEN:
        path = request.url.path
        # Protect the admin API (except health) and the admin SPA shell.
        is_admin_api = path.startswith("/api/") and path != "/api/health"
        is_admin_shell = path == "/" or path == "/index.html"
        if is_admin_api or is_admin_shell:
            token = request.headers.get("X-Admin-Token") or request.cookies.get("rag_admin")
            if token != ADMIN_TOKEN:
                if is_admin_shell:
                    # Let the SPA load so it can show its own token prompt.
                    return await call_next(request)
                return JSONResponse({"detail": "admin authentication required"}, status_code=401)
    return await call_next(request)


# ---------------- Helpers ----------------

def get_session(
    response: Response,
    rag_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> SessionState:
    state = sessions.get_or_create(rag_session)
    # Refresh cookie on every request
    response.set_cookie(
        key=SESSION_COOKIE,
        value=state.session_id,
        max_age=SESSION_TTL,
        httponly=True,
        samesite="lax",
    )
    return state


def _mask_secrets(settings: dict[str, Any]) -> dict[str, Any]:
    """Replace credential-like values with a placeholder before sending to the UI."""
    masked = {}
    for k, v in settings.items():
        if any(s in k.lower() for s in ("key", "secret", "token", "password")):
            masked[k] = "••••" if v else ""
        else:
            masked[k] = v
    return masked


def _build_pipeline_for(state: SessionState) -> RAGPipeline:
    """(Re)build the per-session pipeline using the current credentials and overrides."""
    overrides = {
        "backends": {
            "llm": state.llm_provider,
            "llm_settings": state.llm_settings,
            "embedder": state.embedder_provider,
            "embedder_settings": state.embedder_settings,
            "vector_store": state.vector_store_provider,
            "vector_store_settings": state.vector_store_settings,
        },
        # Pipeline-level chunker selection (separate from the YAML chunk_size).
        "pipeline": {
            "chunker_strategy": state.chunker_strategy,
            "chunker_settings": state.chunker_settings,
        },
    }
    if state.pipeline_overrides:
        # Merge UI-side tuning overrides (chunk_size, thresholds, etc.) into the
        # top-level YAML config, not the backends section.
        for section, vals in state.pipeline_overrides.items():
            overrides.setdefault(section, {}).update(vals)
    return load_pipeline(CONFIG_PATH, overrides=overrides)


def get_pipeline(state: SessionState):
    if state.pipeline is None or state.pipeline_dirty:
        state.pipeline = _build_pipeline_for(state)
        state.pipeline_dirty = False
        # Re-ingest existing sources into the new pipeline
        if state.sources:
            docs = [_source_to_doc(s) for s in state.sources]
            state.pipeline.ingest_documents(docs)
    return state.pipeline


def _source_to_doc(src: dict[str, Any]) -> Document:
    return Document(
        doc_id=src["doc_id"],
        source=src["source"],
        content=src["content"],
        doc_type=src["doc_type"],
        created_at=datetime.fromisoformat(src["created_at"]),
        metadata=src.get("metadata", {}),
        version=src.get("version", 1),
    )


# ---------------- Request models ----------------

class ProviderConfig(BaseModel):
    provider: str
    settings: dict[str, Any] = Field(default_factory=dict)


class PipelineTuning(BaseModel):
    pipeline: dict[str, Any] | None = None
    confidence: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None


class TextIngestion(BaseModel):
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class UrlIngestion(BaseModel):
    url: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PathIngestion(BaseModel):
    path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FilterClauseIn(BaseModel):
    field: str
    op: str = "eq"
    value: Any = None


class QueryRequest(BaseModel):
    query: str
    filters: list[FilterClauseIn] = Field(default_factory=list)


# ---------------- Routes ----------------

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "sessions": sessions.count()}


@app.get("/api/providers")
def get_providers() -> dict:
    """Catalog of providers + which Python packages are actually installed.

    Uses importlib.util.find_spec instead of __import__ — we only need to know
    whether the package exists, not load it. Real imports here triggered native
    library init (FAISS / numpy / chromadb all load libomp.dylib) which
    intermittently crashes uvicorn on macOS with a duplicate-libomp error.
    """
    import importlib.util as _iutil
    catalog = provider_catalog()
    installed: dict[str, bool] = {}
    for category in catalog.values():
        for p in category:
            pkg = p["requires_package"]
            if pkg is None:
                p["installed"] = True
                continue
            if pkg not in installed:
                try:
                    installed[pkg] = _iutil.find_spec(pkg) is not None
                except (ImportError, ValueError):
                    # find_spec raises if a parent package is missing while
                    # probing a dotted name (e.g. azure.search.documents).
                    installed[pkg] = False
            p["installed"] = installed[pkg]
    return catalog


@app.get("/api/session")
def get_session_info(state: SessionState = Depends(get_session)) -> dict:
    # Report current index size (chunks in the active vector store) when the
    # pipeline is built. Cheap call — most stores cache the count.
    n_chunks = None
    if state.pipeline is not None:
        store = getattr(state.pipeline.retriever, "vector_store", None)
        if store is not None and hasattr(store, "size"):
            try:
                n_chunks = store.size()
            except Exception:
                n_chunks = None
    return {
        "session_id": state.session_id,
        "llm": {
            "provider": state.llm_provider,
            "settings": _mask_secrets(state.llm_settings),
        },
        "embedder": {
            "provider": state.embedder_provider,
            "settings": _mask_secrets(state.embedder_settings),
        },
        "vector_store": {
            "provider": state.vector_store_provider,
            "settings": _mask_secrets(state.vector_store_settings),
        },
        "pipeline_overrides": state.pipeline_overrides,
        "n_sources": len(state.sources),
        "n_chunks": n_chunks,
        "pipeline_built": state.pipeline is not None,
        "history_count": len(state.history),
    }


@app.post("/api/session/llm")
def set_llm(cfg: ProviderConfig, state: SessionState = Depends(get_session)) -> dict:
    if cfg.provider not in LLM_PROVIDERS:
        raise HTTPException(400, f"Unknown LLM provider: {cfg.provider}")
    spec = LLM_PROVIDERS[cfg.provider]
    for field_def in spec.fields:
        if field_def.required and not cfg.settings.get(field_def.key):
            raise HTTPException(400, f"Missing required field: {field_def.key}")
    state.llm_provider = cfg.provider
    state.llm_settings = dict(cfg.settings)
    state.pipeline_dirty = True
    return {"ok": True, "provider": cfg.provider}


@app.post("/api/session/embedder")
def set_embedder(cfg: ProviderConfig, state: SessionState = Depends(get_session)) -> dict:
    if cfg.provider not in EMBEDDER_PROVIDERS:
        raise HTTPException(400, f"Unknown embedder provider: {cfg.provider}")
    spec = EMBEDDER_PROVIDERS[cfg.provider]
    for field_def in spec.fields:
        if field_def.required and not cfg.settings.get(field_def.key):
            raise HTTPException(400, f"Missing required field: {field_def.key}")
    # No-op if nothing actually changed — avoids forcing an expensive re-embed
    # on a redundant Save click.
    if (
        state.embedder_provider == cfg.provider
        and state.embedder_settings == dict(cfg.settings)
    ):
        return {"ok": True, "provider": cfg.provider, "changed": False}
    state.embedder_provider = cfg.provider
    state.embedder_settings = dict(cfg.settings)
    state.pipeline_dirty = True
    # Drop the cached pipeline so the next query rebuilds and re-embeds
    # state.sources with the new embedder. Sources themselves are preserved.
    state.pipeline = None
    return {"ok": True, "provider": cfg.provider, "changed": True}


@app.post("/api/session/vector_store")
def set_vector_store(cfg: ProviderConfig, state: SessionState = Depends(get_session)) -> dict:
    from ragzero.rag.providers import VECTOR_STORE_PROVIDERS
    if cfg.provider not in VECTOR_STORE_PROVIDERS:
        raise HTTPException(400, f"Unknown vector store provider: {cfg.provider}")
    spec = VECTOR_STORE_PROVIDERS[cfg.provider]
    for field_def in spec.fields:
        if field_def.required and not cfg.settings.get(field_def.key):
            raise HTTPException(400, f"Missing required field: {field_def.key}")
    if (
        state.vector_store_provider == cfg.provider
        and state.vector_store_settings == dict(cfg.settings)
    ):
        return {"ok": True, "provider": cfg.provider, "changed": False}
    state.vector_store_provider = cfg.provider
    state.vector_store_settings = dict(cfg.settings)
    state.pipeline_dirty = True
    # Vectors in the old store are abandoned, but the source docs persist.
    # The next query rebuilds the pipeline and re-ingests state.sources into
    # the new store automatically.
    state.pipeline = None
    return {"ok": True, "provider": cfg.provider, "changed": True}


@app.post("/api/session/test")
def test_connections(state: SessionState = Depends(get_session)) -> dict:
    """Try instantiating the configured providers and embedding a probe string."""
    results: dict[str, Any] = {"llm": None, "embedder": None}
    # Test embedder
    try:
        ctor = EMBEDDERS[state.embedder_provider]
        emb = ctor(state.embedder_settings)
        probe = emb.embed(["connection test"])
        results["embedder"] = {"ok": True, "dim": len(probe[0]) if probe else 0}
    except Exception as e:
        results["embedder"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    # Test LLM
    try:
        ctor = LLMS[state.llm_provider]
        llm = ctor(state.llm_settings)
        # A minimal generation call to verify auth + connectivity
        out = llm.generate(
            prompt="Respond with the single word: ok",
            context=[],
            max_tokens=5,
            temperature=0.0,
        )
        results["llm"] = {"ok": True, "response": out[:80]}
    except Exception as e:
        results["llm"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return results


@app.post("/api/session/pipeline")
def set_pipeline_tuning(tuning: PipelineTuning, state: SessionState = Depends(get_session)) -> dict:
    overrides: dict[str, Any] = {}
    if tuning.pipeline:
        overrides["pipeline"] = tuning.pipeline
    if tuning.confidence:
        overrides["confidence"] = tuning.confidence
    if tuning.fallback:
        overrides["fallback"] = tuning.fallback
    state.pipeline_overrides = overrides
    state.pipeline_dirty = True
    return {"ok": True, "overrides": overrides}


@app.get("/api/session/config")
def get_full_config(state: SessionState = Depends(get_session)) -> dict:
    """Return the resolved config after merging YAML + session overrides."""
    from ragzero.rag.config import _load_yaml, _deep_merge
    base = _load_yaml(CONFIG_PATH)
    merged = _deep_merge(base, state.pipeline_overrides)
    return merged


# ---- Metadata schema ----

@app.get("/api/schema")
def get_schema(state: SessionState = Depends(get_session)) -> dict:
    """Return the active schema + the active vector store's capability map.

    The UI uses `capabilities` to decide which flag checkboxes to enable.
    """
    from ragzero.rag.schema import schema_to_dict, capabilities_for
    return {
        "schema": schema_to_dict(state.metadata_schema),
        "capabilities": capabilities_for(state.vector_store_provider),
        "vector_store_provider": state.vector_store_provider,
    }


class SchemaUpdate(BaseModel):
    fields: list[dict[str, Any]]


@app.post("/api/schema")
def set_schema(req: SchemaUpdate, state: SessionState = Depends(get_session)) -> dict:
    """Replace the session's schema. Built-in fields are preserved automatically
    (the UI doesn't show them as editable), but if the client omits them we
    re-add the defaults so they're always present for filtering."""
    from ragzero.rag.schema import schema_from_dict, default_schema
    incoming = schema_from_dict({"fields": req.fields})
    # Make sure all builtins are present; UI sends only customs in some flows.
    builtins = default_schema().fields
    custom_by_name = {f.name: f for f in incoming.fields if f.origin == "custom"}
    final = list(builtins) + list(custom_by_name.values())
    incoming.fields = final
    state.metadata_schema = incoming
    return {"ok": True, "n_fields": len(final), "n_custom": len(custom_by_name)}


# ---- Chunker ----

@app.get("/api/chunkers")
def list_chunkers() -> dict:
    from ragzero.rag.chunkers import chunker_catalog
    return {"chunkers": chunker_catalog()}


class ChunkerConfig(BaseModel):
    strategy: str
    settings: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/session/chunker")
def set_chunker(cfg: ChunkerConfig, state: SessionState = Depends(get_session)) -> dict:
    from ragzero.rag.chunkers import CHUNKERS
    if cfg.strategy not in CHUNKERS:
        raise HTTPException(400, f"Unknown chunker strategy: {cfg.strategy}")
    if (state.chunker_strategy == cfg.strategy
            and state.chunker_settings == cfg.settings):
        return {"ok": True, "strategy": cfg.strategy, "changed": False}
    state.chunker_strategy = cfg.strategy
    state.chunker_settings = dict(cfg.settings)
    state.pipeline_dirty = True
    state.pipeline = None  # next ingest/query rebuilds with the new chunker
    return {"ok": True, "strategy": cfg.strategy, "changed": True}


@app.delete("/api/session")
def reset_session(
    response: Response,
    state: SessionState = Depends(get_session),
) -> dict:
    sessions.delete(state.session_id)
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


# ---- Ingestion ----

def _apply_metadata(
    state: SessionState, doc: Document, user_metadata: dict | None = None,
) -> Document:
    """Layer provenance + user-supplied metadata onto a freshly ingested Document.

    Built-ins (content_hash, doc_type, char_count, etc.) are already populated
    by the Ingestor. We add provenance (session_id) and merge in the user's
    typed values after coercing them against the active schema.
    """
    from ragzero.rag.schema import coerce_value
    meta = dict(doc.metadata)
    meta.setdefault("session_id", state.session_id)
    meta.setdefault("char_count", len(doc.content))
    if user_metadata:
        for k, raw in user_metadata.items():
            spec = state.metadata_schema.by_name(k)
            ftype = spec.type if spec is not None else "string"
            try:
                v = coerce_value(raw, ftype)
            except ValueError as e:
                raise HTTPException(400, f"Metadata field {k!r}: {e}")
            if v is not None:
                meta[k] = v
    return Document(
        doc_id=doc.doc_id, source=doc.source, content=doc.content,
        doc_type=doc.doc_type, created_at=doc.created_at,
        metadata=meta, version=doc.version,
    )


def _add_source(state: SessionState, doc: Document) -> dict:
    src = {
        "doc_id": doc.doc_id,
        "source": doc.source,
        "content": doc.content,
        "doc_type": doc.doc_type,
        "created_at": doc.created_at.isoformat(),
        "metadata": dict(doc.metadata),
        "version": doc.version,
        "char_count": len(doc.content),
    }
    state.sources.append(src)
    pipeline = get_pipeline(state)
    pipeline.ingest_documents([doc])
    return src


@app.post("/api/ingest/upload")
async def ingest_upload(
    files: list[UploadFile] = File(...),
    state: SessionState = Depends(get_session),
) -> dict:
    added = []
    skipped = []
    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in (".txt", ".md", ".pdf"):
            skipped.append({"name": upload.filename, "reason": "unsupported type"})
            continue
        raw = await upload.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        try:
            doc = state.ingestor.ingest_path(tmp_path)
            if doc is None:
                skipped.append({"name": upload.filename, "reason": "duplicate or empty"})
                continue
            # Replace temp path in source with original filename for the UI
            doc = Document(
                doc_id=doc.doc_id,
                source=f"upload://{upload.filename}",
                content=doc.content,
                doc_type=doc.doc_type,
                created_at=doc.created_at,
                metadata={**doc.metadata, "original_filename": upload.filename},
                version=doc.version,
            )
            doc = _apply_metadata(state, doc, None)
            added.append(_add_source(state, doc))
        finally:
            tmp_path.unlink(missing_ok=True)
    return {"added": added, "skipped": skipped}


# ---- Job-based progress endpoints ----

def _run_ingest_job(job: Job, state: SessionState, doc_payloads: list[dict]) -> None:
    """Worker fn: parse, chunk, embed, and report progress as it goes.

    Runs in a background thread (started from the POST handler).
    `doc_payloads` is a pre-prepared list so the worker doesn't need to touch
    the FastAPI request lifecycle (UploadFile is consumed in the handler).
    """
    try:
        JobManager.mark_running(job, phase="parsing")
        total = len(doc_payloads)
        JobManager.emit(job, "phase", f"Parsing {total} source(s)", phase="parsing", total=total, current=0)

        documents: list[Document] = []
        added_meta = []
        skipped = []
        for i, payload in enumerate(doc_payloads, start=1):
            if JobManager.cancel_requested(job):
                JobManager.mark_failed(job, "Cancelled by user")
                return
            kind = payload["kind"]
            user_meta = payload.get("metadata") or None
            if kind == "upload":
                tmp_path = Path(payload["tmp_path"])
                try:
                    doc = state.ingestor.ingest_path(tmp_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
                if doc is None:
                    skipped.append({"name": payload["display_name"], "reason": "duplicate or empty"})
                else:
                    doc = Document(
                        doc_id=doc.doc_id,
                        source=f"upload://{payload['display_name']}",
                        content=doc.content,
                        doc_type=doc.doc_type,
                        created_at=doc.created_at,
                        metadata={**doc.metadata, "original_filename": payload["display_name"]},
                        version=doc.version,
                    )
                    documents.append(_apply_metadata(state, doc, user_meta))
            elif kind == "path":
                p = Path(payload["path"]).expanduser().resolve()
                if p.is_file():
                    doc = state.ingestor.ingest_path(p)
                    if doc is None:
                        skipped.append({"name": str(p), "reason": "duplicate or empty"})
                    else:
                        documents.append(_apply_metadata(state, doc, user_meta))
                elif p.is_dir():
                    for sub in state.ingestor.ingest_directory(p):
                        documents.append(_apply_metadata(state, sub, user_meta))
                else:
                    skipped.append({"name": str(p), "reason": "path not found"})
            elif kind == "url":
                url = payload["url"]
                try:
                    with urllib.request.urlopen(url, timeout=30) as resp:
                        content_type = resp.headers.get("Content-Type", "")
                        raw = resp.read()
                    text = raw.decode("utf-8", errors="replace")
                    if "html" in content_type.lower():
                        import re as _re
                        text = _re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=_re.S | _re.I)
                        text = _re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=_re.S | _re.I)
                        text = _re.sub(r"<[^>]+>", " ", text)
                        text = _re.sub(r"\s+", " ", text).strip()
                    if not text:
                        skipped.append({"name": url, "reason": "empty content"})
                    else:
                        doc = Document(
                            doc_id=f"url-{abs(hash(url)) % (10**12)}",
                            source=url,
                            content=text,
                            doc_type="html" if "html" in content_type.lower() else "txt",
                            created_at=datetime.now(timezone.utc),
                            metadata={"fetched_from": url, "content_type": content_type},
                            version=1,
                        )
                        documents.append(_apply_metadata(state, doc, user_meta))
                except Exception as e:
                    skipped.append({"name": url, "reason": f"fetch failed: {e}"})
            elif kind == "text":
                title = payload["title"] or "pasted-text"
                content = payload["content"]
                if content.strip():
                    doc = Document(
                        doc_id=f"text-{abs(hash(content)) % (10**12)}",
                        source=f"text://{title}",
                        content=content,
                        doc_type="txt",
                        created_at=datetime.now(timezone.utc),
                        metadata={"title": title},
                        version=1,
                    )
                    documents.append(_apply_metadata(state, doc, user_meta))
            JobManager.emit(
                job, "progress",
                f"Loaded source {i}/{total}: {payload.get('display_name') or payload.get('path') or payload.get('url') or payload.get('title') or '?'}",
                current=i, total=total,
            )

        if not documents:
            JobManager.mark_done(job, result={
                "added": [], "skipped": skipped,
                "summary": "No new documents to index.",
            })
            return

        # Build the pipeline (this may itself take a moment if SDKs need to init)
        JobManager.emit(job, "phase", f"Initializing pipeline ({state.embedder_provider})", phase="initializing")
        pipeline = get_pipeline(state)

        # Track which doc payloads correspond to our `documents` so we can
        # register them in session.sources after embedding completes.
        JobManager.emit(
            job, "phase",
            f"Chunking {len(documents)} document(s)",
            phase="chunking",
            current=0, total=len(documents),
        )

        def on_chunk_done(done: int, total: int, source: str) -> None:
            JobManager.emit(
                job, "progress",
                f"Chunked {done}/{total}: {source}",
                phase="chunking", current=done, total=total,
            )

        def on_embed_batch(done: int, total: int) -> None:
            if JobManager.cancel_requested(job):
                # Best-effort cancel — the current batch still completes
                raise RuntimeError("Cancelled by user")
            JobManager.emit(
                job, "progress",
                f"Embedded {done}/{total} chunks",
                phase="embedding", current=done, total=total,
            )

        # We need to switch to embedding phase. ingest_documents_progress drives
        # chunking + embedding in sequence; we report the embedding total once
        # chunking finishes, which happens inside ingest_documents_progress.
        # Pre-chunk so we can report the embed total upfront.
        all_chunks = []
        for i, doc in enumerate(documents, start=1):
            chunks = pipeline.chunker.chunk(doc)
            all_chunks.extend(chunks)
            on_chunk_done(i, len(documents), doc.source)
        JobManager.emit(
            job, "phase",
            f"Embedding {len(all_chunks)} chunk(s) with {state.embedder_provider}",
            phase="embedding", current=0, total=len(all_chunks),
        )
        try:
            pipeline.retriever.index_batched(all_chunks, batch_size=32, on_batch=on_embed_batch)
        except RuntimeError as e:
            if "Cancelled" in str(e):
                JobManager.mark_failed(job, "Cancelled by user")
                return
            raise

        # Register sources in the session now that they're indexed
        for doc in documents:
            src = {
                "doc_id": doc.doc_id,
                "source": doc.source,
                "content": doc.content,
                "doc_type": doc.doc_type,
                "created_at": doc.created_at.isoformat(),
                "metadata": dict(doc.metadata),
                "version": doc.version,
                "char_count": len(doc.content),
            }
            state.sources.append(src)
            added_meta.append({k: v for k, v in src.items() if k != "content"})

        JobManager.mark_done(job, result={
            "added": added_meta,
            "skipped": skipped,
            "summary": f"Indexed {len(documents)} doc(s) → {len(all_chunks)} chunks",
        })
    except Exception as e:
        logger.exception("Ingest job failed")
        JobManager.mark_failed(job, f"{type(e).__name__}: {e}")


@app.post("/api/jobs/ingest/upload")
async def start_upload_job(
    files: list[UploadFile] = File(...),
    metadata: str | None = Form(default=None),  # JSON-encoded dict (multipart can't carry nested objects)
    state: SessionState = Depends(get_session),
) -> dict:
    """Start a background upload+embedding job. Returns job_id immediately."""
    import json as _json
    user_meta: dict | None = None
    if metadata:
        try:
            user_meta = _json.loads(metadata)
            if not isinstance(user_meta, dict):
                raise ValueError("metadata must be a JSON object")
        except (ValueError, TypeError) as e:
            raise HTTPException(400, f"Invalid metadata: {e}")
    # Drain uploads to disk now — UploadFile won't survive past the request.
    payloads: list[dict] = []
    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in (".txt", ".md", ".pdf"):
            continue
        raw = await upload.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            payloads.append({
                "kind": "upload",
                "tmp_path": tmp.name,
                "display_name": upload.filename,
                "metadata": user_meta,
            })
    if not payloads:
        raise HTTPException(400, "No supported files in upload")

    job = jobs.create(kind="ingest_upload", session_id=state.session_id)
    threading.Thread(
        target=_run_ingest_job, args=(job, state, payloads), daemon=True,
    ).start()
    return {"job_id": job.job_id}


class PathJobReq(BaseModel):
    # Accept either a single `path` (legacy) or a `paths` list for batch ingest.
    path: str | None = None
    paths: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/jobs/ingest/path")
def start_path_job(
    req: PathJobReq, state: SessionState = Depends(get_session),
) -> dict:
    items = list(req.paths or [])
    if req.path:
        items.append(req.path)
    items = [p.strip() for p in items if p and p.strip()]
    if not items:
        raise HTTPException(400, "Provide `path` or `paths` with at least one entry")
    for p in items:
        resolved = Path(p).expanduser().resolve()
        if not resolved.exists():
            raise HTTPException(400, f"Path does not exist: {resolved}")
    payloads = [{"kind": "path", "path": p, "metadata": req.metadata} for p in items]
    job = jobs.create(kind="ingest_path", session_id=state.session_id)
    threading.Thread(
        target=_run_ingest_job, args=(job, state, payloads), daemon=True,
    ).start()
    return {"job_id": job.job_id, "n": len(items)}


class UrlJobReq(BaseModel):
    # Accept either a single `url` (legacy) or a `urls` list for batch ingest.
    url: str | None = None
    urls: list[str] | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/jobs/ingest/url")
def start_url_job(req: UrlJobReq, state: SessionState = Depends(get_session)) -> dict:
    items = list(req.urls or [])
    if req.url:
        items.append(req.url)
    items = [u.strip() for u in items if u and u.strip()]
    if not items:
        raise HTTPException(400, "Provide `url` or `urls` with at least one entry")
    for u in items:
        if not (u.startswith("http://") or u.startswith("https://")):
            raise HTTPException(400, f"URL must start with http:// or https://: {u}")
    payloads = [{"kind": "url", "url": u, "metadata": req.metadata} for u in items]
    job = jobs.create(kind="ingest_url", session_id=state.session_id)
    threading.Thread(
        target=_run_ingest_job, args=(job, state, payloads), daemon=True,
    ).start()
    return {"job_id": job.job_id, "n": len(items)}


class TextJobReq(BaseModel):
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/jobs/ingest/text")
def start_text_job(req: TextJobReq, state: SessionState = Depends(get_session)) -> dict:
    if not req.content.strip():
        raise HTTPException(400, "Content is empty")
    job = jobs.create(kind="ingest_text", session_id=state.session_id)
    threading.Thread(
        target=_run_ingest_job,
        args=(job, state, [{
            "kind": "text", "title": req.title, "content": req.content,
            "metadata": req.metadata,
        }]),
        daemon=True,
    ).start()
    return {"job_id": job.job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, state: SessionState = Depends(get_session)) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.session_id != state.session_id:
        raise HTTPException(403, "Job belongs to another session")
    snap = job.snapshot()
    if job.status == "completed" and job.result is not None:
        snap["result"] = job.result
    return snap


@app.get("/api/jobs/{job_id}/stream")
def stream_job(job_id: str, state: SessionState = Depends(get_session)):
    """SSE stream of job progress events. Replays buffered events first."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.session_id != state.session_id:
        raise HTTPException(403, "Job belongs to another session")

    def event_stream():
        import json as _json
        for event in jobs.subscribe(job):
            yield f"data: {_json.dumps(event, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, state: SessionState = Depends(get_session)) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.session_id != state.session_id:
        raise HTTPException(403, "Job belongs to another session")
    JobManager.request_cancel(job)
    return {"ok": True}


@app.get("/api/jobs")
def list_jobs(state: SessionState = Depends(get_session)) -> dict:
    job_list = jobs.list_for_session(state.session_id)
    return {"jobs": [j.snapshot() for j in job_list]}


@app.post("/api/ingest/path")
def ingest_path(req: PathIngestion, state: SessionState = Depends(get_session)) -> dict:
    path = Path(req.path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(400, f"Path does not exist: {path}")
    added = []
    skipped = []
    if path.is_file():
        doc = state.ingestor.ingest_path(path)
        if doc is None:
            skipped.append({"name": str(path), "reason": "duplicate or empty"})
        else:
            doc = _apply_metadata(state, doc, req.metadata)
            added.append(_add_source(state, doc))
    else:
        for doc in state.ingestor.ingest_directory(path):
            doc = _apply_metadata(state, doc, req.metadata)
            added.append(_add_source(state, doc))
    return {"added": added, "skipped": skipped}


@app.post("/api/ingest/url")
def ingest_url(req: UrlIngestion, state: SessionState = Depends(get_session)) -> dict:
    if not (req.url.startswith("http://") or req.url.startswith("https://")):
        raise HTTPException(400, "URL must start with http:// or https://")
    try:
        with urllib.request.urlopen(req.url, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
    except Exception as e:
        raise HTTPException(400, f"Fetch failed: {e}")
    text = raw.decode("utf-8", errors="replace")
    # Very basic HTML stripping if needed
    if "html" in content_type.lower():
        import re
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise HTTPException(400, "URL returned empty content")
    doc = Document(
        doc_id=f"url-{abs(hash(req.url)) % (10**12)}",
        source=req.url,
        content=text,
        doc_type="html" if "html" in content_type.lower() else "txt",
        created_at=datetime.now(timezone.utc),
        metadata={"fetched_from": req.url, "content_type": content_type},
        version=1,
    )
    doc = _apply_metadata(state, doc, req.metadata)
    added = _add_source(state, doc)
    return {"added": [added], "skipped": []}


@app.post("/api/ingest/text")
def ingest_text(req: TextIngestion, state: SessionState = Depends(get_session)) -> dict:
    if not req.content.strip():
        raise HTTPException(400, "Content is empty")
    title = req.title.strip() or "pasted-text"
    doc = Document(
        doc_id=f"text-{abs(hash(req.content)) % (10**12)}",
        source=f"text://{title}",
        content=req.content,
        doc_type="txt",
        created_at=datetime.now(timezone.utc),
        metadata={"title": title},
        version=1,
    )
    doc = _apply_metadata(state, doc, req.metadata)
    return {"added": [_add_source(state, doc)], "skipped": []}


# ---- Structured ingest: preview + commit ----

@app.post("/api/ingest/structured/preview")
async def structured_preview(
    file: UploadFile = File(...),
    state: SessionState = Depends(get_session),
) -> dict:
    """Parse a CSV/JSON/JSONL upload, infer the schema, return preview without
    indexing. The caller reviews + adjusts field flags and posts to /commit."""
    import secrets as _secrets
    from dataclasses import asdict as _asdict
    from ragzero.rag.structured import infer_schema
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    try:
        preview = infer_schema(raw, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    preview_id = _secrets.token_urlsafe(12)
    state.structured_previews[preview_id] = {
        "filename": file.filename,
        "bytes": raw,
        "preview": preview,
    }
    return {
        "preview_id": preview_id,
        "filename": file.filename,
        "format": preview.format,
        "row_count": preview.row_count,
        "truncated": preview.truncated,
        "columns": [_asdict(c) for c in preview.columns],
        "preview_rows": preview.preview_rows,
    }


class FieldSpecIn(BaseModel):
    name: str
    type: str = "string"
    retrievable: bool = True
    filterable: bool = False
    sortable: bool = False
    facetable: bool = False
    searchable: bool = False


class StructuredCommit(BaseModel):
    preview_id: str
    fields: list[FieldSpecIn]
    content_columns: list[str] = Field(default_factory=list)
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/ingest/structured/commit")
def structured_commit(
    req: StructuredCommit, state: SessionState = Depends(get_session),
) -> dict:
    """Index every row of a previously-previewed structured file. Each row
    becomes a Document; columns become metadata; the chunker is set to
    per_row automatically for this session."""
    import hashlib as _hashlib
    from ragzero.rag.structured import parse_iter, build_row_text
    from ragzero.rag.schema import (
        MetadataSchema, FieldSpec, FieldFlags, default_schema, capabilities_for,
    )
    entry = state.structured_previews.get(req.preview_id)
    if entry is None:
        raise HTTPException(404, "Unknown preview_id — re-upload to refresh")

    preview = entry["preview"]
    file_bytes: bytes = entry["bytes"]
    filename: str = entry["filename"]

    # Merge user-confirmed fields onto the session schema. Drop any old
    # custom fields with the same name; replace from this commit.
    caps = capabilities_for(state.vector_store_provider)
    base = default_schema()
    custom_by_name = {
        f.name: f for f in state.metadata_schema.fields if f.origin == "custom"
    }
    for f in req.fields:
        if base.by_name(f.name):
            continue  # don't shadow builtins
        # Silently drop flags the active store can't honor (UI should have
        # disabled the checkboxes; this is defense in depth).
        custom_by_name[f.name] = FieldSpec(
            name=f.name, type=f.type, origin="custom", category="custom",
            flags=FieldFlags(
                filterable=bool(f.filterable) and bool(caps.get("filterable")),
                sortable=bool(f.sortable) and bool(caps.get("sortable")),
                full_text_searchable=bool(f.searchable) and bool(caps.get("full_text_searchable")),
                retrievable=bool(f.retrievable),
            ),
        )
    state.metadata_schema = MetadataSchema(
        fields=list(base.fields) + list(custom_by_name.values())
    )

    # Force per_row chunker for structured ingest — each row IS one chunk.
    if state.chunker_strategy != "per_row":
        state.chunker_strategy = "per_row"
        state.chunker_settings = {}
        state.pipeline = None
        state.pipeline_dirty = True

    pipeline = get_pipeline(state)

    n_added = 0
    skipped: list[dict] = []
    seen_hashes: set[str] = set()
    for i, row in enumerate(parse_iter(file_bytes, preview.format)):
        text = build_row_text(row, req.content_columns)
        if not text.strip():
            skipped.append({"row": i, "reason": "empty content after column selection"})
            continue
        # Per-row dedup via content hash (in addition to session ingestor's
        # dedup which is path-based and doesn't help here).
        h = _hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if h in seen_hashes:
            skipped.append({"row": i, "reason": "duplicate of an earlier row"})
            continue
        seen_hashes.add(h)

        doc_id = f"{filename}:r{i}:{h[:8]}"
        meta = {
            "session_id": state.session_id,
            "content_hash": h,
            "char_count": len(text),
            "original_filename": filename,
            "row_index": i,
            **{k: v for k, v in row.items() if v is not None and v != ""},
            **req.extra_metadata,
        }
        doc = Document(
            doc_id=doc_id,
            source=f"{filename}#row={i}",
            content=text,
            doc_type=preview.format,
            created_at=datetime.now(timezone.utc),
            metadata=meta,
            version=1,
        )
        _add_source(state, doc)
        n_added += 1

    # Free the cached bytes — commit done.
    state.structured_previews.pop(req.preview_id, None)
    return {
        "ok": True,
        "added": n_added,
        "skipped": skipped,
        "summary": f"Indexed {n_added} row(s) from {filename} ({preview.format})",
    }


@app.delete("/api/ingest/structured/preview/{preview_id}")
def discard_preview(preview_id: str, state: SessionState = Depends(get_session)) -> dict:
    state.structured_previews.pop(preview_id, None)
    return {"ok": True}


@app.get("/api/sources")
def list_sources(state: SessionState = Depends(get_session)) -> dict:
    return {
        "sources": [
            {k: v for k, v in s.items() if k != "content"}  # don't send full text
            for s in state.sources
        ]
    }


@app.get("/api/sources/{doc_id}/chunks")
def source_chunks(
    doc_id: str,
    state: SessionState = Depends(get_session),
) -> dict:
    """Return chunks for a single ingested source, plus embedding/store metadata.

    Re-chunks the stored document text using the current pipeline's chunker
    rather than fetching from the vector store, since not every backend
    supports a `get-by-doc-id` operation. This is the same chunking the
    pipeline uses at index time, so positions/chunk_ids match what's stored.
    """
    src = next((s for s in state.sources if s["doc_id"] == doc_id), None)
    if src is None:
        raise HTTPException(404, f"No source with doc_id={doc_id}")

    pipeline = get_pipeline(state)
    doc = _source_to_doc(src)
    chunks = pipeline.chunker.chunk(doc)

    # Embedder dim (cheap — most embedders cache it). Wrap in try because
    # some embedders compute it lazily on first embed().
    embedder_dim = None
    try:
        embedder_dim = pipeline.retriever.embedder.dim
    except Exception:
        embedder_dim = None

    # Active vector store: report its class + reported size (chunks indexed).
    store = getattr(pipeline.retriever, "vector_store", None)
    store_size = None
    if store is not None and hasattr(store, "size"):
        try:
            store_size = store.size()
        except Exception:
            store_size = None

    return {
        "doc_id": doc_id,
        "source": src["source"],
        "doc_type": src["doc_type"],
        "char_count": src.get("char_count", len(src.get("content", ""))),
        "version": src.get("version", 1),
        "metadata": src.get("metadata", {}),
        "embedder": {
            "provider": state.embedder_provider,
            "model": state.embedder_settings.get("model"),
            "dim": embedder_dim,
        },
        "vector_store": {
            "provider": state.vector_store_provider,
            "class_name": type(store).__name__ if store is not None else None,
            "settings": _mask_secrets(state.vector_store_settings),
            "total_chunks_in_store": store_size,
        },
        "chunker": {
            "strategy": pipeline.config.chunker_strategy,
            "settings": pipeline.config.chunker_settings,
        },
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "position": c.position,
                "page": c.page,
                "char_count": len(c.text),
                "text": c.text,
            }
            for c in chunks
        ],
    }


@app.delete("/api/sources")
def clear_sources(state: SessionState = Depends(get_session)) -> dict:
    """Clear all ingested data for this session.

    - Drops the per-session source list and history
    - Resets the per-session Ingestor so the same files can be re-added
      (otherwise content-hash dedup would mark them as duplicates)
    - Best-effort clears the active vector store (so external persistent
      stores like Chroma/Qdrant don't leak orphaned vectors)
    """
    cleared_store = None
    cleared_store_error = None
    if state.pipeline is not None:
        store = getattr(state.pipeline.retriever, "vector_store", None)
        if store is not None and hasattr(store, "clear"):
            try:
                store.clear()
                cleared_store = type(store).__name__
            except Exception as e:
                cleared_store_error = f"{type(e).__name__}: {e}"
                logger.warning("Vector store clear() failed: %s", cleared_store_error)
    state.sources = []
    state.ingestor = Ingestor()  # fresh dedup state
    state.pipeline_dirty = True
    state.pipeline = None
    return {
        "ok": True,
        "cleared_store": cleared_store,
        "cleared_store_error": cleared_store_error,
    }


# ---- Strategy + Graph ----

class StrategyConfig(BaseModel):
    mode: str = "simple"  # simple | graph | agentic | multilingual
    multilingual_enabled: bool = False
    output_language: str = "auto"
    agent_max_iterations: int = 3
    graph_max_depth: int = 2


@app.get("/api/strategy")
def get_strategy(state: SessionState = Depends(get_session)) -> dict:
    return {
        "mode": state.rag_mode,
        "multilingual_enabled": state.multilingual_enabled,
        "output_language": state.output_language,
        "agent_max_iterations": state.agent_max_iterations,
        "graph_max_depth": state.graph_max_depth,
        "languages": LANGUAGES,
        "presets": list(PRESETS.keys()),
    }


@app.post("/api/strategy")
def set_strategy(cfg: StrategyConfig, state: SessionState = Depends(get_session)) -> dict:
    if cfg.mode not in ("simple", "graph", "agentic", "multilingual"):
        raise HTTPException(400, f"Unknown mode: {cfg.mode}")
    state.rag_mode = cfg.mode
    state.multilingual_enabled = cfg.multilingual_enabled or cfg.mode == "multilingual"
    state.output_language = cfg.output_language
    state.agent_max_iterations = max(1, min(10, cfg.agent_max_iterations))
    state.graph_max_depth = max(1, min(5, cfg.graph_max_depth))
    return {"ok": True}


@app.post("/api/graph/build")
def build_graph(state: SessionState = Depends(get_session)) -> dict:
    """Extract entities + relations from all indexed chunks and build the graph.

    Synchronous (returns when done). For progress streaming, use POST
    /api/jobs/graph/build instead.
    """
    if not state.sources:
        raise HTTPException(400, "No documents indexed. Upload data first.")
    pipeline = get_pipeline(state)
    if state.graph_store is None:
        state.graph_store = NetworkXGraphStore()
    state.graph_store.clear()

    # Re-chunk the indexed sources so we can iterate; alternatively pull from store
    chunker = Chunker(
        chunk_size=pipeline.config.chunk_size,
        overlap=pipeline.config.chunk_overlap,
    )
    extractor = EntityExtractor(pipeline.generator.llm)

    n_entities = 0
    n_relations = 0
    for src in state.sources:
        doc = _source_to_doc(src)
        for chunk in chunker.chunk(doc):
            ents, rels = extractor.extract(chunk)
            for e in ents:
                state.graph_store.upsert_entity(e)
                n_entities += 1
            for r in rels:
                state.graph_store.upsert_relation(r)
                n_relations += 1
    state.graph_built = True
    stats = state.graph_store.stats()
    return {
        "ok": True,
        "extracted": {"entities": n_entities, "relations": n_relations},
        "stats": stats,
    }


def _run_graph_build_job(job: Job, state: SessionState) -> None:
    """Worker: walk chunks, run LLM-based entity extraction, build the graph."""
    try:
        JobManager.mark_running(job, phase="preparing")
        pipeline = get_pipeline(state)
        if state.graph_store is None:
            state.graph_store = NetworkXGraphStore()
        state.graph_store.clear()

        chunker = Chunker(
            chunk_size=pipeline.config.chunk_size,
            overlap=pipeline.config.chunk_overlap,
        )
        extractor = EntityExtractor(pipeline.generator.llm)

        # Pre-compute total chunks so the UI can show a real percentage.
        # Cheap — chunker is just regex on already-loaded text.
        per_source_chunks = []
        for src in state.sources:
            doc = _source_to_doc(src)
            chunks = chunker.chunk(doc)
            per_source_chunks.append((doc, chunks))
        total_chunks = sum(len(c) for _, c in per_source_chunks)
        JobManager.emit(
            job, "phase",
            f"Extracting entities from {total_chunks} chunk(s) across {len(per_source_chunks)} doc(s)",
            phase="extracting", current=0, total=total_chunks,
        )

        n_entities = 0
        n_relations = 0
        chunks_done = 0
        for doc, chunks in per_source_chunks:
            for chunk in chunks:
                if JobManager.cancel_requested(job):
                    JobManager.mark_failed(job, "Cancelled by user")
                    return
                try:
                    ents, rels = extractor.extract(chunk)
                except Exception as e:
                    # An LLM error on one chunk shouldn't kill the whole build.
                    # Surface it as a log event and continue.
                    JobManager.emit(
                        job, "log",
                        f"Extraction failed on chunk {chunk.chunk_id}: {e}",
                        chunk_id=chunk.chunk_id, error=str(e),
                    )
                    chunks_done += 1
                    JobManager.emit(
                        job, "progress",
                        f"Processed {chunks_done}/{total_chunks} chunks (skipped 1)",
                        phase="extracting", current=chunks_done, total=total_chunks,
                    )
                    continue
                for e in ents:
                    state.graph_store.upsert_entity(e)
                    n_entities += 1
                for r in rels:
                    state.graph_store.upsert_relation(r)
                    n_relations += 1
                chunks_done += 1
                # Emit progress every chunk; the UI can throttle rendering.
                JobManager.emit(
                    job, "progress",
                    f"Processed {chunks_done}/{total_chunks} chunks · {n_entities} entities · {n_relations} relations",
                    phase="extracting", current=chunks_done, total=total_chunks,
                    entities=n_entities, relations=n_relations,
                )

        state.graph_built = True
        stats = state.graph_store.stats()
        JobManager.mark_done(job, result={
            "extracted": {"entities": n_entities, "relations": n_relations},
            "stats": stats,
            "summary": f"Built graph: {stats['nodes']} nodes, {stats['edges']} edges",
        })
    except Exception as e:
        logger.exception("Graph build job failed")
        JobManager.mark_failed(job, f"{type(e).__name__}: {e}")


@app.post("/api/jobs/graph/build")
def start_graph_build_job(state: SessionState = Depends(get_session)) -> dict:
    """Start a background graph build with live progress."""
    if not state.sources:
        raise HTTPException(400, "No documents indexed. Upload data first.")
    job = jobs.create(kind="graph_build", session_id=state.session_id)
    threading.Thread(
        target=_run_graph_build_job, args=(job, state), daemon=True,
    ).start()
    return {"job_id": job.job_id}


@app.get("/api/graph/snapshot")
def graph_snapshot(state: SessionState = Depends(get_session)) -> dict:
    if state.graph_store is None:
        return {"nodes": [], "edges": [], "stats": {"nodes": 0, "edges": 0}}
    snap = state.graph_store.snapshot()
    snap["stats"] = state.graph_store.stats()
    return snap


@app.delete("/api/graph")
def clear_graph(state: SessionState = Depends(get_session)) -> dict:
    if state.graph_store is not None:
        state.graph_store.clear()
    state.graph_built = False
    return {"ok": True}


# ---- Query (streamed via SSE) ----

@app.post("/api/query")
def query(req: QueryRequest, state: SessionState = Depends(get_session)) -> dict:
    """Non-streaming query — returns the full event list at once."""
    if not state.sources:
        raise HTTPException(400, "No documents indexed. Upload data first.")
    if not req.query.strip():
        raise HTTPException(400, "Empty query")

    from ragzero.rag.filters import Filter as _Filter
    filters = [_Filter(field=f.field, op=f.op, value=f.value) for f in (req.filters or [])]
    pipeline = get_pipeline(state)
    strategy, ctx = _build_strategy_and_ctx(state, pipeline, filters=filters)

    events = []
    for ev in strategy.run(req.query, ctx):
        events.append({
            "kind": ev.kind, "label": ev.label,
            "timestamp": ev.timestamp, "data": ev.data,
        })

    answer = extract_answer([
        type("E", (), {"kind": e["kind"], "data": e["data"]})() for e in events
    ])
    if answer is None:
        raise HTTPException(500, "Strategy did not produce an answer")

    record = {
        "query": req.query,
        "mode": state.rag_mode,
        "multilingual_enabled": state.multilingual_enabled,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": answer.trace_id,
        "answer": {
            "text": answer.text,
            "refused": answer.refused,
            "refusal_reason": answer.refusal_reason,
            "confidence": answer.confidence,
            "citations": [
                {
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "page": c.page,
                    "chunk_id": c.chunk_id,
                    "timestamp": c.timestamp.isoformat(),
                }
                for c in answer.citations
            ],
        },
        "events": events,
    }
    state.history.append(record)
    state.history = state.history[-50:]
    return record


@app.post("/api/query/stream")
def query_stream(req: QueryRequest, state: SessionState = Depends(get_session)):
    """Stream strategy events via Server-Sent Events for live UI updates."""
    if not state.sources:
        raise HTTPException(400, "No documents indexed. Upload data first.")
    if not req.query.strip():
        raise HTTPException(400, "Empty query")

    from ragzero.rag.filters import Filter as _Filter
    filters = [_Filter(field=f.field, op=f.op, value=f.value) for f in (req.filters or [])]
    pipeline = get_pipeline(state)
    strategy, ctx = _build_strategy_and_ctx(state, pipeline, filters=filters)

    def event_stream():
        import json as _json
        collected = []
        try:
            for ev in strategy.run(req.query, ctx):
                payload = {
                    "kind": ev.kind, "label": ev.label,
                    "timestamp": ev.timestamp,
                    "data": _serialize_event_data(ev.data),
                }
                collected.append(payload)
                yield f"data: {_json.dumps(payload)}\n\n"
            # Record to history after stream completes
            answer = None
            for e in reversed(collected):
                if e["kind"] == "final":
                    answer = e["data"].get("answer")
                    break
            if answer:
                state.history.append({
                    "query": req.query,
                    "mode": state.rag_mode,
                    "multilingual_enabled": state.multilingual_enabled,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "trace_id": answer.get("trace_id"),
                    "answer": answer,
                    "events": collected,
                })
                state.history = state.history[-50:]
        except Exception as e:
            err = {"kind": "error", "label": "Stream error", "timestamp": "",
                   "data": {"message": str(e), "type": type(e).__name__}}
            yield f"data: {_json.dumps(err)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _build_strategy_and_ctx(
    state: SessionState, pipeline: RAGPipeline,
    filters: list | None = None,
    mode_override: str | None = None,
    multilingual_override: bool | None = None,
):
    """Build a strategy + StrategyContext from current session state.

    `mode_override` / `multilingual_override` let callers (e.g. the public
    assistant) choose a mode per-request without mutating the shared state."""
    from ragzero.rag.strategies import StrategyContext
    translator = None
    mode_sel = mode_override if mode_override is not None else state.rag_mode
    base_ml = multilingual_override if multilingual_override is not None else state.multilingual_enabled
    multilingual = bool(base_ml) or mode_sel == "multilingual"
    if multilingual:
        translator = LLMTranslator(pipeline.generator.llm)
    mode = mode_sel if mode_sel != "multilingual" else "simple"
    strategy = build_strategy(
        mode=mode,
        multilingual=multilingual,
        translator=translator,
        output_language=state.output_language,
        agent_max_iterations=state.agent_max_iterations,
        graph_max_depth=state.graph_max_depth,
    )
    options: dict = {}
    if state.graph_store is not None:
        options["graph_store"] = state.graph_store
    if filters:
        options["filters"] = filters
    from ragzero.rag.observability import new_trace_id
    ctx = StrategyContext(
        pipeline=pipeline,
        tracer=pipeline.tracer,
        trace_id=new_trace_id(),
        options=options,
    )
    return strategy, ctx


def _serialize_event_data(data: dict) -> dict:
    """Make event data JSON-serializable, including Answer objects."""
    import dataclasses
    out = {}
    for k, v in data.items():
        if hasattr(v, "__dataclass_fields__"):
            out[k] = dataclasses.asdict(v)
            # Convert datetime fields to iso strings
            _stringify_datetimes(out[k])
        else:
            out[k] = v
    return out


def _stringify_datetimes(obj):
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if hasattr(v, "isoformat"):
                obj[k] = v.isoformat()
            else:
                _stringify_datetimes(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if hasattr(v, "isoformat"):
                obj[i] = v.isoformat()
            else:
                _stringify_datetimes(v)


@app.get("/api/trace/{trace_id}")
def get_trace(trace_id: str, state: SessionState = Depends(get_session)) -> dict:
    pipeline = get_pipeline(state)
    events = []
    if isinstance(pipeline.tracer, InMemoryTracer):
        events = pipeline.tracer.events_for(trace_id)
    if not events:
        raise HTTPException(404, "trace not found")
    return {"trace_id": trace_id, "events": events}


@app.get("/api/history")
def get_history(state: SessionState = Depends(get_session)) -> dict:
    return {"history": state.history}


# ================= Published Space (public ask-only assistant) =================

class SpaceSettings(BaseModel):
    name: str | None = None
    title: str | None = None
    welcome: str | None = None
    accent: str | None = None
    footer: str | None = None
    suggested_questions: list[str] | None = None
    allowed_modes: list[str] | None = None
    default_mode: str | None = None
    show_citations: bool | None = None
    show_confidence: bool | None = None
    streaming: bool | None = None
    access_code: str | None = None  # "" clears, None = unchanged
    rate_limit_per_min: int | None = None
    daily_cap: int | None = None


class PublicQueryRequest(BaseModel):
    query: str
    mode: str | None = None
    access_code: str | None = None


def _snapshot_session(state: SessionState) -> dict:
    """Capture everything the Space needs to answer, independent of the
    host's ephemeral cookie session."""
    return {
        "llm": {"provider": state.llm_provider, "settings": state.llm_settings},
        "embedder": {"provider": state.embedder_provider, "settings": state.embedder_settings},
        "vector_store": {"provider": state.vector_store_provider, "settings": state.vector_store_settings},
        "pipeline_overrides": state.pipeline_overrides,
        "chunker_strategy": state.chunker_strategy,
        "chunker_settings": state.chunker_settings,
        "sources": state.sources,
    }


def _build_space_state(cfg: SpaceConfig) -> SessionState:
    snap = cfg.snapshot or {}
    llm = snap.get("llm") or {}
    emb = snap.get("embedder") or {}
    vs = snap.get("vector_store") or {}
    st = SessionState(session_id="__space__", created_at=datetime.now().timestamp(),
                      last_seen=datetime.now().timestamp())
    st.llm_provider = llm.get("provider", "mock")
    st.llm_settings = llm.get("settings", {}) or {}
    st.embedder_provider = emb.get("provider", "hash")
    st.embedder_settings = emb.get("settings", {}) or {}
    st.vector_store_provider = vs.get("provider", "in_memory")
    st.vector_store_settings = vs.get("settings", {}) or {}
    st.pipeline_overrides = snap.get("pipeline_overrides", {}) or {}
    st.chunker_strategy = snap.get("chunker_strategy", "fixed_size")
    st.chunker_settings = snap.get("chunker_settings", {}) or {}
    st.sources = snap.get("sources", []) or []
    st.rag_mode = cfg.default_mode
    st.pipeline_dirty = True
    return st


def _get_space_state() -> SessionState | None:
    """Lazily build (and cache) the live SessionState backing the published Space."""
    global _space_runtime
    cfg = space_manager.config
    if cfg is None or not cfg.published:
        return None
    with _space_lock:
        if _space_runtime is None:
            _space_runtime = _build_space_state(cfg)
    return _space_runtime


def _reset_space_runtime() -> None:
    global _space_runtime
    with _space_lock:
        _space_runtime = None


@app.get("/api/space")
def get_space(state: SessionState = Depends(get_session)) -> dict:
    cfg = space_manager.config
    current = {
        "n_sources": len(state.sources),
        "llm": state.llm_provider,
        "embedder": state.embedder_provider,
        "vector_store": state.vector_store_provider,
        "real_providers": state.llm_provider != "mock" and state.embedder_provider != "hash",
    }
    return {
        "space": cfg.admin_view() if cfg else None,
        "current": current,
        "admin_token_set": bool(ADMIN_TOKEN),
    }


@app.post("/api/space")
def save_space(settings: SpaceSettings, state: SessionState = Depends(get_session)) -> dict:
    cfg = space_manager.config or SpaceConfig()
    space_manager.apply_settings(cfg, settings.model_dump(exclude_unset=True))
    space_manager.config = cfg
    space_manager.save()
    _reset_space_runtime()
    return {"space": cfg.admin_view()}


@app.post("/api/space/publish")
def publish_space(settings: SpaceSettings, state: SessionState = Depends(get_session)) -> dict:
    if not state.sources:
        raise HTTPException(400, "Add data before publishing — the assistant needs something to answer from.")
    cfg = space_manager.config or SpaceConfig()
    space_manager.apply_settings(cfg, settings.model_dump(exclude_unset=True))
    cfg.snapshot = _snapshot_session(state)
    cfg.published = True
    space_manager.config = cfg
    space_manager.save()
    _reset_space_runtime()
    return {"space": cfg.admin_view()}


@app.post("/api/space/unpublish")
def unpublish_space(state: SessionState = Depends(get_session)) -> dict:
    cfg = space_manager.config
    if cfg:
        cfg.published = False
        space_manager.save()
    _reset_space_runtime()
    return {"ok": True}


@app.get("/api/space/hosting")
def space_hosting(request: Request, state: SessionState = Depends(get_session)) -> dict:
    import socket
    lan_ip = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass
    slug = space_manager.config.slug if space_manager.config else "assistant"
    return {
        "lan_ip": lan_ip,
        "host": request.headers.get("host", ""),
        "slug": slug,
        "admin_token_set": bool(ADMIN_TOKEN),
    }


# ---- Public, locked-down endpoints (no admin powers, no secrets) ----

@app.get("/public/meta")
def public_meta() -> dict:
    cfg = space_manager.config
    if cfg is None or not cfg.published:
        return {"published": False}
    return cfg.public_meta()


@app.post("/public/query")
def public_query(req: PublicQueryRequest, request: Request) -> dict:
    cfg = space_manager.config
    if cfg is None or not cfg.published:
        raise HTTPException(404, "No assistant is published here.")
    if not req.query.strip():
        raise HTTPException(400, "Empty question")
    if not space_manager.check_access(req.access_code):
        raise HTTPException(401, "Access code required or incorrect.")
    ip = request.client.host if request.client else "?"
    ok, reason = space_manager.check_rate(ip)
    if not ok:
        raise HTTPException(429, reason)

    state = _get_space_state()
    if state is None or not state.sources:
        raise HTTPException(503, "Assistant is not ready yet.")
    mode = req.mode if (req.mode in cfg.allowed_modes) else cfg.default_mode
    pipeline = get_pipeline(state)
    # Graph mode needs a graph that the snapshot doesn't carry — degrade safely.
    if mode == "graph" and state.graph_store is None:
        mode = "simple"
    strategy, ctx = _build_strategy_and_ctx(
        state, pipeline, mode_override=mode,
        multilingual_override=(mode == "multilingual"),
    )
    events = list(strategy.run(req.query, ctx))
    answer = extract_answer(events)
    if answer is None:
        raise HTTPException(500, "No answer was produced.")

    out: dict[str, Any] = {
        "text": answer.text,
        "refused": answer.refused,
        "refusal_reason": answer.refusal_reason,
    }
    if cfg.show_confidence:
        out["confidence"] = answer.confidence
    if cfg.show_citations:
        out["citations"] = [{"source": c.source, "page": c.page} for c in answer.citations]
    return {"answer": out, "mode": mode}


# ---------------- Static UI ----------------

if UI_DIST.exists():
    app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="assets")

    @app.get("/")
    def index():
        index_html = UI_DIST / "index.html"
        if index_html.exists():
            return FileResponse(index_html)
        return JSONResponse({"error": "UI not built"}, status_code=404)

    # SPA fallback — anything not /api/* serves index.html
    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404)
        index_html = UI_DIST / "index.html"
        if index_html.exists():
            return FileResponse(index_html)
        raise HTTPException(404)
else:
    @app.get("/")
    def no_ui():
        return JSONResponse(
            {
                "message": "RAG Console API running. UI not built.",
                "build_ui": "cd ui && npm install && npm run build",
                "dev_ui": "cd ui && npm run dev (then open http://localhost:5173)",
                "api_docs": "/docs",
            }
        )


if __name__ == "__main__":
    import socket
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port_env = os.environ.get("PORT")
    if port_env:
        port = int(port_env)
    else:
        # No PORT set — let the OS pick a free port so we never collide
        # with whatever is already bound on the machine.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
            _s.bind((host, 0))
            port = _s.getsockname()[1]
        print(f"PORT not set; using auto-assigned free port {port}")

    uvicorn.run(app, host=host, port=port)