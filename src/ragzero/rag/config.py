"""Config loader.

Reads a YAML config and builds a fully-wired pipeline. Backends are
selected by string keys looked up in registries — to add a new backend,
register it once and reference it by name in YAML. No pipeline code
changes.

Usage:
    pipeline = load_pipeline("config/dev.yaml")
    pipeline = load_pipeline("config/prod.yaml")

Or override individual sections from code:
    pipeline = load_pipeline("config/dev.yaml", overrides={"backends": {"llm": "mock"}})
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .backends.caches import InMemoryTTLCache
from .backends.embedders import HashEmbedder
from .backends.llms import MockLLM
from .backends.stores import BM25KeywordIndex, InMemoryVectorStore
from .cache import CachedEmbedder, EmbeddingCache, QueryCache
from .confidence import ConfidenceConfig
from .fallback import FallbackConfig
from .interfaces import Cache, Embedder, KeywordIndex, LLM, Tracer, VectorStore
from .observability import InMemoryTracer, LoggingTracer
from .pipeline import PipelineConfig, RAGPipeline


# ---------- Backend registries ----------
# Each registry maps a string key (as used in YAML) to a constructor.
# Constructors take a settings dict so backends can be parameterized
# from YAML without changing the registry signature.

EmbedderCtor = Callable[[dict[str, Any]], Embedder]
VectorStoreCtor = Callable[[dict[str, Any], int], VectorStore]
KeywordIndexCtor = Callable[[dict[str, Any]], KeywordIndex]
LLMCtor = Callable[[dict[str, Any]], LLM]
TracerCtor = Callable[[dict[str, Any]], Tracer]
CacheCtor = Callable[[dict[str, Any]], Cache]


def _embedder_hash(settings: dict[str, Any]) -> Embedder:
    return HashEmbedder(dim=int(settings.get("dim", 64)))


def _embedder_sentence_transformers(settings: dict[str, Any]) -> Embedder:  # pragma: no cover
    from .backends.embedders import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(
        model_name=settings.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    )


def _embedder_openai(settings: dict[str, Any]) -> Embedder:  # pragma: no cover
    from .backends.embedders import OpenAIEmbedder
    return OpenAIEmbedder(
        api_key=settings["api_key"],
        model=settings.get("model", "text-embedding-3-small"),
        base_url=settings.get("base_url") or None,
    )


def _embedder_azure_openai(settings: dict[str, Any]) -> Embedder:  # pragma: no cover
    from .backends.embedders import AzureOpenAIEmbedder
    return AzureOpenAIEmbedder(
        api_key=settings["api_key"],
        endpoint=settings["endpoint"],
        deployment=settings["deployment"],
        api_version=settings.get("api_version", "2024-02-01"),
        dim=int(settings.get("dim", 1536)),
    )


def _embedder_voyage(settings: dict[str, Any]) -> Embedder:  # pragma: no cover
    from .backends.embedders import VoyageEmbedder
    return VoyageEmbedder(
        api_key=settings["api_key"],
        model=settings.get("model", "voyage-3"),
    )


def _embedder_gemini(settings: dict[str, Any]) -> Embedder:  # pragma: no cover
    from .backends.embedders import GeminiEmbedder
    dim = settings.get("dim")
    return GeminiEmbedder(
        api_key=settings["api_key"],
        model=settings.get("model", "text-embedding-004"),
        dim=int(dim) if dim else None,
    )


def _embedder_openrouter(settings: dict[str, Any]) -> Embedder:  # pragma: no cover
    from .backends.embedders import OpenRouterEmbedder
    dim = settings.get("dim")
    return OpenRouterEmbedder(
        api_key=settings["api_key"],
        model=settings.get("model", "openai/text-embedding-3-small"),
        dim=int(dim) if dim else None,
        site_url=settings.get("site_url") or None,
        app_name=settings.get("app_name") or None,
    )


def _store_in_memory(settings: dict[str, Any], dim: int) -> VectorStore:
    return InMemoryVectorStore()


def _store_faiss(settings: dict[str, Any], dim: int) -> VectorStore:  # pragma: no cover
    from .backends.stores import FaissVectorStore
    # UI text inputs arrive as strings; faiss's pybind constructor rejects str for M.
    raw_M = settings.get("M", 32)
    try:
        M = int(raw_M) if raw_M not in (None, "") else 32
    except (TypeError, ValueError):
        raise ValueError(f"FAISS 'M' must be an integer, got: {raw_M!r}")
    return FaissVectorStore(dim=dim, M=M)


def _store_qdrant(settings: dict[str, Any], dim: int) -> VectorStore:  # pragma: no cover
    from .backends.vector_stores_external import QdrantVectorStore
    return QdrantVectorStore(
        url=settings.get("url", "http://localhost:6333"),
        collection=settings.get("collection", "ragzero"),
        api_key=settings.get("api_key") or None,
        dim=dim,
    )


def _store_pinecone(settings: dict[str, Any], dim: int) -> VectorStore:  # pragma: no cover
    from .backends.vector_stores_external import PineconeVectorStore
    return PineconeVectorStore(
        api_key=settings["api_key"],
        index_name=settings.get("index_name", "ragzero"),
        dim=dim,
        cloud=settings.get("cloud", "aws"),
        region=settings.get("region", "us-east-1"),
    )


def _store_weaviate(settings: dict[str, Any], dim: int) -> VectorStore:  # pragma: no cover
    from .backends.vector_stores_external import WeaviateVectorStore
    return WeaviateVectorStore(
        url=settings.get("url", "http://localhost:8080"),
        api_key=settings.get("api_key") or None,
        class_name=settings.get("class_name", "RagzeroChunk"),
        dim=dim,
    )


def _store_chroma(settings: dict[str, Any], dim: int) -> VectorStore:  # pragma: no cover
    from .backends.vector_stores_external import ChromaVectorStore
    port = settings.get("port", 8000)
    return ChromaVectorStore(
        collection=settings.get("collection", "ragzero"),
        persist_directory=settings.get("persist_directory") or None,
        host=settings.get("host") or None,
        port=int(port) if port else 8000,
        dim=dim,
    )


def _store_pgvector(settings: dict[str, Any], dim: int) -> VectorStore:  # pragma: no cover
    from .backends.vector_stores_external import PgVectorStore
    return PgVectorStore(
        dsn=settings["dsn"],
        table=settings.get("table", "ragzero_chunks"),
        dim=dim,
        ivfflat_lists=int(settings.get("ivfflat_lists", 100)),
    )


def _store_azure_ai_search(settings: dict[str, Any], dim: int) -> VectorStore:  # pragma: no cover
    from .backends.vector_stores_external import AzureAISearchVectorStore
    return AzureAISearchVectorStore(
        endpoint=settings["endpoint"],
        api_key=settings["api_key"],
        index_name=settings.get("index_name", "ragzero"),
        dim=dim,
    )


def _kw_bm25(settings: dict[str, Any]) -> KeywordIndex:
    return BM25KeywordIndex(k1=settings.get("k1", 1.5), b=settings.get("b", 0.75))


def _llm_mock(settings: dict[str, Any]) -> LLM:
    return MockLLM()


def _llm_anthropic(settings: dict[str, Any]) -> LLM:  # pragma: no cover
    from .backends.llms import AnthropicLLM
    return AnthropicLLM(
        model=settings.get("model", "claude-opus-4-7"),
        api_key=settings.get("api_key"),
    )


def _llm_openai(settings: dict[str, Any]) -> LLM:  # pragma: no cover
    from .backends.llms import OpenAILLM
    return OpenAILLM(
        api_key=settings["api_key"],
        model=settings.get("model", "gpt-4o-mini"),
        base_url=settings.get("base_url") or None,
    )


def _llm_azure_openai(settings: dict[str, Any]) -> LLM:  # pragma: no cover
    from .backends.llms import AzureOpenAILLM
    return AzureOpenAILLM(
        api_key=settings["api_key"],
        endpoint=settings["endpoint"],
        deployment=settings["deployment"],
        api_version=settings.get("api_version", "2024-02-01"),
    )


def _llm_gemini(settings: dict[str, Any]) -> LLM:  # pragma: no cover
    from .backends.llms import GeminiLLM
    return GeminiLLM(
        api_key=settings["api_key"],
        model=settings.get("model", "gemini-2.0-flash"),
    )


def _llm_openrouter(settings: dict[str, Any]) -> LLM:  # pragma: no cover
    from .backends.llms import OpenRouterLLM
    return OpenRouterLLM(
        api_key=settings["api_key"],
        model=settings.get("model", "anthropic/claude-3.5-sonnet"),
        site_url=settings.get("site_url") or None,
        app_name=settings.get("app_name") or None,
    )


def _tracer_in_memory(settings: dict[str, Any]) -> Tracer:
    return InMemoryTracer(max_events=settings.get("max_events", 10_000))


def _tracer_logging(settings: dict[str, Any]) -> Tracer:
    return LoggingTracer(logger_name=settings.get("logger_name", "rag.trace"))


EMBEDDERS: dict[str, EmbedderCtor] = {
    "hash": _embedder_hash,
    "sentence_transformers": _embedder_sentence_transformers,
    "openai": _embedder_openai,
    "azure_openai": _embedder_azure_openai,
    "voyage": _embedder_voyage,
    "gemini": _embedder_gemini,
    "openrouter": _embedder_openrouter,
}

VECTOR_STORES: dict[str, VectorStoreCtor] = {
    "in_memory": _store_in_memory,
    "faiss": _store_faiss,
    "qdrant": _store_qdrant,
    "pinecone": _store_pinecone,
    "weaviate": _store_weaviate,
    "chroma": _store_chroma,
    "pgvector": _store_pgvector,
    "azure_ai_search": _store_azure_ai_search,
}

KEYWORD_INDEXES: dict[str, KeywordIndexCtor] = {
    "bm25": _kw_bm25,
}

LLMS: dict[str, LLMCtor] = {
    "mock": _llm_mock,
    "anthropic": _llm_anthropic,
    "openai": _llm_openai,
    "azure_openai": _llm_azure_openai,
    "gemini": _llm_gemini,
    "openrouter": _llm_openrouter,
}

TRACERS: dict[str, TracerCtor] = {
    "in_memory": _tracer_in_memory,
    "logging": _tracer_logging,
}


def register_embedder(name: str, ctor: EmbedderCtor) -> None:
    """Register a custom embedder so it's available from YAML."""
    EMBEDDERS[name] = ctor


def register_vector_store(name: str, ctor: VectorStoreCtor) -> None:
    VECTOR_STORES[name] = ctor


def register_llm(name: str, ctor: LLMCtor) -> None:
    LLMS[name] = ctor


# ---------- YAML loader ----------

def _load_yaml(path: Path) -> dict[str, Any]:
    """Tiny YAML loader. Uses PyYAML if available, falls back to a minimal
    parser that handles the subset of YAML used in this project's configs
    (mappings, scalars, comments, no flow style, no anchors)."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except ImportError:
        return _minimal_yaml_parse(text)


def _minimal_yaml_parse(text: str) -> dict[str, Any]:
    """Subset YAML parser: mappings, scalars (str/int/float/bool/null), comments.

    Sufficient for the dev/prod config files in this project. For anything
    fancier, install PyYAML.
    """
    # Pre-scan into (indent, key, value) tuples so we can use lookahead
    # to distinguish "empty value = null" from "key opens sub-mapping".
    lines: list[tuple[int, str, str]] = []
    for raw in text.splitlines():
        stripped = raw.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        key, _, value = stripped.strip().partition(":")
        lines.append((indent, key.strip().strip('"').strip("'"), value.strip()))

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for i, (indent, key, value) in enumerate(lines):
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            # Empty value: open a sub-map ONLY if the next line is deeper indented.
            next_is_child = (
                i + 1 < len(lines) and lines[i + 1][0] > indent
            )
            if next_is_child:
                new_map: dict[str, Any] = {}
                parent[key] = new_map
                stack.append((indent, new_map))
            else:
                parent[key] = None
        else:
            parent[key] = _coerce_scalar(value)
    return root


def _coerce_scalar(value: str) -> Any:
    """Convert a YAML scalar string to a Python value."""
    v = value.strip()
    if v.startswith(('"', "'")) and v.endswith(('"', "'")) and len(v) >= 2:
        return v[1:-1]
    low = v.lower()
    if low in ("null", "~", ""):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `overrides` into `base`."""
    out = dict(base)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------- Pipeline builder ----------

def load_pipeline(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> RAGPipeline:
    """Build a fully-wired pipeline from a YAML config file.

    Args:
        config_path: path to a YAML file.
        overrides: optional dict that is deep-merged on top of the YAML
                   (useful for tests or environment-specific tweaks).
    """
    cfg = _load_yaml(Path(config_path))
    if overrides:
        cfg = _deep_merge(cfg, overrides)

    backends_cfg = cfg.get("backends", {})
    pipeline_cfg = cfg.get("pipeline", {})
    confidence_cfg_raw = cfg.get("confidence", {})
    fallback_cfg_raw = cfg.get("fallback", {})
    cache_cfg = cfg.get("cache", {})

    # ConfidenceConfig
    confidence_config = ConfidenceConfig(
        half_life_days=confidence_cfg_raw.get("half_life_days", 365.0),
        source_quality=confidence_cfg_raw.get("source_quality") or {},
        default_source_quality=confidence_cfg_raw.get("default_source_quality", 0.6),
        w_freshness=confidence_cfg_raw.get("w_freshness", 0.25),
        w_source=confidence_cfg_raw.get("w_source", 0.35),
        w_consistency=confidence_cfg_raw.get("w_consistency", 0.40),
        aggregate_top_n=confidence_cfg_raw.get("aggregate_top_n", 5),
    )

    # FallbackConfig
    fallback_config = FallbackConfig(
        min_aggregate_confidence=fallback_cfg_raw.get("min_aggregate_confidence", 0.55),
        min_chunks=fallback_cfg_raw.get("min_chunks", 1),
        require_min_citations=fallback_cfg_raw.get("require_min_citations", 1),
        min_top_retrieval_score=fallback_cfg_raw.get("min_top_retrieval_score", 0.0),
    )

    # PipelineConfig
    pipeline_config = PipelineConfig(
        retriever_k=pipeline_cfg.get("retriever_k", 50),
        final_k=pipeline_cfg.get("final_k", 10),
        chunk_size=pipeline_cfg.get("chunk_size", 800),
        chunk_overlap=pipeline_cfg.get("chunk_overlap", 100),
        confidence=confidence_config,
        fallback=fallback_config,
        max_context_chunks=pipeline_cfg.get("max_context_chunks", 8),
        max_tokens=pipeline_cfg.get("max_tokens", 512),
        chunker_strategy=pipeline_cfg.get("chunker_strategy", "fixed_size"),
        chunker_settings=pipeline_cfg.get("chunker_settings") or None,
    )

    # Build backends from registries
    embedder_key = backends_cfg.get("embedder", "hash")
    embedder_settings = backends_cfg.get("embedder_settings", {})
    embedder = _lookup(EMBEDDERS, embedder_key, "embedder")(embedder_settings)

    store_key = backends_cfg.get("vector_store", "in_memory")
    store_settings = backends_cfg.get("vector_store_settings", {})
    vector_store = _lookup(VECTOR_STORES, store_key, "vector_store")(store_settings, embedder.dim)

    kw_key = backends_cfg.get("keyword_index", "bm25")
    kw_settings = backends_cfg.get("keyword_index_settings", {})
    keyword_index = _lookup(KEYWORD_INDEXES, kw_key, "keyword_index")(kw_settings)

    llm_key = backends_cfg.get("llm", "mock")
    llm_settings = backends_cfg.get("llm_settings", {})
    if "llm_model" in backends_cfg:  # convenience shortcut
        llm_settings = {**llm_settings, "model": backends_cfg["llm_model"]}
    llm = _lookup(LLMS, llm_key, "llm")(llm_settings)

    tracer_key = backends_cfg.get("tracer", "in_memory")
    tracer_settings = backends_cfg.get("tracer_settings", {})
    tracer = _lookup(TRACERS, tracer_key, "tracer")(tracer_settings)

    # Caches
    cache_backend = InMemoryTTLCache(max_entries=cache_cfg.get("max_entries", 10_000))
    query_cache = QueryCache(cache_backend, ttl_seconds=cache_cfg.get("query_ttl_seconds", 3600))
    cached_embedder = CachedEmbedder(
        embedder, EmbeddingCache(cache_backend, model_id=f"{embedder_key}:{embedder.dim}")
    )

    return RAGPipeline(
        embedder=cached_embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        llm=llm,
        tracer=tracer,
        query_cache=query_cache,
        config=pipeline_config,
    )


def _lookup(registry: dict[str, Any], key: str, kind: str) -> Any:
    if key not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown {kind} backend {key!r}. Available: {available}. "
            f"Use rag.config.register_{kind}() to add a new one."
        )
    return registry[key]
