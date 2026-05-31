"""Pipeline factories.

Convenience builders that wire concrete backends into a pipeline. Keeps
config files small — most users just call `build_dev_pipeline()` and
override what they need.
"""
from __future__ import annotations

from .backends.caches import InMemoryTTLCache
from .backends.embedders import HashEmbedder
from .backends.llms import MockLLM
from .backends.stores import BM25KeywordIndex, InMemoryVectorStore
from .cache import EmbeddingCache, QueryCache, CachedEmbedder
from .observability import InMemoryTracer, LoggingTracer
from .pipeline import PipelineConfig, RAGPipeline


def build_dev_pipeline(config: PipelineConfig | None = None) -> RAGPipeline:
    """Fully in-memory pipeline for tests and local dev."""
    cache_backend = InMemoryTTLCache(max_entries=1000)
    embedder = HashEmbedder(dim=64)
    cached_embedder = CachedEmbedder(
        embedder, EmbeddingCache(cache_backend, model_id="hash-64")
    )
    return RAGPipeline(
        embedder=cached_embedder,
        vector_store=InMemoryVectorStore(),
        keyword_index=BM25KeywordIndex(),
        llm=MockLLM(),
        tracer=InMemoryTracer(),
        query_cache=QueryCache(cache_backend, ttl_seconds=3600),
        config=config,
    )


def build_prod_pipeline(config: PipelineConfig | None = None) -> RAGPipeline:  # pragma: no cover
    """Production pipeline: sentence-transformers + FAISS + Anthropic + logging tracer."""
    from .backends.embedders import SentenceTransformerEmbedder
    from .backends.llms import AnthropicLLM
    from .backends.stores import FaissVectorStore

    cache_backend = InMemoryTTLCache(max_entries=100_000)
    embedder = SentenceTransformerEmbedder()
    cached_embedder = CachedEmbedder(
        embedder, EmbeddingCache(cache_backend, model_id="all-MiniLM-L6-v2")
    )
    return RAGPipeline(
        embedder=cached_embedder,
        vector_store=FaissVectorStore(dim=embedder.dim),
        keyword_index=BM25KeywordIndex(),  # swap for OpenSearch in real prod
        llm=AnthropicLLM(model="claude-opus-4-7"),
        tracer=LoggingTracer(),
        query_cache=QueryCache(cache_backend, ttl_seconds=3600),
        config=config,
    )
