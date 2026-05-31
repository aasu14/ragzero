"""ragzero — production-ready RAG with near-zero hallucination.

This package bundles three things:
- `ragzero.rag` — the core RAG library (pipeline, strategies, providers)
- `ragzero.server` — the FastAPI server + UI
- A pre-built React UI in `ragzero/ui/dist`

Quick start:
    >>> from ragzero import build_dev_pipeline
    >>> pipeline = build_dev_pipeline()
    >>> # ... ingest documents and query

Or launch the full web UI:
    $ ragzero serve
"""
__version__ = "0.1.0"

# Re-export the most common library entry points so:
#   from ragzero import RAGPipeline, build_dev_pipeline
# works directly.
from ragzero.rag import (
    Answer,
    Chunk,
    Citation,
    Document,
    PipelineConfig,
    RAGPipeline,
    ScoredChunk,
)
from ragzero.rag.factories import build_dev_pipeline

__all__ = [
    "__version__",
    "Answer",
    "Chunk",
    "Citation",
    "Document",
    "PipelineConfig",
    "RAGPipeline",
    "ScoredChunk",
    "build_dev_pipeline",
]
