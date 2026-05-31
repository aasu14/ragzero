"""Shared pytest fixtures."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:
    import pytest
    _fixture = pytest.fixture
except ImportError:  # pragma: no cover — used by the stdlib test runner
    def _fixture(fn):
        return fn

from ragzero.rag.factories import build_dev_pipeline
from ragzero.rag.interfaces import Document


def make_doc(
    doc_id: str,
    content: str,
    source: str | None = None,
    age_days: int = 0,
    doc_type: str = "txt",
) -> Document:
    return Document(
        doc_id=doc_id,
        source=source or f"internal://{doc_id}.txt",
        content=content,
        doc_type=doc_type,
        created_at=datetime.now(timezone.utc) - timedelta(days=age_days),
        metadata={},
        version=1,
    )


@_fixture
def sample_docs() -> list[Document]:
    return [
        make_doc(
            "rag-overview",
            "Retrieval-augmented generation, or RAG, combines a retrieval system "
            "with a language model. The retriever fetches relevant documents from "
            "a corpus, and the language model generates an answer grounded in "
            "those documents. This approach reduces hallucination compared to "
            "pure parametric models.",
        ),
        make_doc(
            "bm25-overview",
            "BM25 is a bag-of-words retrieval function used by search engines. "
            "It ranks documents based on the query terms appearing in each "
            "document, regardless of inter-relationships. BM25 is keyword-based.",
        ),
        make_doc(
            "vector-search",
            "Vector search uses dense embeddings to find semantically similar "
            "documents. Approximate nearest neighbor algorithms like HNSW make "
            "vector search scalable to millions of documents.",
        ),
        make_doc(
            "stale-doc",
            "This is an old document about RAG written long ago. It mentions RAG "
            "and retrieval but is significantly outdated.",
            age_days=2000,  # very stale
            source="internal://archive/stale.txt",
        ),
        make_doc(
            "unrelated",
            "Penguins are flightless birds native to the Southern Hemisphere. "
            "They are well adapted to aquatic life.",
        ),
    ]


@_fixture
def pipeline(sample_docs):
    p = build_dev_pipeline()
    p.ingest_documents(sample_docs)
    return p
