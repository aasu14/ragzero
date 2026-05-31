"""Tests for Steps 2-3: hybrid retrieval."""
from datetime import datetime, timezone

from ragzero.rag.backends.embedders import HashEmbedder
from ragzero.rag.backends.stores import BM25KeywordIndex, InMemoryVectorStore
from ragzero.rag.ingest import Chunker
from ragzero.rag.interfaces import Document
from ragzero.rag.retrieval import HybridRetriever


def _doc(doc_id: str, content: str) -> Document:
    return Document(
        doc_id=doc_id,
        source=f"internal://{doc_id}",
        content=content,
        doc_type="txt",
        created_at=datetime.now(timezone.utc),
    )


def _build_retriever() -> HybridRetriever:
    return HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=InMemoryVectorStore(),
        keyword_index=BM25KeywordIndex(),
        k=20,
        final_k=5,
    )


def test_retriever_returns_relevant_chunks():
    retriever = _build_retriever()
    chunker = Chunker(chunk_size=200, overlap=20)
    docs = [
        _doc("rag", "Retrieval augmented generation uses retrieval to ground LLMs."),
        _doc("bm25", "BM25 is a classic keyword search ranking function."),
        _doc("birds", "Penguins are flightless birds in the southern hemisphere."),
    ]
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunker.chunk(d))
    retriever.index(all_chunks)

    result = retriever.search("BM25 keyword search")
    assert result.chunks
    # bm25 doc should be in the top result
    assert result.chunks[0].chunk.doc_id == "bm25"


def test_retriever_rrf_fuses_both_lists():
    retriever = _build_retriever()
    chunker = Chunker(chunk_size=200, overlap=20)
    docs = [
        _doc("a", "alpha beta gamma delta epsilon"),
        _doc("b", "alpha alpha alpha"),
        _doc("c", "completely different content here"),
    ]
    chunks = []
    for d in docs:
        chunks.extend(chunker.chunk(d))
    retriever.index(chunks)

    result = retriever.search("alpha")
    # Both BM25 and dense should have produced results
    assert result.bm25_ranks
    assert result.dense_ranks
    # Top result should be one of the alpha-heavy docs
    assert result.chunks[0].chunk.doc_id in {"a", "b"}


def test_retriever_caps_at_final_k():
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=InMemoryVectorStore(),
        keyword_index=BM25KeywordIndex(),
        k=20,
        final_k=2,
    )
    chunker = Chunker(chunk_size=100, overlap=10)
    docs = [_doc(f"d{i}", f"content number {i} with some text") for i in range(10)]
    chunks = []
    for d in docs:
        chunks.extend(chunker.chunk(d))
    retriever.index(chunks)

    result = retriever.search("content text")
    assert len(result.chunks) <= 2


def test_retriever_empty_index_returns_empty():
    retriever = _build_retriever()
    result = retriever.search("anything")
    assert result.chunks == []


def test_bm25_handles_zero_query_terms():
    idx = BM25KeywordIndex()
    chunker = Chunker(chunk_size=100, overlap=10)
    docs = [_doc("x", "some text here")]
    chunks = chunker.chunk(docs[0])
    idx.add(chunks)
    # Punctuation-only query → no tokens
    assert idx.search("!!!", 5) == []
