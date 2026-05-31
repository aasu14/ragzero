"""Tests for Step 1: ingestion and normalization."""
from datetime import datetime, timezone
from pathlib import Path

from ragzero.rag.ingest import Chunker, Ingestor
from ragzero.rag.interfaces import Document


def _doc(doc_id: str, content: str) -> Document:
    return Document(
        doc_id=doc_id,
        source=f"internal://{doc_id}",
        content=content,
        doc_type="txt",
        created_at=datetime.now(timezone.utc),
    )


def test_ingestor_deduplicates_identical_content(tmp_path: Path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("Hello world")
    f2.write_text("Hello world")

    ing = Ingestor()
    d1 = ing.ingest_path(f1)
    d2 = ing.ingest_path(f2)

    assert d1 is not None
    assert d2 is None  # duplicate skipped


def test_ingestor_versions_on_content_change(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("Version one")
    ing = Ingestor()
    d1 = ing.ingest_path(f)
    f.write_text("Version two")
    d2 = ing.ingest_path(f)
    assert d1.version == 1
    assert d2.doc_id == d1.doc_id
    assert d2.version == 2


def test_ingestor_normalizes_whitespace(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("foo    bar\n\n\n\nbaz   ")
    ing = Ingestor()
    d = ing.ingest_path(f)
    assert "    " not in d.content
    assert "\n\n\n" not in d.content


def test_chunker_short_doc_single_chunk():
    doc = _doc("short", "Just one sentence.")
    chunks = Chunker(chunk_size=800, overlap=100).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Just one sentence."
    assert chunks[0].position == 0


def test_chunker_long_doc_overlapping_chunks():
    text = ". ".join(f"sentence {i}" for i in range(200)) + "."
    doc = _doc("long", text)
    chunks = Chunker(chunk_size=200, overlap=40).chunk(doc)
    assert len(chunks) > 1
    # Chunks should be ordered
    for i, c in enumerate(chunks):
        assert c.position == i
    # Adjacent chunks should overlap (last words of i appear early in i+1)
    last_word = chunks[0].text.split()[-1]
    assert last_word in chunks[1].text


def test_chunker_assigns_unique_chunk_ids():
    doc = _doc("x", ". ".join(f"sent {i}" for i in range(50)))
    chunks = Chunker(chunk_size=100, overlap=20).chunk(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunker_invalid_config_raises():
    import pytest
    with pytest.raises(ValueError):
        Chunker(chunk_size=100, overlap=100)
