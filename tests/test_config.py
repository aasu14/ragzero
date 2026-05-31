"""Tests for the YAML config loader and backend registry."""
from pathlib import Path

from ragzero.rag.config import (
    EMBEDDERS, LLMS, VECTOR_STORES,
    load_pipeline, register_embedder, register_llm,
)
from ragzero.rag.interfaces import LLM
from ragzero.rag.pipeline import RAGPipeline


REPO_ROOT = Path(__file__).parent.parent
# Bundled config lives inside the package
CONFIG_DIR = REPO_ROOT / "src" / "ragzero" / "config"


def test_load_dev_config_builds_pipeline():
    pipeline = load_pipeline(CONFIG_DIR / "dev.yaml")
    assert isinstance(pipeline, RAGPipeline)
    # Thresholds from dev.yaml
    assert pipeline.config.fallback.min_aggregate_confidence == 0.55
    assert pipeline.config.chunk_size == 800


def test_config_overrides_take_precedence():
    pipeline = load_pipeline(
        CONFIG_DIR / "dev.yaml",
        overrides={"pipeline": {"chunk_size": 1234}, "fallback": {"min_chunks": 5}},
    )
    assert pipeline.config.chunk_size == 1234
    assert pipeline.config.fallback.min_chunks == 5


def test_unknown_backend_raises_helpful_error():
    import pytest
    with pytest.raises(ValueError) as exc_info:
        load_pipeline(
            CONFIG_DIR / "dev.yaml",
            overrides={"backends": {"embedder": "nonexistent_backend"}},
        )
    msg = str(exc_info.value)
    assert "nonexistent_backend" in msg
    assert "Available" in msg


def test_register_custom_embedder_makes_it_loadable():
    """End-to-end: register a new backend, then load it from YAML."""
    from ragzero.rag.backends.embedders import HashEmbedder

    def _custom(settings):
        return HashEmbedder(dim=settings.get("dim", 128))

    register_embedder("my_custom", _custom)
    try:
        pipeline = load_pipeline(
            CONFIG_DIR / "dev.yaml",
            overrides={
                "backends": {
                    "embedder": "my_custom",
                    "embedder_settings": {"dim": 32},
                }
            },
        )
        # The embedder is wrapped in CachedEmbedder, so we check via the inner
        assert pipeline.retriever.embedder.inner.dim == 32
    finally:
        EMBEDDERS.pop("my_custom", None)


def test_pipeline_built_from_yaml_can_answer():
    """Smoke test: a pipeline built entirely from YAML actually works."""
    from ragzero.rag.ingest import Ingestor
    from datetime import datetime, timezone
    from ragzero.rag.interfaces import Document

    pipeline = load_pipeline(CONFIG_DIR / "dev.yaml")
    doc = Document(
        doc_id="t",
        source="internal://test.txt",
        content="BM25 is a keyword-based retrieval ranking function. "
                "It assigns scores based on term frequency and inverse document frequency.",
        doc_type="txt",
        created_at=datetime.now(timezone.utc),
    )
    pipeline.ingest_documents([doc])
    answer = pipeline.answer("What is BM25?")
    assert not answer.refused
    assert answer.text


def test_minimal_yaml_parser_handles_subset():
    """Verify the stdlib YAML fallback parses the project's config files."""
    from ragzero.rag.config import _minimal_yaml_parse
    text = """\
top_level: value
nested:
  key: 42
  flag: true
  ratio: 0.5
  empty_value:
  string_value: "hello world"
  null_value: null
deep:
  level1:
    level2:
      leaf: done
"""
    result = _minimal_yaml_parse(text)
    assert result["top_level"] == "value"
    assert result["nested"]["key"] == 42
    assert result["nested"]["flag"] is True
    assert result["nested"]["ratio"] == 0.5
    assert result["nested"]["empty_value"] is None
    assert result["nested"]["string_value"] == "hello world"
    assert result["nested"]["null_value"] is None
    assert result["deep"]["level1"]["level2"]["leaf"] == "done"


def test_registry_lists_known_backends():
    # Sanity check: the registries are populated at import
    assert "hash" in EMBEDDERS
    assert "in_memory" in VECTOR_STORES
    assert "mock" in LLMS
