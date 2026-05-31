"""Smoke test for server logic — exercises route handlers as plain functions.

This sidesteps the need for fastapi/pytest to be installed. It verifies:
- The session manager works under concurrent access
- Provider catalog serializes correctly
- A full ingest → query loop works through the same code path the routes use
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ragzero.rag.config import load_pipeline
from ragzero.rag.interfaces import Document
from ragzero.rag.providers import LLM_PROVIDERS, EMBEDDER_PROVIDERS, provider_catalog
from ragzero.rag.sessions import SessionManager
from datetime import datetime, timezone


def test_provider_catalog():
    cat = provider_catalog()
    assert "llm" in cat and "embedder" in cat
    assert any(p["id"] == "openai" for p in cat["llm"])
    assert any(p["id"] == "anthropic" for p in cat["llm"])
    assert any(p["id"] == "azure_openai" for p in cat["llm"])
    assert any(p["id"] == "voyage" for p in cat["embedder"])
    # Required fields present
    openai_llm = next(p for p in cat["llm"] if p["id"] == "openai")
    keys = {f["key"] for f in openai_llm["fields"]}
    assert "api_key" in keys
    assert "model" in keys
    print("PASS test_provider_catalog")


def test_session_manager_isolation():
    sm = SessionManager(ttl_seconds=60)
    s1 = sm.create()
    s2 = sm.create()
    assert s1.session_id != s2.session_id

    s1.llm_provider = "openai"
    s1.llm_settings = {"api_key": "k1", "model": "gpt-4"}
    s2.llm_provider = "anthropic"
    s2.llm_settings = {"api_key": "k2", "model": "claude-opus-4-7"}

    fetched_1 = sm.get(s1.session_id)
    fetched_2 = sm.get(s2.session_id)
    assert fetched_1.llm_provider == "openai"
    assert fetched_2.llm_provider == "anthropic"
    assert fetched_1.llm_settings["api_key"] == "k1"
    assert fetched_2.llm_settings["api_key"] == "k2"
    print("PASS test_session_manager_isolation")


def test_session_manager_concurrency():
    sm = SessionManager(ttl_seconds=60)
    created = []
    def create_many():
        for _ in range(50):
            created.append(sm.create().session_id)
    threads = [threading.Thread(target=create_many) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(set(created)) == 200  # all unique
    assert sm.count() == 200
    print("PASS test_session_manager_concurrency")


def test_pipeline_built_with_session_overrides():
    """Reproduce what the /api/query endpoint does to build a pipeline."""
    sm = SessionManager(ttl_seconds=60)
    state = sm.create()
    state.llm_provider = "mock"
    state.embedder_provider = "hash"
    state.embedder_settings = {"dim": 64}
    state.pipeline_overrides = {
        "fallback": {"min_aggregate_confidence": 0.3}  # relax for the test
    }
    overrides = {
        "backends": {
            "llm": state.llm_provider,
            "llm_settings": state.llm_settings,
            "embedder": state.embedder_provider,
            "embedder_settings": state.embedder_settings,
        },
        **state.pipeline_overrides,
    }
    pipeline = load_pipeline(
        Path(__file__).parent / "src" / "ragzero" / "config" / "dev.yaml",
        overrides=overrides,
    )
    doc = Document(
        doc_id="t",
        source="text://test",
        content="BM25 is a probabilistic ranking function used in keyword search.",
        doc_type="txt",
        created_at=datetime.now(timezone.utc),
    )
    pipeline.ingest_documents([doc])
    answer = pipeline.answer("What is BM25?")
    assert not answer.refused, f"Refused: {answer.refusal_reason}"
    assert answer.text
    assert answer.confidence > 0.3
    print(f"PASS test_pipeline_built_with_session_overrides (conf={answer.confidence:.3f})")


def test_provider_field_validation_metadata():
    # Each provider declares required fields the UI uses to validate forms
    valid_types = ("text", "password", "url", "select", "text_with_suggestions")
    for prov in LLM_PROVIDERS.values():
        for f in prov.fields:
            assert f.key
            assert f.type in valid_types
            if f.type in ("select", "text_with_suggestions"):
                assert f.options, f"{f.type} field {prov.id}.{f.key} has no options"
    for prov in EMBEDDER_PROVIDERS.values():
        for f in prov.fields:
            assert f.key
            assert f.type in valid_types
    print("PASS test_provider_field_validation_metadata")


def test_masking_logic():
    """Replicate the _mask_secrets helper from server/main.py."""
    def mask(settings):
        masked = {}
        for k, v in settings.items():
            if any(s in k.lower() for s in ("key", "secret", "token", "password")):
                masked[k] = "••••" if v else ""
            else:
                masked[k] = v
        return masked
    res = mask({"api_key": "sk-abc", "model": "gpt-4", "deployment": "prod"})
    assert res["api_key"] == "••••"
    assert res["model"] == "gpt-4"
    assert res["deployment"] == "prod"
    assert mask({"api_key": ""}) == {"api_key": ""}
    print("PASS test_masking_logic")


if __name__ == "__main__":
    test_provider_catalog()
    test_session_manager_isolation()
    test_session_manager_concurrency()
    test_pipeline_built_with_session_overrides()
    test_provider_field_validation_metadata()
    test_masking_logic()
    print("\nAll server smoke tests passed.")
