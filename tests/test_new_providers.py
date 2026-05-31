"""Tests for Gemini, OpenRouter, and free-text model handling."""
from ragzero.rag.config import EMBEDDERS, LLMS
from ragzero.rag.providers import EMBEDDER_PROVIDERS, LLM_PROVIDERS, provider_catalog


def test_gemini_in_llm_registry():
    assert "gemini" in LLMS
    assert "gemini" in LLM_PROVIDERS


def test_gemini_in_embedder_registry():
    assert "gemini" in EMBEDDERS
    assert "gemini" in EMBEDDER_PROVIDERS


def test_openrouter_in_llm_registry():
    assert "openrouter" in LLMS
    assert "openrouter" in LLM_PROVIDERS


def test_openrouter_in_embedder_registry():
    # OpenRouter added embeddings support; both roles now available
    assert "openrouter" in EMBEDDERS
    assert "openrouter" in EMBEDDER_PROVIDERS


def test_openrouter_embedder_suggestions_use_provider_prefix():
    """Same as the LLM test — OpenRouter model IDs are provider/model-name."""
    spec = EMBEDDER_PROVIDERS["openrouter"]
    model_field = next(f for f in spec.fields if f.key == "model")
    for opt in model_field.options:
        assert "/" in opt, f"OpenRouter embedding suggestion {opt!r} missing provider prefix"


def test_openrouter_embedder_has_dim_override():
    """User can override the embedding dim — some models support custom dims."""
    spec = EMBEDDER_PROVIDERS["openrouter"]
    field_map = {f.key: f for f in spec.fields}
    assert "dim" in field_map
    assert not field_map["dim"].required  # blank = auto-detect from default


def test_all_model_fields_use_text_with_suggestions():
    """Every provider's 'model' field should be free-text, not constrained select."""
    for prov in list(LLM_PROVIDERS.values()) + list(EMBEDDER_PROVIDERS.values()):
        for f in prov.fields:
            if f.key == "model":
                assert f.type == "text_with_suggestions", (
                    f"{prov.id}.model is type={f.type}, expected text_with_suggestions "
                    "(model names must be free-text)"
                )


def test_text_with_suggestions_has_suggestions():
    """text_with_suggestions fields should have a curated options list."""
    for prov in list(LLM_PROVIDERS.values()) + list(EMBEDDER_PROVIDERS.values()):
        for f in prov.fields:
            if f.type == "text_with_suggestions":
                assert f.options, f"{prov.id}.{f.key} has no suggestions"


def test_provider_catalog_serializes_new_field_type():
    """The catalog endpoint must include the new type for the UI."""
    cat = provider_catalog()
    found_text_with_suggestions = False
    for category in cat.values():
        for p in category:
            for f in p["fields"]:
                if f["type"] == "text_with_suggestions":
                    found_text_with_suggestions = True
                    assert "options" in f
                    assert len(f["options"]) > 0
    assert found_text_with_suggestions


def test_provider_catalog_lists_gemini_and_openrouter():
    cat = provider_catalog()
    llm_ids = {p["id"] for p in cat["llm"]}
    embed_ids = {p["id"] for p in cat["embedder"]}
    assert "gemini" in llm_ids
    assert "openrouter" in llm_ids
    assert "gemini" in embed_ids


def test_gemini_llm_provider_has_api_key():
    spec = LLM_PROVIDERS["gemini"]
    keys = {f.key for f in spec.fields}
    assert "api_key" in keys
    assert "model" in keys


def test_openrouter_provider_has_optional_analytics_fields():
    spec = LLM_PROVIDERS["openrouter"]
    field_map = {f.key: f for f in spec.fields}
    assert "api_key" in field_map and field_map["api_key"].required
    assert "model" in field_map and field_map["model"].required
    # Analytics fields are optional
    assert "site_url" in field_map and not field_map["site_url"].required
    assert "app_name" in field_map and not field_map["app_name"].required


def test_openrouter_model_suggestions_use_provider_prefix():
    """OpenRouter model names must use the 'provider/model-name' format."""
    spec = LLM_PROVIDERS["openrouter"]
    model_field = next(f for f in spec.fields if f.key == "model")
    for opt in model_field.options:
        assert "/" in opt, f"OpenRouter suggestion {opt!r} missing provider prefix"


def test_anthropic_model_field_accepts_arbitrary_string():
    """The anthropic model field should be text_with_suggestions, allowing any value."""
    spec = LLM_PROVIDERS["anthropic"]
    model_field = next(f for f in spec.fields if f.key == "model")
    assert model_field.type == "text_with_suggestions"
    # The default suggestion list is still curated
    assert "claude-opus-4-7" in model_field.options


def test_openai_model_field_accepts_arbitrary_string():
    spec = LLM_PROVIDERS["openai"]
    model_field = next(f for f in spec.fields if f.key == "model")
    assert model_field.type == "text_with_suggestions"


def test_gemini_embedder_dim_override():
    """User should be able to override the embedding dimension."""
    spec = EMBEDDER_PROVIDERS["gemini"]
    field_map = {f.key: f for f in spec.fields}
    assert "dim" in field_map
    assert not field_map["dim"].required  # has a default
