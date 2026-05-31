"""Provider registry.

A declarative catalog of providers (LLM and embedding). Each entry describes
the credential fields the UI should render and how to validate them. The
server reads this to render dynamic forms; the user fills them in once,
they're stored per-session, and then any pipeline built for that session
uses those credentials.

Model fields use the `text_with_suggestions` type — users can type ANY
model identifier, with curated common choices shown as quick-pick suggestions.
This way the UI doesn't have to be updated every time a provider releases
a new model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


FieldType = Literal["text", "password", "url", "select", "text_with_suggestions"]


@dataclass(frozen=True)
class ProviderField:
    """One credential or config field a provider needs."""
    key: str
    label: str
    type: FieldType
    required: bool = True
    placeholder: str = ""
    default: str = ""
    options: list[str] = field(default_factory=list)  # for select / text_with_suggestions
    help: str = ""


@dataclass(frozen=True)
class ProviderSpec:
    """One provider available for either LLM, embedding, or vector_store role."""
    id: str                                # registry key, e.g. "openai"
    label: str                              # human label
    kind: Literal["llm", "embedder", "vector_store"]
    fields: list[ProviderField]
    requires_package: str | None = None     # pip package name, for status check
    description: str = ""


LLM_PROVIDERS: dict[str, ProviderSpec] = {
    "mock": ProviderSpec(
        id="mock",
        label="Mock LLM (for testing — no API key needed)",
        kind="llm",
        fields=[],
        description="Deterministic extractive mock. Useful for verifying the pipeline wiring without API costs.",
    ),
    "anthropic": ProviderSpec(
        id="anthropic",
        label="Anthropic (Claude)",
        kind="llm",
        requires_package="anthropic",
        fields=[
            ProviderField("api_key", "API key", "password", placeholder="sk-ant-..."),
            ProviderField(
                "model", "Model", "text_with_suggestions",
                default="claude-opus-4-7",
                placeholder="claude-opus-4-7",
                options=[
                    "claude-opus-4-7",
                    "claude-opus-4-6",
                    "claude-sonnet-4-6",
                    "claude-haiku-4-5-20251001",
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022",
                ],
                help="Type any Claude model identifier. Suggestions are common picks.",
            ),
        ],
    ),
    "openai": ProviderSpec(
        id="openai",
        label="OpenAI",
        kind="llm",
        requires_package="openai",
        fields=[
            ProviderField("api_key", "API key", "password", placeholder="sk-..."),
            ProviderField(
                "model", "Model", "text_with_suggestions",
                default="gpt-4o-mini",
                placeholder="gpt-4o-mini",
                options=[
                    "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4",
                    "gpt-3.5-turbo", "o1-mini", "o1-preview", "o3-mini",
                ],
                help="Any model your API key has access to. Suggestions are common picks.",
            ),
            ProviderField(
                "base_url", "Base URL (optional)", "url",
                required=False,
                placeholder="https://api.openai.com/v1",
                help="For OpenAI-compatible endpoints like local Ollama, vLLM, Together, etc.",
            ),
        ],
    ),
    "azure_openai": ProviderSpec(
        id="azure_openai",
        label="Azure OpenAI",
        kind="llm",
        requires_package="openai",
        fields=[
            ProviderField("api_key", "API key", "password"),
            ProviderField("endpoint", "Endpoint", "url", placeholder="https://your-resource.openai.azure.com"),
            ProviderField(
                "deployment", "Deployment name", "text",
                placeholder="my-gpt4-deployment",
                help="Your Azure deployment name (NOT the underlying model name).",
            ),
            ProviderField("api_version", "API version", "text", default="2024-02-01", required=False),
        ],
    ),
    "gemini": ProviderSpec(
        id="gemini",
        label="Google Gemini",
        kind="llm",
        requires_package="google.generativeai",
        fields=[
            ProviderField("api_key", "API key", "password", placeholder="AIza..."),
            ProviderField(
                "model", "Model", "text_with_suggestions",
                default="gemini-2.0-flash",
                placeholder="gemini-2.0-flash",
                options=[
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-thinking-exp",
                    "gemini-2.0-pro",
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                    "gemini-1.5-flash-8b",
                ],
                help="Any Gemini model. Get a key at https://aistudio.google.com/apikey.",
            ),
        ],
        description="Google's Gemini models via AI Studio. Free tier available.",
    ),
    "openrouter": ProviderSpec(
        id="openrouter",
        label="OpenRouter (any model)",
        kind="llm",
        requires_package="openai",
        fields=[
            ProviderField("api_key", "API key", "password", placeholder="sk-or-..."),
            ProviderField(
                "model", "Model", "text_with_suggestions",
                default="anthropic/claude-3.5-sonnet",
                placeholder="provider/model-name",
                options=[
                    # A small curated list — OpenRouter hosts hundreds of models.
                    "anthropic/claude-3.5-sonnet",
                    "anthropic/claude-3.5-haiku",
                    "openai/gpt-4o",
                    "openai/gpt-4o-mini",
                    "google/gemini-2.0-flash-exp:free",
                    "google/gemini-pro-1.5",
                    "meta-llama/llama-3.3-70b-instruct",
                    "meta-llama/llama-3.1-405b-instruct",
                    "mistralai/mistral-large",
                    "deepseek/deepseek-chat",
                    "qwen/qwen-2.5-72b-instruct",
                    "x-ai/grok-2-1212",
                ],
                help="Browse all models at https://openrouter.ai/models. Format: provider/model-name.",
            ),
            ProviderField(
                "site_url", "Site URL (optional)", "url", required=False,
                placeholder="https://your-app.com",
                help="Sent as HTTP-Referer for OpenRouter analytics.",
            ),
            ProviderField(
                "app_name", "App name (optional)", "text", required=False,
                placeholder="My RAG App",
                help="Sent as X-Title header for OpenRouter analytics.",
            ),
        ],
        description="One API key, hundreds of models from every major provider.",
    ),
}


EMBEDDER_PROVIDERS: dict[str, ProviderSpec] = {
    "hash": ProviderSpec(
        id="hash",
        label="Hash embedder (for testing — no API key needed)",
        kind="embedder",
        fields=[
            ProviderField("dim", "Dimensions", "text", default="64", required=False),
        ],
        description="Deterministic local embedder. Quality is poor; use for wiring tests only.",
    ),
    "sentence_transformers": ProviderSpec(
        id="sentence_transformers",
        label="Sentence Transformers (local, free)",
        kind="embedder",
        requires_package="sentence_transformers",
        fields=[
            ProviderField(
                "model", "Model", "text_with_suggestions",
                default="sentence-transformers/all-MiniLM-L6-v2",
                placeholder="sentence-transformers/all-MiniLM-L6-v2",
                options=[
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "sentence-transformers/all-MiniLM-L12-v2",
                    "BAAI/bge-small-en-v1.5",
                    "BAAI/bge-base-en-v1.5",
                    "BAAI/bge-large-en-v1.5",
                    "intfloat/e5-large-v2",
                    "thenlper/gte-large",
                ],
                help="Any HuggingFace sentence-transformer-compatible model.",
            ),
        ],
        description="Runs locally on CPU/GPU. First use downloads the model (~100MB-1GB).",
    ),
    "openai": ProviderSpec(
        id="openai",
        label="OpenAI embeddings",
        kind="embedder",
        requires_package="openai",
        fields=[
            ProviderField("api_key", "API key", "password", placeholder="sk-..."),
            ProviderField(
                "model", "Model", "text_with_suggestions",
                default="text-embedding-3-small",
                placeholder="text-embedding-3-small",
                options=[
                    "text-embedding-3-small",
                    "text-embedding-3-large",
                    "text-embedding-ada-002",
                ],
                help="Any embedding model your API key has access to.",
            ),
            ProviderField("base_url", "Base URL (optional)", "url", required=False),
        ],
    ),
    "azure_openai": ProviderSpec(
        id="azure_openai",
        label="Azure OpenAI embeddings",
        kind="embedder",
        requires_package="openai",
        fields=[
            ProviderField("api_key", "API key", "password"),
            ProviderField("endpoint", "Endpoint", "url"),
            ProviderField("deployment", "Deployment name", "text"),
            ProviderField("api_version", "API version", "text", default="2024-02-01", required=False),
            ProviderField("dim", "Dimensions", "text", default="1536", required=False),
        ],
    ),
    "voyage": ProviderSpec(
        id="voyage",
        label="Voyage AI (Anthropic-recommended)",
        kind="embedder",
        requires_package="voyageai",
        fields=[
            ProviderField("api_key", "API key", "password"),
            ProviderField(
                "model", "Model", "text_with_suggestions",
                default="voyage-3",
                placeholder="voyage-3",
                options=["voyage-3", "voyage-3-lite", "voyage-large-2", "voyage-code-2", "voyage-finance-2"],
                help="Any Voyage embedding model.",
            ),
        ],
    ),
    "gemini": ProviderSpec(
        id="gemini",
        label="Google Gemini embeddings",
        kind="embedder",
        requires_package="google.generativeai",
        fields=[
            ProviderField("api_key", "API key", "password", placeholder="AIza..."),
            ProviderField(
                "model", "Model", "text_with_suggestions",
                default="text-embedding-004",
                placeholder="text-embedding-004",
                options=["text-embedding-004", "embedding-001", "text-embedding-005"],
                help="Any Gemini embedding model.",
            ),
            ProviderField(
                "dim", "Dimensions", "text", default="768", required=False,
                help="Override the dimension if using a non-default model.",
            ),
        ],
    ),
    "openrouter": ProviderSpec(
        id="openrouter",
        label="OpenRouter embeddings (any provider)",
        kind="embedder",
        requires_package="openai",
        fields=[
            ProviderField("api_key", "API key", "password", placeholder="sk-or-..."),
            ProviderField(
                "model", "Model", "text_with_suggestions",
                default="openai/text-embedding-3-small",
                placeholder="provider/embedding-model",
                options=[
                    # Curated common picks — OpenRouter hosts more.
                    "openai/text-embedding-3-small",
                    "openai/text-embedding-3-large",
                    "openai/text-embedding-ada-002",
                    "google/gemini-embedding-001",
                    "qwen/qwen3-embedding-8b",
                    "qwen/qwen3-embedding-4b",
                    "baai/bge-m3",
                ],
                help="Browse all embedding models at https://openrouter.ai/models?output_modalities=embeddings",
            ),
            ProviderField(
                "dim", "Dimensions", "text", default="", required=False,
                placeholder="auto-detect from model",
                help="Leave empty to use the model's default. Override only if you know "
                     "the model supports a custom output dimension.",
            ),
            ProviderField(
                "site_url", "Site URL (optional)", "url", required=False,
                placeholder="https://your-app.com",
                help="Sent as HTTP-Referer for OpenRouter analytics.",
            ),
            ProviderField(
                "app_name", "App name (optional)", "text", required=False,
                placeholder="My RAG App",
                help="Sent as X-Title header for OpenRouter analytics.",
            ),
        ],
        description="One API key, multiple embedding providers (OpenAI, Google, Qwen, BGE, etc.)",
    ),
}


VECTOR_STORE_PROVIDERS: dict[str, ProviderSpec] = {
    "in_memory": ProviderSpec(
        id="in_memory",
        label="In-memory (brute force, no dependencies)",
        kind="vector_store",
        fields=[],
        description="Pure-Python cosine search. Fine up to ~50K vectors. Default for dev.",
    ),
    "faiss": ProviderSpec(
        id="faiss",
        label="FAISS (local, HNSW)",
        kind="vector_store",
        requires_package="faiss",
        fields=[
            ProviderField(
                "M", "HNSW M parameter", "text", default="32", required=False,
                help="Graph connectivity. Higher = more accurate, more memory. Default 32.",
            ),
        ],
        description="Facebook AI Similarity Search with HNSW. Scales to millions of vectors locally.",
    ),
    "qdrant": ProviderSpec(
        id="qdrant",
        label="Qdrant",
        kind="vector_store",
        requires_package="qdrant_client",
        fields=[
            ProviderField(
                "url", "Qdrant URL", "url",
                default="http://localhost:6333",
                placeholder="http://localhost:6333 or https://xyz.cloud.qdrant.io",
            ),
            ProviderField(
                "api_key", "API key", "password", required=False,
                help="Required for Qdrant Cloud; leave empty for local Docker.",
            ),
            ProviderField(
                "collection", "Collection name", "text", default="ragzero",
                help="Auto-created if it doesn't exist.",
            ),
        ],
        description="Open-source vector DB. Self-host with Docker or use Qdrant Cloud.",
    ),
    "pinecone": ProviderSpec(
        id="pinecone",
        label="Pinecone (serverless)",
        kind="vector_store",
        requires_package="pinecone",
        fields=[
            ProviderField("api_key", "API key", "password", placeholder="pcsk_..."),
            ProviderField(
                "index_name", "Index name", "text", default="ragzero",
                help="Auto-created on first add. Pinecone billing applies per index.",
            ),
            ProviderField(
                "cloud", "Cloud provider", "select", default="aws",
                options=["aws", "gcp", "azure"],
            ),
            ProviderField(
                "region", "Region", "text", default="us-east-1",
                placeholder="us-east-1, eu-west-1, etc.",
            ),
        ],
        description="Managed serverless vector DB. Free tier available.",
    ),
    "weaviate": ProviderSpec(
        id="weaviate",
        label="Weaviate",
        kind="vector_store",
        requires_package="weaviate",
        fields=[
            ProviderField(
                "url", "Weaviate URL", "url",
                default="http://localhost:8080",
                placeholder="http://localhost:8080 or https://xyz.weaviate.network",
            ),
            ProviderField(
                "api_key", "API key", "password", required=False,
                help="Required for Weaviate Cloud; leave empty for local.",
            ),
            ProviderField(
                "class_name", "Class name", "text", default="RagzeroChunk",
                help="Weaviate class for storing chunks. Must start with a capital letter.",
            ),
        ],
        description="Open-source vector DB with built-in schema support. Self-host or use Weaviate Cloud.",
    ),
    "chroma": ProviderSpec(
        id="chroma",
        label="Chroma",
        kind="vector_store",
        requires_package="chromadb",
        fields=[
            ProviderField(
                "host", "Server host (optional)", "text", required=False,
                placeholder="localhost",
                help="Leave empty to run Chroma in-process.",
            ),
            ProviderField("port", "Server port", "text", default="8000", required=False),
            ProviderField(
                "persist_directory", "Persist directory (optional)", "text", required=False,
                placeholder="./chroma_data",
                help="If set and no host, persists to disk. If empty, ephemeral in-memory.",
            ),
            ProviderField(
                "collection", "Collection name", "text", default="ragzero",
                help="Auto-created if it doesn't exist.",
            ),
        ],
        description="Lightweight embedded vector DB. Runs in-process, persistent on disk, or as a server.",
    ),
    "pgvector": ProviderSpec(
        id="pgvector",
        label="Postgres + pgvector",
        kind="vector_store",
        requires_package="psycopg",
        fields=[
            ProviderField(
                "dsn", "Connection string", "password",
                placeholder="postgresql://user:pass@host:5432/dbname",
                help="Postgres needs the pgvector extension. The adapter runs CREATE EXTENSION IF NOT EXISTS.",
            ),
            ProviderField(
                "table", "Table name", "text", default="ragzero_chunks",
                help="Alphanumeric/underscore only. Auto-created with an IVFFlat index.",
            ),
            ProviderField(
                "ivfflat_lists", "IVFFlat lists", "text", default="100", required=False,
                help="Higher = faster search, slower build. Rule of thumb: rows / 1000.",
            ),
        ],
        description="Use your existing Postgres as a vector DB. Production-grade with point-in-time recovery, replication, etc.",
    ),
    "azure_ai_search": ProviderSpec(
        id="azure_ai_search",
        label="Azure AI Search",
        kind="vector_store",
        requires_package="azure.search.documents",
        fields=[
            ProviderField(
                "endpoint", "Endpoint", "url",
                placeholder="https://your-search.search.windows.net",
            ),
            ProviderField("api_key", "Admin API key", "password"),
            ProviderField(
                "index_name", "Index name", "text", default="ragzero",
                help="Auto-created with HNSW vector field. Lowercase alphanumeric + dashes.",
            ),
        ],
        description="Microsoft Azure's enterprise search service. Includes hybrid search, semantic ranking, and RBAC.",
    ),
}


def provider_catalog() -> dict[str, list[dict]]:
    """Serialized catalog used by the UI."""
    def _ser(p: ProviderSpec) -> dict:
        return {
            "id": p.id,
            "label": p.label,
            "kind": p.kind,
            "description": p.description,
            "requires_package": p.requires_package,
            "fields": [
                {
                    "key": f.key, "label": f.label, "type": f.type,
                    "required": f.required, "placeholder": f.placeholder,
                    "default": f.default, "options": f.options, "help": f.help,
                }
                for f in p.fields
            ],
        }
    return {
        "llm": [_ser(p) for p in LLM_PROVIDERS.values()],
        "embedder": [_ser(p) for p in EMBEDDER_PROVIDERS.values()],
        "vector_store": [_ser(p) for p in VECTOR_STORE_PROVIDERS.values()],
    }
