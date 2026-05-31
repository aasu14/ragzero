# ragzero architecture

This is the architectural reference. For install/run instructions see `README.md`.

## The 10-stage near-zero-hallucination pipeline

The spine of `ragzero` is a hard-gated answering pipeline. Every query passes through
these stages in `RAGPipeline.answer()` (`src/ragzero/rag/pipeline.py`); each stage emits
a tracer event for observability.

| # | Stage | Where it lives |
|---|---|---|
| 1 | **Ingest + normalize** — parse files, dedup by content hash, version stably | `ragzero/rag/ingest.py` |
| 2 | **Chunk** — split docs into overlapping windows (pluggable strategy) | `ragzero/rag/chunkers/` |
| 3 | **Hybrid retrieve** — dense (vector) + sparse (BM25) with RRF fusion | `ragzero/rag/retrieval.py` |
| 4 | **Confidence score** — per-chunk and aggregate scoring | `ragzero/rag/confidence.py` |
| 5 | **Early-refuse gate** — refuse before generation if confidence is too low | `pipeline.py` (`fallback.early` event) |
| 6 | **Constrained generation** — LLM call locked to retrieved context | `ragzero/rag/generation.py` |
| 7 | **Citation extraction** — pull citations out of the model's response | `generation.py` + `ConstrainedGenerator` |
| 8 | **Final fallback gate** — refuse if citations missing or evidence insufficient | `ragzero/rag/fallback.py` |
| 9 | **Cache** — store final answer keyed by query (skipped when filters present) | `ragzero/rag/cache.py` |
| 10 | **Trace + finalize** — emit observability events, return `Answer` | `ragzero/rag/observability.py` |

These stages are the contract. Everything below is an **extension point** — pluggable
without touching the spine.

## Extension points

Each registry maps a string id → constructor. Pick which implementation to use via YAML
config or the API; defaults work without configuration.

### LLMs — `ragzero/rag/config.py::LLMS`
- `mock`, `anthropic`, `openai`, `azure_openai`, `gemini`, `openrouter`
- Settable via `POST /api/session/llm`

### Embedders — `ragzero/rag/config.py::EMBEDDERS`
- `hash`, `sentence_transformers`, `openai`, `azure_openai`, `voyage`, `gemini`, `openrouter`
- Settable via `POST /api/session/embedder`

### Vector stores — `ragzero/rag/config.py::VECTOR_STORES`
- `in_memory`, `faiss`, `qdrant`, `pinecone`, `weaviate`, `chroma`, `pgvector`, `azure_ai_search`
- Settable via `POST /api/session/vector_store`
- Each implements `add`, `search(query_vector, k, filters=None)`, `size`, `clear`

### Chunkers — `ragzero/rag/chunkers/__init__.py::CHUNKERS`
- `fixed_size`, `sentence`, `paragraph`, `markdown_header`, `token`, `recursive`, `per_row`
- Settable via `POST /api/session/chunker` or the `pipeline.chunker_strategy` YAML key
- New strategy = add a class with `.chunk(doc) -> list[Chunk]` + register it

### Retrieval filters — `ragzero/rag/filters.py`
- Unified format: `[{field, op, value}]` with ops `eq/ne/gt/gte/lt/lte/in/not_in/contains`
- `pipeline.answer(query, filters=[...])` plumbs them through
- Backends with native support translate; others use over-fetch + in-process post-filter
- Chroma has native `where`-clause translation (`vector_stores_external.py::_filters_to_chroma_where`)

### Metadata schema — `ragzero/rag/schema.py`
- Optional. Built-in fields (`source`, `doc_type`, `content_hash`, `created_at`, ...) auto-populated
- Custom fields settable via `POST /api/schema`
- `CAPABILITY_MATRIX` documents which vector stores support which flags

### Structured ingest — `ragzero/rag/structured.py`
- Format detection + type inference for CSV / JSON-array / JSONL
- `POST /api/ingest/structured/preview` returns inferred schema
- `POST /api/ingest/structured/commit` indexes each row as one Document with the
  `per_row` chunker

### Strategies — `ragzero/rag/strategies/`
- `simple`, `graph`, `agentic`, `multilingual`
- Each strategy reads filters from `ctx.options["filters"]` and forwards to
  `pipeline.answer()` or `retriever.search()` directly

## Sessions

`SessionState` (`ragzero/rag/sessions.py`) is the per-user request scope:
- Provider configs (LLM / embedder / vector store)
- Pipeline overrides
- Per-session `Ingestor` (so content-hash dedup doesn't leak across users and resets on Clear)
- `metadata_schema`, `chunker_strategy`, `chunker_settings`
- Cached `RAGPipeline` (rebuilt when provider changes; sources auto-re-ingested)

Switching providers no longer wipes `state.sources` — the pipeline rebuilds on next query
and re-ingests stored sources into the new backend transparently. `clear_sources` does
purge the active store (calling each backend's `clear()`).

## Adding a new vector store

1. Implement `VectorStore` in `ragzero/rag/backends/vector_stores_external.py` (add, search,
   size, clear). `search(query_vector, k, filters=None)` is optional; without it the
   retriever falls back to over-fetch + post-filter.
2. Register a constructor in `ragzero/rag/config.py::VECTOR_STORES`.
3. Add a `ProviderSpec` to `ragzero/rag/providers.py::VECTOR_STORE_PROVIDERS` so the
   UI catalog picks it up.
4. (Optional) Add a row to `schema.py::CAPABILITY_MATRIX` so the UI knows which flags
   to honor.

## Adding a new chunker

1. Implement `ChunkerLike` (just `.chunk(doc) -> list[Chunk]`) in
   `ragzero/rag/chunkers/<your_strategy>.py`.
2. Register a constructor in `ragzero/rag/chunkers/__init__.py::CHUNKERS` and add a
   `chunker_catalog` entry so the UI / `GET /api/chunkers` knows about it.

## Adding a new LLM / embedder

1. Implement the interface in `ragzero/rag/backends/llms.py` / `embedders.py`.
2. Register in `ragzero/rag/config.py::LLMS` or `EMBEDDERS`.
3. Add a `ProviderSpec` to `ragzero/rag/providers.py::LLM_PROVIDERS` / `EMBEDDER_PROVIDERS`.
