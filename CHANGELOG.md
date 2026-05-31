# Changelog

All notable changes to ragzero will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-31

Initial PyPI release.

### Added
- Core RAG library with 10-step pipeline (ingest → retrieve → score → constrained-generate → fallback)
- Four RAG strategies: Simple, Graph, Agentic, Multilingual (composable)
- Pluggable LLM providers: Anthropic, OpenAI, Azure OpenAI, Gemini, OpenRouter, Mock
- Pluggable embedding providers: OpenAI, Azure OpenAI, Gemini, Voyage, OpenRouter, sentence-transformers, hash
- Free-text model selection with curated suggestions (no hard-coded model list)
- FastAPI server with multi-user session isolation
- React UI bundled in the package (no separate npm build needed for end users)
- Live progress streaming via SSE for long-running ingestion and graph-build jobs
- Knowledge graph backend (NetworkX in-memory + Neo4j stub)
- Numeric citation labels + grounding-based fallback (no false-refusal bug)
- **Publish a Space** — turn a configured session into a public, ask-only assistant
  served at `/a/<slug>`. Visitors can only ask questions and read answers; data,
  models, keys, and config are never exposed. The host controls the name, theme,
  welcome message, suggested questions, allowed modes, and which fields
  (citations / confidence) appear in answers.
- **Admin token** (`ragzero serve --admin-token` or `RAGZERO_ADMIN_TOKEN`) gating the
  console + `/api/*`; the public assistant stays open. Per-visitor rate limit + daily
  cap, plus an optional access code, for published assistants.
- Locked-down public API: `GET /public/meta`, `POST /public/query`.
- `ragzero` console script with `serve`, `info`, `query`, `version` subcommands
- 150 passing tests
