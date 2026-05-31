<div align="center">

# ragzero

**Production-ready RAG with near-zero hallucination — install and run in 60 seconds.**

[![PyPI version](https://img.shields.io/pypi/v/ragzero.svg)](https://pypi.org/project/ragzero/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-150%20passing-success.svg)](#testing)

</div>

A complete RAG system in one pip install: pluggable LLM/embedding providers, four composable retrieval strategies (Simple, Graph, Agentic, Multilingual), a FastAPI server, and a **pre-built React UI bundled inside the package**. No npm, no Docker, no configuration required for the basic case.

## Install and run

```bash
pip install "ragzero[all]"
ragzero serve
# → open http://127.0.0.1:8000
```

That's it. The web UI opens, you paste an API key, upload some documents, and ask questions.

## What you get

- **🚫 Near-zero hallucination** by design — constrained generation, citation enforcement, refusal-first fallback when evidence is thin
- **🔌 Pluggable providers** for both LLMs (Anthropic, OpenAI, Azure, Gemini, OpenRouter) and embeddings (OpenAI, Azure, Gemini, Voyage, OpenRouter, sentence-transformers) — pick any model name, not a hard-coded list
- **🧠 Four RAG strategies**, composable: Simple, Graph (vector + entity graph), Agentic (plan → search → reflect), Multilingual (ask in any language, answer in any language)
- **📊 Live progress UI** — see chunk-by-chunk embedding progress and graph build progress in real time via SSE
- **👥 Multi-user isolation** — each browser session has its own credentials, corpus, and config
- **📣 Publish a public assistant** — turn your configured setup into a shareable, ask-only page (`/a/<slug>`); visitors only ask and read answers, never seeing your data, models, or keys
- **🧪 150 tests** covering every pipeline step

## Install with the providers you actually use

The base install includes the server and the core library. Provider SDKs are extras:

```bash
# Minimal — only mock providers (good for testing)
pip install ragzero

# With one specific provider
pip install "ragzero[openai]"           # OpenAI, Azure OpenAI, and OpenRouter
pip install "ragzero[anthropic]"        # Claude
pip install "ragzero[gemini]"           # Google Gemini
pip install "ragzero[voyage]"           # Voyage AI embeddings
pip install "ragzero[local]"            # sentence-transformers (local embeddings)
pip install "ragzero[faiss]"            # FAISS vector store
pip install "ragzero[graph]"            # NetworkX for Graph RAG

# Everything
pip install "ragzero[all]"
```

## Usage

### As a CLI tool

```bash
ragzero serve --port 8000              # web UI + API
ragzero serve --admin-token SECRET     # gate the admin console (see "Publish" below)
ragzero info                           # which provider SDKs are installed
ragzero query --docs ./my-docs --query "What is RAG?"   # one-shot CLI query
ragzero version
```

### As a Python library

```python
from ragzero import build_dev_pipeline, Document
from datetime import datetime, timezone

pipeline = build_dev_pipeline()
pipeline.ingest_documents([
    Document(
        doc_id="d1",
        source="my-notes.txt",
        content="The sky is blue because of Rayleigh scattering...",
        doc_type="txt",
        created_at=datetime.now(timezone.utc),
    )
])
answer = pipeline.answer("Why is the sky blue?")
print(answer.text)
print(f"Confidence: {answer.confidence}")
for c in answer.citations:
    print(f"  - {c.source} (chunk {c.chunk_id})")
```

For real providers, build the pipeline from YAML:

```python
from ragzero.rag.config import load_pipeline

pipeline = load_pipeline("my_config.yaml", overrides={
    "backends": {
        "llm": "anthropic",
        "llm_settings": {"api_key": "sk-ant-...", "model": "claude-opus-4-7"},
        "embedder": "openai",
        "embedder_settings": {"api_key": "sk-...", "model": "text-embedding-3-large"},
    },
})
```

## The four RAG strategies

| Strategy | Best for |
|----------|----------|
| **Simple** | Standard Q&A over well-structured documents |
| **Graph** | Multi-hop questions ("what companies did X's co-founders work at?") |
| **Agentic** | Complex questions requiring decomposition + iterative search |
| **Multilingual** | Cross-language Q&A — wraps any other strategy |

All four compose: `Multilingual( Agentic( pipeline ) )` lets you ask in Hindi and have the agent reason in English and return the answer in Hindi.

## Publish a public assistant

Once you've connected a model and indexed your data, you can publish a **public, ask-only assistant** for other people — they only ask questions and read answers, and never see your data, models, keys, or settings.

1. Start the server with an admin token so the console is protected:
   ```bash
   ragzero serve --admin-token YOUR_SECRET        # or set RAGZERO_ADMIN_TOKEN
   ```
2. In the console, open the **Publish** tab. Set the name, accent color, welcome
   message, suggested questions, allowed modes, and whether citations/confidence are
   shown. Optionally add an access code and tune the per-visitor rate limit + daily cap.
3. Click **Publish**. Share the public link: `http://<host>/a/<slug>`.

The public page talks only to a locked-down API (`GET /public/meta`, `POST /public/query`)
that returns just the fields you allowed. To share beyond your own machine, bind to all
interfaces (`--host 0.0.0.0`) for your LAN, use a tunnel (`cloudflared tunnel --url
http://localhost:8000`) for the internet, or run it behind a reverse proxy on a server.

> ⚠️ Always set `--admin-token` before exposing the server beyond `localhost` — otherwise
> the admin console (and your data/keys) would be reachable too. The server warns you if
> you bind to a non-local address without one.

## Configuration

The bundled `config/dev.yaml` is loaded by default. Override with `--config` or `RAGZERO_CONFIG`:

```bash
ragzero serve --config /etc/ragzero/prod.yaml
# or
RAGZERO_CONFIG=/etc/ragzero/prod.yaml ragzero serve
```

Key knobs (all editable from the UI's **Config** tab too):

```yaml
pipeline:
  chunk_size: 1000
  chunk_overlap: 150

confidence:
  half_life_days: 180          # how fast freshness decays
  w_freshness: 0.30
  w_source: 0.40
  w_consistency: 0.30

fallback:
  min_aggregate_confidence: 0.65   # below this → refuse
  require_min_citations: 2
```

## Testing

```bash
git clone https://github.com/aasu14/ragzero.git
cd ragzero
pip install -e ".[dev,all]"
pytest                                  # 150 tests
python3 run_tests.py                    # alternative stdlib runner (no pytest)
```

## Security notes

- Session API keys are kept in memory only — never written to disk or logs
- HTTP-only `SameSite=Lax` session cookies
- **Admin console + `/api/*` can be gated** with `--admin-token` / `RAGZERO_ADMIN_TOKEN`; set it before exposing the server beyond `localhost`
- The **public assistant** has per-visitor rate limiting + a daily cap and an optional access code; the `/public/*` API never returns data, models, or keys
- A published Space is persisted to `~/.ragzero/space.json` (mode `0600`) and **includes the provider credentials** from the snapshot — protect that file / the host machine accordingly
- `POST /api/ingest/path` reads server-side filesystem — guard access in production
- No built-in CSRF — add tokens if exposing the admin API across origins; for heavy public traffic put nginx/Cloudflare in front

## License

MIT — see [LICENSE](LICENSE).
