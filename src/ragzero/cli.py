"""ragzero command-line interface.

Subcommands:
    ragzero serve              Start the web server (UI + API)
    ragzero version            Print the version
    ragzero info               Show installed providers and bundled UI status
    ragzero query --docs DIR --query "..."   One-shot CLI query (dev convenience)

The `serve` command is the primary entry point. The rest are convenience tools.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from ragzero import __version__


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print(
            "uvicorn not installed. Install with:\n"
            "  pip install 'ragzero[server]'\n"
            "or just:\n"
            "  pip install uvicorn",
            file=sys.stderr,
        )
        return 1

    # Allow config-path / ui-dist overrides via CLI flags as well as env vars
    if args.config:
        os.environ["RAGZERO_CONFIG"] = str(Path(args.config).resolve())
    if args.ui_dist:
        os.environ["RAGZERO_UI_DIST"] = str(Path(args.ui_dist).resolve())

    # Admin token gates the console + /api/*. Set via flag or RAGZERO_ADMIN_TOKEN.
    if args.admin_token:
        os.environ["RAGZERO_ADMIN_TOKEN"] = args.admin_token
    admin_token = os.environ.get("RAGZERO_ADMIN_TOKEN")

    # Safety: exposing the admin console on a non-local interface without a
    # token would also expose your data, model keys, and config to anyone who
    # can reach it. Warn loudly (the public assistant is fine; the admin isn't).
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    if args.host not in local_hosts and not admin_token:
        print(
            "\n" + "!" * 72 +
            "\n  WARNING: binding to a non-local address without an admin token.\n"
            f"  Anyone who can reach {args.host} can open the admin console and\n"
            "  see your data, models, and API keys.\n\n"
            "  Set one before sharing:  ragzero serve --admin-token <secret> ...\n"
            "  (or export RAGZERO_ADMIN_TOKEN). The public /a/<slug> page stays open.\n"
            + "!" * 72 + "\n",
            file=sys.stderr,
        )

    import uvicorn

    # Resolve the port: explicit --port wins; otherwise let the OS pick a free
    # one so we never collide with a stale process holding the old default.
    if args.port is not None:
        port = args.port
    else:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
            _s.bind((args.host, 0))
            port = _s.getsockname()[1]
        print(f"No --port given; serving on auto-assigned free port {port}")

    uvicorn.run(
        "ragzero.server.main:app",
        host=args.host,
        port=port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


def _info(args: argparse.Namespace) -> int:
    print(f"ragzero {__version__}")
    print()
    # Check which optional provider SDKs are available
    print("Provider SDK status:")
    sdks = [
        ("anthropic",            "anthropic",            "Anthropic Claude"),
        ("openai",               "openai",               "OpenAI / Azure OpenAI / OpenRouter"),
        ("google-generativeai",  "google.generativeai",  "Google Gemini"),
        ("voyageai",             "voyageai",             "Voyage AI embeddings"),
        ("sentence-transformers","sentence_transformers","Local embeddings"),
        ("faiss-cpu",            "faiss",                "FAISS vector store"),
        ("networkx",             "networkx",             "Graph RAG (in-memory)"),
        ("neo4j",                "neo4j",                "Graph RAG (Neo4j)"),
    ]
    for pip_name, import_name, label in sdks:
        try:
            spec = importlib.util.find_spec(import_name)
        except (ImportError, ValueError):
            # find_spec raises if a parent package (e.g. `google`) is missing
            # while probing a dotted name (e.g. `google.generativeai`).
            spec = None
        status = "installed" if spec else "not installed"
        marker = "✓" if spec else " "
        print(f"  [{marker}] {label:36s} ({pip_name}) — {status}")

    print()
    # Check bundled UI — tolerate a missing fastapi (so `ragzero info` works
    # even on a minimal install without server deps).
    try:
        from ragzero.server.main import UI_DIST
        ui_path = UI_DIST
    except ImportError:
        # fastapi not installed; compute the path directly
        from pathlib import Path as _P
        import ragzero as _r
        ui_path = _P(_r.__file__).parent / "ui" / "dist"
    if ui_path.exists() and (ui_path / "index.html").exists():
        print(f"Bundled UI: present at {ui_path}")
    else:
        print(f"Bundled UI: NOT FOUND at {ui_path}")
        print("  (run `cd ui && npm run build` to generate it, then rebuild the package)")
    return 0


def _version(args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _query(args: argparse.Namespace) -> int:
    """One-shot query against a directory of documents. Useful for sanity checks."""
    from ragzero.rag.factories import build_dev_pipeline

    docs = Path(args.docs)
    if not docs.exists():
        print(f"Path does not exist: {docs}", file=sys.stderr)
        return 1

    pipeline = build_dev_pipeline()
    n = pipeline.ingest_directory(docs)
    print(f"Indexed {n} document(s) from {docs}", file=sys.stderr)

    answer = pipeline.answer(args.query)
    if answer.refused:
        print(f"[refused] {answer.refusal_reason}")
        return 0
    print(answer.text)
    print()
    for c in answer.citations:
        print(f"  - {c.source} ({c.chunk_id})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ragzero",
        description="ragzero — production RAG with near-zero hallucination.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="Launch the web server (UI + API)")
    p_serve.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=None, help="port (default: auto-pick a free port; set explicitly to override)")
    p_serve.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev mode)")
    p_serve.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    p_serve.add_argument("--config", help="override YAML config path (default: bundled config/dev.yaml)")
    p_serve.add_argument("--ui-dist", help="override path to pre-built UI files")
    p_serve.add_argument("--admin-token", help="secret that gates the admin console + /api/* (required for safe non-local hosting)")
    p_serve.set_defaults(func=_serve)

    p_info = sub.add_parser("info", help="Show installed providers and bundled UI status")
    p_info.set_defaults(func=_info)

    p_ver = sub.add_parser("version", help="Print version and exit")
    p_ver.set_defaults(func=_version)

    p_q = sub.add_parser("query", help="One-shot CLI query against a directory of documents")
    p_q.add_argument("--docs", required=True, help="directory containing .txt/.md/.pdf files")
    p_q.add_argument("--query", required=True, help="the question to ask")
    p_q.set_defaults(func=_query)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())