"""CLI demo: ingest a directory and answer queries interactively.

Usage:
    python -m rag.cli --docs ./data --query "What is RAG?"
    python -m rag.cli --docs ./data --config config/dev.yaml
    python -m rag.cli --docs ./data    # default: dev config
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_pipeline
from .factories import build_dev_pipeline


def _print_answer(answer) -> None:
    print(f"\n--- trace_id={answer.trace_id} confidence={answer.confidence:.3f} ---")
    if answer.refused:
        print(f"[REFUSED] {answer.refusal_reason}")
        return
    print(answer.text)
    if answer.citations:
        print("\nCitations:")
        seen = set()
        for c in answer.citations:
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            page = f", page {c.page}" if c.page is not None else ""
            print(f"  - {c.source}{page} ({c.chunk_id})")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG pipeline demo")
    parser.add_argument("--docs", type=Path, required=True, help="Directory of documents to ingest")
    parser.add_argument("--query", type=str, help="Single query (omit for interactive mode)")
    parser.add_argument("--config", type=Path, help="YAML config file (default: dev factory)")
    args = parser.parse_args()

    if not args.docs.exists():
        print(f"Error: docs directory {args.docs} does not exist", file=sys.stderr)
        return 1

    if args.config:
        print(f"Loading config from {args.config}...")
        pipeline = load_pipeline(args.config)
    else:
        pipeline = build_dev_pipeline()

    print(f"Ingesting documents from {args.docs}...")
    n = pipeline.ingest_directory(args.docs)
    print(f"Indexed {n} document(s).")

    if args.query:
        _print_answer(pipeline.answer(args.query))
        return 0

    print("\nInteractive mode. Type 'quit' to exit.")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if query.lower() in ("quit", "exit", "q"):
            return 0
        if not query:
            continue
        _print_answer(pipeline.answer(query))


if __name__ == "__main__":
    sys.exit(main())
