"""Simple RAG strategy.

The plain pipeline (retrieve → score → constrain → fallback). This is the
strategy v1 implemented. We expose its internal stages as events so the UI
treats it uniformly with the others.
"""
from __future__ import annotations

from typing import Iterator

from ..observability import InMemoryTracer
from .base import Strategy, StrategyContext, StrategyEvent, make_event


class SimpleStrategy(Strategy):
    id = "simple"
    label = "Simple RAG (retrieve → constrain → answer)"

    def run(self, query: str, ctx: StrategyContext) -> Iterator[StrategyEvent]:
        yield make_event("info", "Starting simple RAG", query=query)

        # Run the pipeline (filters set by UI on ctx.options)
        filters = ctx.options.get("filters") if ctx.options else None
        answer = ctx.pipeline.answer(query, filters=filters)

        # Replay tracer events as strategy events for the UI
        if isinstance(ctx.pipeline.tracer, InMemoryTracer):
            for ev in ctx.pipeline.tracer.events_for(answer.trace_id or ""):
                kind: str = "info"
                if ev["stage"] == "retrieval":
                    kind = "retrieval"
                elif ev["stage"] == "generation":
                    kind = "generation"
                elif ev["stage"] in ("fallback.early", "fallback.final"):
                    kind = "info"
                yield make_event(kind, ev["stage"], **ev["data"])

        yield make_event("final", "Done", answer=answer)
