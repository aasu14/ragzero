"""Strategy interface — the contract every RAG mode implements.

A Strategy takes a query + context (the pipeline's services) and emits a
sequence of StrategyEvent objects, ending with one of type "final" that
carries the Answer.

This lets the UI stream the agent's "thinking" in real time. Even
non-streaming strategies (SimpleStrategy) emit the same event shape, so
the UI is consistent across modes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Literal

from ..interfaces import Answer


EventKind = Literal[
    "thought",        # the agent's internal reasoning
    "tool_call",      # invoking a sub-step (retrieve, search graph, translate)
    "tool_result",    # the result of a tool call
    "retrieval",      # retrieval results with chunk IDs
    "translation",    # multilingual: query/answer translation
    "graph_hop",      # graph traversal step
    "generation",     # final LLM call payload
    "info",           # general status update
    "final",          # carries the Answer
    "error",
]


@dataclass(frozen=True)
class StrategyEvent:
    """One step of work the strategy did. Streamable to UI."""
    kind: EventKind
    timestamp: str
    label: str
    data: dict[str, Any] = field(default_factory=dict)


def make_event(kind: EventKind, label: str, **data: Any) -> StrategyEvent:
    return StrategyEvent(
        kind=kind,
        timestamp=datetime.now(timezone.utc).isoformat(),
        label=label,
        data=data,
    )


@dataclass
class StrategyContext:
    """Services + state the strategy can use. Built fresh per query."""
    pipeline: Any                          # the underlying RAGPipeline (steps 1-7)
    tracer: Any                            # InMemoryTracer
    trace_id: str
    options: dict[str, Any] = field(default_factory=dict)  # strategy-specific knobs


class Strategy(ABC):
    """Base class for RAG strategies. Implementations yield events."""

    id: str = "base"
    label: str = "Base"

    @abstractmethod
    def run(self, query: str, ctx: StrategyContext) -> Iterator[StrategyEvent]:
        """Execute the strategy for one query.

        The final event MUST be kind='final' with data['answer'] = Answer.
        """


def extract_answer(events: list[StrategyEvent]) -> Answer | None:
    """Pull the Answer out of an event stream."""
    for e in reversed(events):
        if e.kind == "final" and "answer" in e.data:
            return e.data["answer"]
    return None
