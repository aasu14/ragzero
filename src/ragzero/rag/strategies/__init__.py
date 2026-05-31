"""Strategy registry + preset compositions.

The 4 named presets the UI exposes:
- simple
- graph
- agentic
- multilingual (wraps simple by default; can wrap any other strategy)

The compose() helper lets users build arbitrary stacks (e.g. multilingual
on top of agentic on top of graph).
"""
from __future__ import annotations

from typing import Any

from .agentic import AgenticStrategy
from .base import (
    Strategy,
    StrategyContext,
    StrategyEvent,
    extract_answer,
    make_event,
)
from .graph import GraphStrategy
from .multilingual import MultilingualStrategy
from .simple import SimpleStrategy


# Registry of base (non-decorator) strategies
BASE_STRATEGIES: dict[str, type[Strategy]] = {
    "simple": SimpleStrategy,
    "graph": GraphStrategy,
    "agentic": AgenticStrategy,
}


def build_strategy(
    mode: str,
    *,
    multilingual: bool = False,
    translator=None,
    output_language: str = "auto",
    agent_max_iterations: int = 3,
    graph_max_depth: int = 2,
) -> Strategy:
    """Construct a strategy stack from named knobs.

    `mode` is one of: 'simple', 'graph', 'agentic'.
    `multilingual=True` wraps the chosen mode with translation.
    """
    if mode not in BASE_STRATEGIES:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(BASE_STRATEGIES)}.")

    if mode == "agentic":
        inner: Strategy = AgenticStrategy(max_iterations=agent_max_iterations)
    elif mode == "graph":
        inner = GraphStrategy(max_graph_depth=graph_max_depth)
    else:
        inner = SimpleStrategy()

    if multilingual:
        if translator is None:
            raise ValueError("multilingual=True requires a translator")
        return MultilingualStrategy(
            inner=inner,
            translator=translator,
            output_language=output_language,
        )
    return inner


# Preset configurations the UI exposes as one-click choices
PRESETS = {
    "simple": {"mode": "simple", "multilingual": False},
    "graph": {"mode": "graph", "multilingual": False},
    "agentic": {"mode": "agentic", "multilingual": False},
    "multilingual": {"mode": "simple", "multilingual": True},
}


__all__ = [
    "AgenticStrategy",
    "BASE_STRATEGIES",
    "GraphStrategy",
    "MultilingualStrategy",
    "PRESETS",
    "SimpleStrategy",
    "Strategy",
    "StrategyContext",
    "StrategyEvent",
    "build_strategy",
    "extract_answer",
    "make_event",
]
