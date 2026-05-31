"""Step 10: Observability.

Every request gets a `trace_id`. Every pipeline stage emits a structured
event with that trace_id. This lets you reconstruct exactly what
happened on any given request:

  trace.events_for(trace_id) ->
    [ingest, retrieval, confidence, generation, fallback, finalize]

The default backend is in-memory (bounded ring buffer) for tests and
local dev. In production, swap to OpenTelemetry, Datadog, or your
logging stack of choice via the Tracer interface.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

from .interfaces import Tracer


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


class InMemoryTracer(Tracer):
    """Bounded in-memory tracer suitable for tests and small deployments."""

    def __init__(self, max_events: int = 10_000) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def emit(self, trace_id: str, stage: str, data: dict[str, Any]) -> None:
        event = {
            "trace_id": trace_id,
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        with self._lock:
            self._events.append(event)

    def events_for(self, trace_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self._events if e["trace_id"] == trace_id]

    def all_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)


class LoggingTracer(Tracer):
    """Tracer that pipes events to stdlib logging as structured JSON.

    Use this as the production default — your log aggregator (Datadog,
    Loki, CloudWatch) will index the JSON fields.
    """

    def __init__(self, logger_name: str = "rag.trace") -> None:
        self.logger = logging.getLogger(logger_name)

    def emit(self, trace_id: str, stage: str, data: dict[str, Any]) -> None:
        payload = {
            "trace_id": trace_id,
            "stage": stage,
            "ts": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        self.logger.info(json.dumps(payload, default=str))

    def events_for(self, trace_id: str) -> list[dict[str, Any]]:
        # LoggingTracer is fire-and-forget — events live in the log pipeline.
        return []
