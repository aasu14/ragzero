"""Job manager.

A lightweight, thread-safe progress tracker for long-running operations.
Each job has:
- A unique ID and a kind ("ingest", "graph_build", etc.)
- A current phase (e.g. "parsing", "embedding", "extracting")
- Progress numerator + denominator (so the UI can render a percentage)
- A rolling log of events with timestamps
- A terminal state: completed, failed, or cancelled

The UI subscribes via SSE: while the job is running, new events stream as
they're emitted. Jobs are retained in memory for `retention_seconds` after
completion so the UI can fetch the final state if it reconnects late.

Design choice: jobs use a `queue.Queue` per subscriber rather than a single
shared queue. This lets multiple browser tabs watch the same job, and lets
a late-joining subscriber replay the buffered events before going live.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal


JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


@dataclass
class JobEvent:
    """One step of progress."""
    kind: str                          # "phase", "progress", "log", "done", "error"
    timestamp: float
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "timestamp": self.timestamp,
            "message": self.message,
            "data": self.data,
        }


@dataclass
class Job:
    job_id: str
    kind: str                          # "ingest_upload", "graph_build", ...
    session_id: str
    status: JobStatus = "pending"
    phase: str = "queued"
    current: int = 0                   # progress numerator
    total: int = 0                     # progress denominator (0 = indeterminate)
    started_at: float = 0.0
    finished_at: float = 0.0
    result: Any = None                 # set on success
    error: str | None = None           # set on failure
    events: list[JobEvent] = field(default_factory=list)
    # Threading primitives — Queue per subscriber so multi-watcher works
    _subscribers: list[queue.Queue[JobEvent | None]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cancel_requested: bool = False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": self.job_id,
                "kind": self.kind,
                "status": self.status,
                "phase": self.phase,
                "current": self.current,
                "total": self.total,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": self.error,
                # Result is omitted from the snapshot to keep it small.
                # Callers fetch it explicitly via `Job.result` after status == completed.
            }


class JobManager:
    """Thread-safe registry of running and recently-finished jobs."""

    def __init__(self, retention_seconds: int = 600, max_buffered_events: int = 5000) -> None:
        self.retention_seconds = retention_seconds
        self.max_buffered_events = max_buffered_events
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, kind: str, session_id: str) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex[:16],
            kind=kind,
            session_id=session_id,
            started_at=time.time(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_for_session(self, session_id: str) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.session_id == session_id]

    def gc(self) -> int:
        cutoff = time.time() - self.retention_seconds
        removed = 0
        with self._lock:
            for jid in list(self._jobs.keys()):
                j = self._jobs[jid]
                if j.status in ("completed", "failed", "cancelled") and j.finished_at < cutoff:
                    del self._jobs[jid]
                    removed += 1
        return removed

    # ---- Job-side helpers used by the worker code ----

    @staticmethod
    def emit(
        job: Job,
        kind: str,
        message: str,
        *,
        phase: str | None = None,
        current: int | None = None,
        total: int | None = None,
        **data: Any,
    ) -> None:
        """Record an event AND fan it out to all subscribers."""
        with job._lock:
            if phase is not None:
                job.phase = phase
            if current is not None:
                job.current = current
            if total is not None:
                job.total = total
            event = JobEvent(
                kind=kind,
                timestamp=time.time(),
                message=message,
                data={
                    **data,
                    "phase": job.phase,
                    "current": job.current,
                    "total": job.total,
                },
            )
            job.events.append(event)
            # Trim retained event log if it grows large
            if len(job.events) > 5000:
                job.events = job.events[-2000:]
            subscribers = list(job._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # slow subscriber; drop

    @staticmethod
    def mark_running(job: Job, phase: str = "running") -> None:
        with job._lock:
            job.status = "running"
            job.phase = phase

    @staticmethod
    def mark_done(job: Job, result: Any = None) -> None:
        with job._lock:
            job.status = "completed"
            job.phase = "done"
            job.finished_at = time.time()
            job.result = result
            subscribers = list(job._subscribers)
        JobManager.emit(job, "done", "Job completed", phase="done")
        # Signal end-of-stream to all subscribers
        for q in subscribers:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

    @staticmethod
    def mark_failed(job: Job, error: str) -> None:
        with job._lock:
            job.status = "failed"
            job.phase = "failed"
            job.finished_at = time.time()
            job.error = error
            subscribers = list(job._subscribers)
        JobManager.emit(job, "error", error, phase="failed", error=error)
        for q in subscribers:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

    @staticmethod
    def request_cancel(job: Job) -> None:
        with job._lock:
            job._cancel_requested = True

    @staticmethod
    def cancel_requested(job: Job) -> bool:
        with job._lock:
            return job._cancel_requested

    # ---- Subscriber-side: SSE generator ----

    def subscribe(self, job: Job) -> Iterator[dict[str, Any]]:
        """Yield events for a job until it terminates.

        Replays the existing buffered events first so a late subscriber sees
        the full timeline, then streams live events until status changes to
        a terminal state.
        """
        q: queue.Queue[JobEvent | None] = queue.Queue(maxsize=self.max_buffered_events)
        # Replay buffered events under the lock — and check status atomically.
        # If we don't snapshot status while holding the lock, a job could finish
        # between "replay events" and "register subscriber" and we'd hang waiting
        # for a sentinel that already fired.
        with job._lock:
            replay = list(job.events)
            already_done = job.status in ("completed", "failed", "cancelled")
            if not already_done:
                job._subscribers.append(q)

        for ev in replay:
            yield ev.to_dict()

        if already_done:
            # Emit a synthetic terminal event so the UI's "stream finished" handler runs
            yield {
                "kind": "terminal",
                "timestamp": time.time(),
                "message": f"Job is {job.status}",
                "data": {"status": job.status, "error": job.error},
            }
            return

        try:
            while True:
                try:
                    ev = q.get(timeout=30.0)
                except queue.Empty:
                    # Heartbeat so proxies don't close the connection
                    yield {"kind": "ping", "timestamp": time.time(), "message": "", "data": {}}
                    continue
                if ev is None:  # sentinel — job is done
                    yield {
                        "kind": "terminal",
                        "timestamp": time.time(),
                        "message": f"Job {job.status}",
                        "data": {"status": job.status, "error": job.error},
                    }
                    return
                yield ev.to_dict()
        finally:
            with job._lock:
                if q in job._subscribers:
                    job._subscribers.remove(q)
