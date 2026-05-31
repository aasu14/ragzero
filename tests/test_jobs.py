"""Tests for the job manager + progress-aware indexing path."""
import threading
import time
from datetime import datetime, timezone

from ragzero.rag.factories import build_dev_pipeline
from ragzero.rag.interfaces import Document
from ragzero.rag.jobs import Job, JobManager


def _docs(n: int) -> list[Document]:
    now = datetime.now(timezone.utc)
    return [
        Document(
            doc_id=f"d{i}",
            source=f"internal://d{i}",
            content=f"Document number {i} about topic {i % 3}.",
            doc_type="txt",
            created_at=now,
        )
        for i in range(n)
    ]


def test_job_manager_creates_and_retrieves():
    jm = JobManager()
    job = jm.create("ingest_upload", session_id="sess1")
    assert jm.get(job.job_id) is job
    assert jm.get("nonexistent") is None


def test_job_session_isolation():
    """Jobs are scoped to sessions — list_for_session should not leak."""
    jm = JobManager()
    a = jm.create("ingest", "sess_a")
    b = jm.create("ingest", "sess_b")
    sess_a = jm.list_for_session("sess_a")
    assert a in sess_a
    assert b not in sess_a


def test_job_emits_replay_buffered_events():
    """A subscriber that joins late should see all prior events."""
    jm = JobManager()
    job = jm.create("graph_build", "sess1")
    JobManager.mark_running(job)
    JobManager.emit(job, "progress", "step 1", current=1, total=10)
    JobManager.emit(job, "progress", "step 2", current=2, total=10)

    events = []
    def consume():
        for ev in jm.subscribe(job):
            events.append(ev)
            if len(events) >= 5:
                break

    t = threading.Thread(target=consume)
    t.start()
    time.sleep(0.1)
    JobManager.emit(job, "progress", "step 3", current=3, total=10)
    JobManager.mark_done(job, result={"foo": "bar"})
    t.join(timeout=2.0)

    # Should see at least the two replayed events + the live one + done + terminal
    kinds = [e["kind"] for e in events]
    assert "progress" in kinds
    assert "done" in kinds or "terminal" in kinds


def test_job_emits_terminal_on_finished_job():
    """Subscribing to an already-finished job yields a terminal event immediately."""
    jm = JobManager()
    job = jm.create("ingest", "sess1")
    JobManager.mark_done(job, result={"ok": True})
    events = list(jm.subscribe(job))
    assert any(e["kind"] == "terminal" for e in events)
    assert events[-1]["data"]["status"] == "completed"


def test_job_cancel_flag():
    jm = JobManager()
    job = jm.create("ingest", "sess1")
    assert not JobManager.cancel_requested(job)
    JobManager.request_cancel(job)
    assert JobManager.cancel_requested(job)


def test_job_snapshot_includes_progress():
    jm = JobManager()
    job = jm.create("ingest", "sess1")
    JobManager.emit(job, "progress", "halfway", current=5, total=10)
    snap = job.snapshot()
    assert snap["current"] == 5
    assert snap["total"] == 10


def test_job_mark_failed_sets_error():
    jm = JobManager()
    job = jm.create("ingest", "sess1")
    JobManager.mark_failed(job, "thing exploded")
    assert job.status == "failed"
    assert job.error == "thing exploded"


def test_index_batched_invokes_callback_per_batch():
    """The progress-aware indexing path should fire the callback at each batch boundary."""
    pipeline = build_dev_pipeline()
    chunks = []
    docs = _docs(40)  # enough to span multiple batches
    for d in docs:
        chunks.extend(pipeline.chunker.chunk(d))

    calls = []
    def on_batch(done, total):
        calls.append((done, total))

    pipeline.retriever.index_batched(chunks, batch_size=10, on_batch=on_batch)
    # With batch_size=10 and ~40 chunks we should get at least 4 callbacks
    assert len(calls) >= 4
    # The final call should report all chunks done
    assert calls[-1][0] == calls[-1][1]
    assert calls[-1][0] == len(chunks)


def test_index_batched_handles_empty():
    pipeline = build_dev_pipeline()
    calls = []
    pipeline.retriever.index_batched([], batch_size=10, on_batch=lambda d, t: calls.append((d, t)))
    assert calls == []


def test_ingest_documents_progress_reports_both_phases():
    pipeline = build_dev_pipeline()
    chunk_calls = []
    embed_calls = []
    docs = _docs(5)
    pipeline.ingest_documents_progress(
        docs,
        on_chunk_done=lambda done, total, src: chunk_calls.append((done, total)),
        on_embed_batch=lambda done, total: embed_calls.append((done, total)),
        batch_size=5,
    )
    # Five docs → five chunk callbacks; one or two embed callbacks
    assert len(chunk_calls) == 5
    assert chunk_calls[-1] == (5, 5)
    assert embed_calls
    assert embed_calls[-1][0] == embed_calls[-1][1]


def test_job_concurrent_subscribers_both_see_events():
    """Multiple browser tabs watching the same job."""
    jm = JobManager()
    job = jm.create("ingest", "sess1")
    JobManager.mark_running(job)

    a_events: list = []
    b_events: list = []

    def watch(out):
        for ev in jm.subscribe(job):
            out.append(ev)
            if ev["kind"] == "terminal":
                break

    t_a = threading.Thread(target=watch, args=(a_events,))
    t_b = threading.Thread(target=watch, args=(b_events,))
    t_a.start()
    t_b.start()
    time.sleep(0.1)
    JobManager.emit(job, "progress", "live event", current=1, total=1)
    JobManager.mark_done(job)
    t_a.join(timeout=2.0)
    t_b.join(timeout=2.0)

    # Both subscribers should have seen the live event
    a_kinds = [e["kind"] for e in a_events]
    b_kinds = [e["kind"] for e in b_events]
    assert "progress" in a_kinds
    assert "progress" in b_kinds
    assert a_events[-1]["kind"] == "terminal"
    assert b_events[-1]["kind"] == "terminal"
