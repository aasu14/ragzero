"""Tests for Steps 9-10: caching and observability."""
import time
from datetime import datetime, timezone

from ragzero.rag.backends.caches import InMemoryTTLCache
from ragzero.rag.backends.embedders import HashEmbedder
from ragzero.rag.cache import CachedEmbedder, EmbeddingCache, QueryCache
from ragzero.rag.interfaces import Answer
from ragzero.rag.observability import InMemoryTracer, new_trace_id


def test_inmemory_cache_ttl_expires():
    cache = InMemoryTTLCache(max_entries=10)
    cache.set("k", "v", ttl_seconds=1)
    assert cache.get("k") == "v"
    time.sleep(1.1)
    assert cache.get("k") is None


def test_inmemory_cache_lru_eviction():
    cache = InMemoryTTLCache(max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # evicts "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_query_cache_roundtrips_answer():
    cache_backend = InMemoryTTLCache()
    qc = QueryCache(cache_backend, ttl_seconds=60)
    answer = Answer(
        text="hi", citations=[], confidence=0.8, refused=False, trace_id="t1"
    )
    qc.set("hello?", answer)
    retrieved = qc.get("hello?")
    assert retrieved is not None
    assert retrieved.text == "hi"
    assert retrieved.confidence == 0.8


def test_query_cache_miss_returns_none():
    qc = QueryCache(InMemoryTTLCache())
    assert qc.get("anything") is None


def test_embedding_cache_namespaced_by_model():
    backend = InMemoryTTLCache()
    cache_a = EmbeddingCache(backend, model_id="model-a")
    cache_b = EmbeddingCache(backend, model_id="model-b")
    cache_a.set("hello", [1.0, 2.0])
    # Different model_id → no hit
    assert cache_b.get("hello") is None
    assert cache_a.get("hello") == [1.0, 2.0]


def test_cached_embedder_only_calls_inner_for_misses():
    class CountingEmbedder(HashEmbedder):
        def __init__(self):
            super().__init__(dim=8)
            self.calls = 0
        def embed(self, texts):
            self.calls += 1
            return super().embed(texts)

    inner = CountingEmbedder()
    cache_backend = InMemoryTTLCache()
    cached = CachedEmbedder(inner, EmbeddingCache(cache_backend, "test"))

    cached.embed(["hello", "world"])
    assert inner.calls == 1
    cached.embed(["hello", "world"])  # both cached
    assert inner.calls == 1
    cached.embed(["hello", "new"])
    assert inner.calls == 2


def test_tracer_records_events_per_trace_id():
    tracer = InMemoryTracer()
    tid_a = new_trace_id()
    tid_b = new_trace_id()
    tracer.emit(tid_a, "stage1", {"x": 1})
    tracer.emit(tid_a, "stage2", {"x": 2})
    tracer.emit(tid_b, "stage1", {"x": 3})

    events_a = tracer.events_for(tid_a)
    assert len(events_a) == 2
    assert events_a[0]["stage"] == "stage1"
    assert events_a[1]["stage"] == "stage2"

    events_b = tracer.events_for(tid_b)
    assert len(events_b) == 1
    assert events_b[0]["data"]["x"] == 3


def test_tracer_thread_safety():
    import threading
    tracer = InMemoryTracer()
    tid = new_trace_id()
    def emit_many():
        for i in range(100):
            tracer.emit(tid, f"stage{i}", {"i": i})
    threads = [threading.Thread(target=emit_many) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(tracer.events_for(tid)) == 400


def test_tracer_bounded_capacity():
    tracer = InMemoryTracer(max_events=5)
    tid = new_trace_id()
    for i in range(10):
        tracer.emit(tid, "s", {"i": i})
    assert len(tracer.all_events()) == 5  # ring buffer
