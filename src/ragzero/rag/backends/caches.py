"""Cache backends.

- InMemoryTTLCache: thread-safe, used in tests and small deployments.
- RedisCache: production stub — fill in the Redis client of your choice.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from ..interfaces import Cache


class InMemoryTTLCache(Cache):
    """Thread-safe in-memory cache with TTL and LRU eviction."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self.max_entries = max_entries
        self._store: dict[str, tuple[Any, float | None]] = {}
        self._access_order: list[str] = []
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and time.time() > expires_at:
                self._store.pop(key, None)
                if key in self._access_order:
                    self._access_order.remove(key)
                return None
            # Touch (move to end for LRU)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        with self._lock:
            expires_at = time.time() + ttl_seconds if ttl_seconds else None
            self._store[key] = (value, expires_at)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            while len(self._access_order) > self.max_entries:
                oldest = self._access_order.pop(0)
                self._store.pop(oldest, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._access_order.clear()


class RedisCache(Cache):  # pragma: no cover
    """Redis-backed cache. Stub — wire up your redis client of choice.

    For a production deployment, add JSON (or pickle) serialization,
    and ensure deserialized values come back as the right dataclass type.
    """

    def __init__(self, redis_url: str) -> None:
        try:
            import redis  # type: ignore
        except ImportError as e:
            raise ImportError("redis package not installed") from e
        self._client = redis.from_url(redis_url)

    def get(self, key: str) -> Any | None:
        raw = self._client.get(key)
        return raw  # caller responsible for deserialization

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        if ttl_seconds:
            self._client.setex(key, ttl_seconds, value)
        else:
            self._client.set(key, value)

    def clear(self) -> None:
        self._client.flushdb()
