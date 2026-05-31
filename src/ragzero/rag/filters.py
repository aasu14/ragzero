"""Unified filter language + Python-side predicate evaluator.

Used as a fallback when the active vector store doesn't support native
filtering (FAISS, in_memory) — we fetch more candidates from the store and
post-filter by matching against chunk metadata.

For stores that DO support native filtering (Chroma, Qdrant, etc.), each
backend's search() translates this list into its native filter format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


FilterOp = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "contains"]


@dataclass
class Filter:
    field: str
    op: FilterOp = "eq"
    value: Any = None

    @classmethod
    def from_dict(cls, d: dict) -> "Filter":
        return cls(field=d["field"], op=d.get("op", "eq"), value=d.get("value"))


def _resolve(metadata: dict, source_field_lookup: dict, field: str) -> Any:
    """Look up a value on a Chunk's flattened metadata view.

    We check the metadata dict first (per-chunk fields) then a few common
    Chunk attributes that aren't in metadata (source, doc_id, doc_type).
    """
    if field in metadata:
        return metadata[field]
    return source_field_lookup.get(field)


def matches(filters: list[Filter], metadata: dict, extras: dict) -> bool:
    """AND-combine a list of filters against a flat metadata dict."""
    for f in filters:
        v = _resolve(metadata, extras, f.field)
        if not _match_one(v, f.op, f.value):
            return False
    return True


def _match_one(value: Any, op: str, target: Any) -> bool:
    # If the field is missing from the chunk, only "ne" / "not_in" can pass.
    if value is None:
        return op in ("ne", "not_in")
    try:
        if op == "eq": return value == target
        if op == "ne": return value != target
        if op == "gt": return value > target
        if op == "gte": return value >= target
        if op == "lt": return value < target
        if op == "lte": return value <= target
        if op == "in":
            if isinstance(target, (list, tuple, set)):
                return value in target
            return value == target
        if op == "not_in":
            if isinstance(target, (list, tuple, set)):
                return value not in target
            return value != target
        if op == "contains":
            # String substring OR list membership
            if isinstance(value, (list, tuple)):
                return target in value
            return str(target) in str(value)
    except TypeError:
        # Mismatched types (e.g. comparing str to int) — count as no-match
        return False
    return False


def parse_filters(raw: list[dict] | None) -> list[Filter]:
    if not raw:
        return []
    return [Filter.from_dict(r) for r in raw]
