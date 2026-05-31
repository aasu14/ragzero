"""Metadata schema model + per-backend capability matrix.

This module lets users define structured metadata for their corpus:
  - Field name, type, and required flag
  - Capability flags (filterable / sortable / full-text searchable / retrievable)

Each vector store implements a subset of capabilities — `CAPABILITY_MATRIX`
exposes that subset so the UI can show users what their active store can
actually do, instead of silently ignoring flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Literal


# Supported scalar types. Lists and dicts ride through as JSON for backends
# that don't accept them natively (Chroma is the strictest — see
# vector_stores_external._sanitize_for_chroma).
FieldType = Literal["string", "int", "float", "bool", "date", "list", "json"]
FIELD_TYPES: tuple[FieldType, ...] = (
    "string", "int", "float", "bool", "date", "list", "json",
)


@dataclass
class FieldFlags:
    """Per-field capability flags.

    Filterable / sortable / full_text_searchable are honored by backends that
    support them (see CAPABILITY_MATRIX). Retrievable is always True today —
    every backend ships metadata back with search hits.
    """
    filterable: bool = False
    sortable: bool = False
    full_text_searchable: bool = False
    retrievable: bool = True


@dataclass
class FieldSpec:
    name: str
    type: FieldType = "string"
    required: bool = False
    default: Any = None
    description: str | None = None
    flags: FieldFlags = field(default_factory=FieldFlags)
    # Where this field originates. "builtin" = auto-populated by the ingest
    # pipeline (always present); "custom" = user-defined for this session.
    origin: Literal["builtin", "custom"] = "custom"
    # High-level category for grouping in the UI.
    # builtin field rows set this; custom fields default to "document".
    category: Literal[
        "document", "structural", "temporal", "provenance",
        "chunk", "custom",
    ] = "custom"


@dataclass
class MetadataSchema:
    """A session's metadata schema. Per-session so different users can model
    different corpora without affecting each other."""
    fields: list[FieldSpec] = field(default_factory=list)

    def by_name(self, name: str) -> FieldSpec | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def custom_fields(self) -> list[FieldSpec]:
        return [f for f in self.fields if f.origin == "custom"]

    def filterable_fields(self) -> list[FieldSpec]:
        return [f for f in self.fields if f.flags.filterable]


# ---------- Built-in field definitions ----------
# Always present on every Document; auto-populated by the ingestor. The UI
# shows them as read-only so users see the schema and can filter on them.

BUILTIN_FIELDS: list[FieldSpec] = [
    # Document
    FieldSpec(name="source", type="string", origin="builtin", category="document",
              description="Original URI / filename / paste-text title",
              flags=FieldFlags(filterable=True, sortable=True, full_text_searchable=True)),
    FieldSpec(name="doc_type", type="string", origin="builtin", category="document",
              description="File extension (txt, md, pdf, html, ...)",
              flags=FieldFlags(filterable=True)),
    FieldSpec(name="original_filename", type="string", origin="builtin", category="document",
              description="Filename as uploaded (set for uploads only)",
              flags=FieldFlags(filterable=True, full_text_searchable=True)),

    # Structural
    FieldSpec(name="char_count", type="int", origin="builtin", category="structural",
              description="Document size in characters",
              flags=FieldFlags(filterable=True, sortable=True)),
    FieldSpec(name="pages", type="int", origin="builtin", category="structural",
              description="Page count (PDFs only)",
              flags=FieldFlags(filterable=True, sortable=True)),
    FieldSpec(name="content_type", type="string", origin="builtin", category="structural",
              description="MIME type from HTTP fetch (URL ingest only)",
              flags=FieldFlags(filterable=True)),

    # Temporal
    FieldSpec(name="created_at", type="date", origin="builtin", category="temporal",
              description="When this document was ingested",
              flags=FieldFlags(filterable=True, sortable=True)),

    # Provenance
    FieldSpec(name="content_hash", type="string", origin="builtin", category="provenance",
              description="SHA-256 prefix of normalized text — used for dedup",
              flags=FieldFlags(filterable=True)),
    FieldSpec(name="session_id", type="string", origin="builtin", category="provenance",
              description="Session that ingested this doc",
              flags=FieldFlags(filterable=True)),
    FieldSpec(name="fetched_from", type="string", origin="builtin", category="provenance",
              description="Source URL (URL ingest only)",
              flags=FieldFlags(filterable=True, full_text_searchable=True)),

    # Chunk-level (populated per chunk, not per doc)
    FieldSpec(name="doc_version", type="int", origin="builtin", category="chunk",
              description="Increments when a doc with the same path is re-ingested with new content",
              flags=FieldFlags(filterable=True, sortable=True)),
]


def default_schema() -> MetadataSchema:
    """Schema with all built-in fields, no user-defined custom fields."""
    return MetadataSchema(fields=[
        FieldSpec(
            name=f.name, type=f.type, required=f.required, default=f.default,
            description=f.description,
            flags=FieldFlags(**asdict(f.flags)),
            origin=f.origin, category=f.category,
        )
        for f in BUILTIN_FIELDS
    ])


# ---------- Per-backend capability matrix ----------
# Maps vector_store provider id -> which flags it actually honors.
# Filterable is the most universal; full_text_searchable + sortable are rare.

CAPABILITY_MATRIX: dict[str, dict[str, bool]] = {
    "in_memory":      {"filterable": True,  "sortable": False, "full_text_searchable": False, "retrievable": True},
    "faiss":          {"filterable": False, "sortable": False, "full_text_searchable": False, "retrievable": True},
    "chroma":         {"filterable": True,  "sortable": False, "full_text_searchable": False, "retrievable": True},
    "qdrant":         {"filterable": True,  "sortable": False, "full_text_searchable": False, "retrievable": True},
    "pinecone":       {"filterable": True,  "sortable": False, "full_text_searchable": False, "retrievable": True},
    "weaviate":       {"filterable": True,  "sortable": True,  "full_text_searchable": True,  "retrievable": True},
    "pgvector":       {"filterable": True,  "sortable": True,  "full_text_searchable": True,  "retrievable": True},
    "azure_ai_search":{"filterable": True,  "sortable": True,  "full_text_searchable": True,  "retrievable": True},
}


def capabilities_for(provider: str) -> dict[str, bool]:
    """Return the capability map for a provider, falling back to retrievable-only."""
    return CAPABILITY_MATRIX.get(provider, {
        "filterable": False, "sortable": False,
        "full_text_searchable": False, "retrievable": True,
    })


# ---------- Filter format ----------

FilterOp = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "contains"]
FILTER_OPS: tuple[FilterOp, ...] = (
    "eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "contains",
)


@dataclass
class FilterClause:
    """One filter atom. Combined with AND across the list passed to search()."""
    field: str
    op: FilterOp = "eq"
    value: Any = None


def coerce_value(raw: Any, ftype: FieldType) -> Any:
    """Coerce a UI-supplied value (often string) to the field's declared type.

    Raises ValueError with a clean message if coercion fails — surfaces to UI.
    """
    if raw is None or raw == "":
        return None
    try:
        if ftype == "string":
            return str(raw)
        if ftype == "int":
            return int(raw)
        if ftype == "float":
            return float(raw)
        if ftype == "bool":
            if isinstance(raw, bool):
                return raw
            s = str(raw).strip().lower()
            if s in ("true", "1", "yes", "y"): return True
            if s in ("false", "0", "no", "n"): return False
            raise ValueError(f"not a boolean: {raw!r}")
        if ftype == "date":
            # Accept ISO 8601 or YYYY-MM-DD; return ISO string for portability
            if isinstance(raw, datetime):
                return raw.isoformat()
            return datetime.fromisoformat(str(raw)).isoformat()
        if ftype == "list":
            if isinstance(raw, list):
                return raw
            # Comma-separated string -> list
            return [s.strip() for s in str(raw).split(",") if s.strip()]
        if ftype == "json":
            if isinstance(raw, (dict, list)):
                return raw
            import json
            return json.loads(str(raw))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot coerce {raw!r} to {ftype}: {e}")
    return raw


def schema_to_dict(s: MetadataSchema) -> dict:
    """JSON-serializable representation for the API."""
    return {"fields": [asdict(f) for f in s.fields]}


def schema_from_dict(d: dict) -> MetadataSchema:
    """Reverse of schema_to_dict. Tolerant of missing optional fields."""
    fields: list[FieldSpec] = []
    for raw in d.get("fields", []):
        flags_raw = raw.get("flags") or {}
        flags = FieldFlags(
            filterable=bool(flags_raw.get("filterable", False)),
            sortable=bool(flags_raw.get("sortable", False)),
            full_text_searchable=bool(flags_raw.get("full_text_searchable", False)),
            retrievable=bool(flags_raw.get("retrievable", True)),
        )
        fields.append(FieldSpec(
            name=raw["name"],
            type=raw.get("type", "string"),
            required=bool(raw.get("required", False)),
            default=raw.get("default"),
            description=raw.get("description"),
            flags=flags,
            origin=raw.get("origin", "custom"),
            category=raw.get("category", "custom"),
        ))
    return MetadataSchema(fields=fields)
