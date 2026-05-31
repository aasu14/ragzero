"""Graph store interface.

A GraphStore stores entities (nodes) and relations (edges) extracted from
documents. Used by GraphStrategy.

Two implementations:
- NetworkXGraphStore: in-memory, zero ops, good up to ~100K nodes
- Neo4jGraphStore: production-grade, requires a running Neo4j instance
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Entity:
    """A named entity extracted from documents."""
    name: str
    entity_type: str                       # PERSON, ORG, CONCEPT, etc.
    doc_ids: tuple[str, ...] = ()          # which docs mention it
    chunk_ids: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class Relation:
    """A directed edge between entities."""
    source: str
    target: str
    rel_type: str                          # e.g. "founded", "located_in", "knows"
    doc_ids: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    evidence: str = ""                     # short quote supporting the relation


@dataclass(frozen=True)
class Path:
    """A traversal result: ordered nodes + the edges between them."""
    nodes: list[Entity]
    edges: list[Relation]


class GraphStore(ABC):
    """Pluggable graph backend."""

    @abstractmethod
    def upsert_entity(self, entity: Entity) -> None: ...

    @abstractmethod
    def upsert_relation(self, relation: Relation) -> None: ...

    @abstractmethod
    def find_entities(self, terms: list[str], limit: int = 10) -> list[Entity]:
        """Substring-match entities by name. Returns top matches."""

    @abstractmethod
    def neighbors(self, entity_name: str, max_depth: int = 2) -> list[Path]:
        """Return paths from this entity outward up to max_depth."""

    @abstractmethod
    def stats(self) -> dict[str, int]:
        """Number of nodes, edges, etc. for the UI."""

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Serialize the graph for the UI viz."""
