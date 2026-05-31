"""NetworkX-backed graph store. In-memory, fast, dependency on networkx
optional — if not installed, we fall back to a minimal dict-of-sets impl
so the tests still run.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from .interface import Entity, GraphStore, Path, Relation


class NetworkXGraphStore(GraphStore):
    """In-memory graph store. Uses networkx if installed; falls back to dicts."""

    def __init__(self) -> None:
        try:
            import networkx as nx  # noqa: F401
            self._has_nx = True
            import networkx as nx_mod
            self._graph = nx_mod.MultiDiGraph()
        except ImportError:
            self._has_nx = False
            # Fallback adjacency: node_name -> Entity, edges as list of Relations
            self._entities: dict[str, Entity] = {}
            self._adj: dict[str, list[Relation]] = defaultdict(list)
            self._reverse_adj: dict[str, list[Relation]] = defaultdict(list)

    # ---- writes ----
    def upsert_entity(self, entity: Entity) -> None:
        if self._has_nx:
            if self._graph.has_node(entity.name):
                # Merge doc/chunk references
                existing = self._graph.nodes[entity.name]["entity"]
                merged = Entity(
                    name=entity.name,
                    entity_type=entity.entity_type or existing.entity_type,
                    doc_ids=tuple(sorted(set(existing.doc_ids) | set(entity.doc_ids))),
                    chunk_ids=tuple(sorted(set(existing.chunk_ids) | set(entity.chunk_ids))),
                    description=entity.description or existing.description,
                )
                self._graph.nodes[entity.name]["entity"] = merged
            else:
                self._graph.add_node(entity.name, entity=entity)
        else:
            existing = self._entities.get(entity.name)
            if existing:
                self._entities[entity.name] = Entity(
                    name=entity.name,
                    entity_type=entity.entity_type or existing.entity_type,
                    doc_ids=tuple(sorted(set(existing.doc_ids) | set(entity.doc_ids))),
                    chunk_ids=tuple(sorted(set(existing.chunk_ids) | set(entity.chunk_ids))),
                    description=entity.description or existing.description,
                )
            else:
                self._entities[entity.name] = entity

    def upsert_relation(self, relation: Relation) -> None:
        # Make sure both endpoints exist
        for name in (relation.source, relation.target):
            if self._has_nx:
                if not self._graph.has_node(name):
                    self._graph.add_node(name, entity=Entity(name=name, entity_type="UNKNOWN"))
            else:
                if name not in self._entities:
                    self._entities[name] = Entity(name=name, entity_type="UNKNOWN")
        if self._has_nx:
            self._graph.add_edge(relation.source, relation.target, relation=relation)
        else:
            self._adj[relation.source].append(relation)
            self._reverse_adj[relation.target].append(relation)

    # ---- reads ----
    def find_entities(self, terms: list[str], limit: int = 10) -> list[Entity]:
        terms_lower = [t.lower() for t in terms if t]
        if not terms_lower:
            return []
        results = []
        if self._has_nx:
            for name, data in self._graph.nodes(data=True):
                lname = name.lower()
                if any(t in lname for t in terms_lower):
                    results.append((self._score_name(lname, terms_lower), data["entity"]))
        else:
            for name, ent in self._entities.items():
                lname = name.lower()
                if any(t in lname for t in terms_lower):
                    results.append((self._score_name(lname, terms_lower), ent))
        results.sort(key=lambda r: r[0], reverse=True)
        return [e for _, e in results[:limit]]

    @staticmethod
    def _score_name(name: str, terms: list[str]) -> float:
        """Prefer exact matches over substring matches."""
        score = 0.0
        for t in terms:
            if name == t:
                score += 10
            elif name.startswith(t):
                score += 5
            elif t in name:
                score += 1
        return score

    def neighbors(self, entity_name: str, max_depth: int = 2) -> list[Path]:
        """BFS from entity_name, returning paths up to max_depth edges."""
        if max_depth < 1:
            return []
        paths: list[Path] = []
        visited: set[str] = {entity_name}

        # BFS queue stores: (current_node_name, path_nodes_list, path_edges_list, depth)
        start_entity = self._get_entity(entity_name)
        if start_entity is None:
            return []
        queue: deque = deque([(entity_name, [start_entity], [], 0)])

        while queue:
            current, path_nodes, path_edges, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for rel in self._outgoing(current):
                if rel.target in visited:
                    continue
                next_entity = self._get_entity(rel.target)
                if next_entity is None:
                    continue
                new_nodes = path_nodes + [next_entity]
                new_edges = path_edges + [rel]
                paths.append(Path(nodes=new_nodes, edges=new_edges))
                visited.add(rel.target)
                queue.append((rel.target, new_nodes, new_edges, depth + 1))
        return paths

    def _outgoing(self, name: str) -> list[Relation]:
        if self._has_nx:
            edges = []
            for _, _, data in self._graph.out_edges(name, data=True):
                edges.append(data["relation"])
            return edges
        return list(self._adj.get(name, []))

    def _get_entity(self, name: str) -> Entity | None:
        if self._has_nx:
            if self._graph.has_node(name):
                return self._graph.nodes[name]["entity"]
            return None
        return self._entities.get(name)

    def stats(self) -> dict[str, int]:
        if self._has_nx:
            return {
                "nodes": self._graph.number_of_nodes(),
                "edges": self._graph.number_of_edges(),
            }
        return {
            "nodes": len(self._entities),
            "edges": sum(len(v) for v in self._adj.values()),
        }

    def clear(self) -> None:
        if self._has_nx:
            self._graph.clear()
        else:
            self._entities.clear()
            self._adj.clear()
            self._reverse_adj.clear()

    def snapshot(self) -> dict[str, Any]:
        """Cytoscape-style node/edge lists for UI viz."""
        nodes = []
        edges = []
        if self._has_nx:
            for name, data in self._graph.nodes(data=True):
                ent = data["entity"]
                nodes.append({
                    "id": name,
                    "type": ent.entity_type,
                    "desc": ent.description[:200],
                    "doc_count": len(ent.doc_ids),
                })
            for src, tgt, data in self._graph.edges(data=True):
                rel = data["relation"]
                edges.append({
                    "source": src,
                    "target": tgt,
                    "type": rel.rel_type,
                    "evidence": rel.evidence[:200],
                })
        else:
            for name, ent in self._entities.items():
                nodes.append({
                    "id": name,
                    "type": ent.entity_type,
                    "desc": ent.description[:200],
                    "doc_count": len(ent.doc_ids),
                })
            for src, rels in self._adj.items():
                for rel in rels:
                    edges.append({
                        "source": src,
                        "target": rel.target,
                        "type": rel.rel_type,
                        "evidence": rel.evidence[:200],
                    })
        return {"nodes": nodes, "edges": edges}
