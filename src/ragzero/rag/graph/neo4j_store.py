"""Neo4j-backed graph store. Production backend for Graph RAG.

Requires `neo4j` driver package. Configured via standard env vars or
explicit constructor args.
"""
from __future__ import annotations

from typing import Any

from .interface import Entity, GraphStore, Path, Relation


class Neo4jGraphStore(GraphStore):  # pragma: no cover
    """Neo4j-backed graph. Lazily imports the driver."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            raise ImportError(
                "neo4j driver not installed. pip install neo4j"
            ) from e
        self._driver: Any = GraphDatabase.driver(uri, auth=(user, password))
        self._db = database

    def close(self) -> None:
        self._driver.close()

    def _session(self) -> Any:
        return self._driver.session(database=self._db)

    def upsert_entity(self, entity: Entity) -> None:
        with self._session() as session:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.entity_type = $type, e.description = $desc,
                              e.doc_ids = $docs, e.chunk_ids = $chunks
                ON MATCH SET e.doc_ids = apoc.coll.toSet(coalesce(e.doc_ids, []) + $docs),
                             e.chunk_ids = apoc.coll.toSet(coalesce(e.chunk_ids, []) + $chunks),
                             e.entity_type = coalesce(e.entity_type, $type),
                             e.description = coalesce(e.description, $desc)
                """,
                name=entity.name,
                type=entity.entity_type,
                desc=entity.description,
                docs=list(entity.doc_ids),
                chunks=list(entity.chunk_ids),
            )

    def upsert_relation(self, relation: Relation) -> None:
        with self._session() as session:
            session.run(
                """
                MERGE (s:Entity {name: $source})
                MERGE (t:Entity {name: $target})
                MERGE (s)-[r:REL {rel_type: $rtype}]->(t)
                ON CREATE SET r.doc_ids = $docs, r.chunk_ids = $chunks, r.evidence = $ev
                ON MATCH SET r.doc_ids = apoc.coll.toSet(coalesce(r.doc_ids, []) + $docs)
                """,
                source=relation.source,
                target=relation.target,
                rtype=relation.rel_type,
                docs=list(relation.doc_ids),
                chunks=list(relation.chunk_ids),
                ev=relation.evidence,
            )

    def find_entities(self, terms: list[str], limit: int = 10) -> list[Entity]:
        if not terms:
            return []
        regex = "(?i).*(" + "|".join(t for t in terms if t) + ").*"
        with self._session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name =~ $regex
                RETURN e
                LIMIT $limit
                """,
                regex=regex,
                limit=limit,
            )
            return [self._record_to_entity(r["e"]) for r in result]

    def neighbors(self, entity_name: str, max_depth: int = 2) -> list[Path]:
        with self._session() as session:
            result = session.run(
                """
                MATCH path = (start:Entity {name: $name})-[*1..$d]->(end:Entity)
                RETURN nodes(path) AS nodes, relationships(path) AS rels
                LIMIT 100
                """,
                name=entity_name,
                d=max_depth,
            )
            paths: list[Path] = []
            for rec in result:
                nodes = [self._record_to_entity(n) for n in rec["nodes"]]
                edges = [self._record_to_relation(r) for r in rec["rels"]]
                paths.append(Path(nodes=nodes, edges=edges))
            return paths

    @staticmethod
    def _record_to_entity(node) -> Entity:
        return Entity(
            name=node["name"],
            entity_type=node.get("entity_type", "UNKNOWN"),
            doc_ids=tuple(node.get("doc_ids", [])),
            chunk_ids=tuple(node.get("chunk_ids", [])),
            description=node.get("description", ""),
        )

    @staticmethod
    def _record_to_relation(edge) -> Relation:
        return Relation(
            source=edge.start_node["name"],
            target=edge.end_node["name"],
            rel_type=edge.get("rel_type", "REL"),
            doc_ids=tuple(edge.get("doc_ids", [])),
            chunk_ids=tuple(edge.get("chunk_ids", [])),
            evidence=edge.get("evidence", ""),
        )

    def stats(self) -> dict[str, int]:
        with self._session() as session:
            n = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            e = session.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]
            return {"nodes": n, "edges": e}

    def clear(self) -> None:
        with self._session() as session:
            session.run("MATCH (n:Entity) DETACH DELETE n")

    def snapshot(self) -> dict[str, Any]:
        with self._session() as session:
            nodes_res = session.run("MATCH (e:Entity) RETURN e LIMIT 500")
            edges_res = session.run(
                "MATCH (s:Entity)-[r:REL]->(t:Entity) RETURN s.name AS source, t.name AS target, r LIMIT 1000"
            )
            nodes = []
            for r in nodes_res:
                e = r["e"]
                nodes.append({
                    "id": e["name"],
                    "type": e.get("entity_type", ""),
                    "desc": (e.get("description") or "")[:200],
                    "doc_count": len(e.get("doc_ids", [])),
                })
            edges = []
            for r in edges_res:
                rel = r["r"]
                edges.append({
                    "source": r["source"],
                    "target": r["target"],
                    "type": rel.get("rel_type", "REL"),
                    "evidence": (rel.get("evidence") or "")[:200],
                })
            return {"nodes": nodes, "edges": edges}
