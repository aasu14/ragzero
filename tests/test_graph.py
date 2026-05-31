"""Tests for graph store + entity extraction."""
from datetime import datetime, timezone

from ragzero.rag.graph import (
    Entity, EntityExtractor, NetworkXGraphStore, Relation,
)
from ragzero.rag.interfaces import Chunk


def _chunk(chunk_id: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        page=None,
        position=0,
        source=f"internal://{doc_id}",
        created_at=datetime.now(timezone.utc),
    )


def test_graph_upsert_and_neighbors():
    g = NetworkXGraphStore()
    g.upsert_entity(Entity(name="Alice", entity_type="PERSON", doc_ids=("d1",)))
    g.upsert_entity(Entity(name="Acme Corp", entity_type="ORG", doc_ids=("d1",)))
    g.upsert_relation(Relation(
        source="Alice", target="Acme Corp", rel_type="works_at",
        doc_ids=("d1",), chunk_ids=("c1",), evidence="Alice works at Acme Corp."
    ))

    paths = g.neighbors("Alice", max_depth=2)
    assert len(paths) >= 1
    p = paths[0]
    assert p.nodes[0].name == "Alice"
    assert p.nodes[-1].name == "Acme Corp"
    assert p.edges[0].rel_type == "works_at"


def test_graph_multi_hop():
    g = NetworkXGraphStore()
    g.upsert_relation(Relation(source="A", target="B", rel_type="r1"))
    g.upsert_relation(Relation(source="B", target="C", rel_type="r2"))
    g.upsert_relation(Relation(source="C", target="D", rel_type="r3"))

    paths_2 = g.neighbors("A", max_depth=2)
    # Should reach B (depth 1) and C (depth 2), not D
    reachable = {p.nodes[-1].name for p in paths_2}
    assert "B" in reachable
    assert "C" in reachable
    assert "D" not in reachable


def test_graph_find_entities_prefers_exact_match():
    g = NetworkXGraphStore()
    g.upsert_entity(Entity(name="Java", entity_type="LANG"))
    g.upsert_entity(Entity(name="JavaScript", entity_type="LANG"))
    g.upsert_entity(Entity(name="Javanese coffee", entity_type="CONCEPT"))

    results = g.find_entities(["Java"], limit=10)
    assert len(results) >= 2
    # Exact match should come first
    assert results[0].name == "Java"


def test_graph_entity_merge():
    g = NetworkXGraphStore()
    g.upsert_entity(Entity(name="Alice", entity_type="PERSON", doc_ids=("d1",), chunk_ids=("c1",)))
    g.upsert_entity(Entity(name="Alice", entity_type="", doc_ids=("d2",), chunk_ids=("c2",)))
    # Should now have one node with both docs/chunks
    ents = g.find_entities(["Alice"])
    assert len(ents) == 1
    assert set(ents[0].doc_ids) == {"d1", "d2"}
    assert set(ents[0].chunk_ids) == {"c1", "c2"}
    # Entity type from first set is preserved
    assert ents[0].entity_type == "PERSON"


def test_graph_snapshot_for_ui():
    g = NetworkXGraphStore()
    g.upsert_relation(Relation(source="X", target="Y", rel_type="z"))
    snap = g.snapshot()
    assert {n["id"] for n in snap["nodes"]} == {"X", "Y"}
    assert any(e["source"] == "X" and e["target"] == "Y" for e in snap["edges"])


def test_extractor_regex_fallback_with_mock_llm():
    """When the LLM doesn't return valid JSON, fall back to regex extraction."""
    from ragzero.rag.backends.llms import MockLLM
    extractor = EntityExtractor(MockLLM())
    chunk = _chunk(
        "c1", "d1",
        "Alice Johnson works at Acme Corp. Acme Corp was founded by Bob Smith.",
    )
    ents, rels = extractor.extract(chunk)
    # Regex path should find capitalized phrases mentioned at least once with multi-word names
    names = {e.name for e in ents}
    assert "Alice Johnson" in names or "Acme Corp" in names
    # Should be tagged with the chunk
    for e in ents:
        assert "c1" in e.chunk_ids


def test_extractor_skips_sentence_starters():
    """Capitalized stopwords like 'The', 'When' shouldn't become entities."""
    from ragzero.rag.backends.llms import MockLLM
    extractor = EntityExtractor(MockLLM())
    chunk = _chunk(
        "c1", "d1",
        "The Eiffel Tower stands in Paris. When tourists visit, they see it.",
    )
    ents, _ = extractor.extract(chunk)
    names = {e.name for e in ents}
    assert "The" not in names
    assert "When" not in names


def test_extractor_handles_llm_json():
    """If the LLM returns valid JSON, use it directly."""
    class JsonLLM:
        def generate(self, prompt, context, max_tokens=512, temperature=0.0):
            return '''{
              "entities": [
                {"name": "Marie Curie", "type": "PERSON", "description": "physicist"},
                {"name": "Sorbonne", "type": "ORG", "description": "university"}
              ],
              "relations": [
                {"source": "Marie Curie", "target": "Sorbonne", "type": "studied_at", "evidence": "studied at the Sorbonne"}
              ]
            }'''
    extractor = EntityExtractor(JsonLLM())
    chunk = _chunk("c1", "d1", "Marie Curie studied at the Sorbonne.")
    ents, rels = extractor.extract(chunk)
    assert len(ents) == 2
    assert {e.name for e in ents} == {"Marie Curie", "Sorbonne"}
    assert len(rels) == 1
    assert rels[0].source == "Marie Curie"
    assert rels[0].rel_type == "studied_at"


def test_graph_clear():
    g = NetworkXGraphStore()
    g.upsert_relation(Relation(source="A", target="B", rel_type="r"))
    assert g.stats()["nodes"] == 2
    g.clear()
    assert g.stats()["nodes"] == 0
    assert g.stats()["edges"] == 0
