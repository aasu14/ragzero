"""Entity and relation extractor.

Takes a chunk of text and extracts (entity, entity_type) tuples and
(source, relation, target) triples. Uses the LLM for general text;
falls back to a regex/heuristic extractor for the mock LLM path so tests
work without API access.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from ..interfaces import Chunk, LLM
from .interface import Entity, Relation


EXTRACTION_PROMPT = """\
Extract entities and relations from the text below.

Return ONLY a JSON object with this exact structure (no prose, no markdown):
{{
  "entities": [
    {{"name": "<entity name>", "type": "<PERSON|ORG|LOCATION|CONCEPT|PRODUCT|EVENT|OTHER>", "description": "<one short sentence>"}}
  ],
  "relations": [
    {{"source": "<entity name>", "target": "<entity name>", "type": "<short verb phrase, snake_case>", "evidence": "<exact quote from text>"}}
  ]
}}

Rules:
- Only include entities that are explicitly named.
- Only include relations supported by the text — provide the exact supporting quote in "evidence".
- Use the same exact entity names in relations as you do in entities.
- Maximum 10 entities and 10 relations per chunk.

Text:
\"\"\"
{text}
\"\"\"

JSON:"""


_CAPITALIZED_PHRASE = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3}\b")
_BASIC_VERB_RE = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\s+"
    r"(is|was|are|were|founded|created|invented|built|wrote|leads|owns|works at|joined|developed|uses)\s+"
    r"(?:a |an |the )?([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3})",
    re.IGNORECASE,
)

# Skip these as "entities" — they're sentence starters, not named entities
_STOPWORD_ENTITIES = frozenset({
    "The", "A", "An", "This", "That", "These", "Those", "It", "He", "She", "They",
    "When", "Where", "What", "Why", "How", "Who", "Which", "In", "On", "At",
    "And", "Or", "But", "If", "Then", "While", "Some", "Many", "Most", "All",
    "Each", "Every", "Both", "First", "Second", "Third", "There", "Here",
})


class EntityExtractor:
    """Extracts entities and relations from chunks.

    Strategy:
    1. If the LLM returns valid JSON, use that.
    2. Otherwise (e.g., MockLLM), fall back to a regex extractor.
       The regex path is deliberately conservative — it surfaces only
       clearly-marked named entities and a handful of common verb patterns.
    """

    def __init__(self, llm: LLM, max_chars_per_chunk: int = 2000) -> None:
        self.llm = llm
        self.max_chars_per_chunk = max_chars_per_chunk

    def extract(self, chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
        text = chunk.text[: self.max_chars_per_chunk]
        # Try the LLM first
        prompt = EXTRACTION_PROMPT.format(text=text)
        try:
            raw = self.llm.generate(
                prompt=prompt, context=[text], max_tokens=1024, temperature=0.0
            )
            parsed = self._parse_json(raw)
            if parsed:
                return self._from_json(parsed, chunk)
        except Exception:
            pass
        # Fallback: regex extractor
        return self._regex_extract(text, chunk)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """Find the first JSON object in the response."""
        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            pass
        # Find a JSON-looking block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None

    @staticmethod
    def _from_json(payload: dict[str, Any], chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
        entities: list[Entity] = []
        for e in payload.get("entities", []) or []:
            name = (e.get("name") or "").strip()
            if not name or name in _STOPWORD_ENTITIES:
                continue
            entities.append(
                Entity(
                    name=name,
                    entity_type=(e.get("type") or "OTHER").upper(),
                    description=e.get("description", "")[:300],
                    doc_ids=(chunk.doc_id,),
                    chunk_ids=(chunk.chunk_id,),
                )
            )
        relations: list[Relation] = []
        entity_names = {e.name for e in entities}
        for r in payload.get("relations", []) or []:
            src = (r.get("source") or "").strip()
            tgt = (r.get("target") or "").strip()
            if not src or not tgt or src == tgt:
                continue
            # Allow relations to reference entities that weren't formally listed
            relations.append(
                Relation(
                    source=src,
                    target=tgt,
                    rel_type=(r.get("type") or "related_to").lower().replace(" ", "_"),
                    doc_ids=(chunk.doc_id,),
                    chunk_ids=(chunk.chunk_id,),
                    evidence=r.get("evidence", "")[:300],
                )
            )
        return entities, relations

    @staticmethod
    def _regex_extract(text: str, chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
        # Entities = capitalized noun phrases that aren't stopwords
        candidates = _CAPITALIZED_PHRASE.findall(text)
        counts: dict[str, int] = defaultdict(int)
        for c in candidates:
            head = c.split()[0]
            if head in _STOPWORD_ENTITIES:
                continue
            counts[c] += 1
        # Keep entities mentioned more than once OR with multi-word names
        entities: list[Entity] = []
        for name, freq in counts.items():
            if freq >= 1 and (freq >= 2 or " " in name):
                entities.append(
                    Entity(
                        name=name,
                        entity_type="OTHER",
                        doc_ids=(chunk.doc_id,),
                        chunk_ids=(chunk.chunk_id,),
                    )
                )

        # Relations = matched verb patterns
        relations: list[Relation] = []
        for m in _BASIC_VERB_RE.finditer(text):
            src, verb, tgt = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            if src.split()[0] in _STOPWORD_ENTITIES or tgt.split()[0] in _STOPWORD_ENTITIES:
                continue
            if src == tgt:
                continue
            relations.append(
                Relation(
                    source=src,
                    target=tgt,
                    rel_type=verb.lower().replace(" ", "_"),
                    doc_ids=(chunk.doc_id,),
                    chunk_ids=(chunk.chunk_id,),
                    evidence=m.group(0),
                )
            )
        return entities, relations
