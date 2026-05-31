"""Token-based chunker using OpenAI's tiktoken.

Falls back to a 4-chars-per-token char-count approximation if tiktoken isn't
installed, so the chunker still works (with a clear note in chunk metadata).
"""
from __future__ import annotations

from ..interfaces import Chunk, Document


class TokenChunker:
    def __init__(
        self,
        tokens_per_chunk: int = 400,
        overlap_tokens: int = 50,
        model: str = "cl100k_base",
    ) -> None:
        if tokens_per_chunk < 16:
            raise ValueError("tokens_per_chunk must be >= 16")
        if overlap_tokens >= tokens_per_chunk:
            raise ValueError("overlap_tokens must be < tokens_per_chunk")
        self.n = tokens_per_chunk
        self.overlap = overlap_tokens
        self.model = model

        # Lazy import — tiktoken is optional.
        self._enc = None
        self._using_approximation = False
        try:
            import tiktoken
            try:
                self._enc = tiktoken.get_encoding(model)
            except Exception:
                # Caller passed a model name (e.g. "gpt-4o-mini") instead of
                # an encoding name — try that fallback.
                self._enc = tiktoken.encoding_for_model(model)
        except ImportError:
            self._using_approximation = True

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content.strip()
        if not text:
            return []

        if self._using_approximation:
            # ~4 chars per English token is the standard rule of thumb.
            return self._chunk_chars(doc, target_chars=self.n * 4, overlap_chars=self.overlap * 4)

        # Real token encoding + decode round-trip per window.
        tokens = self._enc.encode(text)
        chunks: list[Chunk] = []
        step = max(1, self.n - self.overlap)
        position = 0
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.n]
            if not window:
                break
            block = self._enc.decode(window).strip()
            if block:
                chunks.append(_make_chunk(doc, block, position, doc.version, approx=False))
                position += 1
            if start + self.n >= len(tokens):
                break
        return chunks

    def _chunk_chars(self, doc: Document, target_chars: int, overlap_chars: int) -> list[Chunk]:
        text = doc.content
        step = max(1, target_chars - overlap_chars)
        chunks: list[Chunk] = []
        position = 0
        for start in range(0, len(text), step):
            window = text[start : start + target_chars].strip()
            if window:
                chunks.append(_make_chunk(doc, window, position, doc.version, approx=True))
                position += 1
            if start + target_chars >= len(text):
                break
        return chunks


def _make_chunk(doc: Document, text: str, position: int, version: int, approx: bool) -> Chunk:
    meta = {"doc_type": doc.doc_type, "doc_version": version, "chunker": "token"}
    if approx:
        meta["token_approximation"] = True
    return Chunk(
        chunk_id=f"{doc.doc_id}:{position}:v{version}",
        doc_id=doc.doc_id,
        text=text,
        page=None,
        position=position,
        source=doc.source,
        created_at=doc.created_at,
        metadata=meta,
    )
