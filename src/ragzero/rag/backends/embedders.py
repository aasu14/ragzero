"""Embedder backends.

- HashEmbedder: deterministic, dependency-free, used for tests.
- SentenceTransformerEmbedder: local model via sentence-transformers.
- OpenAIEmbedder: OpenAI embeddings API.
- AzureOpenAIEmbedder: Azure OpenAI embeddings.
- VoyageEmbedder: Voyage AI (Anthropic's recommended embedding partner).
- GeminiEmbedder: Google Gemini embeddings via google-generativeai SDK.
- OpenRouterEmbedder: Any embedding model on OpenRouter via OpenAI-compatible API.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

from ..interfaces import Embedder


class HashEmbedder(Embedder):
    """Deterministic embedder using token hashing. Used for tests."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = text.lower().split()
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            for i in range(self._dim):
                bit = (h >> (i * 3)) & 0xFF
                vec[i] += (bit - 128) / 128.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class SentenceTransformerEmbedder(Embedder):  # pragma: no cover
    """Wraps sentence-transformers. Imported lazily."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from e
        self._model: Any = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class OpenAIEmbedder(Embedder):  # pragma: no cover
    """OpenAI embeddings API. Defaults to text-embedding-3-small."""

    _DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai not installed. pip install openai") from e
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client: Any = OpenAI(**kwargs)
        self.model = model
        self._dim = self._DIMS.get(model, 1536)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [list(d.embedding) for d in resp.data]


class AzureOpenAIEmbedder(Embedder):  # pragma: no cover
    """Azure OpenAI embeddings. Uses deployment name as the model identifier."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment: str,
        api_version: str = "2024-02-01",
        dim: int = 1536,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise ImportError("openai not installed. pip install openai") from e
        self._client: Any = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.deployment = deployment
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.deployment, input=texts)
        return [list(d.embedding) for d in resp.data]


class VoyageEmbedder(Embedder):  # pragma: no cover
    """Voyage AI embeddings — Anthropic's recommended embedding partner."""

    _DIMS = {
        "voyage-3": 1024,
        "voyage-3-lite": 512,
        "voyage-large-2": 1536,
    }

    def __init__(self, api_key: str, model: str = "voyage-3") -> None:
        try:
            import voyageai
        except ImportError as e:
            raise ImportError("voyageai not installed. pip install voyageai") from e
        self._client: Any = voyageai.Client(api_key=api_key)
        self.model = model
        self._dim = self._DIMS.get(model, 1024)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embed(texts=texts, model=self.model)
        return [list(v) for v in resp.embeddings]


class GeminiEmbedder(Embedder):  # pragma: no cover
    """Google Gemini embeddings via google-generativeai SDK.

    Default model is 'text-embedding-004' (768 dims). Any model name accepted —
    the SDK validates at call time.
    """

    _DEFAULT_DIMS = {
        "text-embedding-004": 768,
        "embedding-001": 768,
        "text-embedding-005": 768,
    }

    def __init__(self, api_key: str, model: str = "text-embedding-004", dim: int | None = None) -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as e:
            raise ImportError(
                "google-generativeai not installed. pip install google-generativeai"
            ) from e
        genai.configure(api_key=api_key)
        self._genai: Any = genai
        self.model_name = model
        # Dim must be known up-front for the vector store. We accept an explicit
        # override; otherwise look up the curated default; otherwise 768.
        self._dim = dim if dim is not None else self._DEFAULT_DIMS.get(model, 768)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for text in texts:
            # The Gemini API embeds one document at a time
            resp = self._genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document",
            )
            # Response shape: {"embedding": [...]}
            vec = resp.get("embedding") if isinstance(resp, dict) else getattr(resp, "embedding", None)
            if vec is None:
                raise RuntimeError(f"Unexpected Gemini embedding response: {resp}")
            results.append(list(vec))
        return results


class OpenRouterEmbedder(Embedder):  # pragma: no cover
    """OpenRouter embeddings — proxies multiple embedding providers via one API.

    Uses the OpenAI-compatible SDK pointed at OpenRouter's embeddings endpoint.
    Specify the model as 'provider/model-name', e.g.:
    - 'openai/text-embedding-3-small' (1536 dims)
    - 'openai/text-embedding-3-large' (3072 dims)
    - 'qwen/qwen3-embedding-8b' (1024 dims)
    - 'google/gemini-embedding-001' (3072 dims, supports custom dims)
    - 'baai/bge-m3' (1024 dims)

    Dimensions vary by model; pass `dim` explicitly because the vector store
    needs to know the dimension at construction time. Some models also accept
    a `dimensions` request parameter to override their default output size.
    """

    # Curated defaults for common models. Users can override via the `dim` arg.
    _DEFAULT_DIMS = {
        "openai/text-embedding-3-small": 1536,
        "openai/text-embedding-3-large": 3072,
        "openai/text-embedding-ada-002": 1536,
        "qwen/qwen3-embedding-8b": 1024,
        "qwen/qwen3-embedding-4b": 1024,
        "google/gemini-embedding-001": 3072,
        "baai/bge-m3": 1024,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "openai/text-embedding-3-small",
        dim: int | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai not installed. pip install openai") from e
        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-Title"] = app_name
        self._client: Any = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=default_headers or None,
        )
        self.model = model
        # Resolve the dimension up front. Explicit `dim` wins; otherwise look up
        # the curated default; final fallback is 1536 (a common embedding size).
        # If the chosen model supports the `dimensions` request parameter (e.g.
        # OpenAI v3 models, Gemini), we'll also pass it so the API truncates to
        # our requested size — guaranteeing the vector store sees consistent
        # dimensionality even when the underlying model defaults differ.
        self._dim = dim if dim is not None else self._DEFAULT_DIMS.get(model, 1536)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Try requesting our target dim via the optional `dimensions` parameter.
        # If the model doesn't support it, retry without — OpenRouter rejects
        # the parameter for some providers (BAAI/BGE, etc.) and we don't want
        # to fail when the natural dim already matches.
        try:
            resp = self._client.embeddings.create(
                model=self.model, input=texts, dimensions=self._dim,
            )
        except TypeError:
            # Older openai SDK versions don't accept `dimensions=` kwarg
            resp = self._client.embeddings.create(model=self.model, input=texts)
        except Exception as e:
            # Provider may reject `dimensions`; retry without
            if "dimensions" in str(e).lower() or "param" in str(e).lower():
                resp = self._client.embeddings.create(model=self.model, input=texts)
            else:
                raise
        vectors = [list(d.embedding) for d in resp.data]
        # If the model returned a different dim than declared, surface it loudly —
        # silent dim mismatches lead to ANN search corruption later.
        if vectors and len(vectors[0]) != self._dim:
            raise RuntimeError(
                f"OpenRouter model {self.model!r} returned vectors of dim "
                f"{len(vectors[0])} but embedder was constructed for dim {self._dim}. "
                f"Reconfigure the embedder with dim={len(vectors[0])} or pick a model "
                f"that supports the `dimensions` parameter."
            )
        return vectors
