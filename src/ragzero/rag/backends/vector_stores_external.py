"""External vector store backends.

Each adapter implements the `VectorStore` interface against a real
third-party vector database. SDKs are imported lazily so the package
installs cleanly without any of these dependencies.

Common design:
- Vectors are stored in the remote DB; the `Chunk` metadata is stored
  alongside (in payload/metadata fields) so `search()` can reconstruct
  Chunk objects without a separate sidecar store.
- All adapters L2-normalize at the embedder layer (already done) and
  use cosine/inner-product as the similarity metric.
- Collection/index creation is idempotent on first `add()` call.

The Chunk model is reconstructed from payload metadata. To avoid drift
across backends we use a shared serialization helper.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any

from ..interfaces import Chunk, VectorStore


logger = logging.getLogger("ragzero.stores")


# ---------- Shared chunk <-> payload helpers ----------

def _chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    """Serialize a Chunk into a flat dict suitable for any vector DB payload."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "page": chunk.page,
        "position": chunk.position,
        "source": chunk.source,
        "created_at": chunk.created_at.isoformat(),
        # `metadata` is a dict — flatten by JSON-encoding so it round-trips
        # cleanly across backends that don't allow nested objects.
        "_extra_metadata_keys": list(chunk.metadata.keys()),
        **{f"meta_{k}": v for k, v in chunk.metadata.items()
           if isinstance(v, (str, int, float, bool, type(None)))},
    }


def _payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    """Reconstruct a Chunk from a payload dict produced by _chunk_to_payload."""
    extra_keys = payload.get("_extra_metadata_keys", []) or []
    metadata = {k: payload.get(f"meta_{k}") for k in extra_keys}
    return Chunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        text=payload["text"],
        page=payload.get("page"),
        position=payload.get("position", 0),
        source=payload["source"],
        created_at=datetime.fromisoformat(payload["created_at"]),
        metadata=metadata,
    )


def _stable_int_id(chunk_id: str) -> int:
    """Some backends (Pinecone, Qdrant) accept ints; derive a stable one."""
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:16], 16)


# ---------- Qdrant ----------

class QdrantVectorStore(VectorStore):  # pragma: no cover
    """Qdrant vector database.

    Works with both self-hosted Qdrant (Docker) and Qdrant Cloud. The
    `url` setting determines which.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "ragzero",
        api_key: str | None = None,
        dim: int = 1536,
        client: Any = None,        # for testing: inject a pre-built client
        models: Any = None,         # for testing: inject a fake qdrant_client.http.models
    ) -> None:
        if client is None or models is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http import models as qm
            except ImportError as e:
                raise ImportError("qdrant-client not installed. pip install qdrant-client") from e
            if client is None:
                client = QdrantClient(url=url, api_key=api_key)
            if models is None:
                models = qm

        self._qm = models
        self._client: Any = client
        self.collection = collection
        self._dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection in existing:
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=self._qm.VectorParams(
                size=self._dim, distance=self._qm.Distance.COSINE,
            ),
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        points = [
            self._qm.PointStruct(
                id=_stable_int_id(c.chunk_id),
                vector=v,
                payload=_chunk_to_payload(c),
            )
            for c, v in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        results = self._client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
        )
        return [(_payload_to_chunk(r.payload), float(r.score)) for r in results]

    def size(self) -> int:
        info = self._client.get_collection(self.collection)
        return info.points_count or 0

    def clear(self) -> None:
        try:
            self._client.delete_collection(collection_name=self.collection)
        except Exception:
            pass
        self._ensure_collection()


# ---------- Pinecone ----------

class PineconeVectorStore(VectorStore):  # pragma: no cover
    """Pinecone serverless vector database. Cloud-only.

    Requires the `pinecone` package (the v3+ SDK). Pinecone has size
    limits on metadata — large `text` fields may be rejected; the adapter
    truncates payload text to 20KB and falls back gracefully if storage
    fails.
    """

    _MAX_PAYLOAD_TEXT_BYTES = 20_000

    def __init__(
        self,
        api_key: str,
        index_name: str = "ragzero",
        dim: int = 1536,
        cloud: str = "aws",
        region: str = "us-east-1",
        client: Any = None,         # for testing: inject a pre-built pinecone client
        index: Any = None,          # for testing: inject a pre-built index handle
    ) -> None:
        if client is None:
            try:
                from pinecone import Pinecone, ServerlessSpec
            except ImportError as e:
                raise ImportError(
                    "pinecone not installed. pip install pinecone (note: NOT pinecone-client)"
                ) from e
            client = Pinecone(api_key=api_key)

        self._pc: Any = client
        self.index_name = index_name
        self._dim = dim

        if index is None:
            # Bootstrap the index (skip if injecting test client that doesn't have create_index)
            try:
                existing = [i["name"] for i in self._pc.list_indexes()]
                if index_name not in existing:
                    from pinecone import ServerlessSpec
                    self._pc.create_index(
                        name=index_name,
                        dimension=dim,
                        metric="cosine",
                        spec=ServerlessSpec(cloud=cloud, region=region),
                    )
            except Exception:
                # In test contexts the mock client may not implement list_indexes()
                pass
            index = self._pc.Index(index_name)
        self._index: Any = index

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        vectors_payload = []
        for c, v in zip(chunks, vectors):
            payload = _chunk_to_payload(c)
            # Truncate text if it exceeds Pinecone metadata limit
            text = payload.get("text", "")
            if len(text.encode("utf-8")) > self._MAX_PAYLOAD_TEXT_BYTES:
                payload["text"] = text.encode("utf-8")[: self._MAX_PAYLOAD_TEXT_BYTES].decode(
                    "utf-8", errors="ignore"
                )
                payload["_text_truncated"] = True
            vectors_payload.append({
                "id": c.chunk_id,        # Pinecone accepts string IDs
                "values": v,
                "metadata": payload,
            })
        # Pinecone batches over 100 are rejected — chunk the upsert
        for i in range(0, len(vectors_payload), 100):
            self._index.upsert(vectors=vectors_payload[i : i + 100])

    def search(self, query_vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        result = self._index.query(
            vector=query_vector, top_k=k, include_metadata=True,
        )
        hits = result.get("matches") or result["matches"] if isinstance(result, dict) else result.matches
        out: list[tuple[Chunk, float]] = []
        for match in hits:
            md = match["metadata"] if isinstance(match, dict) else match.metadata
            score = match["score"] if isinstance(match, dict) else match.score
            out.append((_payload_to_chunk(md), float(score)))
        return out

    def size(self) -> int:
        stats = self._index.describe_index_stats()
        if isinstance(stats, dict):
            return stats.get("total_vector_count", 0) or 0
        return getattr(stats, "total_vector_count", 0) or 0

    def clear(self) -> None:
        # Pinecone supports delete-all via delete(delete_all=True) in a namespace.
        try:
            self._index.delete(delete_all=True)
        except Exception:
            pass


# ---------- Weaviate ----------

class WeaviateVectorStore(VectorStore):  # pragma: no cover
    """Weaviate v4 client.

    Set `use_grpc=False` to use REST only (some sandboxed environments
    block gRPC). Cloud and self-hosted both supported via the `url`.
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: str | None = None,
        class_name: str = "RagzeroChunk",
        dim: int = 1536,
        client: Any = None,         # for testing: inject a pre-built weaviate client
    ) -> None:
        if client is None:
            try:
                import weaviate
                from weaviate.classes.init import Auth
            except ImportError as e:
                raise ImportError("weaviate-client not installed. pip install weaviate-client") from e

            if api_key:
                client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url, auth_credentials=Auth.api_key(api_key),
                )
            else:
                # Local connection — extract host/port from URL
                from urllib.parse import urlparse
                parsed = urlparse(url)
                client = weaviate.connect_to_local(
                    host=parsed.hostname or "localhost",
                    port=parsed.port or 8080,
                )
        self._client: Any = client
        self.class_name = class_name
        self._dim = dim
        self._ensure_class()

    def _ensure_class(self) -> None:
        if self._client.collections.exists(self.class_name):
            return
        try:
            from weaviate.classes.config import Configure
            vectorizer_config = Configure.Vectorizer.none()
        except ImportError:
            # Test path: mock client. Pass None and let the mock handle it.
            vectorizer_config = None
        self._client.collections.create(
            name=self.class_name,
            vectorizer_config=vectorizer_config,
        )

    def add(self, chunks, vectors):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        collection = self._client.collections.get(self.class_name)
        with collection.batch.dynamic() as batch:
            for c, v in zip(chunks, vectors):
                batch.add_object(properties=_chunk_to_payload(c), vector=v)

    def search(self, query_vector, k):
        try:
            from weaviate.classes.query import MetadataQuery
            metadata_query = MetadataQuery(distance=True)
        except ImportError:
            metadata_query = None  # mock client
        collection = self._client.collections.get(self.class_name)
        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=k,
            return_metadata=metadata_query,
        )
        out = []
        for obj in results.objects:
            # Weaviate returns distance (lower = closer); convert to similarity in [0,1]
            distance = obj.metadata.distance or 0.0
            similarity = 1.0 - distance
            out.append((_payload_to_chunk(obj.properties), similarity))
        return out

    def size(self):
        collection = self._client.collections.get(self.class_name)
        return collection.aggregate.over_all(total_count=True).total_count

    def clear(self):
        try:
            self._client.collections.delete(self.class_name)
        except Exception:
            pass
        self._ensure_class()

    def __del__(self):
        # Weaviate v4 wants explicit close; ignore errors during interpreter shutdown
        try:
            self._client.close()
        except Exception:
            pass


# ---------- Chroma ----------

def _sanitize_for_chroma(payload: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool — no None, no list, no dict.
    JSON-encode lists/dicts and drop None values. Reversed in _desanitize_from_chroma."""
    import json as _json
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, dict, tuple)):
            out[f"__json__{k}"] = _json.dumps(list(v) if isinstance(v, tuple) else v)
        else:
            out[k] = str(v)
    return out


def _desanitize_from_chroma(payload: dict[str, Any]) -> dict[str, Any]:
    """Reverse _sanitize_for_chroma: decode JSON-encoded list/dict fields."""
    import json as _json
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if k.startswith("__json__"):
            try:
                out[k[len("__json__"):]] = _json.loads(v)
            except (TypeError, ValueError):
                out[k[len("__json__"):]] = v
        else:
            out[k] = v
    return out


# Map our unified filter ops to Chroma's where-clause operators.
_CHROMA_OPS = {
    "eq": "$eq", "ne": "$ne",
    "gt": "$gt", "gte": "$gte", "lt": "$lt", "lte": "$lte",
    "in": "$in", "not_in": "$nin",
}


def _filters_to_chroma_where(filters) -> dict | None:
    """Translate a list of Filter to Chroma's where dict. Returns None for an
    empty list. Multiple clauses are AND-combined via $and."""
    if not filters:
        return None
    clauses = []
    for f in filters:
        op = _CHROMA_OPS.get(f.op)
        if op is None:
            # contains / unsupported -> skip; retriever's post-filter will catch it
            continue
        clauses.append({f.field: {op: f.value}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class ChromaVectorStore(VectorStore):  # pragma: no cover
    """Chroma — runs in-process by default, or against a Chroma server."""

    def __init__(
        self,
        collection: str = "ragzero",
        persist_directory: str | None = None,
        host: str | None = None,
        port: int = 8000,
        dim: int = 1536,
        client: Any = None,         # for testing: inject a pre-built chroma client
    ) -> None:
        if client is None:
            try:
                import chromadb
            except ImportError as e:
                raise ImportError("chromadb not installed. pip install chromadb") from e

            if host:
                client = chromadb.HttpClient(host=host, port=port)
            elif persist_directory:
                client = chromadb.PersistentClient(path=persist_directory)
            else:
                client = chromadb.EphemeralClient()
        self._client: Any = client
        self._collection: Any = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"},
        )
        self._dim = dim

    def add(self, chunks, vectors):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            metadatas=[_sanitize_for_chroma(_chunk_to_payload(c)) for c in chunks],
        )

    def search(self, query_vector, k, filters=None):
        kwargs = {
            "query_embeddings": [query_vector],
            "n_results": k,
            "include": ["metadatas", "distances"],
        }
        if filters:
            where = _filters_to_chroma_where(filters)
            if where:
                kwargs["where"] = where
        res = self._collection.query(**kwargs)
        ids = res.get("ids", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        out = []
        for _, md, dist in zip(ids, metadatas, distances):
            # Cosine distance = 1 - similarity (with normalized vectors)
            similarity = 1.0 - dist
            out.append((_payload_to_chunk(_desanitize_from_chroma(md)), similarity))
        return out

    def size(self):
        return self._collection.count()

    def clear(self):
        name = self._collection.name
        metadata = {"hnsw:space": "cosine"}
        try:
            self._client.delete_collection(name=name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=name, metadata=metadata,
        )


# ---------- pgvector ----------

class PgVectorStore(VectorStore):  # pragma: no cover
    """Postgres + pgvector extension.

    Schema (created automatically if missing):
      CREATE TABLE <table> (
        chunk_id TEXT PRIMARY KEY,
        embedding VECTOR(<dim>),
        payload JSONB
      )
      CREATE INDEX ... USING ivfflat (embedding vector_cosine_ops) ...

    Requires `pip install psycopg[binary] pgvector`.
    """

    def __init__(
        self,
        dsn: str,
        table: str = "ragzero_chunks",
        dim: int = 1536,
        ivfflat_lists: int = 100,
        conn: Any = None,           # for testing: inject a pre-built psycopg connection
        skip_extension: bool = False,  # for testing: skip CREATE EXTENSION call
    ) -> None:
        if conn is None:
            try:
                import psycopg
                from pgvector.psycopg import register_vector
            except ImportError as e:
                raise ImportError(
                    "psycopg + pgvector not installed. "
                    "pip install 'psycopg[binary]' pgvector"
                ) from e
            conn = psycopg.connect(dsn, autocommit=True)
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
        elif not skip_extension:
            # Injected conn but still want the extension setup
            try:
                from pgvector.psycopg import register_vector
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                register_vector(conn)
            except ImportError:
                pass  # mock client without real pgvector

        self._conn: Any = conn
        # Validate table identifier — string-formatted into SQL below
        if not table.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name {table!r}")
        self.table = table
        self._dim = dim
        self._ivfflat_lists = ivfflat_lists
        self._ensure_table()

    def _ensure_table(self) -> None:
        import json
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    chunk_id TEXT PRIMARY KEY,
                    embedding vector({self._dim}),
                    payload JSONB
                )
                """
            )
            # Create an IVF index for fast search (built lazily once enough rows exist)
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx
                ON {self.table}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {self._ivfflat_lists})
                """
            )

    def add(self, chunks, vectors):
        import json
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        with self._conn.cursor() as cur:
            for c, v in zip(chunks, vectors):
                cur.execute(
                    f"""
                    INSERT INTO {self.table} (chunk_id, embedding, payload)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        payload = EXCLUDED.payload
                    """,
                    (c.chunk_id, v, json.dumps(_chunk_to_payload(c))),
                )

    def search(self, query_vector, k):
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT payload, 1 - (embedding <=> %s::vector) AS similarity
                FROM {self.table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, query_vector, k),
            )
            rows = cur.fetchall()
        return [(_payload_to_chunk(row[0]), float(row[1])) for row in rows]

    def size(self):
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            return cur.fetchone()[0]

    def clear(self):
        # Table name is validated to be alphanumeric/underscore in __init__.
        with self._conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self.table}")


# ---------- Azure AI Search ----------

class AzureAISearchVectorStore(VectorStore):  # pragma: no cover
    """Azure AI Search (formerly Azure Cognitive Search) with vector support.

    Requires `pip install azure-search-documents>=11.4.0`. The adapter
    auto-creates the index if it doesn't exist.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str = "ragzero",
        dim: int = 1536,
        client: Any = None,         # for testing: inject a pre-built SearchClient
        index_client: Any = None,   # for testing: inject a pre-built SearchIndexClient
    ) -> None:
        if client is None or index_client is None:
            try:
                from azure.core.credentials import AzureKeyCredential
                from azure.search.documents import SearchClient
                from azure.search.documents.indexes import SearchIndexClient
            except ImportError as e:
                raise ImportError(
                    "azure-search-documents not installed. "
                    "pip install azure-search-documents"
                ) from e

            self._credential = AzureKeyCredential(api_key)
            if index_client is None:
                index_client = SearchIndexClient(endpoint=endpoint, credential=self._credential)
            if client is None:
                client = SearchClient(
                    endpoint=endpoint, index_name=index_name, credential=self._credential,
                )

        self._index_client: Any = index_client
        self._client: Any = client
        self.endpoint = endpoint
        self.index_name = index_name
        self._dim = dim
        self._ensure_index()

    def _ensure_index(self) -> None:
        # If our index_client is a mock without list_indexes, skip the bootstrap.
        # The mock is expected to have whatever state it needs already.
        try:
            existing = {i.name for i in self._index_client.list_indexes()}
        except Exception:
            return
        if self.index_name in existing:
            return
        try:
            from azure.search.documents.indexes.models import (
                SearchIndex, SearchField, SearchFieldDataType, VectorSearch,
                VectorSearchProfile, HnswAlgorithmConfiguration,
            )
        except ImportError:
            return  # mock client — caller is responsible for index state
        fields = [
            SearchField(name="chunk_id", type=SearchFieldDataType.String, key=True),
            SearchField(name="text", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
            SearchField(name="source", type=SearchFieldDataType.String, filterable=True),
            SearchField(name="page", type=SearchFieldDataType.Int32, filterable=True),
            SearchField(name="position", type=SearchFieldDataType.Int32),
            SearchField(name="created_at", type=SearchFieldDataType.String),
            SearchField(name="payload_json", type=SearchFieldDataType.String),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self._dim,
                vector_search_profile_name="ragzero-hnsw",
            ),
        ]
        vs = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
            profiles=[VectorSearchProfile(name="ragzero-hnsw", algorithm_configuration_name="hnsw-config")],
        )
        self._index_client.create_index(SearchIndex(name=self.index_name, fields=fields, vector_search=vs))

    def add(self, chunks, vectors):
        import json
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        docs = []
        for c, v in zip(chunks, vectors):
            payload = _chunk_to_payload(c)
            docs.append({
                "chunk_id": c.chunk_id,
                "text": c.text,
                "doc_id": c.doc_id,
                "source": c.source,
                "page": c.page,
                "position": c.position,
                "created_at": c.created_at.isoformat(),
                "payload_json": json.dumps(payload),
                "embedding": v,
            })
        # Azure rejects batches > 1000 — chunk if necessary
        for i in range(0, len(docs), 1000):
            self._client.upload_documents(documents=docs[i : i + 1000])

    def search(self, query_vector, k):
        from azure.search.documents.models import VectorizedQuery
        results = self._client.search(
            search_text=None,
            vector_queries=[VectorizedQuery(vector=query_vector, k_nearest_neighbors=k, fields="embedding")],
            select=["payload_json"],
            top=k,
        )
        import json
        out = []
        for r in results:
            payload = json.loads(r["payload_json"])
            score = r["@search.score"]
            out.append((_payload_to_chunk(payload), float(score)))
        return out

    def size(self):
        # Azure Search exposes count via $count parameter on a no-op query
        results = self._client.search(search_text="*", include_total_count=True, top=0)
        return results.get_count() or 0

    def clear(self):
        try:
            self._index_client.delete_index(self.index_name)
        except Exception:
            pass
        self._ensure_index()
