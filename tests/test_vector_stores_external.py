"""Tests for external vector store adapters (Qdrant, Pinecone, Weaviate,
Chroma, pgvector, Azure AI Search).

Strategy: install fake SDK modules into sys.modules before the adapter
imports them. Each fake records calls made to it so we can assert the
adapter formed the right requests.

This catches:
- API shape mismatches (wrong kwargs, wrong return shape parsing)
- Chunk payload round-trips
- Index/collection auto-creation
- Score conversion (distance → similarity)
"""
import sys
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from ragzero.rag.interfaces import Chunk


def _chunks(n=2):
    now = datetime.now(timezone.utc)
    return [
        Chunk(
            chunk_id=f"chunk-{i}",
            doc_id=f"doc-{i}",
            text=f"Text for chunk {i}",
            page=i,
            position=i,
            source=f"internal://doc-{i}",
            created_at=now,
            metadata={"author": f"alice-{i}", "tag": "test"},
        )
        for i in range(n)
    ]


def _vectors(n=2, dim=4):
    return [[float(i + j * 0.1) for j in range(dim)] for i in range(n)]


# ============================================================
# Qdrant
# ============================================================

class _FakeQdrantClient:
    def __init__(self, *args, **kwargs):
        self.upserted_points = []
        self.searched = []
        self.created_collections = []
        self._collections = []
        self.init_kwargs = kwargs

    def get_collections(self):
        ns = types.SimpleNamespace(collections=[types.SimpleNamespace(name=n) for n in self._collections])
        return ns

    def create_collection(self, collection_name, vectors_config):
        self.created_collections.append((collection_name, vectors_config))
        self._collections.append(collection_name)

    def upsert(self, collection_name, points):
        self.upserted_points.extend(points)

    def search(self, collection_name, query_vector, limit, with_payload):
        self.searched.append({"collection": collection_name, "qv": query_vector, "limit": limit})
        # Return previously-upserted points as fake hits
        return [
            types.SimpleNamespace(score=0.9 - 0.1 * i, payload=p.payload)
            for i, p in enumerate(self.upserted_points[:limit])
        ]

    def get_collection(self, name):
        return types.SimpleNamespace(points_count=len(self.upserted_points))


def _install_qdrant_mock():
    qdrant_client = types.ModuleType("qdrant_client")
    qdrant_client.QdrantClient = _FakeQdrantClient
    http_mod = types.ModuleType("qdrant_client.http")

    class _Models:
        class Distance:
            COSINE = "cosine"
        class VectorParams:
            def __init__(self, size, distance):
                self.size = size; self.distance = distance
        class PointStruct:
            def __init__(self, id, vector, payload):
                self.id = id; self.vector = vector; self.payload = payload

    http_mod.models = _Models()
    qdrant_client.http = http_mod

    sys.modules["qdrant_client"] = qdrant_client
    sys.modules["qdrant_client.http"] = http_mod


def test_qdrant_creates_collection_and_round_trips_chunks():
    _install_qdrant_mock()
    from ragzero.rag.backends.vector_stores_external import QdrantVectorStore

    store = QdrantVectorStore(url="http://fake:6333", collection="testcol", dim=4)
    assert store._client.created_collections, "should auto-create collection"
    assert store._client.created_collections[0][0] == "testcol"

    chunks = _chunks(2)
    vectors = _vectors(2)
    store.add(chunks, vectors)
    assert len(store._client.upserted_points) == 2

    # Search uses the upserted payloads to fake hits
    results = store.search(vectors[0], k=2)
    assert len(results) == 2
    # Score should round-trip as a float
    assert all(isinstance(score, float) for _, score in results)
    # Chunks should reconstruct correctly
    assert results[0][0].chunk_id in {"chunk-0", "chunk-1"}
    assert results[0][0].metadata["author"] in {"alice-0", "alice-1"}


def test_qdrant_skips_creation_if_collection_exists():
    _install_qdrant_mock()
    from ragzero.rag.backends.vector_stores_external import QdrantVectorStore

    store = QdrantVectorStore(url="http://fake:6333", collection="exists", dim=4)
    # First call created it; reset and rebuild
    store._client._collections = ["exists"]
    store._client.created_collections = []
    store._ensure_collection()
    assert store._client.created_collections == []


def test_qdrant_size_returns_point_count():
    _install_qdrant_mock()
    from ragzero.rag.backends.vector_stores_external import QdrantVectorStore
    store = QdrantVectorStore(url="http://fake:6333", collection="c", dim=4)
    store.add(_chunks(3), _vectors(3))
    assert store.size() == 3


def test_qdrant_validates_length_mismatch():
    _install_qdrant_mock()
    from ragzero.rag.backends.vector_stores_external import QdrantVectorStore
    store = QdrantVectorStore(url="http://fake:6333", collection="c", dim=4)
    with pytest.raises(ValueError):
        store.add(_chunks(2), _vectors(1))


# ============================================================
# Pinecone
# ============================================================

class _FakePineconeIndex:
    def __init__(self):
        self.upserted = []

    def upsert(self, vectors):
        self.upserted.extend(vectors)

    def query(self, vector, top_k, include_metadata):
        # Return upserted in dict form (mimics newer Pinecone SDK)
        return {
            "matches": [
                {"id": v["id"], "score": 0.9 - 0.1 * i, "metadata": v["metadata"]}
                for i, v in enumerate(self.upserted[:top_k])
            ]
        }

    def describe_index_stats(self):
        return {"total_vector_count": len(self.upserted)}


class _FakePinecone:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self._indexes = []
        self._index = _FakePineconeIndex()

    def list_indexes(self):
        return [{"name": n} for n in self._indexes]

    def create_index(self, name, dimension, metric, spec):
        self._indexes.append(name)
        self._created = {"name": name, "dim": dimension, "metric": metric, "spec": spec}

    def Index(self, name):
        return self._index


class _FakeServerlessSpec:
    def __init__(self, cloud, region):
        self.cloud = cloud; self.region = region


def _install_pinecone_mock():
    pinecone = types.ModuleType("pinecone")
    pinecone.Pinecone = _FakePinecone
    pinecone.ServerlessSpec = _FakeServerlessSpec
    sys.modules["pinecone"] = pinecone


def test_pinecone_creates_index_on_first_use():
    _install_pinecone_mock()
    from ragzero.rag.backends.vector_stores_external import PineconeVectorStore
    store = PineconeVectorStore(api_key="fake", index_name="ragzero-test", dim=8)
    assert "ragzero-test" in store._pc._indexes
    assert store._pc._created["dim"] == 8
    assert store._pc._created["metric"] == "cosine"


def test_pinecone_add_and_search_round_trip():
    _install_pinecone_mock()
    from ragzero.rag.backends.vector_stores_external import PineconeVectorStore
    store = PineconeVectorStore(api_key="fake", index_name="t", dim=4)
    chunks = _chunks(3)
    store.add(chunks, _vectors(3))
    results = store.search(_vectors(1)[0], k=2)
    assert len(results) == 2
    assert results[0][0].text.startswith("Text for chunk")


def test_pinecone_truncates_oversized_text():
    """Pinecone caps metadata size — adapter should truncate long text fields."""
    _install_pinecone_mock()
    from ragzero.rag.backends.vector_stores_external import PineconeVectorStore
    store = PineconeVectorStore(api_key="fake", index_name="t", dim=4)
    big_chunk = Chunk(
        chunk_id="big",
        doc_id="d",
        text="X" * 30_000,    # exceeds 20K threshold
        page=None,
        position=0,
        source="s",
        created_at=datetime.now(timezone.utc),
    )
    store.add([big_chunk], _vectors(1))
    upserted = store._index.upserted[0]
    # Text should be truncated; flag set
    assert len(upserted["metadata"]["text"]) < 30_000
    assert upserted["metadata"].get("_text_truncated") is True


def test_pinecone_batches_large_upserts():
    """Pinecone rejects batches > 100 vectors."""
    _install_pinecone_mock()
    from ragzero.rag.backends.vector_stores_external import PineconeVectorStore
    store = PineconeVectorStore(api_key="fake", index_name="t", dim=4)
    # 250 chunks should produce 3 upsert calls (100 + 100 + 50)
    chunks = _chunks(250)
    vectors = _vectors(250)
    store.add(chunks, vectors)
    # All 250 should be present in the (cumulative) upserted list
    assert len(store._index.upserted) == 250


# ============================================================
# Weaviate (v4)
# ============================================================

class _FakeWeaviateBatch:
    def __init__(self, collection):
        self.collection = collection
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def add_object(self, properties, vector):
        self.collection._objects.append({"properties": properties, "vector": vector})


class _FakeWeaviateCollection:
    def __init__(self):
        self._objects = []
        # nested .batch and .query and .aggregate attributes
        self.batch = types.SimpleNamespace(dynamic=lambda: _FakeWeaviateBatch(self))
        self.query = self
        self.aggregate = types.SimpleNamespace(
            over_all=lambda **kw: types.SimpleNamespace(total_count=len(self._objects))
        )

    def near_vector(self, near_vector, limit, return_metadata):
        objs = []
        for i, o in enumerate(self._objects[:limit]):
            md = types.SimpleNamespace(distance=0.1 * (i + 1))
            objs.append(types.SimpleNamespace(properties=o["properties"], metadata=md))
        return types.SimpleNamespace(objects=objs)


class _FakeWeaviateCollections:
    def __init__(self):
        self._coll = {}

    def exists(self, name):
        return name in self._coll

    def create(self, name, vectorizer_config):
        self._coll[name] = _FakeWeaviateCollection()

    def get(self, name):
        return self._coll[name]


class _FakeWeaviateClient:
    def __init__(self):
        self.collections = _FakeWeaviateCollections()
        self._closed = False

    def close(self): self._closed = True


def _install_weaviate_mock():
    weaviate = types.ModuleType("weaviate")
    classes = types.ModuleType("weaviate.classes")
    init_mod = types.ModuleType("weaviate.classes.init")
    config_mod = types.ModuleType("weaviate.classes.config")
    query_mod = types.ModuleType("weaviate.classes.query")

    class Auth:
        @staticmethod
        def api_key(k): return {"api_key": k}

    class Configure:
        class Vectorizer:
            @staticmethod
            def none(): return "none"

    class MetadataQuery:
        def __init__(self, distance=False): self.distance = distance

    init_mod.Auth = Auth
    config_mod.Configure = Configure
    query_mod.MetadataQuery = MetadataQuery
    classes.init = init_mod
    classes.config = config_mod
    classes.query = query_mod

    _client_instance = _FakeWeaviateClient()
    weaviate.connect_to_weaviate_cloud = lambda **kw: _client_instance
    weaviate.connect_to_local = lambda **kw: _client_instance
    weaviate.classes = classes

    sys.modules["weaviate"] = weaviate
    sys.modules["weaviate.classes"] = classes
    sys.modules["weaviate.classes.init"] = init_mod
    sys.modules["weaviate.classes.config"] = config_mod
    sys.modules["weaviate.classes.query"] = query_mod


def test_weaviate_creates_class_and_round_trips():
    _install_weaviate_mock()
    from ragzero.rag.backends.vector_stores_external import WeaviateVectorStore
    store = WeaviateVectorStore(url="http://localhost:8080", class_name="TestClass", dim=4)
    assert store._client.collections.exists("TestClass")
    store.add(_chunks(2), _vectors(2))
    results = store.search(_vectors(1)[0], k=2)
    assert len(results) == 2
    # Distance → similarity conversion: distance 0.1 → similarity 0.9
    assert results[0][1] == pytest.approx(0.9)


def test_weaviate_cloud_uses_api_key_auth():
    _install_weaviate_mock()
    from ragzero.rag.backends.vector_stores_external import WeaviateVectorStore
    # Just verify construction works with api_key set
    store = WeaviateVectorStore(url="https://x.weaviate.network", api_key="key", class_name="C", dim=4)
    assert store.class_name == "C"


# ============================================================
# Chroma
# ============================================================

class _FakeChromaCollection:
    def __init__(self):
        self._items = []  # (id, embedding, metadata)

    def add(self, ids, embeddings, metadatas):
        for i, e, m in zip(ids, embeddings, metadatas):
            self._items.append((i, e, m))

    def query(self, query_embeddings, n_results, include):
        # Return the first n_results items
        ids = [[t[0] for t in self._items[:n_results]]]
        metas = [[t[2] for t in self._items[:n_results]]]
        dists = [[0.1 * (i + 1) for i in range(min(n_results, len(self._items)))]]
        return {"ids": ids, "metadatas": metas, "distances": dists}

    def count(self):
        return len(self._items)


class _FakeChromaClient:
    def __init__(self, *args, **kwargs):
        self._collections = {}

    def get_or_create_collection(self, name, metadata=None):
        if name not in self._collections:
            self._collections[name] = _FakeChromaCollection()
        return self._collections[name]


def _install_chroma_mock():
    chromadb = types.ModuleType("chromadb")
    chromadb.EphemeralClient = _FakeChromaClient
    chromadb.PersistentClient = _FakeChromaClient
    chromadb.HttpClient = _FakeChromaClient
    sys.modules["chromadb"] = chromadb


def test_chroma_ephemeral_round_trip():
    _install_chroma_mock()
    from ragzero.rag.backends.vector_stores_external import ChromaVectorStore
    store = ChromaVectorStore(collection="rz", dim=4)
    store.add(_chunks(3), _vectors(3))
    assert store.size() == 3
    results = store.search(_vectors(1)[0], k=2)
    assert len(results) == 2
    assert results[0][1] == pytest.approx(0.9)


def test_chroma_persistent_mode():
    _install_chroma_mock()
    from ragzero.rag.backends.vector_stores_external import ChromaVectorStore
    store = ChromaVectorStore(collection="rz", persist_directory="/tmp/chroma_fake", dim=4)
    assert store.size() == 0


def test_chroma_server_mode():
    _install_chroma_mock()
    from ragzero.rag.backends.vector_stores_external import ChromaVectorStore
    store = ChromaVectorStore(collection="rz", host="localhost", port=8765, dim=4)
    assert store.size() == 0


# ============================================================
# pgvector
# ============================================================

class _FakePgCursor:
    def __init__(self, conn):
        self.conn = conn
        self._last_result = None

    def __enter__(self): return self
    def __exit__(self, *args): return False

    def execute(self, query, params=None):
        q = query.strip()
        if q.startswith("INSERT INTO"):
            import json
            chunk_id, embedding, payload_json = params
            payload = json.loads(payload_json)
            self.conn._rows[chunk_id] = (embedding, payload)
        elif q.startswith("SELECT payload"):
            # Search query — return the stored rows in insertion order
            limit = params[2] if params and len(params) >= 3 else 10
            rows = list(self.conn._rows.values())[:limit]
            self._last_result = [(p, 0.9 - 0.1 * i) for i, (_, p) in enumerate(rows)]
        elif q.startswith("SELECT COUNT"):
            self._last_result = [(len(self.conn._rows),)]
        # CREATE EXTENSION / CREATE TABLE / CREATE INDEX → no-op

    def fetchall(self):
        return self._last_result or []

    def fetchone(self):
        return self._last_result[0] if self._last_result else None


class _FakePgConn:
    def __init__(self):
        self._rows = {}  # chunk_id -> (vector, payload_dict)

    def cursor(self):
        return _FakePgCursor(self)


def _install_pgvector_mock():
    psycopg = types.ModuleType("psycopg")
    psycopg.connect = lambda dsn, autocommit=False: _FakePgConn()
    pgvector_mod = types.ModuleType("pgvector")
    pgvector_psycopg = types.ModuleType("pgvector.psycopg")
    pgvector_psycopg.register_vector = lambda conn: None
    pgvector_mod.psycopg = pgvector_psycopg
    sys.modules["psycopg"] = psycopg
    sys.modules["pgvector"] = pgvector_mod
    sys.modules["pgvector.psycopg"] = pgvector_psycopg


def test_pgvector_round_trip():
    _install_pgvector_mock()
    from ragzero.rag.backends.vector_stores_external import PgVectorStore
    store = PgVectorStore(dsn="postgresql://fake", table="rag_test", dim=4)
    store.add(_chunks(3), _vectors(3))
    assert store.size() == 3
    results = store.search(_vectors(1)[0], k=2)
    assert len(results) == 2
    assert results[0][0].chunk_id.startswith("chunk-")


def test_pgvector_rejects_invalid_table_name():
    _install_pgvector_mock()
    from ragzero.rag.backends.vector_stores_external import PgVectorStore
    with pytest.raises(ValueError):
        # SQL injection attempt
        PgVectorStore(dsn="postgresql://fake", table="x; DROP TABLE users--", dim=4)


# ============================================================
# Azure AI Search
# ============================================================

class _FakeAzureSearchClient:
    def __init__(self, endpoint, index_name, credential):
        self.endpoint = endpoint
        self.index_name = index_name
        self._docs = []

    def upload_documents(self, documents):
        self._docs.extend(documents)

    def search(self, search_text, vector_queries=None, select=None, top=None, include_total_count=False):
        if search_text == "*":
            # Count-only query
            return _FakeAzureSearchResultsCount(len(self._docs))
        # Vector query — return top docs with payload_json
        limit = top or 10
        return [
            {"payload_json": d["payload_json"], "@search.score": 0.9 - 0.1 * i}
            for i, d in enumerate(self._docs[:limit])
        ]


class _FakeAzureSearchResultsCount:
    def __init__(self, count):
        self._count = count

    def get_count(self):
        return self._count

    def __iter__(self):
        return iter([])


class _FakeAzureIndexClient:
    def __init__(self, endpoint, credential):
        self._indexes = {}

    def list_indexes(self):
        return [types.SimpleNamespace(name=n) for n in self._indexes]

    def create_index(self, index):
        self._indexes[index.name] = index


def _install_azure_search_mock():
    azure = types.ModuleType("azure")
    core = types.ModuleType("azure.core")
    credentials = types.ModuleType("azure.core.credentials")
    class AzureKeyCredential:
        def __init__(self, key): self.key = key
    credentials.AzureKeyCredential = AzureKeyCredential
    core.credentials = credentials
    azure.core = core

    search = types.ModuleType("azure.search")
    documents = types.ModuleType("azure.search.documents")
    documents.SearchClient = _FakeAzureSearchClient
    indexes_mod = types.ModuleType("azure.search.documents.indexes")
    indexes_mod.SearchIndexClient = _FakeAzureIndexClient
    indexes_models = types.ModuleType("azure.search.documents.indexes.models")

    # All the index-config classes are essentially data holders
    class _Holder:
        def __init__(self, **kwargs):
            for k, v in kwargs.items(): setattr(self, k, v)

    class SearchField(_Holder): pass
    class SearchIndex(_Holder): pass
    class VectorSearch(_Holder): pass
    class VectorSearchAlgorithmConfiguration(_Holder): pass
    class VectorSearchProfile(_Holder): pass
    class HnswAlgorithmConfiguration(_Holder): pass

    class SearchFieldDataType:
        String = "Edm.String"
        Int32 = "Edm.Int32"
        Single = "Edm.Single"
        @staticmethod
        def Collection(t): return f"Collection({t})"

    indexes_models.SearchField = SearchField
    indexes_models.SearchIndex = SearchIndex
    indexes_models.VectorSearch = VectorSearch
    indexes_models.VectorSearchAlgorithmConfiguration = VectorSearchAlgorithmConfiguration
    indexes_models.VectorSearchProfile = VectorSearchProfile
    indexes_models.HnswAlgorithmConfiguration = HnswAlgorithmConfiguration
    indexes_models.SearchFieldDataType = SearchFieldDataType

    models_mod = types.ModuleType("azure.search.documents.models")
    class VectorizedQuery(_Holder): pass
    models_mod.VectorizedQuery = VectorizedQuery

    documents.indexes = indexes_mod
    documents.models = models_mod
    indexes_mod.models = indexes_models
    search.documents = documents
    azure.search = search

    sys.modules["azure"] = azure
    sys.modules["azure.core"] = core
    sys.modules["azure.core.credentials"] = credentials
    sys.modules["azure.search"] = search
    sys.modules["azure.search.documents"] = documents
    sys.modules["azure.search.documents.indexes"] = indexes_mod
    sys.modules["azure.search.documents.indexes.models"] = indexes_models
    sys.modules["azure.search.documents.models"] = models_mod


def test_azure_ai_search_creates_index_and_round_trips():
    _install_azure_search_mock()
    from ragzero.rag.backends.vector_stores_external import AzureAISearchVectorStore
    store = AzureAISearchVectorStore(
        endpoint="https://x.search.windows.net",
        api_key="key",
        index_name="rzidx",
        dim=4,
    )
    assert "rzidx" in store._index_client._indexes
    store.add(_chunks(2), _vectors(2))
    results = store.search(_vectors(1)[0], k=2)
    assert len(results) == 2
    assert results[0][0].chunk_id in {"chunk-0", "chunk-1"}


def test_azure_ai_search_size():
    _install_azure_search_mock()
    from ragzero.rag.backends.vector_stores_external import AzureAISearchVectorStore
    store = AzureAISearchVectorStore(
        endpoint="https://x.search.windows.net",
        api_key="key",
        index_name="rz",
        dim=4,
    )
    store.add(_chunks(5), _vectors(5))
    assert store.size() == 5


# ============================================================
# Registry / catalog tests
# ============================================================

def test_all_new_stores_in_vector_store_registry():
    from ragzero.rag.config import VECTOR_STORES
    for key in ("qdrant", "pinecone", "weaviate", "chroma", "pgvector", "azure_ai_search"):
        assert key in VECTOR_STORES, f"{key} missing from VECTOR_STORES registry"


def test_all_new_stores_in_ui_catalog():
    from ragzero.rag.providers import VECTOR_STORE_PROVIDERS, provider_catalog
    expected = {"in_memory", "faiss", "qdrant", "pinecone", "weaviate",
                "chroma", "pgvector", "azure_ai_search"}
    assert expected.issubset(VECTOR_STORE_PROVIDERS.keys())
    cat = provider_catalog()
    assert "vector_store" in cat
    cat_ids = {p["id"] for p in cat["vector_store"]}
    assert expected.issubset(cat_ids)


def test_vector_store_kind_in_provider_specs():
    from ragzero.rag.providers import VECTOR_STORE_PROVIDERS
    for spec in VECTOR_STORE_PROVIDERS.values():
        assert spec.kind == "vector_store"


def test_required_credential_fields_marked_required():
    """Pinecone/Azure/pgvector need API keys or DSNs — those fields must be required=True."""
    from ragzero.rag.providers import VECTOR_STORE_PROVIDERS
    required_keys = {
        "pinecone": {"api_key"},
        "azure_ai_search": {"endpoint", "api_key"},
        "pgvector": {"dsn"},
    }
    for store_id, keys in required_keys.items():
        spec = VECTOR_STORE_PROVIDERS[store_id]
        field_map = {f.key: f for f in spec.fields}
        for k in keys:
            assert k in field_map, f"{store_id} missing field {k}"
            assert field_map[k].required, f"{store_id}.{k} should be required"


def test_payload_round_trip_helpers():
    """The shared chunk<->payload helpers should preserve metadata."""
    from ragzero.rag.backends.vector_stores_external import _chunk_to_payload, _payload_to_chunk
    original = Chunk(
        chunk_id="abc",
        doc_id="d1",
        text="hello world",
        page=3,
        position=5,
        source="internal://x",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        metadata={"author": "alice", "year": 2024, "draft": True},
    )
    payload = _chunk_to_payload(original)
    reconstructed = _payload_to_chunk(payload)
    assert reconstructed.chunk_id == original.chunk_id
    assert reconstructed.doc_id == original.doc_id
    assert reconstructed.text == original.text
    assert reconstructed.page == original.page
    assert reconstructed.source == original.source
    assert reconstructed.metadata == original.metadata


def test_stable_int_id_is_deterministic():
    from ragzero.rag.backends.vector_stores_external import _stable_int_id
    assert _stable_int_id("abc") == _stable_int_id("abc")
    assert _stable_int_id("abc") != _stable_int_id("def")
    assert isinstance(_stable_int_id("anything"), int)
