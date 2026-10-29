"""Unit tests for bounded AstraDBVectorStore text insertion."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from langchain_astradb.vectorstores import AstraDBVectorStore


class RecordingEmbeddings:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text))] for text in texts]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


class RecordingCollection:
    def __init__(self) -> None:
        self.batches: list[list[dict[str, Any]]] = []

    def insert_many(
        self, documents: list[dict[str, Any]], **_kwargs: Any
    ) -> SimpleNamespace:
        self.batches.append(list(documents))
        return SimpleNamespace(inserted_ids=[doc["_id"] for doc in documents])


class AsyncRecordingCollection(RecordingCollection):
    async def insert_many(
        self, documents: list[dict[str, Any]], **_kwargs: Any
    ) -> SimpleNamespace:
        return super().insert_many(documents, **_kwargs)


def make_store(
    *, server_side_embeddings: bool
) -> tuple[AstraDBVectorStore, RecordingCollection, RecordingEmbeddings]:
    """Build a store without network, using in-memory collections."""
    store = object.__new__(AstraDBVectorStore)
    stub: Any = store
    embeddings = RecordingEmbeddings()
    collection = RecordingCollection()
    async_collection = AsyncRecordingCollection()

    async def noop_async() -> None:
        pass

    stub.batch_size = 2
    stub.bulk_insert_batch_concurrency = 2
    stub.bulk_insert_overwrite_concurrency = 2
    stub.embedding = None if server_side_embeddings else embeddings
    stub.document_codec = SimpleNamespace(
        server_side_embeddings=server_side_embeddings,
        encode=lambda *, content, document_id, vector, metadata: {
            "_id": document_id,
            "content": content,
            "$vector": vector,
            "metadata": metadata,
        },
        get_id=lambda doc: doc["_id"],
    )
    stub.astra_env = SimpleNamespace(
        ensure_db_setup=lambda: None,
        aensure_db_setup=noop_async,
        collection=collection,
        async_collection=async_collection,
    )
    return store, collection, embeddings


@pytest.mark.parametrize("server_side_embeddings", [False, True])
def test_add_texts_streams_one_shot_generators(server_side_embeddings: bool) -> None:
    store, collection, embeddings = make_store(
        server_side_embeddings=server_side_embeddings
    )
    result = store.add_texts(
        (f"text-{i}" for i in range(5)),
        metadatas=({"position": i} for i in range(5)),
        ids=(f"doc-{i}" for i in range(5)),
        batch_size=2,
    )

    assert result == [f"doc-{i}" for i in range(5)]
    assert [len(batch) for batch in collection.batches] == [2, 2, 1]
    inserted = [doc for batch in collection.batches for doc in batch]
    assert [doc["content"] for doc in inserted] == [f"text-{i}" for i in range(5)]
    assert [doc["metadata"] for doc in inserted] == [
        {"position": i} for i in range(5)
    ]
    assert all((doc["$vector"] is None) == server_side_embeddings for doc in inserted)
    assert [len(batch) for batch in embeddings.batches] == (
        [] if server_side_embeddings else [2, 2, 1]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("server_side_embeddings", [False, True])
async def test_aadd_texts_streams_one_shot_generators(
    server_side_embeddings: bool,
) -> None:
    store, _, embeddings = make_store(
        server_side_embeddings=server_side_embeddings
    )
    result = await store.aadd_texts(
        (f"text-{i}" for i in range(5)),
        metadatas=({"position": i} for i in range(5)),
        ids=(f"doc-{i}" for i in range(5)),
        batch_size=2,
    )
    assert result == [f"doc-{i}" for i in range(5)]
    inserted_batches = cast("Any", store.astra_env.async_collection).batches
    assert [len(batch) for batch in inserted_batches] == [2, 2, 1]
    inserted = [doc for batch in inserted_batches for doc in batch]
    assert [doc["metadata"] for doc in inserted] == [
        {"position": i} for i in range(5)
    ]
    assert [len(batch) for batch in embeddings.batches] == (
        [] if server_side_embeddings else [2, 2, 1]
    )


def test_add_texts_streams_large_list_in_bounded_batches() -> None:
    store, collection, _ = make_store(server_side_embeddings=True)
    ids = store.add_texts(
        [f"text-{i}" for i in range(5)],
        ids=[f"doc-{i}" for i in range(5)],
        batch_size=2,
    )
    assert ids == [f"doc-{i}" for i in range(5)]
    assert [len(batch) for batch in collection.batches] == [2, 2, 1]


def test_add_texts_empty_generator_is_noop() -> None:
    store, collection, _ = make_store(server_side_embeddings=True)
    assert store.add_texts(iter(()), batch_size=2) == []
    assert collection.batches == []


def test_add_texts_rejects_short_metadata_iterator_before_inserting_batch() -> None:
    store, collection, _ = make_store(server_side_embeddings=True)
    with pytest.raises(ValueError, match="metadatas must have the same number"):
        store.add_texts(iter(["a", "b"]), metadatas=iter([{}]), batch_size=2)
    assert collection.batches == []


def test_add_texts_rejects_extra_ids() -> None:
    store, _, _ = make_store(server_side_embeddings=True)
    with pytest.raises(ValueError, match="ids must have the same number"):
        store.add_texts(iter(["a"]), ids=iter(["one", "two"]), batch_size=2)


def test_add_texts_rejects_nonpositive_batch_size() -> None:
    store, _, _ = make_store(server_side_embeddings=True)
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        store.add_texts(iter(["a"]), batch_size=-1)
