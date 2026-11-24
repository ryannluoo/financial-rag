from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from financial_rag.retrieval.retriever import (
    QdrantRetriever,
    RetrievalConfig,
    RetrievedChunk,
    build_document_filter,
)


def _make_chunk(
    chunk_id: str = "doc1:p1:c0",
    *,
    document_id: str = "doc1",
    page: int = 1,
    chunk_index: int = 0,
    text: str = "Revenue grew 10%.",
    score: float = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        page=page,
        chunk_index=chunk_index,
        token_count=5,
        char_start=0,
        char_end=len(text),
        text=text,
        score=score,
    )


def _make_point(point_id: str, chunk: RetrievedChunk) -> models.ScoredPoint:
    return models.ScoredPoint(
        id=point_id,
        version=1,
        score=chunk.score,
        payload=chunk.model_dump(exclude={"score"}),
    )


def test_build_document_filter():
    assert build_document_filter([]) is None

    doc_filter = build_document_filter(["doc_b", "doc_a", "doc_b"])
    assert doc_filter == models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=["doc_b", "doc_a"]),
            )
        ]
    )

    with pytest.raises(TypeError, match="sequence of strings"):
        build_document_filter("doc_a")
    with pytest.raises(ValueError, match="non-empty"):
        build_document_filter(["doc_a", "  "])


def test_retrieval_config_validation():
    config = RetrievalConfig(collection_name="col_v1")
    assert config.top_k == 20
    assert config.vector_name == "dense"
    assert config.batch_size == 32
    assert config.bm25_weight == 0.0

    with pytest.raises(ValidationError):
        config.top_k = 10
    with pytest.raises(ValidationError):
        RetrievalConfig(collection_name="")
    with pytest.raises(ValidationError):
        RetrievalConfig(collection_name="col_v1", top_k=0)
    with pytest.raises(ValidationError):
        RetrievalConfig(collection_name="col_v1", batch_size=0)
    for invalid_weight in (-0.5, float("inf"), float("nan")):
        with pytest.raises(ValidationError, match="bm25_weight"):
            RetrievalConfig(collection_name="col_v1", bm25_weight=invalid_weight)


def test_qdrant_retriever_retrieve_and_batching():
    chunk_a = _make_chunk("doc_a:p1:c0", document_id="doc_a", score=0.9)
    chunk_b = _make_chunk("doc_b:p2:c0", document_id="doc_b", page=2, score=0.7)

    embedder = Mock()
    embedder.dimension = 2
    embedder.embed_query.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    client = Mock(spec=QdrantClient)
    client.query_batch_points.side_effect = [
        [SimpleNamespace(points=[_make_point("1", chunk_a)])],
        [SimpleNamespace(points=[_make_point("2", chunk_b)])],
    ]

    retriever = QdrantRetriever(
        RetrievalConfig(collection_name="col_v1", top_k=5, batch_size=1),
        client,
        embedder,
    )
    doc_filter = build_document_filter(["doc_a"])

    results = retriever.retrieve_batch(
        ["Q1", "Q2"],
        top_k=3,
        query_filters=[doc_filter, None],
    )

    assert results == [[chunk_a], [chunk_b]]
    embedder.embed_query.assert_called_once_with(["Q1", "Q2"])
    assert client.query_batch_points.call_count == 2

    first_req = client.query_batch_points.call_args_list[0].kwargs["requests"][0]
    assert first_req.using == "dense"
    assert first_req.filter == doc_filter
    assert first_req.limit == 3
    assert first_req.prefetch is None
    assert first_req.with_vector is False
    assert first_req.with_payload is True

    embedder.embed_query.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
    client.query_batch_points.side_effect = [[SimpleNamespace(points=[_make_point("1", chunk_a)])]]
    assert retriever.retrieve("Q1", query_filter=doc_filter) == [chunk_a]


@pytest.mark.parametrize(
    ("query", "bm25_weight", "expected_first"),
    [
        ("Revenue", 0.25, "dense-first"),
        ("Revenue", 2.0, "bm25-first"),
        ("Revenue cash", 1.0, "shared-hit"),
    ],
)
def test_qdrant_hybrid_fusion_respects_weight_filter_and_top_k(query, bm25_weight, expected_first):
    collection = "reports"
    client = QdrantClient(":memory:")
    client.create_collection(
        collection,
        vectors_config={"dense": models.VectorParams(size=2, distance=models.Distance.COSINE)},
    )
    client.create_payload_index(
        collection,
        "text",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
            scoring=True,
        ),
    )
    rows = [
        ([1.0, 0.0], _make_chunk("dense-first", text="Liabilities on the balance sheet")),
        ([0.8, 0.6], _make_chunk("shared-hit", text="Cash flow statement attached")),
        ([0.0, 1.0], _make_chunk("bm25-first", text="Revenue revenue revenue")),
        ([-1.0, 0.0], _make_chunk("bm25-second", text="Gross revenue is split across divisions")),
        ([1.0, 0.0], _make_chunk("other-report", document_id="doc2", text="Revenue")),
    ]
    client.upsert(
        collection,
        points=[
            models.PointStruct(
                id=point_id, vector={"dense": vector}, payload=chunk.model_dump(exclude={"score"})
            )
            for point_id, (vector, chunk) in enumerate(rows, start=1)
        ],
    )
    embedder = Mock(dimension=2)
    embedder.embed_query.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
    retriever = QdrantRetriever(
        RetrievalConfig(collection_name=collection, bm25_weight=bm25_weight), client, embedder
    )

    result = retriever.retrieve(query, top_k=2, query_filter=build_document_filter(["doc1"]))

    assert len(result) == 2
    assert len({chunk.chunk_id for chunk in result}) == 2
    assert {chunk.document_id for chunk in result} == {"doc1"}
    assert result[0].chunk_id == expected_first


def test_qdrant_retriever_empty_queries_skips_embedder_and_client():
    embedder = Mock(dimension=2)
    client = Mock(spec=QdrantClient)
    retriever = QdrantRetriever(RetrievalConfig(collection_name="col_v1"), client, embedder)

    assert retriever.retrieve_batch([]) == []
    embedder.embed_query.assert_not_called()
    client.query_batch_points.assert_not_called()


def test_qdrant_retriever_validation_and_error_cases():
    embedder = Mock(dimension=2)
    client = Mock(spec=QdrantClient)
    retriever = QdrantRetriever(RetrievalConfig(collection_name="col_v1"), client, embedder)

    with pytest.raises(TypeError, match="list of strings"):
        retriever.retrieve_batch("single query")
    with pytest.raises(ValueError, match="non-empty"):
        retriever.retrieve_batch(["valid", "   "])
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve_batch(["valid"], top_k=0)
    with pytest.raises(ValueError, match="query_filters"):
        retriever.retrieve_batch(["q1", "q2"], query_filters=[None])

    embedder.embed_query.return_value = np.ones((1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        retriever.retrieve("q1")

    embedder.embed_query.return_value = np.array([[np.nan, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        retriever.retrieve("q1")

    embedder.embed_query.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
    client.query_batch_points.return_value = []
    with pytest.raises(RuntimeError, match="different number of responses"):
        retriever.retrieve("q1")

    bad_point = models.ScoredPoint(id="bad", version=1, score=0.5, payload={"chunk_id": "only"})
    client.query_batch_points.return_value = [SimpleNamespace(points=[bad_point])]
    with pytest.raises(ValueError, match="Invalid payload"):
        retriever.retrieve("q1")
