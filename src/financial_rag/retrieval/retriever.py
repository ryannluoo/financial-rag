from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from qdrant_client import QdrantClient, models

from financial_rag.embedding.embedder import Embedder


def build_document_filter(document_ids: Sequence[str]) -> models.Filter | None:
    if isinstance(document_ids, str):
        raise TypeError("document_ids must be a sequence of strings, not a single str.")
    if any(not doc_id.strip() for doc_id in document_ids):
        raise ValueError("document_ids must contain only non-empty strings.")
    if not document_ids:
        return None

    unique_ids = list(dict.fromkeys(document_ids))
    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=unique_ids),
            )
        ]
    )


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    collection_name: str = Field(min_length=1)
    top_k: int = Field(default=20, ge=1)
    vector_name: str = Field(default="dense", min_length=1)
    batch_size: int = Field(default=32, ge=1)
    bm25_weight: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description="BM25 fusion weight relative to a dense weight of 1.0. Zero selects dense search.",
    )


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    page: int
    chunk_index: int
    token_count: int
    char_start: int
    char_end: int
    text: str
    score: float = Field(description="Cosine similarity in dense search or the RRF score in hybrid search.")


class QdrantRetriever:
    def __init__(self, config: RetrievalConfig, client: QdrantClient, embedder: Embedder):
        self.config = config
        self._client = client
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        return self.retrieve_batch(
            [query],
            top_k=top_k,
            query_filters=[query_filter],
        )[0]

    def retrieve_batch(
        self,
        queries: list[str],
        *,
        top_k: int | None = None,
        query_filters: list[models.Filter | None] | None = None,
    ) -> list[list[RetrievedChunk]]:
        if isinstance(queries, str):
            raise TypeError(
                "queries must be a list of strings. Use retrieve() for a single string query."
            )
        if any(not q.strip() for q in queries):
            raise ValueError("queries must contain only non-empty strings.")

        limit = self.config.top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be at least 1.")

        filters = [None] * len(queries) if query_filters is None else query_filters
        if len(filters) != len(queries):
            raise ValueError("query_filters must have one entry per query.")

        if not queries:
            return []

        query_embeddings = self._embedder.embed_query(queries)
        if (
            query_embeddings.ndim != 2
            or query_embeddings.shape[0] != len(queries)
            or query_embeddings.shape[1] != self._embedder.dimension
        ):
            raise ValueError(
                f"Expected query_embeddings to have shape ({len(queries)}, {self._embedder.dimension}), "
                f"but got shape {query_embeddings.shape}."
            )
        if not np.isfinite(query_embeddings).all():
            raise ValueError("query_embeddings contain non-finite values.")

        requests = [
            self._build_query_request(query, vector.tolist(), query_filter, limit)
            for query, vector, query_filter in zip(queries, query_embeddings, filters, strict=True)
        ]
        results: list[list[RetrievedChunk]] = []
        for start in range(0, len(requests), self.config.batch_size):
            batch_requests = requests[start : start + self.config.batch_size]
            responses = self._client.query_batch_points(
                collection_name=self.config.collection_name,
                requests=batch_requests,
            )
            if len(responses) != len(batch_requests):
                raise RuntimeError(
                    "Qdrant returned a different number of responses than requests."
                )
            results.extend(
                [self._to_retrieved_chunk(point) for point in response.points]
                for response in responses
            )

        return results

    def _build_query_request(
        self,
        query: str,
        vector: list[float],
        query_filter: models.Filter | None,
        limit: int,
    ) -> models.QueryRequest:
        bm25_weight = self.config.bm25_weight
        if bm25_weight == 0.0:
            return models.QueryRequest(
                query=vector,
                using=self.config.vector_name,
                filter=query_filter,
                limit=limit,
                with_vector=False,
                with_payload=True,
            )

        return models.QueryRequest(
            prefetch=[
                models.Prefetch(
                    query=vector,
                    using=self.config.vector_name,
                    filter=query_filter,
                    limit=limit,
                ),
                models.Prefetch(
                    query=models.TextQuery(text=query),
                    using="text",
                    filter=query_filter,
                    limit=limit,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(weights=[1.0, bm25_weight])),
            limit=limit,
            with_vector=False,
            with_payload=True,
        )

    @staticmethod
    def _to_retrieved_chunk(point: models.ScoredPoint) -> RetrievedChunk:
        payload = point.payload or {}
        try:
            return RetrievedChunk.model_validate({**payload, "score": point.score})
        except ValidationError as exc:
            raise ValueError(f"Invalid payload for Qdrant point {point.id}.") from exc
