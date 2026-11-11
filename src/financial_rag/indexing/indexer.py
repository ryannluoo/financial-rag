from uuid import NAMESPACE_URL, uuid5

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient, models


class IndexingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    collection_name: str = Field(min_length=1)
    vector_size: int = Field(ge=1)
    batch_size: int = Field(default=256, ge=1)


class QdrantIndexer:
    def __init__(self, config: IndexingConfig, client: QdrantClient):
        self.config = config
        self._client = client

    def create_collection(self) -> None:
        self._client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.config.vector_size,
                    distance=models.Distance.COSINE,
                ),
            },
        )

        self._client.create_payload_index(
            collection_name=self.config.collection_name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

        self._client.create_payload_index(
            collection_name=self.config.collection_name,
            field_name="text",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
                scoring=True,
            ),
            wait=True,
        )

    def validate_vectors(self, vectors: np.ndarray, chunk_count: int) -> None:
        """Validate an embedding matrix before any points are uploaded."""
        expected_shape = (chunk_count, self.config.vector_size)
        if vectors.shape != expected_shape:
            raise ValueError(
                f"Expected embedding shape {expected_shape}, got {vectors.shape}"
            )
        if vectors.dtype != np.float32:
            raise ValueError(f"Expected float32 embeddings, got {vectors.dtype}")
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain non-finite values")
        if np.any(np.all(vectors == 0, axis=1)):
            raise ValueError("Embeddings contain zero-length vectors")

    def upload_chunks(self, chunks: list[dict], vectors: np.ndarray) -> None:
        self.validate_vectors(vectors, len(chunks))
        if not chunks:
            return
        
        points = (
            models.PointStruct(
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"financial-rag:{chunk['metadata']['chunk_id']}",
                    )
                ),
                vector={"dense": vector.tolist()},
                payload={
                    **chunk['metadata'],
                    "text": chunk['text'],
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        )

        self._client.upload_points(
            collection_name=self.config.collection_name,
            points=points,
            batch_size=self.config.batch_size,
            wait=True,
        )
