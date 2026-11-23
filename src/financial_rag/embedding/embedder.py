from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer
from transformers.utils.hub import cached_file, extract_commit_hash


class EmbeddingModel(StrEnum):
    QWEN3_0_6B = "Qwen/Qwen3-Embedding-0.6B"
    YUAN_2_0_EN = "IEITYuan/Yuan-embedding-2.0-en"


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    embedding_model: EmbeddingModel = EmbeddingModel.QWEN3_0_6B
    batch_size: int | None = Field(default=None, ge=1)
    normalize_embeddings: bool = True
    device: str | None = None
    model_dtype: Literal["float32", "float16"] | None = None


class Embedder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        model_id = config.embedding_model.value
        revision = extract_commit_hash(cached_file(model_id, "config.json"), None)
        if revision is None:
            raise ValueError(f"Could not resolve a model revision for {model_id}")
        self.revision = revision
        self._model = SentenceTransformer(
            model_id,
            revision=revision,
            device=config.device,
            model_kwargs={"dtype": config.model_dtype} if config.model_dtype else None,
        )
        dimension = self._model.get_embedding_dimension()
        if dimension is None or dimension <= 0:
            raise ValueError(f"Could not determine the embedding dimension for {model_id}")
        self.dimension = dimension

    def embed_chunks(self, chunks: list[str]) -> np.ndarray:
        if not chunks:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._model.encode(
            chunks,
            normalize_embeddings=self.config.normalize_embeddings,
            batch_size=self.config.batch_size or 32,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)

    def embed_query(
        self, queries: str | list[str], prompt_name: str | None = "query"
    ) -> np.ndarray:
        if prompt_name and prompt_name not in self._model.prompts:
            raise ValueError(
                f"Prompt name {prompt_name} not found in model prompts: {list(self._model.prompts)}"
            )
        return self._model.encode(
            queries,
            prompt_name=prompt_name,
            normalize_embeddings=self.config.normalize_embeddings,
            batch_size=self.config.batch_size or 32,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)
