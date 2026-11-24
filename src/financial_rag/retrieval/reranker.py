from collections.abc import Sequence
from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import CrossEncoder

from financial_rag.retrieval.retriever import RetrievedChunk


class RerankingModel(StrEnum):
    QWEN3_0_6B = "Qwen/Qwen3-Reranker-0.6B"
    ZERANK_1_SMALL = "zeroentropy/zerank-1-small-reranker"


_REMOTE_CODE_MODELS = frozenset({RerankingModel.ZERANK_1_SMALL})


class RerankingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    reranking_model: RerankingModel = RerankingModel.QWEN3_0_6B
    top_n: int = Field(default=5, ge=1)
    batch_size: int = Field(default=16, ge=1)
    device: str | None = None
    model_dtype: Literal["float32", "float16"] | None = None


class RerankedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: RetrievedChunk
    rerank_score: float


class CrossEncoderReranker:
    def __init__(self, config: RerankingConfig):
        self.config = config
        self._model = CrossEncoder(
            config.reranking_model.value,
            device=config.device,
            trust_remote_code=config.reranking_model in _REMOTE_CODE_MODELS,
            model_kwargs={"dtype": config.model_dtype} if config.model_dtype else None,
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_n: int | None = None,
    ) -> list[RerankedChunk]:
        if not query.strip():
            raise ValueError("query must be a non-empty string.")

        limit = self.config.top_n if top_n is None else top_n
        if limit < 1:
            raise ValueError("top_n must be at least 1.")

        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        raw_scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        if scores.shape != (len(candidates),):
            raise ValueError(
                f"Expected reranker scores to have shape ({len(candidates)},), "
                f"but got shape {scores.shape}."
            )
        if not np.isfinite(scores).all():
            raise ValueError("Reranker scores contain non-finite values.")

        reranked = [
            RerankedChunk(chunk=candidate, rerank_score=float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        return reranked[:limit]
