from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError

from financial_rag.retrieval.reranker import (
    CrossEncoderReranker,
    RerankedChunk,
    RerankingConfig,
    RerankingModel,
)
from financial_rag.retrieval.retriever import RetrievedChunk


def _make_chunk(
    chunk_id: str = "doc1:p1:c0",
    *,
    text: str = "Revenue grew 10%.",
    score: float = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc1",
        page=1,
        chunk_index=0,
        token_count=5,
        char_start=0,
        char_end=len(text),
        text=text,
        score=score,
    )


@pytest.fixture
def patched_cross_encoder(monkeypatch):
    model = Mock()
    ctor = Mock(return_value=model)
    monkeypatch.setattr("financial_rag.retrieval.reranker.CrossEncoder", ctor)
    return model, ctor


def test_reranking_config_validation():
    config = RerankingConfig()
    assert config.reranking_model is RerankingModel.QWEN3_0_6B
    assert config.top_n == 5
    assert config.batch_size == 16
    assert config.device is None
    assert config.model_dtype is None

    with pytest.raises(ValidationError):
        config.top_n = 3
    with pytest.raises(ValidationError):
        RerankingConfig(top_n=0)
    with pytest.raises(ValidationError):
        RerankingConfig(batch_size=0)


@pytest.mark.parametrize(
    "reranking_model,expected_trust_remote_code",
    [
        (RerankingModel.QWEN3_0_6B, False),
        (RerankingModel.ZERANK_1_SMALL, True),
    ],
)
def test_cross_encoder_reranker_init_options(
    patched_cross_encoder, reranking_model, expected_trust_remote_code
):
    _, ctor = patched_cross_encoder
    CrossEncoderReranker(
        RerankingConfig(
            reranking_model=reranking_model,
            device="cuda",
            model_dtype="float16",
        )
    )

    ctor.assert_called_once_with(
        reranking_model.value,
        device="cuda",
        trust_remote_code=expected_trust_remote_code,
        model_kwargs={"dtype": "float16"},
    )


@pytest.mark.parametrize("raw_scores", [[1.5, 4.0, 4.0], np.array([1.5, 4.0, 4.0], dtype=np.float32)])
def test_cross_encoder_reranker_sorts_stably_and_truncates(patched_cross_encoder, raw_scores):
    model, _ = patched_cross_encoder
    c1 = _make_chunk("doc1:p1:c0", text="First chunk", score=0.9)
    c2 = _make_chunk("doc1:p2:c0", text="Second chunk", score=0.8)
    c3 = _make_chunk("doc1:p3:c0", text="Third chunk", score=0.7)
    model.predict.return_value = raw_scores

    reranker = CrossEncoderReranker(RerankingConfig(top_n=2, batch_size=8))
    results = reranker.rerank("What was revenue?", [c1, c2, c3])

    model.predict.assert_called_once_with(
        [
            ("What was revenue?", "First chunk"),
            ("What was revenue?", "Second chunk"),
            ("What was revenue?", "Third chunk"),
        ],
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    assert results == [
        RerankedChunk(chunk=c2, rerank_score=4.0),
        RerankedChunk(chunk=c3, rerank_score=4.0),
    ]

    model.predict.return_value = np.array([[0.1], [2.5], [1.2]], dtype=np.float32)
    override = reranker.rerank("What was revenue?", [c1, c2, c3], top_n=1)
    assert override == [RerankedChunk(chunk=c2, rerank_score=2.5)]


def test_cross_encoder_reranker_empty_and_error_cases(patched_cross_encoder):
    model, _ = patched_cross_encoder
    reranker = CrossEncoderReranker(RerankingConfig())
    candidate = _make_chunk()

    assert reranker.rerank("What was revenue?", []) == []
    model.predict.assert_not_called()

    with pytest.raises(ValueError, match="non-empty"):
        reranker.rerank("   ", [candidate])
    with pytest.raises(ValueError, match="top_n"):
        reranker.rerank("What was revenue?", [candidate], top_n=0)

    model.predict.return_value = np.array([0.5, 0.2], dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        reranker.rerank("What was revenue?", [candidate])

    model.predict.return_value = np.array([np.nan], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        reranker.rerank("What was revenue?", [candidate])
