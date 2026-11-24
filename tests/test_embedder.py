import hashlib
import json
import sys
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError

from financial_rag.embedding import MODEL_NAMES
from financial_rag.embedding import __main__ as cli
from financial_rag.embedding.embedder import Embedder, EmbeddingConfig, EmbeddingModel


class _FakeModel:
    def __init__(self, dimension: int | None = 3, prompts: dict[str, str] | None = None):
        self._dimension = dimension
        self.prompts = {"query": "Instruct: "} if prompts is None else prompts
        self.encode = Mock(side_effect=self._encode)

    def get_embedding_dimension(self) -> int | None:
        return self._dimension

    def _encode(self, inputs, **kwargs):
        dim = self._dimension or 0
        if isinstance(inputs, str):
            vec = np.zeros(dim, dtype=np.float64)
            vec[0] = 1.0
            return vec
        arr = np.zeros((len(inputs), dim), dtype=np.float64)
        if len(inputs):
            arr[:, 0] = 1.0
        return arr


@pytest.fixture
def patched_embedder_deps(monkeypatch):
    model = _FakeModel(dimension=3)
    ctor = Mock(return_value=model)
    monkeypatch.setattr("financial_rag.embedding.embedder.cached_file", Mock(return_value="/cache/config.json"))
    monkeypatch.setattr("financial_rag.embedding.embedder.extract_commit_hash", Mock(return_value="abc123def"))
    monkeypatch.setattr("financial_rag.embedding.embedder.SentenceTransformer", ctor)
    return model, ctor


def test_embedding_config_validation():
    config = EmbeddingConfig()
    assert config.embedding_model is EmbeddingModel.QWEN3_0_6B
    assert config.normalize_embeddings is True
    with pytest.raises(ValidationError):
        config.batch_size = 16
    with pytest.raises(ValidationError):
        EmbeddingConfig(batch_size=0)


def test_embedder_init_passes_revision_device_and_dtype(patched_embedder_deps):
    _, ctor = patched_embedder_deps
    embedder = Embedder(EmbeddingConfig(device="cuda", model_dtype="float16", batch_size=16))

    assert embedder.revision == "abc123def"
    assert embedder.dimension == 3
    ctor.assert_called_once_with(
        EmbeddingModel.QWEN3_0_6B.value,
        revision="abc123def",
        device="cuda",
        model_kwargs={"dtype": "float16"},
    )


def test_embedder_init_rejects_missing_revision_or_invalid_dimension(monkeypatch, patched_embedder_deps):
    monkeypatch.setattr("financial_rag.embedding.embedder.extract_commit_hash", Mock(return_value=None))
    with pytest.raises(ValueError, match="revision"):
        Embedder(EmbeddingConfig())

    monkeypatch.setattr("financial_rag.embedding.embedder.extract_commit_hash", Mock(return_value="abc123"))
    for bad_dim in (None, 0, -4):
        monkeypatch.setattr(
            "financial_rag.embedding.embedder.SentenceTransformer",
            Mock(return_value=_FakeModel(dimension=bad_dim)),
        )
        with pytest.raises(ValueError, match="dimension"):
            Embedder(EmbeddingConfig())


def test_embed_chunks_returns_float32_and_handles_empty(patched_embedder_deps):
    model, _ = patched_embedder_deps
    embedder = Embedder(EmbeddingConfig())

    empty = embedder.embed_chunks([])
    assert empty.shape == (0, 3)
    assert empty.dtype == np.float32
    model.encode.assert_not_called()

    vectors = embedder.embed_chunks(["Revenue grew 10%.", "Operating margin improved."])
    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    model.encode.assert_called_once_with(
        ["Revenue grew 10%.", "Operating margin improved."],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def test_embed_query_supports_single_batch_and_prompt_validation(patched_embedder_deps):
    model, _ = patched_embedder_deps
    embedder = Embedder(EmbeddingConfig(batch_size=8, normalize_embeddings=False))

    single = embedder.embed_query("What was revenue?")
    assert single.shape == (3,)
    assert single.dtype == np.float32
    assert model.encode.call_args.kwargs == {
        "prompt_name": "query",
        "normalize_embeddings": False,
        "batch_size": 8,
        "convert_to_numpy": True,
    }

    batch = embedder.embed_query(["Q1", "Q2"], prompt_name=None)
    assert batch.shape == (2, 3)
    assert batch.dtype == np.float32
    assert model.encode.call_args.kwargs["prompt_name"] is None

    with pytest.raises(ValueError, match="Prompt name unknown"):
        embedder.embed_query("What was revenue?", prompt_name="unknown")


@pytest.fixture
def cli_run(tmp_path, monkeypatch, patched_embedder_deps):
    chunks_dir = tmp_path / "c384_o64"
    output_dir = tmp_path / "embeddings"
    chunks_dir.mkdir()
    records = [
        {"metadata": {"chunk_id": "alpha:p1:c0", "document_id": "alpha"}, "text": "Revenue $10M"},
        {"metadata": {"chunk_id": "alpha:p1:c1", "document_id": "alpha"}, "text": "Net income $2M"},
    ]
    alpha_bytes = "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")
    (chunks_dir / "alpha.jsonl").write_bytes(alpha_bytes)
    (chunks_dir / "empty.jsonl").write_bytes(b"")
    monkeypatch.setattr(
        sys,
        "argv",
        ["embedding", "--input-dir", str(chunks_dir), "--output-dir", str(output_dir)],
    )
    return tmp_path, chunks_dir, output_dir, alpha_bytes, patched_embedder_deps[0]


@pytest.mark.parametrize("model", list(EmbeddingModel))
def test_cli_embeds_reports_and_publishes_manifest(cli_run, monkeypatch, model):
    _, chunks_dir, output_dir, alpha_bytes, _ = cli_run
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embedding",
            "--input-dir",
            str(chunks_dir),
            "--output-dir",
            str(output_dir),
            "--embedding-model",
            model.value,
        ],
    )
    assert cli.main() == 0

    run_dir = output_dir / MODEL_NAMES[model.value] / "c384_o64"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == model.value
    assert manifest["model_revision"] == "abc123def"
    assert manifest["embedding_dimension"] == 3
    assert manifest["dtype"] == "float32"
    assert manifest["normalize_embeddings"] is True
    assert manifest["source_chunk_dir"] == chunks_dir.resolve().as_posix()
    assert [a["source_filename"] for a in manifest["artifacts"]] == ["alpha.jsonl", "empty.jsonl"]
    assert manifest["artifacts"][0]["source_sha256"] == hashlib.sha256(alpha_bytes).hexdigest()
    assert manifest["artifacts"][0]["chunk_count"] == 2
    assert manifest["artifacts"][1]["chunk_count"] == 0

    alpha_vectors = np.load(run_dir / "alpha.npy", allow_pickle=False)
    empty_vectors = np.load(run_dir / "empty.npy", allow_pickle=False)
    assert alpha_vectors.shape == (2, 3) and alpha_vectors.dtype == np.float32
    assert empty_vectors.shape == (0, 3) and empty_vectors.dtype == np.float32


@pytest.mark.parametrize("damage", ["whitespace_text", "duplicate_id", "wrong_document_id", "empty_dir"])
def test_cli_rejects_invalid_chunk_inputs_without_publishing_manifest(cli_run, damage):
    _, chunks_dir, output_dir, _, _ = cli_run
    run_dir = output_dir / MODEL_NAMES[EmbeddingModel.QWEN3_0_6B.value] / "c384_o64"
    run_dir.mkdir(parents=True)
    stale_manifest = run_dir / "manifest.json"
    stale_manifest.write_text("{}", encoding="utf-8")

    if damage == "whitespace_text":
        bad = {"metadata": {"chunk_id": "alpha:p1:c0", "document_id": "alpha"}, "text": "   "}
        (chunks_dir / "alpha.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    elif damage == "duplicate_id":
        dup = {"metadata": {"chunk_id": "alpha:p1:c0", "document_id": "empty"}, "text": "Duplicate"}
        (chunks_dir / "empty.jsonl").write_text(json.dumps(dup) + "\n", encoding="utf-8")
    elif damage == "wrong_document_id":
        bad = {"metadata": {"chunk_id": "other:p1:c0", "document_id": "other"}, "text": "Mismatch"}
        (chunks_dir / "alpha.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    else:
        for file in chunks_dir.glob("*.jsonl"):
            file.unlink()

    assert cli.main() == 1
    if damage != "empty_dir":
        assert not stale_manifest.exists()


@pytest.mark.parametrize("bad_vectors", ["shape", "nan", "zero", "unnormalized"])
def test_cli_rejects_invalid_output_vectors(cli_run, bad_vectors):
    _, _, output_dir, _, model = cli_run
    if bad_vectors == "shape":
        model.encode.side_effect = lambda *a, **k: np.ones((1, 3), dtype=np.float32)
    elif bad_vectors == "nan":
        model.encode.side_effect = lambda *a, **k: np.full((2, 3), np.nan, dtype=np.float32)
    elif bad_vectors == "zero":
        model.encode.side_effect = lambda *a, **k: np.zeros((2, 3), dtype=np.float32)
    else:
        model.encode.side_effect = lambda *a, **k: np.full((2, 3), 2.0, dtype=np.float32)

    assert cli.main() == 1
    assert not list(output_dir.glob("*/*/manifest.json"))
