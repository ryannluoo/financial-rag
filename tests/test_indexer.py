import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from financial_rag.indexing import __main__ as cli
from financial_rag.indexing.indexer import IndexingConfig, QdrantIndexer


def test_indexing_config_validation():
    config = IndexingConfig(collection_name="test", vector_size=4)
    assert config.batch_size == 256
    with pytest.raises(ValidationError):
        config.batch_size = 128
    with pytest.raises(ValidationError):
        IndexingConfig(collection_name="", vector_size=4)
    with pytest.raises(ValidationError):
        IndexingConfig(collection_name="test", vector_size=0)
    with pytest.raises(ValidationError):
        IndexingConfig(collection_name="test", vector_size=4, batch_size=0)


def test_create_collection_configures_dense_vector_and_payload_indexes():
    client = Mock(spec=QdrantClient)
    indexer = QdrantIndexer(IndexingConfig(collection_name="col_v1", vector_size=1024), client)

    indexer.create_collection()

    client.create_collection.assert_called_once_with(
        collection_name="col_v1",
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
        },
    )
    assert client.create_payload_index.call_count == 2
    doc_index, text_index = client.create_payload_index.call_args_list
    assert doc_index.kwargs == {
        "collection_name": "col_v1",
        "field_name": "document_id",
        "field_schema": models.PayloadSchemaType.KEYWORD,
        "wait": True,
    }
    assert text_index.kwargs["collection_name"] == "col_v1"
    assert text_index.kwargs["field_name"] == "text"
    assert text_index.kwargs["wait"] is True
    assert text_index.kwargs["field_schema"] == models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        lowercase=True,
        scoring=True,
    )


@pytest.mark.parametrize(
    "vectors,chunk_count,match",
    [
        (np.ones((2, 2), dtype=np.float32), 1, "shape"),
        (np.ones((1, 2), dtype=np.float64), 1, "float32"),
        (np.array([[np.nan, 0.0]], dtype=np.float32), 1, "non-finite"),
        (np.array([[np.inf, 0.0]], dtype=np.float32), 1, "non-finite"),
        (np.zeros((1, 2), dtype=np.float32), 1, "zero-length"),
    ],
)
def test_validate_vectors_rejects_invalid_arrays(vectors, chunk_count, match):
    indexer = QdrantIndexer(IndexingConfig(collection_name="test", vector_size=2), Mock(spec=QdrantClient))
    with pytest.raises(ValueError, match=match):
        indexer.validate_vectors(vectors, chunk_count)


def test_upload_chunks_skips_empty_input():
    client = Mock(spec=QdrantClient)
    indexer = QdrantIndexer(IndexingConfig(collection_name="test", vector_size=2), client)
    indexer.upload_chunks([], np.empty((0, 2), dtype=np.float32))
    client.upload_points.assert_not_called()


@pytest.fixture
def run_input(tmp_path, monkeypatch):
    chunks_dir = tmp_path / "c384_o64"
    vectors_dir = tmp_path / "vectors"
    chunks_dir.mkdir()
    vectors_dir.mkdir()
    chunk = {
        "metadata": {"chunk_id": "report:p1:c0", "document_id": "report",
                     "page": 1, "chunk_index": 0, "token_count": 1,
                     "char_start": 0, "char_end": 7},
        "text": "Revenue",
    }
    source = (json.dumps(chunk) + "\n").encode()
    (chunks_dir / "report.jsonl").write_bytes(source)
    np.save(vectors_dir / "report.npy", np.array([[1, 0]], dtype=np.float32))
    manifest = {
        "model_id": "Qwen/Qwen3-Embedding-0.6B", "model_revision": "revision",
        "dtype": "float32", "normalize_embeddings": True,
        "embedding_dimension": 2, "source_chunk_dir": str(chunks_dir),
        "artifacts": [{"source_filename": "report.jsonl",
                       "source_sha256": hashlib.sha256(source).hexdigest(),
                       "vector_filename": "report.npy", "chunk_count": 1}],
    }
    manifest_path = vectors_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = Mock(spec=QdrantClient)
    client.count.return_value.count = 1
    client.upload_points.side_effect = lambda **kwargs: list(kwargs["points"])
    monkeypatch.setattr(cli, "QdrantClient", Mock(return_value=client))
    monkeypatch.setattr(sys, "argv", ["indexing", "--input-dir", str(vectors_dir),
                                    "--output-dir", str(tmp_path / "indexes")])
    return tmp_path, manifest_path, manifest, client


@pytest.mark.parametrize("model_id,prefix", list(cli.MODEL_NAMES.items()))
def test_completed_run_publishes_verified_manifest(run_input, model_id, prefix):
    root, path, manifest, client = run_input
    manifest["model_id"] = model_id
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert cli.main() == 0
    output = root / "indexes" / f"{prefix}_c384_o64_v1" / "manifest.json"
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["point_count"] == result["report_count"] == 1
    assert result["model_id"] == model_id
    assert result["embedding_manifest"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = result["artifacts"][0]
    assert list(artifact) == [
        "source_filename", "source_sha256", "vector_filename", "vector_sha256", "chunk_count",
    ]
    assert artifact["vector_sha256"] == hashlib.sha256((path.parent / "report.npy").read_bytes()).hexdigest()
    indexes = client.create_payload_index.call_args_list
    assert indexes[0].kwargs["field_schema"] == models.PayloadSchemaType.KEYWORD
    assert indexes[1].kwargs["field_schema"].scoring is True
    assert client.count.call_args.kwargs["exact"] is True
    assert client.upload_points.call_args.kwargs["wait"] is True
    client.close.assert_called_once()
    assert list(output.parent.iterdir()) == [output]
    client.reset_mock()
    assert cli.main() == 1
    client.create_collection.assert_not_called()
    assert json.loads(output.read_text(encoding="utf-8")) == result


@pytest.mark.parametrize("damage", ["source", "shape", "nan", "duplicate", "path", "metadata"])
def test_invalid_inputs_do_not_create_collection(run_input, damage):
    root, path, manifest, client = run_input
    if damage == "source":
        (Path(manifest["source_chunk_dir"]) / "report.jsonl").write_text("changed")
    elif damage in ("shape", "nan"):
        vectors = np.ones((2, 2), dtype=np.float32) if damage == "shape" else np.array([[np.nan, 0]], dtype=np.float32)
        np.save(path.parent / "report.npy", vectors)
    elif damage == "duplicate":
        manifest["artifacts"] *= 2
    elif damage == "path":
        manifest["artifacts"][0]["vector_filename"] = "../report.npy"
    else:
        source_path = Path(manifest["source_chunk_dir"]) / "report.jsonl"
        chunk = json.loads(source_path.read_text())
        del chunk["metadata"]["page"]
        source_path.write_text(json.dumps(chunk), encoding="utf-8")
        manifest["artifacts"][0]["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert cli.main() == 1
    client.create_collection.assert_not_called()
    assert not list(root.glob("indexes/*/manifest.json"))


@pytest.mark.parametrize("failure", ["upload", "count", "create"])
def test_failed_remote_operations_never_publish_manifest(run_input, failure):
    root, _, _, client = run_input
    if failure == "count":
        client.count.return_value.count = 0
    else:
        getattr(client, "upload_points" if failure == "upload" else "create_collection").side_effect = RuntimeError("unavailable")
    assert cli.main() == 1
    assert not list(root.glob("indexes/*/manifest.json"))
    client.close.assert_called_once()


def test_point_ids_and_payload_support_repeatable_report_filtering():
    client = Mock(spec=QdrantClient)
    captured = []
    client.upload_points.side_effect = lambda **kwargs: captured.extend(kwargs["points"])
    indexer = QdrantIndexer(IndexingConfig(collection_name="test", vector_size=2), client)
    chunks = [{"metadata": {"chunk_id": "report:p1:c0", "document_id": "report", "page": 1}, "text": "Revenue"}]
    for _ in range(2):
        indexer.upload_chunks(chunks, np.array([[1, 0]], dtype=np.float32))
    assert captured[0].id == captured[1].id
    assert captured[0].payload["document_id"] == "report"
    assert captured[0].payload["text"] == "Revenue"
    assert captured[0].vector == {"dense": [1, 0]}


def test_changed_vectors_after_preflight_do_not_upload(run_input):
    root, path, _, client = run_input
    client.create_collection.side_effect = lambda **kwargs: np.save(
        path.parent / "report.npy", np.array([[0, 1]], dtype=np.float32)
    )
    assert cli.main() == 1
    client.upload_points.assert_not_called()
    assert not list(root.glob("indexes/*/manifest.json"))


def test_empty_report_is_valid(run_input):
    root, path, manifest, client = run_input
    (Path(manifest["source_chunk_dir"]) / "report.jsonl").write_bytes(b"")
    artifact = manifest["artifacts"][0]
    artifact["chunk_count"] = 0
    artifact["source_sha256"] = hashlib.sha256(b"").hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    np.save(path.parent / "report.npy", np.empty((0, 2), dtype=np.float32))
    client.count.return_value.count = 0
    assert cli.main() == 0
    client.upload_points.assert_not_called()
    output = root / "indexes/qwen3_emb_06b_c384_o64_v1/manifest.json"
    assert json.loads(output.read_text())["point_count"] == 0
