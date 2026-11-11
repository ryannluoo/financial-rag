import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from financial_rag.indexing import __main__ as cli
from financial_rag.indexing.indexer import IndexingConfig, QdrantIndexer


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
    assert result["artifacts"][0]["vector_sha256"] == hashlib.sha256((path.parent / "report.npy").read_bytes()).hexdigest()
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
