"""Validate an embedding run, index its chunks, and publish completion."""

import argparse
import hashlib
from io import BytesIO
import json
import logging
from pathlib import Path
import re
from time import perf_counter

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from financial_rag.embedding import MODEL_NAMES
from financial_rag.indexing.indexer import IndexingConfig, QdrantIndexer
from financial_rag.indexing.manifest import write_manifest


DEFAULT_INPUT_DIR = Path(
    "data/processed/embeddings/qwen3_emb_06b/c384_o64"
)
DEFAULT_OUTPUT_DIR = Path("data/processed/indexes")


def _filename(value: str, suffix: str) -> str:
    if (
        not isinstance(value, str)
        or Path(value).name != value
        or "/" in value or "\\" in value or ":" in value
        or not value.endswith(suffix)
        or value == suffix
    ):
        raise ValueError(f"Expected a plain {suffix} filename, got {value!r}")
    return value


def _read_report(
    artifact: dict,
    source_dir: Path,
    vector_dir: Path,
    indexer: QdrantIndexer,
    normalize: bool,
    seen_ids: set[str],
) -> tuple[list[dict], np.ndarray, str]:
    source_path = source_dir / artifact["source_filename"]
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != artifact["source_sha256"]:
        raise ValueError(f"{source_path}: source hash differs from the embedding run")
    chunks = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        try:
            chunk = json.loads(line)
            text, metadata = chunk["text"], chunk["metadata"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("chunk text must be a non-empty string")
            if metadata["document_id"] != source_path.stem:
                raise ValueError("document ID must match the source filename")
            for field, minimum in (
                ("page", 1), ("chunk_index", 0), ("token_count", 1),
                ("char_start", 0), ("char_end", 1),
            ):
                if type(metadata[field]) is not int or metadata[field] < minimum:
                    raise ValueError(f"invalid citation metadata: {field}")
            if metadata["char_end"] - metadata["char_start"] != len(text):
                raise ValueError("character offsets must span the chunk text")
            chunk_id = metadata["chunk_id"]
            expected_id = (
                f"{source_path.stem}:p{metadata['page']}:c{metadata['chunk_index']}"
            )
            if chunk_id != expected_id or chunk_id in seen_ids:
                raise ValueError(f"invalid or duplicate chunk ID: {chunk_id}")
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{source_path}:{line_number}: {error}") from error
        seen_ids.add(chunk_id)
        chunks.append(chunk)
    if len(chunks) != artifact["chunk_count"]:
        raise ValueError(f"{source_path}: chunk count differs from the manifest")

    vector_path = vector_dir / artifact["vector_filename"]
    vector_source = vector_path.read_bytes()
    vector_hash = hashlib.sha256(vector_source).hexdigest()
    if artifact.get("vector_sha256", vector_hash) != vector_hash:
        raise ValueError(f"{vector_path}: vectors changed after input validation")
    vectors = np.load(BytesIO(vector_source), allow_pickle=False)
    if not isinstance(vectors, np.ndarray):
        vectors.close()
        raise ValueError(f"{vector_path}: expected a NumPy array")
    indexer.validate_vectors(vectors, len(chunks))
    if normalize and not np.allclose(
        np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=1e-3
    ):
        raise ValueError(f"{vector_path}: normalized vectors must have unit length")
    return chunks, vectors, vector_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--chunks-dir", type=Path, help="Override the manifest's source path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    client = None
    try:
        if args.version < 1:
            raise ValueError("Version must be positive")
        input_dir = args.input_dir.resolve()
        embedding_manifest = input_dir / "manifest.json"
        manifest_source = embedding_manifest.read_bytes()
        manifest = json.loads(manifest_source)
        model_id = manifest["model_id"]
        if model_id not in MODEL_NAMES:
            raise ValueError(f"Unsupported embedding model: {model_id}")
        model_name = MODEL_NAMES[model_id]
        source_dir = Path(args.chunks_dir or manifest["source_chunk_dir"]).resolve()
        if source_dir.name != Path(manifest["source_chunk_dir"]).name:
            raise ValueError("Chunk override must preserve the manifest's configuration directory name")
        chunk_config = re.fullmatch(r"c([0-9]+)_o([0-9]+)", source_dir.name)
        if chunk_config is None:
            raise ValueError("Chunk directory must be named c<size>_o<overlap>")
        chunk_size, overlap = map(int, chunk_config.groups())
        if chunk_size < 1 or not 0 <= overlap < chunk_size:
            raise ValueError("Require chunk size > overlap >= 0")
        if manifest["dtype"] != "float32":
            raise ValueError("Embedding manifest must declare float32 vectors")
        if type(manifest["normalize_embeddings"]) is not bool:
            raise ValueError("normalize_embeddings must be a boolean")
        if not isinstance(manifest["model_revision"], str) or not manifest["model_revision"]:
            raise ValueError("Embedding model revision is required")
        config = IndexingConfig(
            collection_name=f"{model_name}_c{chunk_size}_o{overlap}_v{args.version}",
            vector_size=manifest["embedding_dimension"],
            batch_size=args.batch_size,
        )
        output_path = args.output_dir / config.collection_name / "manifest.json"
        if output_path.exists():
            raise ValueError(f"Index manifest already exists: {output_path}; use a new version")
        artifacts = manifest["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("Embedding manifest must contain artifacts")
        seen_sources: set[str] = set()
        for artifact in artifacts:
            source_name = _filename(artifact["source_filename"], ".jsonl")
            vector_name = _filename(artifact["vector_filename"], ".npy")
            if Path(source_name).stem != Path(vector_name).stem or source_name in seen_sources:
                raise ValueError("Artifact filenames must be unique, matching report pairs")
            if type(artifact["chunk_count"]) is not int or artifact["chunk_count"] < 0:
                raise ValueError("Artifact chunk_count must be a non-negative integer")
            seen_sources.add(source_name)

        client = QdrantClient(url=args.qdrant_url, timeout=60)
        indexer = QdrantIndexer(config, client)
        validated = []
        seen_ids: set[str] = set()
        for artifact in artifacts:
            _, _, vector_hash = _read_report(
                artifact, source_dir, input_dir, indexer,
                manifest["normalize_embeddings"], seen_ids,
            )
            validated.append({**artifact, "vector_sha256": vector_hash})
        logging.info("Validated %d reports and %d chunks", len(validated), len(seen_ids))

        started = perf_counter()
        indexer.create_collection()
        seen_ids = set()
        for position, artifact in enumerate(validated, start=1):
            chunks, vectors, _ = _read_report(
                artifact, source_dir, input_dir, indexer,
                manifest["normalize_embeddings"], seen_ids,
            )
            indexer.upload_chunks(chunks, vectors)
            logging.info("Indexed %d/%d: %s (%d chunks)",
                         position, len(validated), artifact["source_filename"], len(chunks))
        actual_count = client.count(collection_name=config.collection_name, exact=True).count
        if actual_count != len(seen_ids):
            raise ValueError(f"Expected {len(seen_ids)} points, found {actual_count}")
        write_manifest(output_path, {
            "schema_version": 1,
            "collection_name": config.collection_name,
            "version": args.version,
            "model_id": manifest["model_id"],
            "model_revision": manifest["model_revision"],
            "normalize_embeddings": manifest["normalize_embeddings"],
            "embedding_dimension": config.vector_size,
            "dtype": "float32",
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "dense_vector": {"name": "dense", "distance": "Cosine"},
            "payload_indexes": {
                "document_id": "keyword",
                "text": {
                    "type": "text", "scoring": "bm25",
                    "tokenizer": "word", "lowercase": True,
                },
            },
            "embedding_manifest": {
                "path": embedding_manifest.as_posix(),
                "sha256": hashlib.sha256(manifest_source).hexdigest(),
            },
            "source_chunk_dir": source_dir.as_posix(),
            "report_count": len(validated),
            "point_count": actual_count,
            "batch_size": config.batch_size,
            "indexing_runtime_seconds": round(perf_counter() - started, 3),
            "artifacts": validated,
        })
        logging.info("Published %s", output_path)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError,
            UnexpectedResponse, ResponseHandlingException) as error:
        logging.error("%s", error)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
