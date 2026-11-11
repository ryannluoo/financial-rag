"""Embed report chunks and publish a manifest for the completed run."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter

import numpy as np

from financial_rag.embedding import MODEL_NAMES
from financial_rag.embedding.embedder import Embedder, EmbeddingConfig, EmbeddingModel
from financial_rag.embedding.manifest import write_manifest


DEFAULT_INPUT_DIR = Path("data/processed/chunks/c384_o64")
DEFAULT_OUTPUT_DIR = Path("data/processed/embeddings")


def _read_chunks(path: Path, seen_ids: set[str]) -> tuple[list[str], str]:
    source = path.read_bytes()
    texts = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        try:
            chunk = json.loads(line)
            text = chunk["text"]
            metadata = chunk["metadata"]
            chunk_id = metadata["chunk_id"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("chunk text must be a non-empty string")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("chunk ID must be a non-empty string")
            if chunk_id in seen_ids:
                raise ValueError(f"duplicate chunk ID: {chunk_id}")
            if metadata["document_id"] != path.stem:
                raise ValueError("document ID must match the source filename stem")
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        seen_ids.add(chunk_id)
        texts.append(text)
    return texts, hashlib.sha256(source).hexdigest()


def _validate_vectors(
    vectors: np.ndarray, expected_shape: tuple[int, int], normalize: bool
) -> None:
    if vectors.shape != expected_shape:
        raise ValueError(f"expected embedding shape {expected_shape}, got {vectors.shape}")
    if vectors.dtype != np.float32:
        raise ValueError(f"expected float32 embeddings, got {vectors.dtype}")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("embeddings contain zero-length vectors")
    if normalize and not np.allclose(norms, 1, rtol=0, atol=1e-3):
        raise ValueError("normalized embeddings must have approximately unit length")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--embedding-model", type=EmbeddingModel, choices=list(EmbeddingModel),
        default=EmbeddingModel.QWEN3_0_6B,
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--model-dtype", choices=("float32", "float16"))
    parser.add_argument(
        "--normalize-embeddings", action=argparse.BooleanOptionalAction, default=True,
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        config = EmbeddingConfig(
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            model_dtype=args.model_dtype,
            normalize_embeddings=args.normalize_embeddings,
        )
        input_dir = args.input_dir.resolve()
        paths = sorted(path for path in input_dir.glob("*.jsonl") if path.is_file())
        if not paths:
            raise ValueError(f"No chunk JSONL files found in {input_dir}")
        model_name = MODEL_NAMES[config.embedding_model.value]
        output_dir = args.output_dir / model_name / input_dir.name
        logging.info("Loading %s to embed %d reports", config.embedding_model, len(paths))
        embedder = Embedder(config)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        manifest_path.unlink(missing_ok=True)

        artifacts = []
        seen_ids: set[str] = set()
        started = None
        for index, path in enumerate(paths, start=1):
            texts, source_hash = _read_chunks(path, seen_ids)
            if started is None:
                started = perf_counter()
            vectors = embedder.embed_chunks(texts)
            finished = perf_counter()
            try:
                _validate_vectors(
                    vectors, (len(texts), embedder.dimension), config.normalize_embeddings
                )
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from error
            vector_path = output_dir / f"{path.stem}.npy"
            temporary = vector_path.with_suffix(".npy.tmp")
            with temporary.open("wb") as file:
                np.save(file, vectors, allow_pickle=False)
            temporary.replace(vector_path)
            artifacts.append({
                "source_filename": path.name,
                "source_sha256": source_hash,
                "vector_filename": vector_path.name,
                "chunk_count": len(texts),
            })
            logging.info("Embedded %d/%d: %s (%d chunks)", index, len(paths), path.stem, len(texts))

        for artifact in artifacts:
            vector_path = output_dir / artifact["vector_filename"]
            if not vector_path.is_file():
                raise ValueError(f"Missing embedding output: {vector_path}")

        write_manifest(manifest_path, {
            "model_id": config.embedding_model.value,
            "model_revision": embedder.revision,
            "model_dtype": config.model_dtype,
            "normalize_embeddings": config.normalize_embeddings,
            "embedding_dimension": embedder.dimension,
            "dtype": "float32",
            "source_chunk_dir": input_dir.as_posix(),
            "embedding_runtime_seconds": round(finished - started, 3),
            "artifacts": artifacts,
        })
    except (OSError, ValueError, RuntimeError) as error:
        logging.error("%s", error)
        return 1

    logging.info("Published %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
