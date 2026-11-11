# Report indexing specification

The CLI consumes the completed embedding manifest in `--input-dir` and only
indexes the listed JSONL/NumPy pairs, preserving source record order.
The default input is `data/processed/embeddings/qwen3_emb_06b/c384_o64`.

## Collection naming

Names follow `<model>_c<size>_o<overlap>_v<version>`. The model mapping is:

| Embedding model | Prefix |
| --- | --- |
| `Qwen/Qwen3-Embedding-0.6B` | `qwen3_emb_06b` |
| `IEITYuan/Yuan-embedding-2.0-en` | `yuan_emb_20_en` |

Chunk settings come from the source directory name `c<size>_o<overlap>`.
The positive integer `--version` defaults to 1. `--chunks-dir` overrides the
manifest's source path after a repository move; source hashes remain mandatory.

## Validation and indexing

Before creating the collection, validate manifest settings, unique matching
artifact filenames, source hashes, chunk counts, positional chunk IDs,
citation metadata, and vector shape, float32 dtype, finite values, and nonzero
norms. Normalized vectors must have unit norms within 0.001.
Pickling is disabled. Capture vector hashes in preflight and revalidate all
report inputs during upload. Memory is bounded by report size.

The CLI invokes `QdrantIndexer.create_collection()` once and `upload_chunks()`
per report. Each point has a deterministic UUID derived from its chunk ID,
a named `dense` cosine vector, and flat chunk metadata plus `text` payload.
Create a keyword index on `document_id` for report filtering and a full-text
index on `text` using word tokenization, lowercasing, and BM25 scoring.
Qdrant server/client 1.20 are required. SDK uploads use `wait=True` with
`--batch-size` defaulting to 256. `--qdrant-url` defaults to `http://localhost:6333`.

## Completion and manifest

The exact server point count must equal the validated unique chunk count.
Then atomically publish one `manifest.json` in `--output-dir/<collection_name>`;
the output parent defaults to `data/processed/indexes`.
The manifest records model ID/revision, dimension, dtype, normalization,
chunk settings, index definitions, source paths, embedding-manifest SHA-256,
report and verified point counts, batch size, indexing runtime, and artifact
inventory with source and vector SHA-256 hashes. Runtime excludes preflight
and includes collection creation, upload, and count verification.

Existing collections and manifests fail without replacement. Failures return
a nonzero exit status and never publish a new completion manifest. A remote
failure may leave a partial collection; a publication failure may leave an
unpublished complete collection or temporary file. Cleanup is explicit.
