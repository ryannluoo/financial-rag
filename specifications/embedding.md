# Report embedding specification

## Scope

The embedding stage converts the text in report chunk JSONL files into dense vectors for indexing. The CLI loads one `Embedder` from `financial_rag.embedding.embedder`, processes reports individually, and publishes a manifest for the completed run. The original JSONL remains the source of chunk text and citation metadata.

## Input

The default input directory is `data/processed/chunks/c384_o64/`. The command processes regular `*.jsonl` files directly inside the selected directory in lexicographic path order. Each file represents one report and follows the [chunking specification](chunking.md). The command uses three fields from each record.

| Field | Requirement |
| --- | --- |
| `text` | String with at least one non-whitespace character. The original string is passed to the model. |
| `metadata.chunk_id` | String with at least one non-whitespace character, unique across the selected files. |
| `metadata.document_id` | Must equal the source filename stem. |

The command processes records in file order and hashes the exact source bytes with SHA-256. A zero-byte input file represents a report with zero chunks.

## Configuration and operation

```powershell
uv run --cache-dir .cache/uv --locked python -m financial_rag.embedding --device cuda --model-dtype float16
```

| Option | Default | Behavior |
| --- | --- | --- |
| `--input-dir` | `data/processed/chunks/c384_o64` | Select a directory of report JSONL files. |
| `--output-dir` | `data/processed/embeddings` | Select the parent directory for embedding outputs. |
| `--embedding-model` | `Qwen/Qwen3-Embedding-0.6B` | Also accepts `IEITYuan/Yuan-embedding-2.0-en`. The corpus baseline uses Qwen. |
| `--batch-size` | `32` | Positive integer for batches within each report. |
| `--device` | Automatic | Pass a device such as `cuda` or `cpu` to Sentence Transformers. |
| `--model-dtype` | Model loader default | Accepts `float16` or `float32` for inference. Stored vectors are always float32. |
| `--normalize-embeddings`, `--no-normalize-embeddings` | Enabled | Apply unit-length normalization during encoding. |

The embedder resolves the Hugging Face commit from the model configuration, then loads Sentence Transformers at that revision. Model files must be in the local Hugging Face cache or downloadable. With a complete cache, set `HF_HUB_OFFLINE=1` to run offline.

`embed_chunks(chunks: list[str])` returns a float32 NumPy array with shape `(len(chunks), D)`, where `D` is the model's reported embedding dimension. It uses the model's default document encoding behavior. `embed_query(queries: str | list[str], prompt_name="query")` returns a float32 vector of shape `(D,)` for one query or a matrix of shape `(Q, D)` for a non-empty batch of `Q` queries. It validates any non-empty prompt name against the loaded model. Query embeddings are produced in memory.

## Vector files

The default output directory is `data/processed/embeddings/qwen3_emb_06b/c384_o64/`. The command appends the shared model alias and the resolved input directory's basename to the output parent. The model aliases are `qwen3_emb_06b` for `Qwen/Qwen3-Embedding-0.6B` and `yuan_emb_20_en` for `IEITYuan/Yuan-embedding-2.0-en`; embedding directories and collection names use the same mapping.

Each `<document_id>.npy` contains one float32 matrix of shape `(N, D)`, with `N` equal to the number of source records. Row `i` embeds record `i` of the matching JSONL file, using zero-based indices. Empty reports produce shape `(0, D)`. NumPy files are written with pickling disabled. Generated artifacts are Git-ignored under the default output parent.

## Manifest

The command writes one `manifest.json` beside the vector files after all selected reports have passed validation and their output files exist.

| Field | Meaning |
| --- | --- |
| `model_id` | Hugging Face model ID. |
| `model_revision` | Commit hash resolved and used for loading. |
| `model_dtype` | Requested inference dtype, or `null` when the model loader selects it. |
| `normalize_embeddings` | Whether encoding applied normalization. |
| `embedding_dimension` | Model-reported dimension `D`. |
| `dtype` | Stored array dtype, always `float32`. |
| `source_chunk_dir` | Absolute path to the resolved chunk directory. |
| `embedding_runtime_seconds` | Wall-clock seconds from immediately before the first report's embedding call to immediately after the last. This includes work between calls and excludes model loading, the first source read, and final validation and publication. |
| `artifacts` | One entry per report in processing order. |

Each artifact entry contains `source_filename` relative to the source directory, `source_sha256` for its raw bytes, `vector_filename` relative to the manifest directory, and `chunk_count` for the number of rows. Indexing should use this inventory and verify the source hashes before joining arrays to records. Rechunking may change the source text while retaining the same positional chunk IDs.

## Validation and publication

The command validates each source record before encoding its report. It checks every output matrix for the expected shape, float32 dtype, finite values, and nonzero row norms. With normalization enabled, each norm must be within `0.001` of one. After writing all matrices, it checks that each selected report's vector file exists.

Once the model has loaded, the command removes any prior manifest before replacing vector files. Each matrix and the final manifest are written to temporary files and then moved into place. Failures return a nonzero exit status. A failed run can leave completed matrices or temporary files. Treat the manifest as the completed run's inventory because unrelated files in the output directory are left in place.

## Baseline verification

The run with `Qwen/Qwen3-Embedding-0.6B` at revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` used CUDA, float16 inference, batches of 32, and normalization. It produced 53,330 vectors of dimension 1,024 across 100 reports in 1,240.037 seconds. Reading back the files verified source hashes, row counts, shape, dtype, and finite values. The maximum absolute deviation from a unit norm was approximately 0.000484.
