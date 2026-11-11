# Financial RAG

A compact financial-report RAG engineering project built on the 100 annual reports from Enterprise RAG Challenge Round 2. Evaluation uses unseen questions over this fixed corpus; it does not measure generalization to unseen documents.

## Status

The PDF parsing baseline, page renderer, token-based chunker, and embedding stage are implemented. Parsing produces structured JSON with OCR, layout-aware reading order, native table extraction, and conservative year/value chart recovery. Rendering turns each parsed report into markdown page text. Chunking splits each page into overlapping text chunks with source offsets and citation metadata. Embedding converts those chunks into dense vectors. Indexing validates and uploads the embedding run to Qdrant with dense vectors, citation payloads, report filtering, and BM25 text indexing. Retrieval and generation are pending.

## Setup

Use Python 3.12 or later and `uv`. PDF parsing also requires an NVIDIA GPU whose driver supports CUDA 12.6:

```powershell
uv sync --cache-dir .cache/uv --locked --group dev
uv run --cache-dir .cache/uv --locked docling-tools models download -o data/models/docling layout tableformer easyocr --easyocr-lang en
uv run --cache-dir .cache/uv --locked python -m pytest -q
```

Linux and Windows install CUDA 12.6 PyTorch from the PyTorch wheel index. Model artifacts are local and untracked. Runtime and development dependencies are reproducibly locked in `uv.lock`.

## Parsing

Place the source PDFs in `data/raw/reports/`, then run:

```powershell
uv run --cache-dir .cache/uv --locked python -m financial_rag.parsing
```

The parser writes one JSON report per PDF under `data/processed/parsed_reports/`. A directory parse first checks `data/raw/dataset_manifest.json`, then writes `manifest.json` beside the reports only after every JSON report matches that inventory. Explicit input files skip corpus publication:

```powershell
uv run --cache-dir .cache/uv --locked python -m financial_rag.parsing report.pdf --output-dir data/processed/parsed_reports --threads 4
```

See the [PDF parsing specification](specifications/parsing.md) for the fixed composition, output contract, provenance, and failure policy.

## Rendering

```powershell
uv run --cache-dir .cache/uv --locked python -m financial_rag.rendering
```

The renderer reads `data/processed/parsed_reports/` and writes `data/processed/rendered_reports/<document_id>.json`. Each file has `metadata.document_id`, `metadata.page_count`, and `pages`, a list of `{page, text}` in PDF page order. Blocks on a page are separated by a blank line. Table rows and list items stay on single lines. Pages that contain only footers, pictures, or no blocks are omitted. `page_count` still counts every PDF page.

## Chunking

```powershell
uv run --cache-dir .cache/uv --locked python -m financial_rag.chunking
```

The chunker reads `data/processed/rendered_reports/*.json` and writes one `data/processed/chunks/c<chunk_size>_o<chunk_overlap>/<document_id>.jsonl` per report. Each line contains `metadata` and `text`. Metadata records the chunk ID, document ID, original PDF page, page-local chunk index, token count, and character offsets into the rendered page text.

The baseline uses recursive splitting with the `Qwen/Qwen3-Embedding-0.6B` tokenizer, a chunk size of 384 tokens, and an overlap target of 64 tokens. Each page is split independently; chunks and overlap never cross page boundaries. Only the tokenizer is loaded, so chunking does not require a GPU or embedding-model weights. The tokenizer must be cached locally or downloadable from Hugging Face on first use.

Process selected reports or override the defaults:

```powershell
uv run --cache-dir .cache/uv --locked python -m financial_rag.chunking data/processed/rendered_reports/<document_id>.json
uv run --cache-dir .cache/uv --locked python -m financial_rag.chunking --chunk-size 512 --chunk-overlap 64
```

The second command writes `data/processed/chunks/c512_o64/`. The default corpus run produced 53,330 chunks across 100 JSONL files in `data/processed/chunks/c384_o64/`. Generated files remain local and Git-ignored. This stage does not publish a manifest or perform corpus-level validation. See the [chunking specification](specifications/chunking.md) for the schema, splitting behavior, and CLI contract.

## Embedding

```powershell
uv run --cache-dir .cache/uv --locked python -m financial_rag.embedding --device cuda --model-dtype float16
```

The default model is `Qwen/Qwen3-Embedding-0.6B`. Its weights and tokenizer must be cached locally or downloadable from Hugging Face. The command above uses CUDA and float16 inference. Use `--device cpu --model-dtype float32` for CPU inference.

By default, the command reads `data/processed/chunks/c384_o64/` and writes one `<document_id>.npy` per report under `data/processed/embeddings/qwen3_emb_06b/c384_o64/`. Each matrix stores normalized float32 vectors in source JSONL record order. Use `--input-dir` to select a different chunk directory.

The baseline run produced 53,330 vectors of 1,024 dimensions across 100 reports. The command publishes `manifest.json` after validating the selected run. The manifest records the model revision, vector settings, source hashes, output inventory, and embedding runtime. The source JSONL files retain the chunk text and citation metadata. See the [embedding specification](specifications/embedding.md) for the CLI, artifact schema, and validation rules.

## Indexing

Start Qdrant 1.20 and index a completed embedding run:

```powershell
docker compose up -d
uv run --cache-dir .cache/uv --locked python -m financial_rag.indexing
```

The default collection is `qwen3_emb_06b_c384_o64_v1`. A verified run publishes
`data/processed/indexes/qwen3_emb_06b_c384_o64_v1/manifest.json`.
To index Yuan embeddings, pass `--input-dir data/processed/embeddings/yuan_emb_20_en/c384_o64`;
its collection prefix is `yuan_emb_20_en`. Use `--version 2` for a new run.
Existing collections and published manifests are never overwritten.

If the repository moved since embedding, pass `--chunks-dir data/processed/chunks/c384_o64`.
Source hashes must still match. See the [indexing specification](specifications/indexing.md).

## Evaluation

The committed benchmark uses a reproducible 70-question development split and 30-question holdout split. See [evaluation policy](evaluation/README.md).
