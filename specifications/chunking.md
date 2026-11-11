# Report chunking specification

## Scope

Chunking converts rendered financial-report pages into overlapping text chunks. Rendering produces the input. The [embedding stage](embedding.md) consumes the output JSONL files in record order.

The baseline is `financial_rag.chunking.chunker.RecursiveTokenChunker`, configured with `ChunkingConfig`. It does not publish a manifest or perform corpus-level inventory or integrity validation.

## Input

Each input is a rendered report as defined in the [page rendering specification](rendering.md). Chunking reads:

- `metadata.document_id`: the report identifier;
- `pages`: an array of `{page, text}` objects;
- `page`: the original 1-based PDF page number;
- `text`: the rendered Markdown page text.

Other fields, including `metadata.page_count`, are ignored. Pages are processed in their input order. Original page numbers are preserved, including gaps from pages omitted during rendering. The CLI requires `metadata.document_id` to match the input filename stem.

## Configuration

| Setting | Default | Constraint |
| --- | --- | --- |
| `chunk_size` | `384` | Integer from 32 through 2048, measured in tokenizer tokens. |
| `chunk_overlap` | `64` | Non-negative integer, at most `floor(chunk_size * 0.30)`. |
| `tokenizer_model` | `Qwen/Qwen3-Embedding-0.6B` | Hugging Face tokenizer model ID or local tokenizer path. |

The tokenizer is loaded with `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`. Tokenizer files must be available in the local Hugging Face cache, at the supplied local path, or downloadable on first use. Chunking loads no embedding-model weights and requires no GPU.

## Splitting

The chunker uses LangChain's `RecursiveCharacterTextSplitter.from_huggingface_tokenizer`. The tokenizer measures split lengths; this is recursive text splitting with token-based length measurement, rather than slicing a token-ID sequence.

Separators are tried in this order:

```python
["\n\n", "\n", ". ", "; ", " ", ""]
```

Separators are retained at the end of their splits (`keep_separator="end"`). The splitter strips leading and trailing whitespace from assembled chunks. Internal Markdown, whitespace, and line breaks remain part of the source text.

Each page is split independently. Overlap is applied only within a page. Chunk size and overlap guide the splitter's assembly of text segments; chunks can be shorter than the configured size, and adjacent chunks need not overlap by exactly the configured token count. The implementation does not apply a second splitting pass or enforce a post-split token limit.

Empty or whitespace-only pages produce no chunks. A report with no chunks still produces an empty output file. Tables use the same splitting rules as other rendered text: there is no special table preservation, row grouping, header repetition, or cross-page assembly.

## Output schema

Each chunk is one JSON object with two top-level members, `metadata` and `text`. Metadata fields are written in the order below.

| Field | Type | Meaning |
| --- | --- | --- |
| `metadata.chunk_id` | string | `<document_id>:p<page>:c<chunk_index>`. |
| `metadata.document_id` | string | Report identifier copied unchanged from the input. |
| `metadata.page` | integer | Original 1-based PDF page number. |
| `metadata.chunk_index` | integer | Zero-based chunk position within the page; resets on each page. |
| `metadata.token_count` | integer | `len(tokenizer.encode(text, add_special_tokens=False))`. |
| `metadata.char_start` | integer | Inclusive offset into the original rendered page text. |
| `metadata.char_end` | integer | Exclusive offset into the original rendered page text. |
| `text` | string | Non-empty chunk content, including any overlap. |

Character offsets use Python string indexing, not UTF-8 byte positions or PDF coordinates. Every chunk must satisfy:

```python
chunk["text"] == page_text[
    chunk["metadata"]["char_start"]:chunk["metadata"]["char_end"]
]
```

Source lookup moves forward while allowing token-bounded overlap. It accounts for repeated content, including table cells, and anchors the final chunk to its last source occurrence. A trailing chunk may be fully contained in the preceding chunk's overlap. Failure to locate a chunk raises `ValueError`.

Chunk IDs are deterministic for the same rendered input, configuration, dependencies, and tokenizer artifacts. They identify positions in a particular chunking result; rechunking can change the text associated with an ID. Benchmark page citations are derived as `<document_id>:<page>`.

## Files

The default input directory is `data/processed/rendered_reports`. A directory run selects every `*.json` file directly in that directory, in lexicographic path order. It does not recurse or exclude a file named `manifest.json`.

The default parent directory is `data/processed/chunks`. Each run writes into a subdirectory named from the active chunk size and overlap:

```text
data/processed/chunks/c<chunk_size>_o<chunk_overlap>/
```

`--output-dir` selects that parent. The size and overlap in the directory name always come from the configuration used for the run. Each selected `<document_id>.json` produces `<document_id>.jsonl` in that subdirectory, with one chunk object per physical line. Newlines inside chunk text are JSON-escaped. Files use UTF-8 without a BOM, preserved Unicode, LF line endings, and a final newline for each record. Empty reports produce zero-byte files.

The command creates the output directory when needed and replaces existing files of the same name. Other output files remain untouched. Writes are not transactional: a failed run can leave completed reports or a partially written output file. No manifest or other publication artifact is written.

Generated chunks are Git-ignored under `data/processed/`.

## Operation

Chunk all rendered reports with the defaults:

```powershell
uv run --cache-dir .cache/uv --locked python -m financial_rag.chunking
```

Process selected reports:

```powershell
uv run --cache-dir .cache/uv --locked python -m financial_rag.chunking data/processed/rendered_reports/<document_id>.json
```

Override paths and splitting settings:

```powershell
uv run --cache-dir .cache/uv --locked python -m financial_rag.chunking --input-dir data/processed/rendered_reports --output-dir data/processed/chunks --chunk-size 512 --chunk-overlap 64 --tokenizer-model Qwen/Qwen3-Embedding-0.6B
```

That override writes `data/processed/chunks/c512_o64/`. Positional paths replace directory selection and are processed in the order supplied. The CLI accepts `--input-dir`, `--output-dir`, `--chunk-size`, `--chunk-overlap`, and `--tokenizer-model`. It logs the chunk count for each report and prints total report and chunk counts on success.

A missing input directory, no selected reports, invalid configuration, unreadable or malformed input, filename/ID mismatch, tokenizer loading failure, source lookup failure, or output write failure causes a nonzero exit. Previously written outputs remain.

## Baseline verification

The default run over the current 100 rendered reports wrote `data/processed/chunks/c384_o64/`: 100 JSONL files containing 53,330 chunks. Checks confirmed one-to-one filenames, unique chunk IDs, exact source slices, complete coverage of non-whitespace page text, and a maximum observed chunk length of 384 tokens. These are observations for this corpus run, not corpus-level validation implemented in the command.

`tests/test_chunker.py` covers page boundaries, token counts, Unicode offsets, repeated content, zero overlap, empty pages, trailing overlap and table delimiters, JSONL serialization, and missing input handling. Tests use a local in-memory tokenizer and require no model downloads.
