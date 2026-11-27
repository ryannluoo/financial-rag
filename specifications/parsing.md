# PDF parsing specification

## Scope

The baseline parser converts financial-report PDFs into structured JSON reports. Rendering, chunking, embedding, indexing, retrieval, and generation are separate stages.

## Provenance

The Docling pipeline configuration is adopted from the PDF parser in the [Enterprise RAG Challenge Round 2 winning submission](https://github.com/IlyaRice/RAG-Challenge-2) at commit `452d688d1e0d6c3dc5150f9e42c2facb477178b7`. The output contract derives its structure from that submission's report format and is defined by this project; field names follow this project's conventions where they differ.

The baseline parser is not a reproduction of the submission's pipeline or results, and its configuration has not been tuned or compared against alternative parsers on this corpus. The implementation, API names, module layout, and command-line interface are specific to this project.

## Composition

The baseline parser is `financial_rag.parsing.parser.FinancialReportParser`. It uses:

- Docling's maintained `DoclingParseDocumentBackend` and `StandardPdfPipeline`;
- EasyOCR with English recognition and selective OCR;
- TableFormer in accurate mode with PDF-cell matching;
- CUDA acceleration (`device=cuda`) with four CPU threads by default;
- layout and OCR batch size 1, to fit an 8GB GPU;
- CUDA 12.6 PyTorch wheels on Linux and Windows;
- local model artifacts under `data/models/docling`;
- GitHub-flavored Markdown table rendering through `tabulate`.

Parsing requires an NVIDIA GPU and a driver that supports CUDA 12.6. PyTorch does not publish CUDA wheels for macOS; that platform is not a supported parsing host.

Built-in chart extraction is disabled. A deterministic postprocessor promotes a picture to a table only when its text contains consecutive year labels and an equal number of unambiguous currency or percentage values.

The parser rejects output containing a material concentration of unresolved `/gid` identifiers. A rejected or failed conversion produces no output files.

## Output contract

Each input PDF produces `<document_id>.json` in `data/processed/parsed_reports` unless another output directory is selected.

The document ID is the filename stem of the source PDF. In the ERC2 corpus it equals the `document_id` in `data/raw/dataset_manifest.json`, which is the SHA-1 of the PDF bytes. The parser takes the ID from the filename. Benchmark references pair this ID with a zero-based physical PDF index. Parsed page 1 corresponds to index 0.

The JSON object has four top-level members, in order:

1. `metadata`: `document_id` and document-level item counts;
2. `content`: page-grouped reading-order references to text, tables, and pictures;
3. `tables`: page, bounds, dimensions, Markdown, HTML, and the Docling table object;
4. `pictures`: page, bounds, and recognized child text.

UTF-8, LF newlines, two-space JSON indentation, and preserved Unicode make outputs deterministic for a fixed input, dependency lock, and model set.

## Corpus publication

Parsing an input directory, with no explicit PDF arguments, checks `data/raw/dataset_manifest.json` before the parser starts. The manifest `corpus_fingerprint` must match the canonical digest in `data/raw/README.md`. The PDF filenames in the input directory must be exactly the manifest inventory, and each file's SHA-1 must equal that document's `document_id` and `sha1`. A mismatch raises `ValidationError` and does not start conversion.

After every report converts successfully, the parsed directory is checked before publication. Each source document must have exactly one `<document_id>.json`, with no other JSON report files. `metadata.document_id` must equal the source document id and the output stem. A mismatch raises `ValidationError` and does not write a publication manifest.

On success the command writes `manifest.json` in the output directory, beside the reports. That file records the corpus (`document_count`, `page_count`), source (`manifest_path`, `corpus_fingerprint`), parser (`class_path`, `package_version` from `pyproject.toml`), and the SHA-256 of each JSON report. A directory parse removes any existing publication manifest before conversion, so the file exists only for the outputs that just passed validation.

Explicit PDF arguments skip these corpus checks and do not write `manifest.json`.

## Dependencies

Runtime versions are locked in `uv.lock`. Direct parser dependencies are:

- `docling[easyocr]==2.126.0`
- `tabulate==0.10.0`
- `torch==2.14.0` from the PyTorch CUDA 12.6 index on Linux and Windows
- `torchvision==0.29.0` from the same index

PyMuPDF is not a direct or transitive parser dependency.

## Operation

Install the environment and local models:

```powershell
uv sync --cache-dir .cache/uv --locked --group dev
uv run --cache-dir .cache/uv --locked docling-tools models download -o data/models/docling layout tableformer easyocr --easyocr-lang en
```

Parse the corpus:

```powershell
uv run --cache-dir .cache/uv --locked python -m financial_rag.parsing
```

Parse selected PDFs or override paths:

```powershell
uv run --cache-dir .cache/uv --locked python -m financial_rag.parsing report-a.pdf report-b.pdf --output-dir data/processed/parsed_reports
```

The command exits nonzero when corpus validation, conversion, parsed-output validation, or an output-quality check fails.
