# Evaluation benchmark specification

Benchmark v1 pairs each record in `data/benchmark/v1/questions.json` with an entry keyed by its exact question text in `data/benchmark/v1/answers.json`. The zero-based question position determines the stable ID, from `benchmark-v1-q000` to `benchmark-v1-q099`. Each generated JSONL record contains `question_id`, `question`, `kind`, `answers`, and `reference_pools`. Only the first three fields are inputs to the RAG pipeline.

## Gold answers

The answer key stores accepted answers as strings in an array. Examples from the benchmark illustrate the encoding for three question kinds.

| Kind | Encoding | Example |
| --- | --- | --- |
| `number` | Decimal string in the requested unit, without a currency symbol or thousands separators | `["348000.0"]` |
| `names` | One accepted list encoded as a comma-separated string | `["forming fabrics,pressing fabrics,drying fabrics"]` |
| `boolean` | Capitalized Boolean string | `["False"]` |

The prompt can request `N/A` when the report provides insufficient evidence. The gold answer determines the stratum. For q077, both years' dividends are disclosed and the comparison yields `False`.

## Evidence and split membership

Each reference is `<document_id>:<zero_based_pdf_index>`, counting from the first physical page of the PDF. A complete evidence set covers every pool with at least one cited page from that pool. For q064, one pool supports the two-year total and the other supports openings in 2022. An empty pool list indicates that the question has no gold retrieval references.

The split configuration pins the 30 holdout IDs. The other 70 IDs belong to development. The generator checks both source hashes and the expected counts by question kind, answer category, and reference availability before publishing the datasets. See the [evaluation README](../evaluation/README.md) for the stratum counts and usage protocol.
