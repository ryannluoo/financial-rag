# Benchmark data

This benchmark contains 100 questions and gold answers over the project's fixed corpus of 100 annual reports. The original questions and answer key come from [Enterprise RAG Challenge Round 2](https://github.com/trustbit/enterprise-rag-challenge/tree/main/round2).

| Version | Contents |
| --- | --- |
| [v0](v0/) | Untouched upstream `questions.json` and `answers.json`, kept as the public reference. |
| [v1](v1/) | Current benchmark. Contains the corrected `questions.json` and `answers.json` used to generate the evaluation splits. |

## Source file schema

| File | Structure |
| --- | --- |
| `questions.json` | Ordered array of objects with `text` and `kind`. The zero-based position gives the question number. |
| `answers.json` | Object keyed by the exact question text. Each entry contains the matching `kind`, `answers` as an array of accepted strings, and `reference_pools` as an array of citation groups. |

`kind` is `number`, `boolean`, `name`, or `names`. In v1, a citation is `<document_id>:<zero_based_pdf_index>`, where `document_id` is the report PDF's SHA-1 filename stem. A complete citation set includes at least one page from each pool. `[]` means the item has no gold retrieval references.

See the [changelog](CHANGELOG.md) for the v0-to-v1 changes and the [evaluation README](../../evaluation/README.md) for split generation and usage.
