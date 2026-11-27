# Evaluation benchmark

This directory defines the reproducible development and holdout splits for the active benchmark. Both splits use the same fixed collection of annual reports.

## Files

- `split_config.json` is the authoritative, human-authored split policy.
- `generate_split.py` validates the frozen benchmark and generates the split artifacts.
- `split_manifest.json` records source and output hashes plus realized composition.
- `datasets/development.jsonl` contains the 70 development QA records.
- `datasets/holdout.jsonl` contains the 30 holdout QA records.

The generator reads the active [questions](../data/benchmark/v1/questions.json) and [answer key](../data/benchmark/v1/answers.json) and publishes both JSONL snapshots.

## Split design

The original question-level split used seed 42. The configuration now pins its 30 holdout question IDs. The other 70 IDs form the development split. The generator validates question kind, answer category, and reference availability within each split.

| Stratum | Total | Development | Holdout |
| --- | ---: | ---: | ---: |
| `number/value` | 26 | 18 | 8 |
| `number/na` | 33 | 24 | 9 |
| `boolean/true` | 16 | 10 | 6 |
| `boolean/false` with references | 2 | 1 | 1 |
| `boolean/false` without references | 6 | 5 | 1 |
| `name/value` | 8 | 6 | 2 |
| `name/na` | 1 | 1 | 0 |
| `names/value` | 6 | 4 | 2 |
| `names/na` | 2 | 1 | 1 |
| **Total** | **100** | **70** | **30** |

Reference-backed records are distributed as follows:

| Reference status | Total | Development | Holdout |
| --- | ---: | ---: | ---: |
| With references | 58 | 39 | 19 |
| Without references | 42 | 31 | 11 |

The detailed targets are defined in `split_config.json`. See the [evaluation specification](../specifications/evaluation.md) for record formats and reference-pool semantics.

## Source integrity and question IDs

`split_config.json` pins the active source paths, benchmark version, and SHA-256 hashes. Generation stops if either source hash differs from its expected value. Git attributes and the generator pin line endings so the hashes are consistent across checkouts.

Question IDs are generated from source positions as `benchmark-v1-q000` through `benchmark-v1-q099`. Integrate pre-experiment corrections into v1. After a batch of corrections, refresh the source hashes and regenerate the splits and manifest once. After experiments begin, version subsequent source changes separately. Use the version recorded in `split_manifest.json` when reporting scores.

## Usage protocol

Use the development split for repeated system iteration, category analysis, ablations, and failure analysis. Report category results as counts with percentages because the strata are small.

Evaluate the holdout once after the final system configuration is frozen. Report one aggregate score as a fraction with a 95% Wilson interval. Holdout category slices are too small for reliable conclusions.

Gold answers and reference pools are committed in both datasets for reproducibility. The evaluation layer may read them, but the RAG pipeline must receive only the question ID, question text, and kind.

Experiment outputs belong under `experiments/<run-name>/`, not in this directory. A run should reference question IDs and store its configuration, predictions, and metrics without copying the benchmark records.

## Regeneration

From the repository root, run:

```console
python evaluation/generate_split.py
```

The command validates source hashes and composition before replacing the generated datasets and manifest. To check whether a checkout is in sync without rewriting files, run `python -m pytest -q tests/test_evaluation_split.py`. The tests compare every split record with its source by question ID and check the file hashes against the manifest.
