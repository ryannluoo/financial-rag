"""Generate deterministic development and holdout datasets from benchmark v1."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("split_config.json")


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def answer_category(kind: str, answers: list[Any]) -> str:
    """Return the answer category used by the split configuration."""

    if not answers:
        raise ValueError("Each benchmark question must have at least one answer.")

    normalized = [str(value).strip().lower() for value in answers]

    if kind == "boolean":
        values = set(normalized)
        if values == {"true"}:
            return "true"
        if values == {"false"}:
            return "false"
        raise ValueError(f"Unexpected boolean answers: {answers}")

    is_na = [value == "n/a" for value in normalized]
    if all(is_na):
        return "na"
    if any(is_na):
        raise ValueError(f"N/A cannot be mixed with value answers: {answers}")
    return "value"


def stratum_key(
    kind: str,
    category: str,
    has_reference: bool,
) -> tuple[str, str, bool]:
    return kind, category, has_reference


def load_and_validate_sources(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    sources = config["source_files"]
    paths = {
        name: ROOT / source["path"]
        for name, source in sources.items()
    }
    actual_hashes = {name: sha256(path) for name, path in paths.items()}

    for name, source in sources.items():
        expected = source["expected_sha256"].lower()
        actual = actual_hashes[name]
        if actual != expected:
            raise ValueError(
                f"Frozen source hash mismatch for {source['path']}: "
                f"expected {expected}, found {actual}"
            )

    questions = load_json(paths["questions"])
    answers = load_json(paths["answers"])

    question_texts = [question["text"] for question in questions]
    if len(question_texts) != len(set(question_texts)):
        raise ValueError("Question text must be unique within benchmark v1.")
    if set(question_texts) != set(answers):
        missing = sorted(set(question_texts) - set(answers))
        extra = sorted(set(answers) - set(question_texts))
        raise ValueError(
            f"Question and answer keys differ. Missing: {missing}; extra: {extra}"
        )

    return questions, answers, actual_hashes


def build_buckets(
    questions: list[dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    question_id_prefix: str,
) -> dict[tuple[str, str, bool], list[dict[str, Any]]]:
    buckets: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)

    for index, question in enumerate(questions):
        text = question["text"]
        kind = question["kind"]
        answer = answers[text]

        if answer["kind"] != kind:
            raise ValueError(
                f"Question and answer kinds differ for source index {index}: "
                f"{kind!r} != {answer['kind']!r}"
            )

        category = answer_category(kind, answer["answers"])
        has_reference = bool(answer["reference_pools"])
        key = stratum_key(kind, category, has_reference)

        buckets[key].append(
            {
                "source_index": index,
                "question_id": f"{question_id_prefix}{index:03d}",
                "question": text,
                "kind": kind,
                "answers": answer["answers"],
                "reference_pools": answer["reference_pools"],
            }
        )

    return buckets


def validate_config(
    config: dict[str, Any],
    buckets: dict[tuple[str, str, bool], list[dict[str, Any]]],
) -> dict[tuple[str, str, bool], dict[str, Any]]:
    expected_dimensions = ["kind", "answer_category", "has_reference"]
    if config["stratify_by"] != expected_dimensions:
        raise ValueError(f"stratify_by must be {expected_dimensions}")

    configured: dict[tuple[str, str, bool], dict[str, Any]] = {}
    for row in config["strata"]:
        key = stratum_key(
            row["kind"],
            row["answer_category"],
            row["has_reference"],
        )
        if key in configured:
            raise ValueError(f"Duplicate configured stratum: {key}")
        if row["development"] + row["holdout"] != row["total"]:
            raise ValueError(f"Configured split counts do not sum for {key}")
        configured[key] = row

    if set(configured) != set(buckets):
        missing = sorted(set(buckets) - set(configured))
        extra = sorted(set(configured) - set(buckets))
        raise ValueError(
            f"Configured and observed strata differ. Missing: {missing}; extra: {extra}"
        )

    for key, row in configured.items():
        actual = len(buckets[key])
        if actual != row["total"]:
            raise ValueError(
                f"Unexpected benchmark count for {key}: "
                f"expected {row['total']}, found {actual}"
            )

    for split_name in ("development", "holdout"):
        configured_total = sum(row[split_name] for row in config["strata"])
        expected_total = config["split_sizes"][split_name]
        if configured_total != expected_total:
            raise ValueError(
                f"Configured {split_name} strata sum to {configured_total}, "
                f"not {expected_total}"
            )

    return configured


def split_records(
    config: dict[str, Any],
    buckets: dict[tuple[str, str, bool], list[dict[str, Any]]],
    configured: dict[tuple[str, str, bool], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(config["seed"])
    splits: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "holdout": [],
    }

    for row in config["strata"]:
        key = stratum_key(
            row["kind"],
            row["answer_category"],
            row["has_reference"],
        )
        entries = list(buckets[key])
        rng.shuffle(entries)
        development_count = configured[key]["development"]

        splits["development"].extend(entries[:development_count])
        splits["holdout"].extend(entries[development_count:])

    for records in splits.values():
        records.sort(key=lambda record: record["source_index"])

    development_ids = {
        record["question_id"] for record in splits["development"]
    }
    holdout_ids = {record["question_id"] for record in splits["holdout"]}
    if development_ids & holdout_ids:
        raise RuntimeError("Development and holdout question IDs overlap.")
    if len(development_ids | holdout_ids) != sum(config["split_sizes"].values()):
        raise RuntimeError("Generated splits do not cover the complete benchmark.")

    return splits


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in records:
        output_record = {
            key: value
            for key, value in record.items()
            if key != "source_index"
        }
        lines.append(json.dumps(output_record, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    config: dict[str, Any],
    actual_source_hashes: dict[str, str],
    split_paths: dict[str, Path],
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    split_summaries = {}
    for split_name, records in splits.items():
        split_summaries[split_name] = {
            "path": config["outputs"][split_name],
            "sha256": sha256(split_paths[split_name]),
            "questions": len(records),
            "with_reference": sum(bool(record["reference_pools"]) for record in records),
            "without_reference": sum(
                not record["reference_pools"] for record in records
            ),
        }

    return {
        "schema_version": config["schema_version"],
        "benchmark_version": config["benchmark_version"],
        "strategy": config["strategy"],
        "seed": config["seed"],
        "question_id_prefix": config["question_id_prefix"],
        "stratify_by": config["stratify_by"],
        "config": {
            "path": CONFIG_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256(CONFIG_PATH),
        },
        "source_files": {
            name: {
                "path": source["path"],
                "sha256": actual_source_hashes[name],
            }
            for name, source in config["source_files"].items()
        },
        "splits": split_summaries,
        "strata": config["strata"],
    }


def main() -> None:
    config = load_json(CONFIG_PATH)
    questions, answers, actual_source_hashes = load_and_validate_sources(config)
    buckets = build_buckets(questions, answers, config["question_id_prefix"])
    configured = validate_config(config, buckets)
    splits = split_records(config, buckets, configured)

    split_paths = {
        split_name: ROOT / config["outputs"][split_name]
        for split_name in ("development", "holdout")
    }
    for split_name, path in split_paths.items():
        write_jsonl(path, splits[split_name])

    manifest = build_manifest(
        config,
        actual_source_hashes,
        split_paths,
        splits,
    )
    manifest_path = ROOT / config["outputs"]["manifest"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    for split_name, records in splits.items():
        with_reference = sum(bool(record["reference_pools"]) for record in records)
        print(
            f"{split_name.capitalize()}: {len(records)} questions "
            f"({with_reference} with references)"
        )


if __name__ == "__main__":
    main()
