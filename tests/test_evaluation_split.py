import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "evaluation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _answer_category(record: dict) -> str:
    normalized = [str(value).strip().lower() for value in record["answers"]]
    if record["kind"] == "boolean":
        return normalized[0]
    return "na" if all(value == "n/a" for value in normalized) else "value"


def test_committed_evaluation_split_matches_configuration():
    config_path = EVALUATION_DIR / "split_config.json"
    manifest_path = EVALUATION_DIR / "split_manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_sizes = {"development": 70, "holdout": 30}
    expected_reference_counts = {"development": 34, "holdout": 15}
    required_fields = {
        "question_id",
        "question",
        "kind",
        "answers",
        "reference_pools",
    }

    assert config["split_sizes"] == expected_sizes
    assert manifest["config"]["sha256"] == _sha256(config_path)
    assert manifest["strata"] == config["strata"]

    for source_name, source in config["source_files"].items():
        source_path = ROOT / source["path"]
        actual_hash = _sha256(source_path)
        assert actual_hash == source["expected_sha256"]
        assert actual_hash == manifest["source_files"][source_name]["sha256"]

    datasets = {}
    for split_name, expected_size in expected_sizes.items():
        dataset_path = ROOT / config["outputs"][split_name]
        records = _load_jsonl(dataset_path)
        datasets[split_name] = records

        assert len(records) == expected_size
        assert all(required_fields <= record.keys() for record in records)
        assert _sha256(dataset_path) == manifest["splits"][split_name]["sha256"]
        assert sum(bool(record["reference_pools"]) for record in records) == (
            expected_reference_counts[split_name]
        )

        actual_strata = Counter(
            (
                record["kind"],
                _answer_category(record),
                bool(record["reference_pools"]),
            )
            for record in records
        )
        expected_strata = Counter(
            {
                (
                    row["kind"],
                    row["answer_category"],
                    row["has_reference"],
                ): row[split_name]
                for row in config["strata"]
                if row[split_name]
            }
        )
        assert actual_strata == expected_strata

    development_ids = {
        record["question_id"] for record in datasets["development"]
    }
    holdout_ids = {record["question_id"] for record in datasets["holdout"]}
    expected_ids = {f"benchmark-v1-q{index:03d}" for index in range(100)}

    assert development_ids.isdisjoint(holdout_ids)
    assert development_ids | holdout_ids == expected_ids
