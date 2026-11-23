"""Checks that source PDFs and parsed reports match the frozen dataset manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from financial_rag.parsing.manifest import MANIFEST_FILENAME

DEFAULT_DATASET_MANIFEST = Path("data/raw/dataset_manifest.json")


class ValidationError(Exception):
    """Raised when the source corpus or parsed reports fail an integrity check."""


def corpus_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 digest of a dataset manifest.

    The digest covers ``dataset_id``, ``document_count``, and ``documents``
    only. Document entries are sorted by ``document_id``. Serialization is
    compact JSON with recursively sorted keys, UTF-8, and no trailing newline.
    """
    try:
        documents = sorted(
            manifest["documents"],
            key=lambda document: document["document_id"],
        )
        payload = {
            "dataset_id": manifest["dataset_id"],
            "document_count": manifest["document_count"],
            "documents": documents,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValidationError(
            f"dataset manifest cannot be fingerprinted: {error}"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raise_if(problems: list[str]) -> None:
    if problems:
        raise ValidationError("\n".join(problems))


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read dataset manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise ValidationError("dataset manifest must be a JSON object")
    return manifest


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicated: set[str] = set()
    for value in values:
        if value in seen:
            duplicated.add(value)
        seen.add(value)
    return sorted(duplicated)


def _identity_problems(documents: list[Any]) -> list[str]:
    problems: list[str] = []
    document_ids: list[str] = []
    filenames: list[str] = []
    for document in documents:
        if not isinstance(document, dict):
            problems.append("dataset manifest document is not an object")
            continue
        document_id = document.get("document_id")
        filename = document.get("filename")
        sha1 = document.get("sha1")
        if not all(
            isinstance(value, str) and value for value in (document_id, filename, sha1)
        ):
            problems.append(
                "dataset manifest document is missing document_id, filename, or sha1"
            )
            continue
        if Path(filename).name != filename or Path(filename).suffix != ".pdf":
            problems.append(f"invalid filename: {filename}")
            continue
        document_ids.append(document_id)
        filenames.append(filename)
        if Path(filename).stem != document_id:
            problems.append(f"filename does not match document_id: {filename}")
        if sha1 != document_id:
            problems.append(f"document_id does not match sha1: {filename}")
    duplicate_ids = _duplicates(document_ids)
    duplicate_names = _duplicates(filenames)
    if duplicate_ids:
        problems.append("duplicate document_id values: " + ", ".join(duplicate_ids))
    if duplicate_names:
        problems.append("duplicate filenames: " + ", ".join(duplicate_names))
    return problems


def validate_source_corpus(
    reports_dir: Path,
    manifest_path: Path = DEFAULT_DATASET_MANIFEST,
) -> dict[str, Any]:
    """Abort unless ``reports_dir`` is exactly the frozen dataset manifest.

    Every PDF filename and SHA-1 must match the manifest, with no missing or
    unexpected PDFs, and ``corpus_fingerprint`` must match the canonical digest.
    """
    reports_dir = Path(reports_dir)
    manifest = _load_manifest(Path(manifest_path))
    recorded = manifest.get("corpus_fingerprint")
    computed = corpus_fingerprint(manifest)
    if recorded != computed:
        raise ValidationError(
            f"corpus fingerprint mismatch: manifest {recorded}, computed {computed}"
        )

    documents = manifest["documents"]
    if manifest["document_count"] != len(documents):
        raise ValidationError("document_count does not match the document inventory")
    _raise_if(_identity_problems(documents))
    if not reports_dir.is_dir():
        raise ValidationError(f"report directory not found: {reports_dir}")

    expected = {document["filename"]: document for document in documents}
    found = {
        path.name: path
        for path in reports_dir.iterdir()
        if path.is_file() and path.suffix == ".pdf"
    }
    problems: list[str] = []
    missing = sorted(set(expected) - set(found))
    unexpected = sorted(set(found) - set(expected))
    if missing:
        problems.append("missing documents: " + ", ".join(missing))
    if unexpected:
        problems.append("unexpected documents: " + ", ".join(unexpected))
    for filename in sorted(set(expected) & set(found)):
        actual = _sha1(found[filename])
        recorded_sha1 = expected[filename]["sha1"]
        if actual != recorded_sha1:
            problems.append(
                f"SHA-1 mismatch for {filename}: manifest {recorded_sha1}, file {actual}"
            )
    _raise_if(problems)
    return manifest


def _page_count(report: Any, json_name: str, problems: list[str]) -> int | None:
    metadata = report.get("metadata") if isinstance(report, dict) else None
    if not isinstance(metadata, dict):
        problems.append(f"document_id mismatch for {json_name}: metadata is missing")
        return None
    document_id = json_name.removesuffix(".json")
    if metadata.get("document_id") != document_id:
        problems.append(
            f"document_id mismatch for {json_name}: "
            f"metadata has {metadata.get('document_id')!r}"
        )
        return None
    page_count = metadata.get("page_count")
    if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 0:
        problems.append(f"invalid page_count for {json_name}")
        return None
    return page_count


def validate_parsed_corpus(
    reports_dir: Path,
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Abort unless each source PDF has one matching JSON report.

    ``metadata.document_id`` must equal the source document id and the output
    filename. ``manifest.json`` is the publication file, not a parsed report.
    """
    reports_dir = Path(reports_dir)
    output_dir = Path(output_dir)
    documents = list(manifest["documents"])
    problems: list[str] = []
    expected_json: set[str] = set()
    verified: list[dict[str, Any]] = []

    for document in documents:
        document_id = document["document_id"]
        filename = document["filename"]
        source = reports_dir / filename
        if not source.is_file() or source.stem != document_id:
            problems.append(f"source document mismatch: {filename}")
        expected_json.add(f"{document_id}.json")

    if not output_dir.is_dir():
        raise ValidationError(f"parsed report directory not found: {output_dir}")

    actual_json = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix == ".json" and path.name != MANIFEST_FILENAME
    }
    missing_json = sorted(expected_json - actual_json)
    unexpected_json = sorted(actual_json - expected_json)
    if missing_json:
        problems.append("missing parsed json: " + ", ".join(missing_json))
    if unexpected_json:
        problems.append("unexpected json: " + ", ".join(unexpected_json))

    for json_name in sorted(expected_json & actual_json):
        json_path = output_dir / json_name
        try:
            report = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"cannot read {json_name}: {error}")
            continue
        page_count = _page_count(report, json_name, problems)
        if page_count is not None:
            verified.append(
                {
                    "document_id": json_name.removesuffix(".json"),
                    "page_count": page_count,
                }
            )

    _raise_if(problems)
    return verified
