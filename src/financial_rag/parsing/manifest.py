"""Publication manifest for a validated parsed corpus."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "manifest.json"
PARSER_CLASS_PATH = "financial_rag.parsing.parser.FinancialReportParser"
_PROJECT_FILE = Path(__file__).resolve().parents[3] / "pyproject.toml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    with _PROJECT_FILE.open("rb") as file:
        return tomllib.load(file)["project"]["version"]


def _provenance_path(path: Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def write_manifest(
    output_dir: Path,
    documents: Sequence[Mapping[str, Any]],
    *,
    dataset_manifest: Path,
    corpus_fingerprint: str,
) -> Path:
    """Write the publication manifest beside a validated set of parsed reports.

    Call this only after ``validate_parsed_corpus`` succeeds. The file's
    presence is the record that the parsed corpus passed that check.
    """
    ordered = sorted(documents, key=lambda document: document["document_id"])
    page_count = 0
    integrity: list[dict[str, Any]] = []
    for document in ordered:
        document_id = document["document_id"]
        pages = document["page_count"]
        if isinstance(pages, bool) or not isinstance(pages, int) or pages < 0:
            raise ValueError(f"invalid page_count for {document_id}")
        page_count += pages
        filename = f"{document_id}.json"
        integrity.append(
            {
                "document_id": document_id,
                "filename": filename,
                "sha256": _sha256(output_dir / filename),
            }
        )

    payload = {
        "corpus": {
            "document_count": len(integrity),
            "page_count": page_count,
        },
        "source": {
            "manifest_path": _provenance_path(dataset_manifest),
            "corpus_fingerprint": corpus_fingerprint,
        },
        "parser": {
            "class_path": PARSER_CLASS_PATH,
            "package_version": _package_version(),
        },
        "artifacts": integrity,
    }
    destination = output_dir / MANIFEST_FILENAME
    temporary = output_dir / f".{MANIFEST_FILENAME}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)
    return destination
