import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from financial_rag.parsing.manifest import MANIFEST_FILENAME, write_manifest

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_MANIFEST = ROOT / "data" / "processed" / "parsed_reports" / "manifest.json"
PUBLISHED_MANIFEST_SHA256 = (
    "6b3a9f5998fb29502cbbf0502f7ab10ff09e577daa0fccd2111a42f90e1bcb71"
)
PARSER_CLASS_PATH = "financial_rag.parsing.parser.FinancialReportParser"


def _package_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]["version"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_write_manifest_records_summary_provenance_and_hashes(tmp_path: Path):
    output = tmp_path / "reports"
    output.mkdir()
    later = "b" * 40
    earlier = "a" * 40
    (output / f"{later}.json").write_bytes(b'{"metadata":{"page_count":4}}')
    (output / f"{earlier}.json").write_bytes(b'{"metadata":{"page_count":10}}')
    dataset_manifest = Path("data/raw/dataset_manifest.json")
    fingerprint = "d8f0c9c6053849fc1406e9fae3538bc8aa1eeefad69f2838f4878e14da4cee2a"

    destination = write_manifest(
        output,
        [
            {"document_id": later, "page_count": 4},
            {"document_id": earlier, "page_count": 10},
        ],
        dataset_manifest=dataset_manifest,
        corpus_fingerprint=fingerprint,
    )

    published = json.loads(destination.read_text(encoding="utf-8"))
    raw = destination.read_bytes()
    assert destination.name == MANIFEST_FILENAME
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    assert not (output / f".{MANIFEST_FILENAME}.tmp").exists()
    assert published["corpus"] == {"document_count": 2, "page_count": 14}
    assert published["source"] == {
        "manifest_path": "data/raw/dataset_manifest.json",
        "corpus_fingerprint": fingerprint,
    }
    assert published["parser"] == {
        "class_path": PARSER_CLASS_PATH,
        "package_version": _package_version(),
    }
    assert [item["document_id"] for item in published["artifacts"]] == [
        earlier,
        later,
    ]
    for item in published["artifacts"]:
        assert item["filename"] == f"{item['document_id']}.json"
        assert item["sha256"] == _sha256(output / item["filename"])


@pytest.mark.skipif(
    not PUBLISHED_MANIFEST.is_file(),
    reason="parsed corpus manifest is not published",
)
def test_published_corpus_manifest_records_validated_inventory():
    raw = PUBLISHED_MANIFEST.read_bytes()
    published = json.loads(raw)
    source = json.loads(
        (ROOT / "data" / "raw" / "dataset_manifest.json").read_text(encoding="utf-8")
    )

    assert hashlib.sha256(raw).hexdigest() == PUBLISHED_MANIFEST_SHA256
    assert published["corpus"] == {"document_count": 100, "page_count": 14454}
    assert published["source"] == {
        "manifest_path": "data/raw/dataset_manifest.json",
        "corpus_fingerprint": source["corpus_fingerprint"],
    }
    assert published["parser"] == {
        "class_path": PARSER_CLASS_PATH,
        "package_version": _package_version(),
    }
    assert [item["document_id"] for item in published["artifacts"]] == [
        document["document_id"] for document in source["documents"]
    ]


def test_write_manifest_leaves_no_publication_when_it_cannot_finish(tmp_path: Path):
    output = tmp_path / "reports"
    output.mkdir()
    (output / MANIFEST_FILENAME).write_text("previous\n", encoding="utf-8")

    with pytest.raises(ValueError, match="page_count"):
        write_manifest(
            output,
            [{"document_id": "a", "page_count": -1}],
            dataset_manifest=tmp_path / "dataset_manifest.json",
            corpus_fingerprint="ab" * 32,
        )
    assert (output / MANIFEST_FILENAME).read_text(encoding="utf-8") == "previous\n"

    with pytest.raises(OSError):
        write_manifest(
            output,
            [{"document_id": "a", "page_count": 1}],
            dataset_manifest=tmp_path / "dataset_manifest.json",
            corpus_fingerprint="ab" * 32,
        )
    assert (output / MANIFEST_FILENAME).read_text(encoding="utf-8") == "previous\n"
    assert not (output / f".{MANIFEST_FILENAME}.tmp").exists()
