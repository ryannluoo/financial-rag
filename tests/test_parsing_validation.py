import hashlib
import json
from pathlib import Path

import pytest

from financial_rag.parsing.__main__ import main
from financial_rag.parsing.manifest import MANIFEST_FILENAME, write_manifest
from financial_rag.parsing.validation import (
    ValidationError,
    corpus_fingerprint,
    validate_parsed_corpus,
    validate_source_corpus,
)

ROOT = Path(__file__).resolve().parents[1]
DOC_ID = "11f6ad8ec52a2984abaafd7c3b516503785c2072"
FINGERPRINT = "58e8c8d5df6d57b01c599dd91ca8b3f2679f41b33b2c1a8d79cbd7c28b6d0225"
COMMITTED_FINGERPRINT = (
    "d8f0c9c6053849fc1406e9fae3538bc8aa1eeefad69f2838f4878e14da4cee2a"
)


def _document(
    content: bytes,
    *,
    document_id: str | None = None,
    sha1: str | None = None,
    filename: str | None = None,
) -> dict:
    digest = hashlib.sha1(content).hexdigest()
    resolved_id = digest if document_id is None else document_id
    return {
        "document_id": resolved_id,
        "filename": f"{resolved_id}.pdf" if filename is None else filename,
        "sha1": digest if sha1 is None else sha1,
        "size_bytes": len(content),
    }


def _write_dataset(
    directory: Path,
    documents: list[dict],
    contents: dict[str, bytes],
    *,
    fingerprint: str | None = None,
) -> tuple[Path, Path, dict]:
    reports = directory / "reports"
    reports.mkdir(parents=True)
    for filename, content in contents.items():
        (reports / filename).write_bytes(content)
    manifest = {
        "dataset_id": "erc2",
        "document_count": len(documents),
        "documents": documents,
    }
    manifest["corpus_fingerprint"] = (
        corpus_fingerprint(manifest) if fingerprint is None else fingerprint
    )
    manifest_path = directory / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return reports, manifest_path, manifest


def _write_report(output_dir: Path, document_id: str, page_count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"metadata": {"document_id": document_id, "page_count": page_count}}
    (output_dir / f"{document_id}.json").write_text(
        json.dumps(report), encoding="utf-8"
    )


def test_corpus_fingerprint_matches_canonical_digest():
    document = {
        "size_bytes": 1,
        "sha1": DOC_ID,
        "filename": f"{DOC_ID}.pdf",
        "document_id": DOC_ID,
    }
    manifest = {"document_count": 1, "dataset_id": "erc2", "documents": [document]}

    assert corpus_fingerprint(manifest) == FINGERPRINT
    assert corpus_fingerprint({**manifest, "corpus_fingerprint": "ignored"}) == FINGERPRINT

    committed = json.loads(
        (ROOT / "data" / "raw" / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert corpus_fingerprint(committed) == COMMITTED_FINGERPRINT
    assert committed["corpus_fingerprint"] == COMMITTED_FINGERPRINT


def test_source_corpus_accepts_exact_inventory(tmp_path: Path):
    content = b"x"
    document = _document(content)
    reports, manifest_path, _ = _write_dataset(
        tmp_path, [document], {document["filename"]: content}
    )
    (reports / "notes.txt").write_text("ignore", encoding="utf-8")

    manifest = validate_source_corpus(reports, manifest_path)

    assert manifest["corpus_fingerprint"] == FINGERPRINT
    assert manifest["documents"] == [document]


def test_source_corpus_rejects_fingerprint_membership_and_hash_mismatches(tmp_path: Path):
    content = b"x"
    document = _document(content)
    reports, manifest_path, _ = _write_dataset(
        tmp_path,
        [document],
        {document["filename"]: b"tampered"},
        fingerprint="0" * 64,
    )
    (reports / "extra.pdf").write_bytes(b"extra")

    with pytest.raises(ValidationError, match="corpus fingerprint mismatch"):
        validate_source_corpus(reports, manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corpus_fingerprint"] = corpus_fingerprint(
        {key: value for key, value in manifest.items() if key != "corpus_fingerprint"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValidationError, match="missing documents") as missing:
        reports.joinpath(document["filename"]).unlink()
        validate_source_corpus(reports, manifest_path)
    assert "unexpected documents: extra.pdf" in str(missing.value)

    (reports / document["filename"]).write_bytes(b"tampered")
    with pytest.raises(ValidationError, match="SHA-1 mismatch"):
        validate_source_corpus(reports, manifest_path)


def test_source_corpus_rejects_document_id_disagreements(tmp_path: Path):
    content = b"x"
    renamed = _document(content, filename="other.pdf")
    reports, manifest_path, _ = _write_dataset(
        tmp_path, [renamed], {renamed["filename"]: content}
    )

    with pytest.raises(ValidationError, match="filename does not match document_id"):
        validate_source_corpus(reports, manifest_path)

    digest = hashlib.sha1(content).hexdigest()
    mismatched = _document(content, sha1="0" * 40)
    mismatched["document_id"] = digest
    mismatched["filename"] = f"{digest}.pdf"
    reports, manifest_path, _ = _write_dataset(
        tmp_path / "sha", [mismatched], {mismatched["filename"]: content}
    )
    with pytest.raises(ValidationError, match="document_id does not match sha1"):
        validate_source_corpus(reports, manifest_path)


def test_published_parsed_corpus_matches_dataset_inventory():
    published_manifest = ROOT / "data" / "processed" / "parsed_reports" / "manifest.json"
    if not published_manifest.is_file():
        pytest.skip("parsed corpus manifest is not published")
    source = json.loads(
        (ROOT / "data" / "raw" / "dataset_manifest.json").read_text(encoding="utf-8")
    )

    verified = validate_parsed_corpus(
        ROOT / "data" / "raw" / "reports",
        published_manifest.parent,
        source,
    )

    assert [item["document_id"] for item in verified] == [
        document["document_id"] for document in source["documents"]
    ]
    assert sum(item["page_count"] for item in verified) == 14454


def test_parsed_corpus_requires_one_report_and_matching_document_id(tmp_path: Path):
    content = b"x"
    document = _document(content)
    reports, _, manifest = _write_dataset(
        tmp_path, [document], {document["filename"]: content}
    )
    output = tmp_path / "parsed"
    _write_report(output, document["document_id"], 4)
    (output / "notes.md").write_text("ignore", encoding="utf-8")
    (output / MANIFEST_FILENAME).write_text("{}\n", encoding="utf-8")

    verified = validate_parsed_corpus(reports, output, manifest)

    assert verified == [{"document_id": document["document_id"], "page_count": 4}]
    write_manifest(
        output,
        verified,
        dataset_manifest=tmp_path / "dataset_manifest.json",
        corpus_fingerprint=manifest["corpus_fingerprint"],
    )
    assert validate_parsed_corpus(reports, output, manifest) == verified

    (output / "extra.json").write_text("{}", encoding="utf-8")
    (output / f"{document['document_id']}.json").unlink()
    with pytest.raises(ValidationError, match="missing parsed json") as error:
        validate_parsed_corpus(reports, output, manifest)
    assert "unexpected json: extra.json" in str(error.value)

    _write_report(output, document["document_id"], 4)
    (output / "extra.json").unlink()
    report = {"metadata": {"document_id": "other", "page_count": 4}}
    (output / f"{document['document_id']}.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    with pytest.raises(ValidationError, match="document_id mismatch"):
        validate_parsed_corpus(reports, output, manifest)


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str], parser) -> int:
    monkeypatch.setattr(
        "financial_rag.parsing.__main__.FinancialReportParser", parser
    )
    monkeypatch.setattr("sys.argv", argv)
    return main()


def test_directory_parse_publishes_manifest_only_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    content = b"x"
    document = _document(content)
    reports, manifest_path, _ = _write_dataset(
        tmp_path, [document], {document["filename"]: content}
    )
    output = tmp_path / "parsed"
    output.mkdir()
    stale = output / MANIFEST_FILENAME
    stale.write_text("stale\n", encoding="utf-8")

    class _Parser:
        constructed = False

        def __init__(self, output_dir, settings):
            _Parser.constructed = True
            self.output_dir = Path(output_dir)

        def parse(self, paths):
            for path in paths:
                _write_report(self.output_dir, Path(path).stem, 6)
            return (len(list(paths)), 0)

    assert _run_main(
        monkeypatch,
        [
            "financial_rag.parsing",
            "--input-dir",
            str(reports),
            "--output-dir",
            str(output),
            "--dataset-manifest",
            str(manifest_path),
        ],
        _Parser,
    ) == 0
    published = json.loads((output / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert published["corpus"] == {"document_count": 1, "page_count": 6}
    assert published["source"]["corpus_fingerprint"] == FINGERPRINT
    assert published["parser"]["package_version"] == "0.1.0"
    assert published["artifacts"][0]["document_id"] == document["document_id"]

    class _BadParser:
        def __init__(self, output_dir, settings):
            self.output_dir = Path(output_dir)

        def parse(self, paths):
            for path in paths:
                document_id = Path(path).stem
                _write_report(self.output_dir, document_id, 6)
                report = {"metadata": {"document_id": "other", "page_count": 6}}
                (self.output_dir / f"{document_id}.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
            return (len(list(paths)), 0)

    assert _run_main(
        monkeypatch,
        [
            "financial_rag.parsing",
            "--input-dir",
            str(reports),
            "--output-dir",
            str(output),
            "--dataset-manifest",
            str(manifest_path),
        ],
        _BadParser,
    ) == 1
    assert not (output / MANIFEST_FILENAME).exists()


def test_failed_source_validation_does_not_start_parser_or_replace_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    content = b"x"
    document = _document(content)
    reports, manifest_path, _ = _write_dataset(
        tmp_path,
        [document],
        {document["filename"]: content},
        fingerprint="0" * 64,
    )
    output = tmp_path / "parsed"
    output.mkdir()
    (output / MANIFEST_FILENAME).write_text("keep\n", encoding="utf-8")

    class _Parser:
        def __init__(self, output_dir, settings):
            raise AssertionError("parser started after source validation failed")

    assert _run_main(
        monkeypatch,
        [
            "financial_rag.parsing",
            "--input-dir",
            str(reports),
            "--output-dir",
            str(output),
            "--dataset-manifest",
            str(manifest_path),
        ],
        _Parser,
    ) == 1
    assert (output / MANIFEST_FILENAME).read_text(encoding="utf-8") == "keep\n"


def test_explicit_pdfs_skip_corpus_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "parsed"
    output.mkdir()
    (output / MANIFEST_FILENAME).write_text("keep\n", encoding="utf-8")
    seen: list[Path] = []

    class _Parser:
        def __init__(self, output_dir, settings):
            self.output_dir = output_dir

        def parse(self, paths):
            seen.extend(paths)
            return (len(list(paths)), 0)

    pdf = tmp_path / "one.pdf"
    assert _run_main(
        monkeypatch,
        ["financial_rag.parsing", str(pdf), "--output-dir", str(output)],
        _Parser,
    ) == 0
    assert seen == [pdf]
    assert (output / MANIFEST_FILENAME).read_text(encoding="utf-8") == "keep\n"
