import json
import sys

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from financial_rag.chunking import __main__ as cli
from financial_rag.chunking.chunker import ChunkingConfig, RecursiveTokenChunker


@pytest.fixture
def chunker(monkeypatch):
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    monkeypatch.setattr(
        "financial_rag.chunking.chunker.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    return RecursiveTokenChunker(ChunkingConfig(chunk_size=32, chunk_overlap=8))


def test_page_boundaries_overlap_and_unicode_offsets(chunker):
    text = "  " + " ".join(f"café{i}" for i in range(100)) + "  "
    report = {
        "metadata": {"document_id": "report"},
        "pages": [{"page": 3, "text": text}, {"page": 7, "text": "Second page."}],
    }
    chunks = chunker.chunk_report(report)
    first_page = [c for c in chunks if c["metadata"]["page"] == 3]
    assert len(first_page) > 1
    for index, chunk in enumerate(first_page):
        metadata = chunk["metadata"]
        assert metadata["chunk_index"] == index
        assert metadata["chunk_id"] == f"report:p3:c{index}"
        assert chunk["text"] == text[metadata["char_start"]:metadata["char_end"]]
        assert metadata["token_count"] == len(chunker._tokenizer.encode(
            chunk["text"], add_special_tokens=False
        ))
        assert 0 < metadata["token_count"] <= 32
    assert first_page[1]["metadata"]["char_start"] < first_page[0]["metadata"]["char_end"]
    assert chunks[-1]["metadata"]["chunk_id"] == "report:p7:c0"
    assert chunks[-1]["text"] == "Second page."


def test_repeated_chunks_advance_through_source(chunker):
    text = "Repeated financial disclosure. " * 100
    report = {"metadata": {"document_id": "report"}, "pages": [{"page": 1, "text": text}]}
    chunks = chunker.chunk_report(report)
    starts = [chunk["metadata"]["char_start"] for chunk in chunks]
    assert starts == sorted(set(starts))
    for chunk in chunks:
        metadata = chunk["metadata"]
        assert text[metadata["char_start"]:metadata["char_end"]] == chunk["text"]
    assert not text[chunks[-1]["metadata"]["char_end"]:].strip()


def test_zero_overlap_does_not_reuse_repeated_source_text(chunker):
    chunker = RecursiveTokenChunker(ChunkingConfig(chunk_size=32, chunk_overlap=0))
    text = "Repeated financial disclosure. " * 100
    chunks = chunker.chunk_report({
        "metadata": {"document_id": "report"},
        "pages": [{"page": 1, "text": text}],
    })
    previous_end = 0
    for chunk in chunks:
        metadata = chunk["metadata"]
        assert metadata["char_start"] >= previous_end
        assert not text[previous_end:metadata["char_start"]].strip()
        previous_end = metadata["char_end"]
    assert not text[previous_end:].strip()


def test_empty_pages_produce_no_chunks(chunker):
    assert chunker.chunk_report({"metadata": {"document_id": "empty"}, "pages": []}) == []
    assert chunker.chunk_report({
        "metadata": {"document_id": "empty"},
        "pages": [{"page": 1, "text": " \n\t"}],
    }) == []


def test_trailing_chunk_can_be_contained_in_previous_overlap(chunker, monkeypatch):
    text = "First disclosure.\n\nTotal revenue."
    monkeypatch.setattr(chunker._splitter, "split_text", lambda _: [text, "Total revenue."])
    chunks = chunker.chunk_report({
        "metadata": {"document_id": "report"},
        "pages": [{"page": 1, "text": text}],
    })
    assert chunks[1]["metadata"]["char_start"] == text.index("Total revenue.")
    assert chunks[0]["metadata"]["char_end"] == chunks[1]["metadata"]["char_end"]


def test_final_table_delimiter_uses_final_source_occurrence(chunker, monkeypatch):
    text = "| Revenue |\n\n|"
    monkeypatch.setattr(chunker._splitter, "split_text", lambda _: ["| Revenue |", "|"])
    chunks = chunker.chunk_report({
        "metadata": {"document_id": "report"},
        "pages": [{"page": 1, "text": text}],
    })
    assert chunks[-1]["metadata"]["char_start"] == len(text) - 1


def test_cli_writes_one_jsonl_per_report(chunker, monkeypatch, tmp_path):
    source = tmp_path / "rendered"
    output = tmp_path / "chunks"
    source.mkdir()
    for document_id, text in [("alpha", "Café revenue.\n\nSecond paragraph."), ("empty", "")]:
        (source / f"{document_id}.json").write_text(json.dumps({
            "metadata": {"document_id": document_id},
            "pages": [{"page": 4, "text": text}],
        }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "chunking", "--input-dir", str(source), "--output-dir", str(output),
    ])
    assert cli.main() == 0
    variant = output / "c384_o64"
    assert sorted(path.name for path in output.iterdir()) == ["c384_o64"]
    assert sorted(path.name for path in variant.iterdir()) == ["alpha.jsonl", "empty.jsonl"]
    raw = (variant / "alpha.jsonl").read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    assert records[0]["text"] == "Café revenue.\n\nSecond paragraph."
    assert set(records[0]) == {"metadata", "text"}
    assert (variant / "empty.jsonl").read_bytes() == b""

    monkeypatch.setattr(sys, "argv", [
        "chunking", "--input-dir", str(source), "--output-dir", str(output),
        "--chunk-size", "64", "--chunk-overlap", "8",
    ])
    assert cli.main() == 0
    assert (output / "c64_o8" / "alpha.jsonl").is_file()
    assert (variant / "alpha.jsonl").is_file()


def test_cli_missing_input_returns_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "chunking", "--input-dir", str(tmp_path / "missing"),
    ])
    assert cli.main() == 1
