import json
from pathlib import Path

import pytest

from financial_rag.rendering.renderer import (
    RenderingError,
    render_directory,
    render_report,
)

_TABLE = "| Metric | 2022 | 2021 |\n|---|---|---|\n| Revenue | 120.5 | 98.1 |"


def _report(pages: list[dict], tables: list[dict] | None = None, **metadata) -> dict:
    report_metadata = {"document_id": "doc", "page_count": 1}
    report_metadata.update(metadata)
    return {
        "metadata": report_metadata,
        "content": pages,
        "tables": tables or [],
    }


def test_render_report_builds_page_markdown():
    report = {
        "metadata": {
            "document_id": "0f111d244aee3d976684995a222fa177a64571c4",
            "page_count": 9,
            "company_name": "NextNav Inc.",
            "table_count": 1,
        },
        "content": [
            {
                "page": 8,
                "content": [{"type": "text", "text": "Notes"}],
            },
            {
                "page": 2,
                "content": [
                    {"type": "picture", "picture_id": 0},
                    {"type": "page_footer", "text": "2"},
                    {"type": "section_header", "text": "Annual Report 2022"},
                    {"type": "section_header", "text": "Directors' Report"},
                    {"type": "text", "text": "The directors present..."},
                    {"type": "text", "text": "Key figures:"},
                    {"type": "table", "table_id": 0},
                    {
                        "type": "list_item",
                        "text": "First bullet",
                        "marker": "·",
                        "enumerated": False,
                    },
                    {
                        "type": "list_item",
                        "text": "Second bullet",
                        "marker": "-",
                        "enumerated": False,
                    },
                ],
            },
            {"page": 5, "content": [{"type": "page_footer", "text": "5"}]},
            {"page": 1, "content": [{"type": "picture", "picture_id": 1}]},
            {"page": 3, "content": []},
        ],
        "tables": [{"table_id": 0, "markdown": _TABLE}],
    }

    rendered = render_report(report)

    assert list(rendered) == ["metadata", "pages"]
    assert list(rendered["metadata"]) == ["document_id", "page_count"]
    assert rendered == {
        "metadata": {
            "document_id": "0f111d244aee3d976684995a222fa177a64571c4",
            "page_count": 9,
        },
        "pages": [
            {
                "page": 2,
                "text": (
                    "# Annual Report 2022\n\n"
                    "## Directors' Report\n\n"
                    "The directors present...\n\n"
                    "Key figures:\n\n"
                    f"{_TABLE}\n\n"
                    "- First bullet\n"
                    "- Second bullet"
                ),
            },
            {"page": 8, "text": "Notes"},
        ],
    }


def test_render_report_formats_headers_lists_and_tables():
    report = _report(
        [
            {
                "page": 4,
                "content": [
                    {"type": "picture", "picture_id": 0},
                    {"type": "page_header", "text": "Table of Contents"},
                    {"type": "section_header", "text": "Item 1"},
                    {"type": "text", "text": "Financing:"},
                    {
                        "type": "list_item",
                        "text": "Secured facility",
                        "marker": "1.",
                        "enumerated": True,
                    },
                    {
                        "type": "list_item",
                        "text": "Warehouse facilities",
                        "marker": "2.",
                        "enumerated": True,
                    },
                    {"type": "table", "table_id": 1},
                    {"type": "footnote", "text": "See note A."},
                    {"type": "checkbox_selected", "text": "☑ Annual report"},
                    {"type": "text", "text": "Before"},
                    {"type": "formula", "text": "  "},
                    {"type": "text", "text": "After"},
                    {
                        "type": "list_item",
                        "text": "One",
                        "marker": "·",
                        "enumerated": False,
                    },
                    {"type": "page_footer", "text": "4"},
                    {
                        "type": "list_item",
                        "text": "Two",
                        "marker": "·",
                        "enumerated": False,
                    },
                ],
            }
        ],
        tables=[{"table_id": 1, "markdown": "| A |\n|---|\n| 1 |"}],
        page_count=10,
    )

    rendered = render_report(report)

    assert rendered["metadata"]["page_count"] == 10
    assert rendered["pages"] == [
        {
            "page": 4,
            "text": (
                "# Table of Contents\n\n"
                "## Item 1\n\n"
                "Financing:\n\n"
                "1. Secured facility\n"
                "2. Warehouse facilities\n\n"
                "| A |\n|---|\n| 1 |\n\n"
                "See note A.\n\n"
                "☑ Annual report\n\n"
                "Before\n\n"
                "After\n\n"
                "- One\n"
                "- Two"
            ),
        }
    ]


def test_headers_after_the_opening_blocks_stay_second_level():
    report = _report(
        [
            {
                "page": 1,
                "content": [
                    {"type": "text", "text": "One"},
                    {"type": "text", "text": "Two"},
                    {"type": "text", "text": "Three"},
                    {"type": "page_header", "text": "Summary"},
                    {"type": "section_header", "text": "Notes"},
                ],
            }
        ]
    )

    assert render_report(report)["pages"][0]["text"] == (
        "One\n\nTwo\n\nThree\n\n## Summary\n\n## Notes"
    )


def test_render_directory_names_files_by_document_id(tmp_path: Path):
    source = tmp_path / "parsed_reports"
    source.mkdir()
    report = _report(
        [{"page": 1, "content": [{"type": "text", "text": "Hello"}]}],
        document_id="abc123",
        company_name="Acme",
    )
    (source / "not-the-id.json").write_text(json.dumps(report), encoding="utf-8")
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "rendered_reports"

    assert render_directory(source, output) == 1

    written = output / "abc123.json"
    raw = written.read_bytes()
    assert written.is_file()
    assert not (output / "manifest.json").exists()
    assert not (output / "not-the-id.json").exists()
    assert b"\r\n" not in raw
    assert json.loads(raw) == render_report(report)


def test_renderer_rejects_incomplete_input(tmp_path: Path):
    with pytest.raises(RenderingError, match="unknown block type"):
        render_report(
            _report([{"page": 1, "content": [{"type": "chart", "text": "x"}]}])
        )
    with pytest.raises(RenderingError, match="table 7"):
        render_report(
            _report([{"page": 1, "content": [{"type": "table", "table_id": 7}]}])
        )
    with pytest.raises(RenderingError, match="page_count"):
        render_report(
            {"metadata": {"document_id": "abc", "page_count": True}, "content": []}
        )
    with pytest.raises(RenderingError, match="parsed report directory not found"):
        render_directory(tmp_path / "missing", tmp_path / "out")

    source = tmp_path / "empty"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RenderingError, match="no parsed reports"):
        render_directory(source, tmp_path / "out")
