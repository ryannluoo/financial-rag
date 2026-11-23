"""Render parsed financial reports as markdown page text."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

DEFAULT_INPUT_DIR = Path("data/processed/parsed_reports")
DEFAULT_OUTPUT_DIR = Path("data/processed/rendered_reports")
_PUBLICATION_MANIFEST = "manifest.json"
_SKIPPED_TYPES = frozenset({"page_footer", "picture"})
_HEADER_TYPES = frozenset({"title", "section_header", "page_header"})
_PLAIN_TYPES = frozenset(
    {
        "text",
        "caption",
        "footnote",
        "formula",
        "code",
        "checkbox_selected",
        "checkbox_unselected",
    }
)


class RenderingError(ValueError):
    """Raised when a parsed report cannot be rendered."""


class _Block(NamedTuple):
    type: str
    text: str


def render_report(report: dict) -> dict:
    """Render one parsed report into markdown pages.

    Blocks on a page are separated by a blank line. Table rows and list items
    stay on single lines. Footers and pictures are dropped. A page left with
    no text is omitted. ``metadata.page_count`` is copied unchanged, so it can
    exceed the number of rendered pages.
    """
    if not isinstance(report, dict):
        raise RenderingError("report must be an object")
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise RenderingError("report metadata is missing")
    document_id = metadata.get("document_id")
    page_count = metadata.get("page_count")
    if not isinstance(document_id, str) or document_id == "":
        raise RenderingError("report metadata is missing document_id")
    if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 0:
        raise RenderingError("report metadata has an invalid page_count")

    content = report.get("content", [])
    if not isinstance(content, list):
        raise RenderingError("report content must be a list")

    tables = _tables_by_id(report)
    pages: list[dict[str, Any]] = []
    for page in content:
        if not isinstance(page, dict):
            raise RenderingError("page entry must be an object")
        number = _page_number(page)
        text = _render_page(number, page.get("content", []), tables)
        if text:
            pages.append({"page": number, "text": text})
    pages.sort(key=lambda item: item["page"])
    return {
        "metadata": {
            "document_id": document_id,
            "page_count": page_count,
        },
        "pages": pages,
    }


def render_directory(input_dir: Path, output_dir: Path) -> int:
    """Render every parsed report in ``input_dir`` except the publication manifest."""
    source = Path(input_dir)
    if not source.is_dir():
        raise RenderingError(f"parsed report directory not found: {source}")
    paths = sorted(
        path
        for path in source.glob("*.json")
        if path.name != _PUBLICATION_MANIFEST
    )
    if not paths:
        raise RenderingError(f"no parsed reports in {source}")
    return render_paths(paths, output_dir)


def render_paths(paths: Sequence[Path], output_dir: Path) -> int:
    """Render specific parsed-report JSON files into ``output_dir``."""
    files = [Path(path) for path in paths]
    if not files:
        raise RenderingError("no parsed reports found")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for path in files:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            _write_report(report, destination)
        except (RenderingError, json.JSONDecodeError) as error:
            raise RenderingError(f"{path.name}: {error}") from error
    return len(files)


def _write_report(report: dict, output_dir: Path) -> None:
    rendered = render_report(report)
    document_id = rendered["metadata"]["document_id"]
    if Path(document_id).name != document_id:
        raise RenderingError(f"invalid document_id: {document_id}")
    (output_dir / f"{document_id}.json").write_text(
        json.dumps(rendered, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )


def _tables_by_id(report: dict) -> dict[int, dict[str, Any]]:
    tables = report.get("tables", [])
    if not isinstance(tables, list):
        raise RenderingError("report tables must be a list")
    indexed: dict[int, dict[str, Any]] = {}
    for table in tables:
        if not isinstance(table, dict):
            continue
        table_id = table.get("table_id")
        if isinstance(table_id, int) and not isinstance(table_id, bool):
            indexed[table_id] = table
    return indexed


def _page_number(page: dict) -> int:
    number = page.get("page")
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise RenderingError(f"invalid page number: {number!r}")
    return number


def _render_page(
    page_number: int,
    blocks: Any,
    tables: dict[int, dict[str, Any]],
) -> str:
    if not isinstance(blocks, list):
        raise RenderingError(f"page {page_number} content must be a list")
    visible = _visible_blocks(page_number, blocks, tables)
    parts: list[str] = []
    index = 0
    while index < len(visible):
        block = visible[index]
        if block.type == "list_item":
            lines: list[str] = []
            while index < len(visible) and visible[index].type == "list_item":
                lines.append(visible[index].text)
                index += 1
            parts.append("\n".join(lines))
            continue
        text = block.text
        if block.type in _HEADER_TYPES:
            text = f"{_heading_prefix(visible, index)}{text}"
        parts.append(text)
        index += 1
    return "\n\n".join(parts)


def _visible_blocks(
    page_number: int,
    blocks: list[Any],
    tables: dict[int, dict[str, Any]],
) -> list[_Block]:
    visible: list[_Block] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise RenderingError(f"page {page_number} has a block that is not an object")
        block_type = block.get("type")
        if block_type in _SKIPPED_TYPES:
            continue
        if block_type == "list_item":
            line = _list_line(block)
            if line:
                visible.append(_Block("list_item", line))
            continue
        if block_type == "table":
            text = _table_text(page_number, block, tables)
        elif block_type in _HEADER_TYPES or block_type in _PLAIN_TYPES:
            text = _block_text(block)
        else:
            raise RenderingError(
                f"unknown block type on page {page_number}: {block_type}"
            )
        if text:
            visible.append(_Block(block_type, text))
    return visible


def _heading_prefix(blocks: Sequence[_Block], index: int) -> str:
    """Promote a heading that opens the page; later headings stay second level."""
    block_type = blocks[index].type
    if block_type == "title":
        return "# "
    page_header_in_first_3 = any(block.type == "page_header" for block in blocks[:3])
    if block_type == "page_header":
        return "# " if index < 3 else "## "
    opens_page = (
        index < 3
        and not page_header_in_first_3
        and not any(block.type == "section_header" for block in blocks[:index])
    )
    if opens_page:
        return "# "
    return "## "


def _list_line(block: dict[str, Any]) -> str:
    text = _block_text(block)
    if not text:
        return ""
    raw_marker = block.get("marker") or ""
    marker = raw_marker.strip() if isinstance(raw_marker, str) else ""
    if marker and text.startswith(marker):
        return text
    if block.get("enumerated") and marker:
        return f"{marker} {text}"
    return f"- {text}"


def _table_text(
    page_number: int,
    block: dict[str, Any],
    tables: dict[int, dict[str, Any]],
) -> str:
    table_id = block.get("table_id")
    if isinstance(table_id, bool) or not isinstance(table_id, int):
        raise RenderingError(f"table on page {page_number} has an invalid table_id")
    try:
        table = tables[table_id]
    except KeyError:
        raise RenderingError(
            f"table {table_id} on page {page_number} was not found"
        ) from None
    markdown = table.get("markdown", "")
    if not isinstance(markdown, str):
        raise RenderingError(f"table {table_id} markdown must be a string")
    return markdown.strip()


def _block_text(block: dict[str, Any]) -> str:
    text = block.get("text", "")
    if text is None:
        return ""
    if not isinstance(text, str):
        raise RenderingError("block text must be a string")
    return text.strip()
