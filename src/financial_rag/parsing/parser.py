"""Local financial-report PDF parser and report exporter."""

from __future__ import annotations

import html
import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMode,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from tabulate import tabulate

LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_DIR = Path("data/raw/reports")
DEFAULT_OUTPUT_DIR = Path("data/processed/parsed_reports")
DEFAULT_ARTIFACTS_DIR = Path("data/models/docling")
DEFAULT_ACCELERATOR_DEVICE = AcceleratorDevice.CUDA
GLYPH_IDENTIFIER = re.compile(r"/gid\d+")
MIN_GLYPH_IDENTIFIERS = 3
MAX_GLYPH_IDENTIFIER_RATIO = 0.01
WIDE_ITEM_RATIO = 0.6
LEFT_EDGE_CLUSTER_RATIO = 0.08
YEAR_LABEL = re.compile(r"^(?:19|20)\d{2}$")
CHART_VALUE = re.compile(
    r"(?:[$€£](?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|(?:\d+(?:\.\d+)?)%)"
)


class ParsingError(RuntimeError):
    """Raised when a PDF cannot produce a valid parser report."""


@dataclass(frozen=True)
class ParserSettings:
    """Stable local configuration for the baseline parser."""

    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    threads: int = 4
    device: AcceleratorDevice = DEFAULT_ACCELERATOR_DEVICE

    def __post_init__(self) -> None:
        if self.threads < 1:
            raise ValueError("threads must be positive")


def build_pipeline_options(settings: ParserSettings) -> PdfPipelineOptions:
    """Build the configured Docling pipeline without initializing its models."""
    options = PdfPipelineOptions(artifacts_path=settings.artifacts_dir)
    options.do_ocr = True
    options.ocr_options = EasyOcrOptions(
        lang=["en"],
        mode=OcrMode.DEFAULT,
        model_storage_directory=str(settings.artifacts_dir / "EasyOcr"),
        download_enabled=False,
    )
    options.do_table_structure = True
    options.table_structure_options.do_cell_matching = True
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    options.do_chart_extraction = False
    options.ocr_batch_size = 1
    options.layout_batch_size = 1
    options.accelerator_options.num_threads = settings.threads
    options.accelerator_options.device = settings.device
    return options


def build_converter(settings: ParserSettings) -> DocumentConverter:
    """Create the single PDF converter used by the project."""
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=build_pipeline_options(settings),
                backend=DoclingParseDocumentBackend,
            )
        }
    )


def _text_item(reference_index: int, data: dict[str, Any]) -> dict[str, Any]:
    source = data["texts"][reference_index]
    item: dict[str, Any] = {
        "text": source.get("text", ""),
        "type": source["label"],
        "text_id": reference_index,
    }
    original = source.get("orig", "")
    if original != source.get("text", ""):
        item["orig"] = original
    for field in ("enumerated", "marker"):
        if field in source:
            item[field] = source[field]
    return item


def _expand_groups(
    body_children: Sequence[dict[str, Any]], groups: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for item in body_children:
        reference = item.get("$ref")
        if not reference:
            expanded.append(item)
            continue
        reference_type, reference_index = reference.split("/")[-2:]
        if reference_type != "groups":
            expanded.append(item)
            continue
        group = groups[int(reference_index)]
        for child in group["children"]:
            expanded.append(
                {
                    **child,
                    "group_id": int(reference_index),
                    "group_name": group.get("name", ""),
                    "group_label": group.get("label", ""),
                }
            )
    return expanded


def _assemble_content(data: dict[str, Any]) -> list[dict[str, Any]]:
    pages: dict[int, dict[str, Any]] = {}
    children = _expand_groups(data["body"]["children"], data.get("groups", []))
    for item in children:
        reference = item.get("$ref")
        if not reference:
            continue
        reference_type, raw_index = reference.split("/")[-2:]
        reference_index = int(raw_index)
        if reference_type == "texts":
            source = data["texts"][reference_index]
            content_item = _text_item(reference_index, data)
            for field in ("group_id", "group_name", "group_label"):
                if field in item:
                    content_item[field] = item[field]
        elif reference_type == "tables":
            source = data["tables"][reference_index]
            content_item = {"type": "table", "table_id": reference_index}
        elif reference_type == "pictures":
            source = data["pictures"][reference_index]
            content_item = {"type": "picture", "picture_id": reference_index}
        else:
            continue
        provenance = source.get("prov", [])
        if not provenance:
            continue
        page_number = int(provenance[0]["page_no"])
        page = pages.setdefault(
            page_number,
            {
                "page": page_number,
                "content": [],
                "page_dimensions": provenance[0].get("bbox", {}),
            },
        )
        page["content"].append(content_item)
    return [pages[number] for number in sorted(pages)]


def _table_markdown(table: dict[str, Any]) -> str:
    rows = [[cell["text"] for cell in row] for row in table["data"]["grid"]]
    if len(rows) > 1 and rows[0]:
        try:
            return tabulate(rows[1:], headers=rows[0], tablefmt="github")
        except ValueError:
            return tabulate(
                rows[1:], headers=rows[0], tablefmt="github", disable_numparse=True
            )
    return tabulate(rows, tablefmt="github")


def _assemble_tables(document: Any, data: dict[str, Any]) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for table, source in zip(document.tables, data.get("tables", []), strict=True):
        table_json = table.model_dump(mode="json")
        provenance = source["prov"][0]
        box = provenance["bbox"]
        reference_index = int(source["self_ref"].split("/")[-1])
        tables.append(
            {
                "table_id": reference_index,
                "page": provenance["page_no"],
                "bbox": [box["l"], box["t"], box["r"], box["b"]],
                "row_count": source["data"]["num_rows"],
                "column_count": source["data"]["num_cols"],
                "markdown": _table_markdown(table_json),
                "html": table.export_to_html(doc=document, add_caption=False),
                "json": table_json,
            }
        )
    return tables


def _assemble_pictures(data: dict[str, Any]) -> list[dict[str, Any]]:
    pictures: list[dict[str, Any]] = []
    for source in data.get("pictures", []):
        children: list[dict[str, Any]] = []
        for child in source.get("children", []):
            reference = child.get("$ref")
            if reference and reference.split("/")[-2] == "texts":
                children.append(_text_item(int(reference.split("/")[-1]), data))
        reference_index = int(source["self_ref"].split("/")[-1])
        provenance = source["prov"][0]
        box = provenance["bbox"]
        pictures.append(
            {
                "picture_id": reference_index,
                "page": provenance["page_no"],
                "bbox": [box["l"], box["t"], box["r"], box["b"]],
                "children": children,
            }
        )
    return pictures


def _release_accelerator_memory() -> None:
    """Free cached CUDA blocks between documents on the 8GB parsing GPU."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def assemble_report(
    document: Any, data: dict[str, Any], document_id: str
) -> dict[str, Any]:
    """Create the project report in the baseline output contract."""
    report = {
        "metadata": {
            "document_id": document_id,
            "page_count": len(data.get("pages", [])),
            "text_block_count": len(data.get("texts", [])),
            "table_count": len(data.get("tables", [])),
            "picture_count": len(data.get("pictures", [])),
            "equation_count": len(data.get("equations", [])),
            "footnote_count": sum(
                item.get("label") == "footnote" for item in data.get("texts", [])
            ),
        },
        "content": _assemble_content(data),
        "tables": _assemble_tables(document, data),
        "pictures": _assemble_pictures(data),
    }
    chart_pages = {
        int(picture["page"])
        for picture in report["pictures"]
        if _chart_series(picture) is not None
    }
    _order_columns(report, data, chart_pages)
    _promote_charts(report)
    _validate_text_quality(report)
    return report


def _item_source(item: dict[str, Any], data: dict[str, Any]) -> dict[str, Any] | None:
    if item.get("type") == "table":
        collection, key = "tables", "table_id"
    elif item.get("type") == "picture":
        collection, key = "pictures", "picture_id"
    else:
        collection, key = "texts", "text_id"
    index = item.get(key)
    values = data.get(collection, [])
    return values[index] if isinstance(index, int) and 0 <= index < len(values) else None


def _item_bbox(item: dict[str, Any], data: dict[str, Any]) -> dict[str, Any] | None:
    source = _item_source(item, data)
    provenance = source.get("prov", []) if source else []
    return provenance[0].get("bbox") if provenance else None


def _page_width(data: dict[str, Any], page_number: int) -> float:
    pages = data.get("pages", {})
    page = pages.get(str(page_number), pages.get(page_number, {}))
    return float(page.get("size", {}).get("width", 0))


def _left_edge_anchors(boxes: Iterable[dict[str, Any]], page_width: float) -> list[float]:
    left_edges = sorted(float(box["l"]) for box in boxes)
    if not left_edges:
        return []
    maximum_gap = page_width * LEFT_EDGE_CLUSTER_RATIO
    clusters: list[list[float]] = [[left_edges[0]]]
    for left_edge in left_edges[1:]:
        if left_edge - clusters[-1][-1] > maximum_gap:
            clusters.append([left_edge])
        else:
            clusters[-1].append(left_edge)
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _order_columns(
    report: dict[str, Any], data: dict[str, Any], page_numbers: set[int]
) -> None:
    for page in report["content"]:
        if int(page["page"]) not in page_numbers:
            continue
        page_width = _page_width(data, int(page["page"]))
        if page_width <= 0:
            continue
        indexed = [
            (index, item, _item_bbox(item, data))
            for index, item in enumerate(page["content"])
        ]
        narrow = [
            entry
            for entry in indexed
            if entry[2] is not None
            and float(entry[2]["r"]) - float(entry[2]["l"])
            < page_width * WIDE_ITEM_RATIO
        ]
        anchors = _left_edge_anchors((entry[2] for entry in narrow), page_width)
        if len(anchors) < 2:
            continue

        def top(box: dict[str, Any]) -> float:
            value = float(box["t"])
            return value if box.get("coord_origin") == "TOPLEFT" else -value

        ordered: list[tuple[int, dict[str, Any], dict[str, Any] | None]] = []
        for anchor in anchors:
            column = [
                entry
                for entry in narrow
                if min(anchors, key=lambda value: abs(float(entry[2]["l"]) - value))
                == anchor
            ]
            ordered.extend(sorted(column, key=lambda entry: (top(entry[2]), entry[0])))
        narrow_ids = {id(entry[1]) for entry in narrow}
        ordered.extend(entry for entry in indexed if id(entry[1]) not in narrow_ids)
        page["content"] = [entry[1] for entry in ordered]


def _chart_series(picture: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
    texts = [
        child["text"].strip()
        for child in picture.get("children", [])
        if isinstance(child.get("text"), str) and child["text"].strip()
    ]
    year_indexes = [index for index, value in enumerate(texts) if YEAR_LABEL.fullmatch(value)]
    if len(year_indexes) < 2:
        return None
    if year_indexes != list(range(year_indexes[0], year_indexes[-1] + 1)):
        return None
    years = [texts[index] for index in year_indexes]

    def is_axis_scale(value: str) -> bool:
        return (
            len(re.findall(r"\d", value)) >= 2
            and not re.sub(r"[\d\s$€£%.,KkMmBb-]", "", value)
        )

    title_tokens = [
        value
        for value in texts[: year_indexes[0]]
        if value not in {"$", "€", "£"}
        and not CHART_VALUE.fullmatch(value)
        and not is_axis_scale(value)
    ]
    title = " ".join(title_tokens).strip()
    if not title:
        return None
    if title.isupper():
        title = title.capitalize()
    payload = "".join(texts[year_indexes[-1] + 1 :]).replace(" ", "")
    values = CHART_VALUE.findall(payload)
    if len(values) != len(years) or CHART_VALUE.sub("", payload):
        return None
    return title, list(zip(years, values, strict=True))


def _table_cell(text: str, row: int, column: int, *, header: bool) -> dict[str, Any]:
    return {
        "bbox": None,
        "row_span": 1,
        "col_span": 1,
        "start_row_offset_idx": row,
        "end_row_offset_idx": row + 1,
        "start_col_offset_idx": column,
        "end_col_offset_idx": column + 1,
        "text": text,
        "column_header": header,
        "row_header": not header and column == 0,
        "row_section": False,
    }


def _chart_table(picture: dict[str, Any], table_id: int) -> dict[str, Any] | None:
    series = _chart_series(picture)
    if series is None:
        return None
    title, rows = series
    markdown = "\n".join(
        [
            f"| Year | {title.replace('|', '\\|')} |",
            "| --- | --- |",
            *(f"| {year} | {value} |" for year, value in rows),
        ]
    )
    html_table = "".join(
        [
            "<table><thead><tr><th>Year</th>",
            f"<th>{html.escape(title)}</th></tr></thead><tbody>",
            *(f"<tr><td>{html.escape(year)}</td><td>{html.escape(value)}</td></tr>" for year, value in rows),
            "</tbody></table>",
        ]
    )
    grid = [
        [_table_cell("Year", 0, 0, header=True), _table_cell(title, 0, 1, header=True)],
        *[
            [_table_cell(year, row, 0, header=False), _table_cell(value, row, 1, header=False)]
            for row, (year, value) in enumerate(rows, start=1)
        ],
    ]
    box = picture.get("bbox", [0, 0, 0, 0])
    cells = [cell for row in grid for cell in row]
    return {
        "table_id": table_id,
        "page": picture.get("page", 1),
        "bbox": box,
        "row_count": len(grid),
        "column_count": 2,
        "markdown": markdown,
        "html": html_table,
        "json": {
            "self_ref": f"#/tables/{table_id}",
            "parent": {"cref": "#/body"},
            "children": [],
            "label": "table",
            "prov": [
                {
                    "page_no": picture.get("page", 1),
                    "bbox": {
                        "l": box[0],
                        "t": box[1],
                        "r": box[2],
                        "b": box[3],
                        "coord_origin": "BOTTOMLEFT",
                    },
                    "charspan": [0, 0],
                }
            ],
            "captions": [],
            "references": [],
            "footnotes": [],
            "image": None,
            "data": {
                "table_cells": cells,
                "num_rows": len(grid),
                "num_cols": 2,
                "grid": grid,
            },
        },
        "source_picture_id": picture["picture_id"],
        "extraction_method": "picture_children_year_value_series",
    }


def _promote_charts(report: dict[str, Any]) -> None:
    tables = report["tables"]
    next_table_id = max((table["table_id"] for table in tables), default=-1) + 1
    promoted: dict[int, int] = {}
    for picture in report["pictures"]:
        table = _chart_table(picture, next_table_id)
        if table is None:
            continue
        tables.append(table)
        promoted[picture["picture_id"]] = next_table_id
        next_table_id += 1
    if not promoted:
        return
    for page in report["content"]:
        page["content"] = [
            {"type": "table", "table_id": promoted[item["picture_id"]]}
            if item.get("type") == "picture" and item.get("picture_id") in promoted
            else item
            for item in page["content"]
        ]
    report["metadata"]["table_count"] = len(tables)


def _retrieval_texts(report: dict[str, Any]) -> Iterable[str]:
    for page in report.get("content", []):
        for item in page.get("content", []):
            if isinstance(item.get("text"), str) and item["text"]:
                yield item["text"]
    for table in report.get("tables", []):
        if isinstance(table.get("markdown"), str) and table["markdown"]:
            yield table["markdown"]


def _validate_text_quality(report: dict[str, Any]) -> None:
    text = "\n".join(_retrieval_texts(report))
    glyph_count = len(GLYPH_IDENTIFIER.findall(text))
    readable_text = GLYPH_IDENTIFIER.sub(" ", text)
    readable_count = len(re.findall(r"\b\w[\w'.-]*\b", readable_text))
    ratio = glyph_count / max(1, glyph_count + readable_count)
    if glyph_count >= MIN_GLYPH_IDENTIFIERS and ratio > MAX_GLYPH_IDENTIFIER_RATIO:
        raise ParsingError(
            f"report contains unresolved glyph identifiers ({glyph_count}, ratio {ratio:.4f})"
        )


class FinancialReportParser:
    """Parse financial PDFs into deterministic JSON reports."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        settings: ParserSettings | None = None,
        converter: Any | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.settings = settings or ParserSettings()
        self.converter = converter or build_converter(self.settings)

    def parse(self, paths: Sequence[Path]) -> tuple[int, int]:
        inputs = sorted(Path(path) for path in paths)
        if not inputs:
            raise ValueError("no PDF inputs found")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        succeeded = 0
        failed = 0
        for path in inputs:
            try:
                results = list(self.converter.convert_all(source=[path]))
            except Exception:
                failed += 1
                LOGGER.exception("Failed to convert %s", path)
                _release_accelerator_memory()
                continue
            if not results or results[0].status != ConversionStatus.SUCCESS:
                failed += 1
                LOGGER.error("Failed to convert %s", path)
                _release_accelerator_memory()
                continue
            document = results[0].document
            document_id = path.stem
            report = assemble_report(document, document.export_to_dict(), document_id)
            (self.output_dir / f"{document_id}.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
                newline="\n",
            )
            succeeded += 1
            _release_accelerator_memory()
        if failed:
            raise ParsingError(f"failed to convert {failed} of {len(inputs)} PDFs")
        return succeeded, failed

    def parse_directory(self, input_dir: Path = DEFAULT_INPUT_DIR) -> tuple[int, int]:
        return self.parse(list(Path(input_dir).glob("*.pdf")))
