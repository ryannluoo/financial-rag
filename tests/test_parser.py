import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, OcrMode, TableFormerMode

from financial_rag.parsing.parser import (
    FinancialReportParser,
    ParserSettings,
    ParsingError,
    _promote_charts,
    _validate_text_quality,
    build_converter,
    build_pipeline_options,
)


def test_pipeline_matches_baseline_configuration(tmp_path: Path):
    settings = ParserSettings(artifacts_dir=tmp_path / "models", threads=6)
    options = build_pipeline_options(settings)

    assert options.artifacts_path == settings.artifacts_dir
    assert options.do_ocr is True
    assert isinstance(options.ocr_options, EasyOcrOptions)
    assert options.ocr_options.lang == ["en"]
    assert options.ocr_options.mode is OcrMode.DEFAULT
    assert options.ocr_options.download_enabled is False
    assert options.ocr_options.model_storage_directory == str(
        settings.artifacts_dir / "EasyOcr"
    )
    assert options.do_table_structure is True
    assert options.table_structure_options.do_cell_matching is True
    assert options.table_structure_options.mode is TableFormerMode.ACCURATE
    assert options.do_chart_extraction is False
    assert options.ocr_batch_size == 1
    assert options.layout_batch_size == 1
    assert ParserSettings().device is AcceleratorDevice.CUDA
    assert options.accelerator_options.num_threads == 6
    assert options.accelerator_options.device is AcceleratorDevice.CUDA

    converter = build_converter(settings)
    pdf_options = converter.format_to_options[InputFormat.PDF]
    assert pdf_options.backend is DoclingParseDocumentBackend


def test_chart_series_becomes_retrieval_table():
    report = {
        "metadata": {"table_count": 0},
        "content": [
            {"page": 1, "content": [{"type": "picture", "picture_id": 0}]}
        ],
        "tables": [],
        "pictures": [
            {
                "picture_id": 0,
                "page": 1,
                "bbox": [1, 2, 3, 4],
                "children": [
                    {"text": "TOTAL ASSETS"},
                    {"text": "$0.8M $1.0M $1.2M"},
                    {"text": "2022"},
                    {"text": "2021"},
                    {"text": "$1,745,530"},
                    {"text": "$1,664,323"},
                ],
            }
        ],
    }

    _promote_charts(report)

    assert report["content"][0]["content"] == [{"type": "table", "table_id": 0}]
    assert report["metadata"]["table_count"] == 1
    assert report["tables"][0]["row_count"] == 3
    assert "| 2022 | $1,745,530 |" in report["tables"][0]["markdown"]
    assert "Total assets $0.8M" not in report["tables"][0]["markdown"]


def test_corrupt_glyph_output_is_rejected():
    report = {
        "content": [
            {
                "content": [
                    {"type": "text", "text": "/gid1 /gid2 /gid3 /gid4"}
                ]
            }
        ],
        "tables": [],
    }

    with pytest.raises(ParsingError, match="glyph"):
        _validate_text_quality(report)


class _FakeDocument:
    tables = []

    def export_to_dict(self):
        return {
            "pages": {"1": {"size": {"width": 612, "height": 792}}},
            "body": {"children": [{"$ref": "#/texts/0"}]},
            "groups": [],
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "label": "text",
                    "text": "Readable text",
                    "orig": "Readable text",
                    "prov": [
                        {
                            "page_no": 1,
                            "bbox": {
                                "l": 10,
                                "t": 20,
                                "r": 200,
                                "b": 10,
                                "coord_origin": "BOTTOMLEFT",
                            },
                        }
                    ],
                }
            ],
            "tables": [],
            "pictures": [],
            "equations": [],
        }


class _FakeConverter:
    def convert_all(self, source):
        assert source == sorted(source)
        return [
            SimpleNamespace(
                status=ConversionStatus.SUCCESS,
                document=_FakeDocument(),
            )
        ]


def test_parser_writes_only_json_report(tmp_path: Path):
    parser = FinancialReportParser(output_dir=tmp_path, converter=_FakeConverter())

    assert parser.parse([Path("sample.pdf")]) == (1, 0)
    assert [path.name for path in tmp_path.iterdir()] == ["sample.json"]
    report = json.loads((tmp_path / "sample.json").read_text(encoding="utf-8"))
    assert list(report) == ["metadata", "content", "tables", "pictures"]
    assert report["metadata"]["document_id"] == "sample"
