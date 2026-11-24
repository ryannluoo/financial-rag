import json

import pytest

from financial_rag.retrieval.router import (
    load_company_to_document_id,
    resolve_document_ids,
)


def test_load_company_to_document_id_loads_catalog(tmp_path):
    catalog = tmp_path / "document_catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "doc_a": {"company_name": "Acme Corp."},
                "doc_b": {"company_name": "Beta Holdings"},
            }
        ),
        encoding="utf-8",
    )

    mapping = load_company_to_document_id(catalog)
    assert mapping == {"Acme Corp.": "doc_a", "Beta Holdings": "doc_b"}


@pytest.mark.parametrize(
    "payload,match",
    [
        (["not-an-object"], "JSON object"),
        ({"  ": {"company_name": "Acme"}}, "non-empty strings"),
        ({"doc_a": "bad-entry"}, "JSON object"),
        ({"doc_a": {"company_name": "  "}}, "company_name"),
        (
            {"doc_a": {"company_name": "Acme"}, "doc_b": {"company_name": "acme"}},
            "Duplicate company name",
        ),
    ],
)
def test_load_company_to_document_id_rejects_invalid_catalog(tmp_path, payload, match):
    catalog = tmp_path / "document_catalog.json"
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_company_to_document_id(catalog)


def test_resolve_document_ids_order_dedup_longest_match_and_boundaries():
    catalog = {
        "Acme": "doc_short",
        "Acme Corp": "doc_long",
        "Beta Inc": "doc_beta",
    }

    assert resolve_document_ids(
        "Compare BETA INC with acme corp and beta inc.", catalog
    ) == ["doc_beta", "doc_long"]
    assert resolve_document_ids("SuperAcmeCorp is unrelated.", catalog) == []
    assert resolve_document_ids("   ", catalog) == []
    assert resolve_document_ids("Acme Corp revenue", {}) == []


def test_resolve_document_ids_uses_the_same_unicode_casefold_for_matching_and_lookup():
    catalog = {"iRobot": "doc_irobot", "Straße AG": "doc_strasse"}

    assert resolve_document_ids("Compare ıRobot and STRASSE AG.", catalog) == ["doc_strasse"]
    assert resolve_document_ids("Compare IROBOT and Straße AG.", catalog) == [
        "doc_irobot",
        "doc_strasse",
    ]
