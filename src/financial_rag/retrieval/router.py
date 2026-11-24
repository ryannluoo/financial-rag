import json
import re
from collections.abc import Mapping
from pathlib import Path


def load_company_to_document_id(catalog_path: Path) -> dict[str, str]:
    """Load a company_name -> document_id mapping from document_catalog.json."""
    raw = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Document catalog must be a JSON object.")

    company_to_document_id: dict[str, str] = {}
    seen_folded: set[str] = set()
    for document_id, entry in raw.items():
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Document catalog keys must be non-empty strings.")
        if not isinstance(entry, dict):
            raise ValueError(f"Catalog entry for {document_id!r} must be a JSON object.")
        company_name = entry.get("company_name")
        if not isinstance(company_name, str) or not company_name.strip():
            raise ValueError(
                f"Catalog entry for {document_id!r} must have a non-empty 'company_name'."
            )
        folded = company_name.casefold()
        if folded in seen_folded:
            raise ValueError(f"Duplicate company name in catalog: {company_name!r}.")
        seen_folded.add(folded)
        company_to_document_id[company_name] = document_id

    return company_to_document_id


def resolve_document_ids(
    question: str,
    company_to_document_id: Mapping[str, str],
) -> list[str]:
    """Return deduplicated document_ids for companies mentioned in a question, in order of appearance."""
    if not question.strip() or not company_to_document_id:
        return []

    folded_lookup = {
        name.casefold(): doc_id
        for name, doc_id in company_to_document_id.items()
        if name.strip()
    }
    if not folded_lookup:
        return []

    ordered_names = sorted(folded_lookup, key=len, reverse=True)
    pattern = re.compile("|".join(rf"(?<!\w){re.escape(name)}(?!\w)" for name in ordered_names))

    document_ids: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(question.casefold()):
        doc_id = folded_lookup[match.group(0)]
        if doc_id not in seen:
            seen.add(doc_id)
            document_ids.append(doc_id)

    return document_ids
