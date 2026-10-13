"""Primary/secondary knowledge-base category metadata helpers.

Documents and chunks remain single records.  Chroma metadata supports scalar
values, so the canonical JSON representation is accompanied by bounded scalar
slots used only for category filtering.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

from routers.services.kb_categories import (
    DEFAULT_KB_CATEGORY,
    KB_CATEGORIES,
    OCCUPATIONAL_SAFETY_CATEGORY,
    canonical_kb_category,
)


MAX_SECONDARY_CATEGORY_SLOTS = 4


def parse_string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                decoded = json.loads(raw)
            except (TypeError, ValueError):
                decoded = None
            if isinstance(decoded, list):
                value = decoded
            else:
                value = re.split(r"[,;|]", raw)
        else:
            value = re.split(r"[,;|]", raw)
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    output: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in output:
            output.append(text)
    return output


def normalize_primary_category(value: Optional[str]) -> str:
    category = canonical_kb_category(value or DEFAULT_KB_CATEGORY)
    if category not in KB_CATEGORIES:
        raise ValueError(f"Unknown knowledge base category: {value}")
    return category


def normalize_secondary_categories(
    value: Any,
    primary_category: str,
) -> list[str]:
    primary = normalize_primary_category(primary_category)
    output: list[str] = []
    for raw in parse_string_list(value):
        category = canonical_kb_category(raw)
        if category not in KB_CATEGORIES:
            raise ValueError(f"Unknown knowledge base category: {raw}")
        if category == primary or category in output:
            continue
        output.append(category)
    if len(output) > MAX_SECONDARY_CATEGORY_SLOTS:
        raise ValueError(
            f"A document can have at most {MAX_SECONDARY_CATEGORY_SLOTS} secondary categories"
        )
    return output


def normalize_document_categories(
    primary_category: Optional[str],
    secondary_categories: Any = None,
) -> tuple[str, list[str]]:
    primary = normalize_primary_category(primary_category)
    return primary, normalize_secondary_categories(secondary_categories, primary)


def secondary_categories_from_metadata(metadata: Optional[dict[str, Any]]) -> list[str]:
    meta = metadata or {}
    raw = meta.get("secondary_categories")
    if raw in (None, ""):
        raw = meta.get("secondary_categories_json")
    values = parse_string_list(raw)
    if not values:
        values = [
            str(meta.get(f"secondary_category_{index}") or "").strip()
            for index in range(1, MAX_SECONDARY_CATEGORY_SLOTS + 1)
        ]
    primary = canonical_kb_category(
        str(meta.get("primary_category") or meta.get("category") or DEFAULT_KB_CATEGORY)
    )
    output: list[str] = []
    for value in values:
        category = canonical_kb_category(value)
        if category in KB_CATEGORIES and category != primary and category not in output:
            output.append(category)
    return output[:MAX_SECONDARY_CATEGORY_SLOTS]


def category_metadata(
    primary_category: Optional[str],
    secondary_categories: Any = None,
) -> dict[str, Any]:
    primary, secondary = normalize_document_categories(
        primary_category,
        secondary_categories,
    )
    metadata: dict[str, Any] = {
        "category": primary,
        "primary_category": primary,
        # Chroma metadata values must remain scalar in the deployed version.
        "secondary_categories": json.dumps(secondary, ensure_ascii=False),
        "secondary_categories_json": json.dumps(secondary, ensure_ascii=False),
        "secondary_category_count": len(secondary),
    }
    for index in range(1, MAX_SECONDARY_CATEGORY_SLOTS + 1):
        metadata[f"secondary_category_{index}"] = (
            secondary[index - 1] if index <= len(secondary) else ""
        )
    return metadata


def json_list(value: Any) -> str:
    return json.dumps(parse_string_list(value), ensure_ascii=False)


def chroma_safe_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Convert metadata to the scalar value types supported by Chroma."""

    output: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            output[str(key)] = ""
        elif isinstance(value, (list, tuple, set, dict)):
            output[str(key)] = json.dumps(value, ensure_ascii=False, default=str)
        elif isinstance(value, (str, int, float, bool)):
            output[str(key)] = value
        else:
            output[str(key)] = str(value)
    return output


def category_match_type(
    metadata: Optional[dict[str, Any]],
    requested_categories: Iterable[str],
) -> str:
    requested = {
        canonical_kb_category(str(value or ""))
        for value in requested_categories or []
        if str(value or "").strip()
    }
    if not requested:
        return "none"
    meta = metadata or {}
    primary = canonical_kb_category(
        str(meta.get("primary_category") or meta.get("category") or DEFAULT_KB_CATEGORY)
    )
    if primary in requested:
        return "primary"
    if requested.intersection(secondary_categories_from_metadata(meta)):
        return "secondary"
    return "none"


def suggest_document_categories(
    *,
    title: str = "",
    law_number: str = "",
    document_type: str = "",
    issuing_agency: str = "",
    text: str = "",
) -> dict[str, Any]:
    """Return review-only deterministic suggestions; never persists changes."""

    blob = " ".join((title, law_number, document_type, issuing_agency, text[:4000])).lower()
    primary = DEFAULT_KB_CATEGORY
    secondary: list[str] = []
    confidence = 0.55
    reasons: list[str] = []

    is_dole = bool(re.search(r"\b(?:dole|department\s+of\s+labor|department\s+order)\b", blob))
    is_advisory = bool(re.search(r"\blabor\s+advisory\b|\badvisory\b", blob))

    if re.search(r"\b(?:ra\s*11058|11058|do(?:le)?\s*(?:no\.?\s*)?(?:198[- ]?18|252[- ]?25)|occupational\s+safety|\bosh\b|ppe)\b", blob):
        primary = OCCUPATIONAL_SAFETY_CATEGORY
        confidence = 0.95
        reasons.append("RA 11058 or OSH terminology detected")
        if is_dole or "department order" in blob:
            secondary.append("DOLE Resources")
        if re.search(r"\b(?:irr|implementing\s+rules|ra\s*11058)\b", blob):
            secondary.append("Core Labor Laws")
    elif re.search(r"\b(?:regional\s+)?wage\s+order\b|\brtwpb\b|regional\s+minimum\s+wage", blob):
        primary = "Employment Benefits"
        secondary.append("DOLE Resources")
        confidence = 0.94
        reasons.append("Regional wage-order signals detected")
    elif re.search(r"\b(?:ra\s*10361|kasambahay|domestic\s+workers?\s+act)\b", blob):
        primary = "Domestic Workers"
        confidence = 0.94
        reasons.append("RA 10361 or Kasambahay signals detected")
        if is_dole or "issuance" in blob or "irr" in blob:
            secondary.append("DOLE Resources")
        secondary.append("Core Labor Laws")
    elif re.search(r"\b(union\s+registration|collective\s+bargaining|\bcba\b|certification\s+election|strike|grievance\s+machinery)\b", blob):
        primary = "Labor Relations"
        confidence = 0.90
        reasons.append("Labor-relations subject detected")
        if is_dole or "department order" in blob:
            secondary.append("DOLE Resources")
    elif is_advisory and re.search(r"\bholiday\s+pay|regular\s+holiday|special\s+day\b", blob):
        primary = "Employment Benefits"
        secondary.extend(["Labor Advisories", "DOLE Resources"])
        confidence = 0.92
        reasons.append("Holiday-pay advisory detected")
    elif is_advisory:
        primary = "Labor Advisories"
        if is_dole:
            secondary.append("DOLE Resources")
        confidence = 0.72
        reasons.append("Advisory document type detected without a narrower subject")
    elif is_dole:
        primary = "DOLE Resources"
        confidence = 0.65
        reasons.append("DOLE issuer detected without a narrower legal subject")

    primary, secondary = normalize_document_categories(primary, secondary)
    return {
        "primary_category": primary,
        "secondary_categories": secondary,
        "confidence": round(confidence, 2),
        "reasons": reasons or ["No strong deterministic subject signal detected"],
        "review_required": True,
    }
