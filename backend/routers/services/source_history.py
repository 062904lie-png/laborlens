"""Topic-aware source-history helpers for saved assistant messages."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def parse_saved_sources(raw_sources: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_sources, list):
        return [item for item in raw_sources if isinstance(item, dict)]
    if isinstance(raw_sources, str) and raw_sources.strip():
        try:
            parsed = json.loads(raw_sources)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def infer_source_followup_topic(query: str) -> str:
    text = query or ""
    if re.search(r"\b(overtime|undertime|hours?\s+of\s+work|article\s*8[2478])\b", text, re.IGNORECASE):
        return "overtime_pay"
    if re.search(r"\b(13th[-\s]*month|thirteenth|pd\s*851)\b", text, re.IGNORECASE):
        return "thirteenth_month_pay"
    if re.search(r"\b(regular\s+holiday|special\s+non[-\s]?working|holiday\s+pay)\b", text, re.IGNORECASE):
        return "holiday_pay"
    if re.search(r"\b(kasambahay|domestic\s+worker|house\s*helper|ra\s*10361|batas\s+kasambahay)\b", text, re.IGNORECASE):
        return "domestic_workers"
    if re.search(r"\b(minimum\s+wage|wage\s+order|salary|wages?)\b", text, re.IGNORECASE):
        return "minimum_wage"
    if re.search(r"\b(termination|dismissal|separation\s+pay)\b", text, re.IGNORECASE):
        return "termination"
    return ""


def _source_blob(source: Dict[str, Any]) -> str:
    return " ".join(
        str(source.get(key) or "")
        for key in (
            "topic",
            "source_topic",
            "document_title",
            "source",
            "source_name",
            "canonical_document_id",
            "law_number",
            "article_number",
            "article_title",
            "citation",
            "section",
            "excerpt",
        )
    )


def infer_topic_from_saved_sources(row: Optional[Dict[str, Any]], sources: List[Dict[str, Any]]) -> str:
    for source in sources or []:
        topic = str(source.get("topic") or source.get("source_topic") or "").strip()
        if topic:
            return topic
    blob = " ".join(_source_blob(source) for source in sources or [])
    text = f"{(row or {}).get('content') or ''} {blob}"
    if re.search(r"\b13th[-\s]*month|thirteenth|pd\s*851\b", text, re.IGNORECASE):
        return "thirteenth_month_pay"
    if re.search(r"\bovertime|undertime|hours\s+of\s+work|article\s*8[2478]\b", text, re.IGNORECASE):
        return "overtime_pay"
    if re.search(r"\bregular\s+holiday|special\s+non[-\s]?working|holiday\s+pay\b", text, re.IGNORECASE):
        return "holiday_pay"
    if re.search(r"\bkasambahay|domestic\s+worker|ra\s*10361\b", text, re.IGNORECASE):
        return "domestic_workers"
    return "general_labor"


def select_relevant_assistant_for_source_followup(
    rows: List[Dict[str, Any]],
    query: str,
    requested_topic: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    requested_topic = requested_topic if requested_topic is not None else infer_source_followup_topic(query)
    fallback = None
    for row in rows or []:
        if not (row.get("content") or "").strip() or row.get("intent") == "translation_request":
            continue
        sources = parse_saved_sources(row.get("sources"))
        if not sources:
            continue
        row_topic = infer_topic_from_saved_sources(row, sources)
        if requested_topic:
            if row_topic == requested_topic:
                return row, requested_topic, False
            continue
        if fallback is None:
            fallback = row
    return fallback, requested_topic, bool(requested_topic and fallback is None)
