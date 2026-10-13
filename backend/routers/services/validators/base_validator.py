"""Reusable validation helpers shared by topic validators."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Set

from routers.services.legal_topic_matrix import (
    collect_supported_concepts,
    essential_concepts_for_topic,
    missing_required_concepts,
    required_concepts_for_topic,
    supporting_concepts_for_topic,
)


def concept_coverage(
    topic: str,
    sources: Iterable[Dict[str, Any]],
    query_text: str = "",
) -> Dict[str, Any]:
    supported: Set[str] = collect_supported_concepts(topic, sources, query_text)
    essential = essential_concepts_for_topic(topic, query_text)
    supporting = supporting_concepts_for_topic(topic, query_text)
    return {
        "topic": topic,
        "essential": essential,
        "supporting": supporting,
        "required": required_concepts_for_topic(topic, query_text),
        "supported": sorted(supported),
        "missing": missing_required_concepts(topic, sources, query_text),
        "essential_supported": any(concept in supported for concept in essential) if essential else bool(supported),
    }
