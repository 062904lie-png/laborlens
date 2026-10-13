"""Bounded, category-first retrieval planning for LaborLens.

This module contains no legal answers.  It only decides where and how to look
for uploaded evidence, keeping the Knowledge Base First policy intact.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from config import settings
from routers.services.kb_categories import normalize_kb_category, routed_kb_categories, LEGACY_KB_CATEGORY_MAP
from routers.services.document_categories import MAX_SECONDARY_CATEGORY_SLOTS
from routers.services.rag_policy import (
    SOURCE_STATUS_DIRECT,
    SOURCE_STATUS_INSUFFICIENT,
    SOURCE_STATUS_PARTIAL,
    rerank_chunks,
    osh_evidence_is_responsive,
    source_match_status,
)
from routers.services.performance import measure


logger = logging.getLogger(__name__)

INITIAL_TOP_K = max(4, min(int(settings.RAG_INITIAL_TOP_K), 30))
EXPANDED_TOP_K = max(INITIAL_TOP_K, min(int(settings.RAG_EXPANDED_TOP_K), 50))
MAX_CONTEXT_CHUNKS = max(3, min(int(settings.RAG_MAX_CONTEXT_CHUNKS), 12))
MAX_RETRIEVAL_STAGES = max(1, min(int(settings.RAG_MAX_RETRIEVAL_STAGES), 3))
MAX_CATEGORIES = max(1, min(int(settings.RAG_MAX_CATEGORIES), 3))
VECTOR_SCORE_FLOOR = max(-1.0, min(float(settings.RAG_VECTOR_SCORE_FLOOR), 1.0))
NEAR_DUPLICATE_THRESHOLD = max(0.75, min(float(settings.RAG_NEAR_DUPLICATE_THRESHOLD), 0.99))

# Task-specific category expansion complements the topic route when a query
# asks about a document or certificate governed by an advisory. Keep this out
# of the primary route so ordinary broad labor questions stay narrowly scoped.
TASK_CATEGORY_EXPANSIONS = {
    ("general_labor", "document_requirement"): ["Labor Advisories"],
}


TOPIC_RETRIEVAL_TERMS = {
    "approved_leave_recall": "approved leave recall cancellation employer requiring work during leave employee availability company policy collective bargaining agreement",
    "minimum_wage": "current regional minimum wage wage order effectivity sector coverage",
    "cost_of_living_allowance": "cost of living allowance COLA entitlement",
    "payment_of_wages": "payment of wages timing method employee rights",
    "wage_deductions": "authorized prohibited wage deductions employee consent",
    "overtime_pay": "overtime work beyond eight hours coverage formula",
    "holiday_pay": "regular holiday pay coverage worked unworked",
    "premium_pay": "special non-working day rest day premium pay",
    "night_shift_differential": "night shift differential covered hours coverage formula",
    "hours_of_work": "normal hours of work meal period coverage",
    "thirteenth_month_pay": "13th month pay coverage basic salary formula deadline",
    "service_incentive_leave": "service incentive leave eligibility days coverage exclusions",
    "service_charges": "service charges distribution covered employees establishments",
    "maternity_leave": "maternity leave eligibility duration benefit",
    "paternity_leave": "paternity leave eligibility duration",
    "solo_parent_leave": "solo parent parental leave days service requirement",
    "special_leave_for_women": "special leave for women eligibility duration",
    "domestic_workers": "Batas Kasambahay rights benefits contract rest periods",
    "workplace_safety": (
        "employer duties occupational safety health safe workplace worker safety rights PPE "
        "safety information OSH program safety committee safety officer RA 11058 DO 252-25"
    ),
    "workplace_injury": "workplace injury reporting compensation benefit",
    "sexual_harassment": "workplace sexual harassment prohibited acts employer duties remedies",
    "safe_spaces": "gender based sexual harassment workplace employer duties",
    "discrimination": "employment discrimination prohibited acts protected ground",
    "termination": "termination just causes authorized causes employee",
    "procedural_due_process": "dismissal procedural due process notice hearing decision",
    "separation_pay": "separation pay authorized cause formula coverage",
    "retirement_pay": "retirement pay eligibility formula exemptions",
    "probationary_employment": "probationary employment period standards regularization",
    "security_floating_status": "DOLE Department Order 150-16 security guards reserved status work pool no service assignment six months separation pay",
    "regular_employment": "regular employment status necessary desirable work",
    "project_employment": "project employment duration completion condition",
    "fixed_term_employment": "fixed term employment validity conditions",
    "union_rights": "right to self organization labor union unfair labor practice",
    "collective_bargaining": "collective bargaining agreement duty bargaining unit",
    "strike_lockout": "strike lockout legal requirements labor relations",
    "labor_complaint": "DOLE complaint SEnA request for assistance conciliation mediation",
    "social_security": "SSS coverage contributions benefits",
    "philhealth": "PhilHealth coverage contributions benefits",
    "pagibig": "Pag-IBIG HDMF coverage contributions benefits",
    "employees_compensation": "employees compensation work contingency benefit",
    "telecommuting": "telecommuting fair treatment minimum labor standards data protection",
    "contracting_subcontracting": "contracting subcontracting legitimate contractor labor only contracting",
    "ofw": "overseas Filipino worker employment rights recruitment",
    "jurisprudence_request": "Supreme Court labor case jurisprudence ruling",
}

EXCEPTION_TERMS_RE = re.compile(
    r"\b(except|exception|exempt|exemption|exclude|excluded|exclusion|unless|"
    r"provided\s+that|provided\s+further|subject\s+to|shall\s+not\s+apply|"
    r"covered|coverage|eligible|eligibility)\b",
    re.IGNORECASE,
)

NEW_PROVISION_FOLLOWUP_RE = re.compile(
    r"\b(region|province|city|regular\s+holiday|special\s+(?:non[-\s]?)?working|"
    r"rest\s+day|managerial|field\s+personnel|exempt|exception|coverage|"
    r"article|section|republic\s+act|\bra\s*\d+|department\s+order|wage\s+order)\b",
    re.IGNORECASE,
)


def _category_filter_values(categories: Iterable[str]) -> List[str]:
    values: List[str] = []
    for category in categories:
        canonical = normalize_kb_category(category)
        names = [canonical, *[name for name, target in LEGACY_KB_CATEGORY_MAP.items() if target == canonical]]
        for name in names:
            values.extend([name, name.lower().replace(" & ", "_").replace(" / ", "_").replace(" ", "_")])
    return list(dict.fromkeys(values))


def build_category_filter(
    categories: List[str],
    query_profile: Optional[Dict[str, Any]] = None,
    *,
    exact_region: bool = False,
    match_type: str = "primary",
) -> Dict[str, Any]:
    """Build a Chroma-compatible filter before vector retrieval."""

    values = _category_filter_values(categories)
    if match_type in {"secondary", "any"}:
        secondary_filters = [
            {f"secondary_category_{index}": {"$in": values}}
            for index in range(1, MAX_SECONDARY_CATEGORY_SLOTS + 1)
        ]
    if match_type == "secondary":
        category_filter: Dict[str, Any] = {
            "$or": secondary_filters
        }
    elif match_type == "any":
        category_filter = {
            "$or": [
                {"category": {"$in": values}},
                *secondary_filters,
            ]
        }
    else:
        # ``category`` remains the primary-category compatibility field for
        # legacy and newly indexed chunks.
        category_filter = {"category": {"$in": values}}
    profile = query_profile or {}
    # A region landing page and its uploaded PDFs have different URLs.
    # Regional applicability is checked from owning-document metadata; an
    # exact URL here would hide the controlling uploaded wage order.
    return category_filter


def build_retrieval_plan(
    query: str,
    query_profile: Dict[str, Any],
    conversation_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    topic = str(query_profile.get("topic") or "general_labor")
    from routers.services.evidence_scope import requested_scope
    compare_government_private = bool(
        query_profile.get("compare_government_private")
        or (
            str(query_profile.get("sector") or "") == "general_private_employment"
            and topic not in {"government_employment", "domestic_workers"}
            and requested_scope(query) is None
        )
    )
    category_override = [
        normalize_kb_category(value)
        for value in query_profile.get("category_override") or []
        if str(value or "").strip()
    ]
    from routers.services.named_issuance import requested_issuance
    if requested_issuance(query):
        # Issuer is a retrieval hint, not the subject category. Keep the
        # topic's legal category first and include DOLE folders as fallbacks.
        category_override = []
    primary = list(dict.fromkeys(category_override))[:MAX_CATEGORIES] or routed_kb_categories(
        topic, expanded=False, limit=MAX_CATEGORIES
    )
    expanded = list(dict.fromkeys([
        *primary,
        *routed_kb_categories(topic, expanded=True, limit=12),
    ]))
    task_categories = TASK_CATEGORY_EXPANSIONS.get(
        (topic, str(query_profile.get("task") or "")), []
    )
    expanded = list(dict.fromkeys([*expanded, *task_categories]))
    if query_profile.get("sector") == "government_employment":
        from routers.services.sector_retrieval import sector_category_route
        primary, expanded = sector_category_route(topic, query_profile["sector"])
    # Search a few strong category hints together on pass one. This remains a
    # retrieval priority, not an evidence allowlist; later category stages can
    # still widen the search when scope or evidence is incomplete.
    from routers.services.kb_categories import KB_CATEGORIES
    issue_topic = str(query_profile.get("issue_topic") or topic)
    preferred = "Employment Benefits" if issue_topic in {
        "overtime_pay", "ordinary_overtime", "rest_day_overtime", "holiday_overtime",
        "regular_holiday_overtime", "special_day_overtime", "holiday_rest_day_overtime",
    } else ""
    special_sector = str(query_profile.get("sector") or "") in {
        "government_employment", "domestic_workers", "caregivers", "government_cos_jo",
    }
    scoped_category = normalize_kb_category(query_profile.get("category")) if query_profile.get("category") else ""
    if special_sector and scoped_category in KB_CATEGORIES:
        primary = [scoped_category]
    elif preferred in KB_CATEGORIES and not special_sector and not query_profile.get("is_regional_wage_query"):
        primary = [preferred, *primary]
    elif not query_profile.get("is_regional_wage_query"):
        primary = [*primary, *(query_profile.get("category_hints") or [])]
    if compare_government_private:
        government_category = "Civil Service / Government Employment"
        # Keep the main topic category plus a government source category in
        # the first vector pass, even when the configured category limit is 2.
        primary = list(dict.fromkeys([*primary, government_category]))
        if government_category not in primary[:MAX_CATEGORIES]:
            primary = [*primary[:max(0, MAX_CATEGORIES - 1)], government_category]
        expanded = list(dict.fromkeys([*expanded, government_category]))
    if query_profile.get("is_regional_wage_query"):
        primary = primary[:1]
    primary = list(dict.fromkeys(normalize_kb_category(value) for value in primary if value))[:MAX_CATEGORIES]
    named_issuance = bool(requested_issuance(query)) and topic == "general_labor"
    if named_issuance:
        primary = list(dict.fromkeys(["DOLE Resources", *primary]))[:MAX_CATEGORIES]
        expanded = list(dict.fromkeys([*expanded, "DOLE Resources", "Labor Advisories"]))
    related = list(dict.fromkeys([
        *[value for value in query_profile.get("category_hints") or [] if value != primary[0]],
        *[value for value in expanded if value not in primary],
    ]))
    authority_categories = [
        value for value in ("Core Labor Laws", "Jurisprudence", "Labor Advisories", "DOLE Resources")
        if value not in primary and value not in related
    ]
    # Only an unclassified/Others route earns a final broad category pass.
    # Known topics never fan out into unrelated folders such as Social Benefits.
    broader = []
    if primary == ["Others"]:
        social_query = re.search(r"\b(sss|philhealth|pag[- ]?ibig|contribution|social security)\b", query, re.I)
        broader = [value for value in KB_CATEGORIES
                   if value not in primary and value not in related and value not in authority_categories
                   and (value != "Social Benefits" or social_query)]
    category_stages = [
        {"stage": "primary_category", "categories": primary},
        {"stage": "related_categories", "categories": related[:MAX_CATEGORIES]},
        {"stage": "authoritative_categories", "categories": authority_categories},
        {"stage": "broader_categories", "categories": broader[:MAX_CATEGORIES]},
    ]
    exact_region = False
    return {
        "topic": topic,
        "language": (conversation_state or {}).get("preferred_language") or "auto",
        "primary_categories": primary,
        "expanded_categories": expanded,
        "category_stages": [stage for stage in category_stages if stage["categories"]],
        "initial_top_k": INITIAL_TOP_K,
        "expanded_top_k": EXPANDED_TOP_K,
        "max_stages": MAX_RETRIEVAL_STAGES,
        "initial_filter": build_category_filter(
            primary, query_profile, exact_region=exact_region, match_type="any"
        ),
        "initial_secondary_filter": build_category_filter(
            primary,
            query_profile,
            exact_region=exact_region,
            match_type="secondary",
        ),
        "expanded_filter": build_category_filter(expanded, query_profile, match_type="any"),
        "expanded_secondary_filter": build_category_filter(expanded, query_profile, match_type="secondary"),
        "exact_region_filter": exact_region,
        "original_query": query,
    }


def rewrite_retrieval_query(query: str, query_profile: Dict[str, Any]) -> str:
    """Create a compact legal search query without appending chat transcripts."""

    topic = str(query_profile.get("topic") or "general_labor")
    from routers.services.scenario_retrieval import scenario_search_terms
    expansion_parts = [scenario_search_terms(query), TOPIC_RETRIEVAL_TERMS.get(topic, topic.replace("_", " "))]
    if query_profile.get("sector") == "government_employment":
        expansion_parts.insert(0, "Civil Service Commission CSC government employment applicable civil service rules")
    if query_profile.get("compare_government_private"):
        expansion_parts.append(
            f"private-sector labor law rules and government employee civil service rules for {topic.replace('_', ' ')}"
        )
    task = str(query_profile.get("task") or "").replace("_", " ").strip()
    if task and task not in {"general explanation", "general question"}:
        expansion_parts.append(task)
    sub_intent = str(query_profile.get("sub_intent_terms") or query_profile.get("sub_intent") or "").replace("_", " ").strip()
    if sub_intent and sub_intent not in expansion_parts:
        expansion_parts.append(sub_intent)
    for key in ("region_code", "region_name"):
        value = str(query_profile.get(key) or "").strip()
        if value:
            expansion_parts.append(value)
    sector = str(query_profile.get("sector") or "").replace("_", " ").strip()
    if sector and sector not in {"general private employment", "unknown employment"}:
        expansion_parts.append(sector)

    # Preserve explicit legal identifiers and the latest question's key words,
    # but never append the complete conversation history.
    identifiers = re.findall(
        r"\b(?:RA|Republic\s+Act|PD|DO|Department\s+Order|Wage\s+Order|"
        r"Article|Art\.?|Section|Sec\.?)\s*(?:No\.?\s*)?[A-Z0-9-]+",
        query or "",
        re.IGNORECASE,
    )
    expansion_parts.extend(identifiers[:3])
    # Keep the complete normalized question (up to the supported query bound)
    # intact. Only secondary search vocabulary is expendable.
    normalized_question = re.sub(r"\s+", " ", query or "").strip()
    expansion = " | ".join(dict.fromkeys(
        re.sub(r"\s+", " ", str(part or "")).strip()
        for part in expansion_parts if str(part or "").strip()
    ))[:300]
    if not normalized_question:
        return expansion[:1200]
    remaining = max(0, 1200 - len(normalized_question) - 3)
    if not remaining:
        return normalized_question
    return normalized_question + (" | " + expansion[:remaining] if expansion else "")


def required_component_queries(query: str, profile: Dict[str, Any], missing: List[str], limit: int = 4) -> List[Dict[str, Any]]:
    """Bounded independent probes; no answer text or topic-specific outcomes."""
    limit = max(0, min(limit, 4))
    if not limit:
        return []
    question = re.sub(r"\s+", " ", query or "").strip()
    identifiers = re.findall(
        r"\b(?:Republic\s+Act|RA|PD|Department\s+Order|DO|Article|Art\.?|Section|Sec\.?)"
        r"\s*(?:No\.?\s*)?\d+[A-Za-z]?(?:[-–][A-Z0-9]+)*", question, re.I,
    )
    probes = [{"components": [], "query": f"{identifier} | {question[:480]}", "exact_identifier": True}
              for identifier in dict.fromkeys(identifiers)][:min(2, limit)]
    for probe, identifier in zip(probes, dict.fromkeys(identifiers)):
        compact = re.sub(r"\s*(?:No\.?)\s*", " ", identifier, flags=re.I)
        match = re.fullmatch(r"(RA|Republic\s+Act|PD|DO|Department\s+Order)\s*(\d+)(?:[-–](\d+))?", compact, re.I)
        if match:
            prefix = match[1].lower()
            authority = (f"dole_do_{match[2]}_{match[3]}" if prefix in {"do", "department order"} and match[3]
                         else f"{'pd' if prefix == 'pd' else 'ra'}_{match[2]}" if prefix not in {"do", "department order"} else "")
            if authority:
                probe["authority_ids"] = [authority]
        elif re.search(r"labor\s+code", question, re.I) and re.match(r"(?:Art|Article)", identifier, re.I):
            probe["authority_ids"] = ["labor_code_article_" + re.search(r"\d+", identifier)[0]]
    concepts = list(dict.fromkeys(c for c in missing if c != "legal_basis"))
    slots = limit - len(probes)
    if slots and not concepts and profile.get('evidence_contract_unresolved'):
        probes.append({'components': [], 'exact_identifier': False,
                       'query': ('operative legal provision obligations conditions exceptions ' + question)[:900]})
    if slots and concepts:
        # Pack all missing parts into a fixed number of small searches rather
        # than silently dropping the last requested part of a long comparison.
        width = (len(concepts) + slots - 1) // slots
        for start in range(0, len(concepts), width):
            group = concepts[start:start + width]
            probes.append({"components": group, "exact_identifier": False,
                           "query": " | ".join([
                               str(profile.get("topic") or "").replace("_", " "),
                               " ".join(c.replace("_", " ") for c in group),
                               question[:480],
                               str(profile.get("sector") or "").replace("_", " "),
                           ])[:900]})
    return probes[:limit]


def expansion_query(base_query: str, query_profile: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
    required = [str(item).replace("_", " ") for item in query_profile.get("required_concepts") or []]
    text = " ".join(str(chunk.get("text") or "") for chunk in chunks)
    additions = required[:6]
    task = str(query_profile.get("task") or "")
    if task in {"coverage_question", "exclusion_question", "eligibility_question"} or (
        EXCEPTION_TERMS_RE.search(base_query or "") and not EXCEPTION_TERMS_RE.search(text)
    ):
        additions.append("coverage exceptions exemptions exclusions unless provided that subject to")
    if str(query_profile.get("topic") or "") == "workplace_safety":
        osh_terms = {
            "employer_duties": "RA 11058 Section 4 duties of employers safe and healthful workplace job safety instructions hazards PPE OSH program DO 252-25",
            "refusal_of_unsafe_work": "RA 11058 Section 6 right to refuse unsafe work imminent danger DOLE determination reprisal DO 252-25",
            "ppe_requirement": "RA 11058 DO 252-25 employer provide personal protective equipment PPE free of charge cost",
            "safety_officer_requirement": "DO 252-25 safety officer requirement number workers risk classification establishment",
            "osh_program_requirement": "DO 252-25 OSH program required contents workplace risk",
            "osh_committee_requirement": "DO 252-25 safety and health committee composition requirement",
        }
        additions.append(osh_terms.get(task, osh_terms["employer_duties"]))
    return " | ".join([base_query, *dict.fromkeys(additions)])[:900]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 2
    }


def _near_duplicate(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    # Several distinct chunks can belong to the same long provision. Identity
    # alone must not discard its formula, definitions, or exceptions.
    if str(left.get("text") or "").strip() == str(right.get("text") or "").strip():
        return True
    left_tokens = _tokens(str(left.get("text") or ""))
    right_tokens = _tokens(str(right.get("text") or ""))
    if min(len(left_tokens), len(right_tokens)) < 8:
        return False
    similarity = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    return similarity >= NEAR_DUPLICATE_THRESHOLD


def deduplicate_retrieval_chunks(
    chunks: List[Dict[str, Any]],
    query_profile: Dict[str, Any],
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Drop exact/provision/near duplicates after legal reranking."""

    with measure("rerank_ms"):
        ranked = rerank_chunks(chunks, query_profile, limit=max(len(chunks or []), limit or 0, 1))
        output: List[Dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for chunk in ranked:
            score = float(chunk.get("score") or 0.0)
            retrieval_method = str((chunk.get("metadata") or {}).get("retrieval_method") or "")
            if score < VECTOR_SCORE_FLOOR and retrieval_method != "required_authority_lookup":
                continue
            chunk_id = str((chunk.get("metadata") or {}).get("chunk_id") or chunk.get("id") or "")
            if chunk_id and chunk_id in seen_chunk_ids:
                continue
            if any(_near_duplicate(chunk, kept) for kept in output):
                continue
            output.append(chunk)
            if chunk_id:
                seen_chunk_ids.add(chunk_id)
            if limit and len(output) >= limit:
                break
        return output


def needs_progressive_expansion(
    chunks: List[Dict[str, Any]],
    query_profile: Dict[str, Any],
    *,
    missing_authorities: Optional[List[str]] = None,
) -> bool:
    if not chunks or missing_authorities:
        return True
    status, _reason = source_match_status(chunks, query_profile)
    if (
        str(query_profile.get("topic") or "") == "workplace_safety"
        and not osh_evidence_is_responsive(chunks, query_profile)
    ):
        return True
    from routers.services.legal_topic_matrix import supported_passage_concepts
    from routers.services.scenario_retrieval import scenario_evidence_gap
    required = set(query_profile.get("required_concepts") or [])
    if not required:
        from routers.services.issue_evidence import unclassified_evidence_responsive
        return not unclassified_evidence_responsive(
            query_profile.get('retrieval_query') or query_profile.get('normalized_query') or '',
            [c.get('text', '') for c in chunks],
        )
    covered = set().union(*(supported_passage_concepts(c, required) for c in chunks))
    if (required and required.issubset(covered)
            and not query_profile.get("is_regional_wage_query")
            and not scenario_evidence_gap(query_profile.get("retrieval_query") or "", [c.get("text", "") for c in chunks])):
        # Similarity confidence is not a missing legal component. Do not scan
        # every KB category just to improve a label after this question's
        # governing sources and requested points have already been found.
        return False
    top_chunk = max(chunks, key=lambda item: float(item.get("score") or 0.0))
    top_metadata = top_chunk.get("metadata") or {}
    top_is_authoritative = bool(
        float(top_chunk.get("score") or 0.0) >= float(settings.RAG_RELIABLE_SOURCE_THRESHOLD)
        and any(
            str(top_metadata.get(key) or "").strip()
            for key in (
                "law_number",
                "article_number",
                "article_section",
                "section",
                "citation",
                "source_url",
            )
        )
    )
    task = str(query_profile.get("task") or "")
    needs_multi_rule_detail = bool(
        task in {
            "coverage_question",
            "exclusion_question",
            "comparison_question",
            "jurisprudence_request",
        }
        or EXCEPTION_TERMS_RE.search(str(query_profile.get("retrieval_query") or ""))
    )
    # A strong official provision is sufficient for an ordinary exact-rule
    # question. Do not broaden retrieval solely to improve a confidence label.
    if top_is_authoritative and not needs_multi_rule_detail and status == SOURCE_STATUS_DIRECT:
        return False
    # A partial match means the first small search found useful evidence but
    # missed at least one required concept/authority.  Expand once within the
    # same routed categories instead of treating that incomplete set as final.
    if status in {SOURCE_STATUS_INSUFFICIENT, SOURCE_STATUS_PARTIAL}:
        return True
    if query_profile.get("is_regional_wage_query"):
        expected = str(query_profile.get("region_source_url") or "").lower()
        if expected and not any(
            str((chunk.get("metadata") or {}).get("source_url") or "").lower() == expected
            for chunk in chunks
        ):
            return True
    return False


def history_source_chunks(history: List[Dict[str, Any]], topic: str) -> List[Dict[str, Any]]:
    """Rebuild compact chunks from the latest saved assistant source payload."""

    for message in reversed(history or []):
        if message.get("role") != "assistant":
            continue
        if message.get("intent") == "needs_clarification" or str(message.get("source_match") or "").lower() in {"insufficient", "conflicting", "not_required", "provider_unavailable"}:
            return []
        raw_sources = message.get("sources")
        if isinstance(raw_sources, str):
            try:
                raw_sources = json.loads(raw_sources)
            except (TypeError, json.JSONDecodeError):
                raw_sources = []
        if not isinstance(raw_sources, list):
            continue
        chunks: List[Dict[str, Any]] = []
        for index, source in enumerate(raw_sources):
            if not isinstance(source, dict) or not source.get("excerpt"):
                continue
            source_topic = str(source.get("topic") or "")
            if source_topic and source_topic != topic:
                continue
            metadata = {
                key: value
                for key, value in source.items()
                if key not in {"label", "excerpt", "score", "rerank_score"}
            }
            metadata["retrieval_method"] = "conversation_source_reuse"
            chunks.append({
                "id": str(source.get("chunk_id") or f"history-{index}"),
                "text": str(source["excerpt"]),
                "metadata": metadata,
                "score": float(source.get("score") or 0.0),
            })
        if chunks:
            return chunks
    return []


def can_reuse_history_sources(
    query: str,
    query_profile: Dict[str, Any],
    history: List[Dict[str, Any]],
    conversation_state: Optional[Dict[str, Any]] = None,
) -> bool:
    # Historical excerpts may explain conversation context, but are not a
    # substitute for checking the currently searchable registry on each turn.
    return False


def log_retrieval_summary(
    *,
    language: str,
    intent: str,
    plan: Dict[str, Any],
    stages: List[Dict[str, Any]],
    candidates: int,
    deduplicated: int,
    final_chunks: int,
    fallback_used: bool,
) -> None:
    logger.info(
        "[LaborLens Retrieval] language=%s intent=%s categories=%s filters=%s "
        "stages=%s candidates=%s after_dedup=%s fallback=%s llm_context=%s",
        language,
        intent,
        plan.get("expanded_categories") if fallback_used else plan.get("primary_categories"),
        [stage.get("filter") for stage in stages],
        len(stages),
        candidates,
        deduplicated,
        fallback_used,
        final_chunks,
    )
