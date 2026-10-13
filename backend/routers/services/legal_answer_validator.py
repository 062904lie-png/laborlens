"""Legal-answer validation and deterministic high-risk answer helpers."""

from __future__ import annotations

import re
import hashlib
from routers.services.document_evidence import is_document_driven
from typing import Any, Dict, List, Optional, Sequence, Set
from routers.services.performance import measure
from routers.services.evidence_limits import unsupported_scenario_claims

from routers.services.rag_policy import (
    DOMESTIC_SECTOR,
    GENERAL_SECTOR,
    GOVERNMENT_SECTOR,
    HOSPITAL_SECTOR,
    MANAGERIAL_SECTOR,
    MANAGERIAL_STAFF_SECTOR,
    FIELD_SECTOR,
    SOURCE_STATUS_DIRECT,
    SOURCE_STATUS_INSUFFICIENT,
    SOURCE_STATUS_NOT_REQUIRED,
    SOURCE_STATUS_PARTIAL,
    canonical_sector,
    is_domestic_worker_overtime_query,
    missing_required_authorities,
    sector_matches,
)
from routers.services.legal_topic_matrix import (
    CONCEPT_PATTERNS,
    canonical_topic,
    collect_supported_concepts,
    essential_concepts_for_topic,
    detect_supported_concepts,
    missing_required_concepts,
    required_concepts_for_topic,
    source_passage_text,
    supported_passage_concepts,
)


OVERTIME_REQUIRED_ARTICLES = {"82", "84", "87", "93", "94"}
OVERTIME_OPTIONAL_ARTICLES = {"88"}

FORBIDDEN_OVERTIME_PATTERNS = (
    re.compile(r"\bconfidential\s+employees?\b.{0,80}\b(excluded|exempt|not\s+covered)\b", re.IGNORECASE),
    re.compile(r"\ball\s+supervisors?\b.{0,80}\b(excluded|exempt|not\s+covered)\b", re.IGNORECASE),
    re.compile(r"\bunauthori[sz]ed\s+overtime\b.{0,100}\b(does\s+not|doesn't|no\s+right|not\s+compensable|unpaid)\b", re.IGNORECASE),
    re.compile(r"\b(written\s+)?authorization\b.{0,80}\b(always|required|must\s+be|required\s+before)\b", re.IGNORECASE),
    re.compile(r"\bholiday\s+overtime\b.{0,80}\b(130\s*%|1\.30|php\s*130|p\s*130|₱\s*130)\b", re.IGNORECASE),
    re.compile(r"\brest[-\s]?day\s+overtime\b.{0,80}\b(130\s*%|1\.30|php\s*130|p\s*130|₱\s*130)\b", re.IGNORECASE),
)

FORBIDDEN_KASAMBAHAY_OVERTIME_PATTERNS = (
    re.compile(r"\b(domestic\s+workers?|kasambahay|kasambhay|house\s*helpers?|househelpers?|household\s+workers?|yaya|maid|housemaid)\b.{0,180}\b(125\s*%|169\s*%|195\s*%|260\s*%|338\s*%)", re.IGNORECASE),
    re.compile(r"\b(domestic\s+workers?|kasambahay|kasambhay|house\s*helpers?|househelpers?|household\s+workers?|yaya|maid|housemaid)\b.{0,220}\bordinary\s+working\s+day\s+overtime\b", re.IGNORECASE),
    re.compile(r"\b(domestic\s+workers?|kasambahay|kasambhay|house\s*helpers?|househelpers?|household\s+workers?|yaya|maid|housemaid)\b.{0,220}\bovertime\s+multiplier\b", re.IGNORECASE),
    re.compile(r"\bordinary\s+working\s+day\b.{0,120}\brest\s+day\b.{0,120}\bspecial\s+non[-\s]?working\b", re.IGNORECASE),
)

UNSUPPORTED_PLACEHOLDER_RE = re.compile(
    r"\b(source\s+unknown|citation\s+needed|source\s+needed)\b|\(source\)",
    re.IGNORECASE,
)
MENTIONED_ARTICLE_RE = re.compile(r"\b(?:art\.?|article)\s*(\d{1,3})\b", re.IGNORECASE)
MENTIONED_RA_RE = re.compile(
    r"\b(?:r\.?\s*a\.?|ra|republic\s+act(?:\s+no\.?)?)\s*(\d{4,6})\b",
    re.IGNORECASE,
)
MENTIONED_DO_RE = re.compile(
    r"\b(?:d\.?\s*o\.?|department\s+order)(?:\s+no\.?)?\s*([A-Z0-9]+(?:[-–][A-Z0-9]+)+)\b",
    re.IGNORECASE,
)
MENTIONED_WAGE_ORDER_RE = re.compile(
    r"\b(?:wage\s+order)(?:\s+no\.?)?\s*([A-Z0-9]+(?:[-–][A-Z0-9]+)+)\b",
    re.IGNORECASE,
)
MENTIONED_GR_RE = re.compile(
    # A G.R. citation must contain a numeric docket identifier.  Requiring
    # ``No.`` (or ``Nos.``) and at least four digits prevents ordinary words
    # such as "grounds", "granted", and "grievance" from being parsed as
    # fabricated Supreme Court case numbers.
    r"\bG\.?\s*R\.?\s+(?:Nos?\.?\s*)?(\d{4,8}(?:[-–]\d{1,8})?)\b",
    re.IGNORECASE,
)


def source_text(source: Dict[str, Any]) -> str:
    return " ".join(
        str(source.get(key) or "")
        for key in (
            "label",
            "source",
            "document_title",
            "source_name",
            "law_number",
            "article_section",
            "citation",
            "article_number",
            "article_title",
            "book",
            "title",
            "chapter",
            "rule",
            "section",
            "rule_title",
            "section_title",
            "excerpt",
            "source_url",
            "official_url",
            "canonical_document_id",
            "legal_status",
            "content_hash",
            "category",
            "doc_type",
            "issuing_agency",
            "effective_date",
            "date_issued",
            "decision_date",
            "case_name",
            "gr_number",
            "holding",
            "ruling",
            "topic",
            "sector",
            "concepts",
        )
    )


def _normalized_text_hash(value: str) -> str:
    normalized = re.sub(r"\s+", " ", (value or "").strip().lower())
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def article_sources(sources: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    found: Dict[str, Dict[str, Any]] = {}
    for source in sources or []:
        blob = source_text(source)
        article_number = str(source.get("article_number") or "").strip()
        if article_number:
            found.setdefault(article_number, source)
        for match in re.finditer(r"\b(?:art\.?|article)\s*(82|84|87|88|89|93|94)\b", blob, re.IGNORECASE):
            found.setdefault(match.group(1), source)
    return found


def source_label(source: Optional[Dict[str, Any]]) -> str:
    if not source:
        return ""
    label = str(source.get("label") or "").strip()
    if label:
        return label
    return ""


def cited_source_numbers(answer: str) -> Set[int]:
    return {
        int(match.group(1))
        for match in re.finditer(r"\bSource\s+(\d+)\b", answer or "", re.IGNORECASE)
        if match.group(1).isdigit()
    }


def answer_has_raw_filename(answer: str) -> bool:
    visible_text = answer or ""
    # Official source URLs commonly end in .pdf. They are allowed and are
    # separately checked against retrieved source metadata. Only inspect text
    # visible to the user so a valid link target is not mistaken for an
    # uploaded filename.
    visible_text = re.sub(
        r"\]\(\s*https?://[^)\s]+(?:\s+['\"][^'\"]*['\"])?\s*\)",
        "]",
        visible_text,
        flags=re.IGNORECASE,
    )
    visible_text = re.sub(r"<https?://[^>]+>", "", visible_text, flags=re.IGNORECASE)
    visible_text = re.sub(r"https?://[^\s)\]]+", "", visible_text, flags=re.IGNORECASE)
    return bool(
        re.search(
            r"\b[\w .()-]+\.(?:pdf|docx?|png|jpe?g|tiff?|bmp|webp)\b",
            visible_text,
            re.IGNORECASE,
        )
    )


def normalize_known_source_filenames(
    answer: str,
    sources: Sequence[Dict[str, Any]],
) -> str:
    """Replace displayed upload filenames with clean source titles.

    URLs are masked during replacement so an official link ending in `.pdf`
    remains byte-for-byte unchanged. Unknown filenames are intentionally left
    alone for ``answer_has_raw_filename`` to reject.
    """

    if not answer or not sources:
        return answer

    masked_urls: List[str] = []

    def mask_url(match: re.Match[str]) -> str:
        masked_urls.append(match.group(0))
        return f"\ufff0URL{len(masked_urls) - 1}\ufff1"

    normalized = re.sub(r"https?://[^\s)\]]+", mask_url, answer, flags=re.IGNORECASE)
    for source in sources:
        clean_title = str(
            source.get("document_title")
            or source.get("source_name")
            or "Official source"
        ).strip()
        clean_title = re.sub(
            r"\.(?:pdf|docx?|png|jpe?g|tiff?|bmp|webp)$",
            "",
            clean_title,
            flags=re.IGNORECASE,
        ).strip() or "Official source"
        for key in ("source", "filename", "original_filename"):
            raw_name = str(source.get(key) or "").strip()
            if not re.search(
                r"\.(?:pdf|docx?|png|jpe?g|tiff?|bmp|webp)$",
                raw_name,
                re.IGNORECASE,
            ):
                continue
            normalized = re.sub(
                re.escape(raw_name),
                lambda _match: clean_title,
                normalized,
                flags=re.IGNORECASE,
            )

    for index, url in enumerate(masked_urls):
        normalized = normalized.replace(f"\ufff0URL{index}\ufff1", url)
    return normalized


def validate_source_ids(answer: str, sources: Sequence[Dict[str, Any]]) -> List[str]:
    max_source = len(sources or [])
    errors = []
    for number in cited_source_numbers(answer):
        if number < 1 or number > max_source:
            errors.append(f"Source {number} was cited but only {max_source} source(s) exist.")
    return errors


def validate_metadata_urls(sources: Sequence[Dict[str, Any]]) -> List[str]:
    errors = []
    for source in sources or []:
        url = str(source.get("source_url") or "").strip()
        if url and not re.match(r"^https?://", url, re.IGNORECASE):
            errors.append(f"{source.get('label', 'A source')} has an invalid source URL.")
    return errors


def validate_answer_urls(answer: str, sources: Sequence[Dict[str, Any]]) -> List[str]:
    allowed_urls = {
        str(source.get("source_url") or "").strip()
        for source in sources or []
        if str(source.get("source_url") or "").strip()
    }
    errors = []
    for url in re.findall(r"https?://[^\s)\]]+", answer or "", re.IGNORECASE):
        clean_url = url.rstrip(".,;")
        if clean_url not in allowed_urls:
            errors.append("The answer contains a URL that was not provided by source metadata.")
    return errors


def validate_mentioned_authorities(answer: str, sources: Sequence[Dict[str, Any]]) -> List[str]:
    """Make sure authorities named in an answer are backed by displayed sources."""

    from .citation_identity import validate_circular_identity
    errors: List[str] = validate_circular_identity(answer, sources)
    source_blob = " ".join(source_text(source) for source in sources or [])
    for article in sorted(set(MENTIONED_ARTICLE_RE.findall(answer or "")), key=lambda item: int(item)):
        if not any(_source_has_article(source, article) for source in sources or []):
            errors.append(f"Article {article} is mentioned but is not present in the displayed Legal Basis/source list.")

    for law_number in sorted(set(MENTIONED_RA_RE.findall(answer or ""))):
        if not re.search(rf"\b(?:ra|r\.?\s*a\.?|republic\s+act(?:\s+no\.?)?)\s*{re.escape(law_number)}\b", source_blob, re.IGNORECASE):
            errors.append(f"RA {law_number} is mentioned but is not present in the displayed Legal Basis/source list.")
    for label, regex in (
        ("Department Order", MENTIONED_DO_RE),
        ("Wage Order", MENTIONED_WAGE_ORDER_RE),
        ("G.R. No.", MENTIONED_GR_RE),
    ):
        for identifier in sorted(set(regex.findall(answer or ""))):
            normalized_identifier = re.sub(r"[^a-z0-9]", "", identifier.lower())
            normalized_sources = re.sub(r"[^a-z0-9]", "", source_blob.lower())
            if normalized_identifier not in normalized_sources:
                errors.append(
                    f"{label} {identifier} is mentioned but is not present in the displayed Legal Basis/source list."
                )
    return errors


def _citation_validation_messages(answer: str, sources: Sequence[Dict[str, Any]]) -> List[str]:
    """Return unsupported authority citations separately from core errors.

    A stray, optional citation must not erase an otherwise supported answer;
    callers expose these messages as warnings and can rewrite the citation.
    Essential concept/authority checks still determine answer eligibility.
    """
    with measure("citation_validation_ms"):
        return validate_mentioned_authorities(answer, sources)


def _unsupported_citation_claim_errors(answer: str, messages: Sequence[str]) -> List[str]:
    """Reject unsupported affirmative claims, but allow explicit disclaimers.

    A sentence such as "Article 100 is not the overtime rule" can safely be
    rewritten without erasing an otherwise supported answer. In contrast,
    "Article 82 excludes ..." attributes a legal proposition to an authority
    absent from the displayed sources and must fail validation.
    """

    errors: List[str] = []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer or "")
    for message in messages or []:
        identifier = str(message).split(" is mentioned", 1)[0].strip()
        if not identifier:
            errors.append(str(message))
            continue
        relevant = [
            sentence
            for sentence in sentences
            if re.search(re.escape(identifier), sentence, re.IGNORECASE)
        ]
        disclaimed = bool(relevant) and all(
            re.search(
                r"\b(?:is|are|was|were|does|do|did|can|could|should|would)\s+not\b|"
                r"\bnot\s+(?:the|an|a|applicable|controlling|relevant|supporting)\b",
                sentence,
                re.IGNORECASE,
            )
            for sentence in relevant
        )
        if not disclaimed:
            errors.append(str(message))
    return errors


def remove_unsupported_authority_citations(answer: str, sources: Sequence[Dict[str, Any]]) -> str:
    """Rewrite unsupported authority mentions instead of returning a bad citation.

    The model's surrounding explanation can remain useful when one optional
    citation is wrong.  Replace only the unverified identifier with a neutral
    phrase; the final validator still reports it in ``unsupported_citations``.
    """
    if not answer:
        return answer
    source_blob = " ".join(source_text(source) for source in sources or [])
    patterns = (MENTIONED_ARTICLE_RE, MENTIONED_RA_RE, MENTIONED_DO_RE, MENTIONED_WAGE_ORDER_RE, MENTIONED_GR_RE)
    result = answer
    for regex in patterns:
        def replace(match: re.Match[str]) -> str:
            identifier = match.group(1)
            if regex is MENTIONED_ARTICLE_RE:
                present = bool(re.search(rf"\b(?:art\.?|article)\s*{re.escape(identifier)}\b", source_blob, re.IGNORECASE))
            else:
                present = re.sub(r"[^a-z0-9]", "", identifier.lower()) in re.sub(r"[^a-z0-9]", "", source_blob.lower())
            return match.group(0) if present else "the applicable legal provision"
        result = regex.sub(replace, result)
    return result


def validate_citation_hierarchy(sources: Sequence[Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []
    for source in sources or []:
        blob = source_text(source).lower()
        if "omnibus" not in blob:
            continue
        missing = []
        if not str(source.get("document_title") or source.get("source") or "").strip():
            missing.append("document title")
        if not str(source.get("book") or "").strip():
            missing.append("book")
        if not str(source.get("rule") or "").strip():
            missing.append("rule")
        if not str(source.get("section") or source.get("article_section") or "").strip():
            missing.append("section")
        if not str(source.get("page") or source.get("page_number") or "").strip():
            missing.append("page number")
        if not str(source.get("source_url") or "").strip():
            missing.append("official URL")
        if missing:
            warnings.append(
                f"{source.get('label', 'An Omnibus Rules source')} has incomplete citation hierarchy: "
                + ", ".join(missing)
                + "."
            )
    return warnings


def validate_jurisprudence_evidence(sources: Sequence[Dict[str, Any]]) -> List[str]:
    """Require an actual decided case, not a statute-only substitute."""

    case_sources = [
        source
        for source in sources or []
        if str(source.get("category") or "").strip().lower() == "jurisprudence"
        or re.search(r"\b(?:supreme\s+court|g\.?\s*r\.?\s*no\.?)\b", source_text(source), re.IGNORECASE)
    ]
    if not case_sources:
        return ["A jurisprudence request requires an actual case source, not only a statute or general guide."]

    for source in case_sources:
        blob = source_text(source)
        has_case_name = bool(
            str(source.get("case_name") or source.get("document_title") or source.get("source") or "").strip()
        )
        has_gr = bool(str(source.get("gr_number") or "").strip()) or bool(MENTIONED_GR_RE.search(blob))
        has_date = bool(str(source.get("decision_date") or source.get("date_issued") or "").strip()) or bool(
            re.search(r"\b(?:19|20)\d{2}\b", blob)
        )
        has_ruling = bool(str(source.get("holding") or source.get("ruling") or "").strip()) or bool(
            re.search(r"\b(?:held|ruled|ruling|decision|dismissal|liability|entitled)\b", blob, re.IGNORECASE)
        )
        if has_case_name and has_gr and has_date and has_ruling:
            return []
    return ["The retrieved case source is missing a case name, G.R. number, decision date, or ruling excerpt."]


def _overtime_allowed_articles_for_task(task: str) -> Set[str]:
    task = str(task or "").strip()
    if task in {"coverage_question", "eligibility_question"}:
        return {"82", "87"}
    if task == "exclusion_question":
        return {"82"}
    if task == "authorization_question":
        return {"84"}
    if task == "undertime_offset_question":
        return {"88"}
    if task == "compulsory_overtime_question":
        return {"89"}
    if task == "threshold_question":
        return {"84", "87"}
    if task in {"rate_question", "formula_question", "calculation_request", "personalized_computation"}:
        return {"84", "87", "93", "94"}
    return {"82", "84", "87", "88", "93", "94"}


def _source_article_number(source: Dict[str, Any]) -> str:
    article_number = str(source.get("article_number") or "").strip()
    if article_number:
        return article_number
    match = re.search(r"\b(?:art\.?|article)\s*(82|84|87|88|89|93|94)\b", source_text(source), re.IGNORECASE)
    return match.group(1) if match else ""


def filter_display_sources_for_answer(
    answer: str,
    sources: Sequence[Dict[str, Any]],
    query_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Remove sources that do not support any displayed claim.

    This intentionally starts conservative. The common noisy case is Article 88:
    it is useful only when the answer actually mentions undertime offsets.
    """

    filtered: List[Dict[str, Any]] = []
    answer_text = answer or ""
    query_text = str((query_profile or {}).get("normalized_query") or "")
    topic = canonical_topic((query_profile or {}).get("topic"))
    task = str((query_profile or {}).get("task") or "")
    query_sector = canonical_sector(str((query_profile or {}).get("sector") or GENERAL_SECTOR))
    document_driven = is_document_driven(query_profile)
    allowed_overtime_articles = _overtime_allowed_articles_for_task(task) if topic == "overtime_pay" else set()
    domestic_overtime = bool(query_profile and is_domestic_worker_overtime_query(query_profile))
    mentions_undertime = bool(
        re.search(r"\bundertime\b|\boffset\b", answer_text, re.IGNORECASE)
        or re.search(r"\bundertime\b|\boffset\b", query_text, re.IGNORECASE)
    )
    cited_numbers = cited_source_numbers(answer_text)
    from routers.services.document_evidence import evidence_origin
    from routers.services.source_integrity import source_passage_identity
    supporting_origins = set((query_profile or {}).get('supporting_source_origins') or [])
    checked_support = document_driven and 'supporting_source_origins' in (query_profile or {})
    seen_sources: Dict[str, Dict[str, Any]] = {}
    for source in sources or []:
        status = str(
            source.get("legal_status")
            or source.get("version_status")
            or source.get("status")
            or ""
        ).strip().lower()
        if status and status not in {"active", "current", "approved", "indexed"}:
            continue
        source_sector = _source_sector(source)
        restrict_sector = not document_driven or (query_profile or {}).get('sector_restricted', True)
        if restrict_sector:
            if document_driven:
                from .evidence_scope import scope_compatible, source_scope
                if not scope_compatible(query_sector, source_scope(source)):
                    continue
            elif query_sector == GENERAL_SECTOR and source_sector != GENERAL_SECTOR:
                continue
            elif query_sector != GENERAL_SECTOR and source_sector and not sector_matches(query_sector, source_sector):
                continue
        if checked_support and evidence_origin(source) not in supporting_origins:
            continue
        if cited_numbers:
            original_label = str(source.get("label") or "")
            label_match = re.search(r"\bSource\s+(\d+)\b", original_label, re.IGNORECASE)
            if label_match and int(label_match.group(1)) not in cited_numbers and evidence_origin(source) not in supporting_origins:
                continue
        article_number = _source_article_number(source)
        blob = source_text(source)
        if domestic_overtime and not document_driven:
            if _source_has_canonical(source, "ra_10361", "ra_10361_irr") or article_number == "82":
                pass
            else:
                continue
        if not document_driven and topic == "overtime_pay" and article_number and article_number not in allowed_overtime_articles:
            continue
        if not document_driven and topic == "overtime_pay" and not article_number:
            if domestic_overtime and _source_has_canonical(source, "ra_10361", "ra_10361_irr"):
                pass
            else:
                supported_for_task = detect_supported_concepts("overtime_pay", source, task=task)
                required_for_task = set(required_concepts_for_topic("overtime_pay", query_text, task=task))
                if required_for_task and not required_for_task.intersection(supported_for_task):
                    continue
        is_article_88 = article_number == "88" or (not article_number and bool(
            re.search(r"\b(?:art\.?|article)\s*88\b", blob, re.IGNORECASE)
        ))
        if not document_driven and is_article_88 and not mentions_undertime:
            continue
        source_key = source_passage_identity(source.get('excerpt') or blob, source, evidence_origin(source))
        aliases = list(source.get('original_labels') or [])
        for label in (source.get('original_label'), source.get('label')):
            if label and label not in aliases:
                aliases.append(label)
        if source_key in seen_sources:
            existing = seen_sources[source_key]
            existing['original_labels'] = list(dict.fromkeys(existing['original_labels'] + aliases))
            continue
        if len(filtered) >= 6 and not document_driven:
            continue
        filtered_source = dict(source)
        filtered_source.setdefault("original_label", filtered_source.get("label"))
        filtered_source['original_labels'] = aliases
        seen_sources[source_key] = filtered_source
        filtered.append(filtered_source)
    for index, source in enumerate(filtered, 1):
        source["label"] = f"Source {index}"
    return filtered


def remap_answer_source_citations(answer: str, sources: Sequence[Dict[str, Any]]) -> str:
    """Keep displayed source citations aligned after source filtering."""

    if not answer or not sources:
        return answer
    label_map: Dict[int, int] = {}
    for source in sources or []:
        new_label = str(source.get("label") or "")
        new_match = re.search(r"\bSource\s+(\d+)\b", new_label, re.IGNORECASE)
        old_labels = source.get('original_labels') or [source.get('original_label') or source.get('label') or '']
        for old_label in old_labels:
            old_match = re.search(r"\bSource\s+(\d+)\b", str(old_label), re.IGNORECASE)
            if old_match and new_match:
                label_map[int(old_match.group(1))] = int(new_match.group(1))
    if not label_map:
        return answer
    def replace_source(match: re.Match[str]) -> str:
        original = int(match.group(1))
        if original in label_map:
            return f"Source {label_map[original]}"
        # Never redirect a removed citation to a different source that now
        # occupies its old number. Source 0 fails subsequent ID validation.
        return "Source 0"

    return re.sub(r"\bSource\s+(\d+)\b", replace_source, answer, flags=re.IGNORECASE)


def validate_overtime_answer(
    answer: str,
    sources: Sequence[Dict[str, Any]],
    query_profile: Dict[str, Any],
) -> Dict[str, Any]:
    if query_profile.get("topic") != "overtime_pay":
        return {
            "source_match": query_profile.get("source_match") or SOURCE_STATUS_PARTIAL,
            "errors": [],
            "missing_articles": [],
        }

    errors = validate_source_ids(answer, sources)
    citation_warnings = _citation_validation_messages(answer, sources)
    errors.extend(_unsupported_citation_claim_errors(answer, citation_warnings))
    errors.extend(validate_metadata_urls(sources))
    errors.extend(validate_answer_urls(answer, sources))
    hierarchy_warnings = validate_citation_hierarchy(sources)
    if answer_has_raw_filename(answer):
        errors.append("The answer shows a raw uploaded filename.")
    if UNSUPPORTED_PLACEHOLDER_RE.search(answer or ""):
        errors.append("The answer contains an unsupported citation placeholder.")
    for pattern in FORBIDDEN_OVERTIME_PATTERNS:
        if pattern.search(answer or ""):
            errors.append("The answer contains an unsupported overtime exclusion, authorization, or rate claim.")
            break

    if query_profile.get("sector") != GENERAL_SECTOR:
        scope = validate_source_scope(sources, query_profile)
        errors.extend(scope["errors"])
        consistency = validate_sector_consistency(answer, sources, query_profile)
        errors.extend(consistency["errors"])
        return {
            "source_match": SOURCE_STATUS_INSUFFICIENT if errors else SOURCE_STATUS_DIRECT,
            "errors": errors,
            "missing_articles": [],
            "warnings": scope["warnings"] + consistency["warnings"] + hierarchy_warnings + citation_warnings,
            "unsupported_citations": citation_warnings,
        }

    found = article_sources(sources)
    task = str(query_profile.get("task") or "")
    required_articles = _overtime_allowed_articles_for_task(task)
    if task in {"rate_question", "formula_question", "calculation_request", "personalized_computation"}:
        required_articles = {"87"}
        query_blob = str(query_profile.get("normalized_query") or "")
        if re.search(r"\b(rest\s+day|special\s+day|special\s+non[-\s]?working)\b", query_blob, re.IGNORECASE):
            required_articles.add("93")
        if re.search(r"\b(regular\s+holiday|holiday)\b", query_blob, re.IGNORECASE):
            required_articles.add("94")
    missing = sorted(article for article in required_articles if article not in found)
    query_text = str(query_profile.get("normalized_query") or "")
    missing_concepts = missing_required_concepts("overtime_pay", sources, query_text, task=task)
    supported_concepts = sorted(collect_supported_concepts("overtime_pay", sources, query_text, task=task))
    essential_concepts = list(
        query_profile.get("essential_concepts")
        or essential_concepts_for_topic("overtime_pay", query_text, task=task)
    )
    essential_supported = any(concept in supported_concepts for concept in essential_concepts) if essential_concepts else bool(supported_concepts)
    has_ordinary_overtime_support = "87" in found or any(
        "ordinary_overtime_rule" in detect_supported_concepts("overtime_pay", source, task=task)
        for source in sources or []
    )
    missing_rate_markers: List[str] = []
    if task in {"general_explanation", "rate_question", "calculation_request", "personalized_computation"}:
        rate_markers = ["125%"]
        if task in {"general_explanation", "calculation_request", "personalized_computation"}:
            rate_markers.extend(["169%", "195%", "260%", "338%"])
        for marker in rate_markers:
            if marker not in (answer or ""):
                missing_rate_markers.append(marker)

    if errors:
        status = SOURCE_STATUS_INSUFFICIENT
    elif not essential_supported:
        if supported_concepts:
            status = SOURCE_STATUS_PARTIAL
        else:
            status = SOURCE_STATUS_INSUFFICIENT
            errors.append("No retrieved source supports the essential overtime concept for this task.")
    elif task in {"general_explanation", "rate_question", "formula_question", "calculation_request", "personalized_computation"} and not has_ordinary_overtime_support:
        status = SOURCE_STATUS_INSUFFICIENT
        errors.append("No retrieved source supports the ordinary overtime rule.")
    elif missing or missing_concepts or missing_rate_markers:
        status = SOURCE_STATUS_PARTIAL
    elif hierarchy_warnings:
        status = SOURCE_STATUS_PARTIAL
    else:
        status = SOURCE_STATUS_DIRECT
    return {
        "source_match": status,
        "errors": errors,
        "missing_articles": missing,
        "warnings": hierarchy_warnings + citation_warnings,
        "unsupported_citations": citation_warnings,
        "missing_concepts": missing_concepts,
        "supported_concepts": supported_concepts,
        "missing_rate_markers": missing_rate_markers,
    }


def _source_sector(source: Dict[str, Any]) -> str:
    return canonical_sector(str(source.get("sector") or source.get("source_sector") or ""))


def validate_source_scope(
    sources: Sequence[Dict[str, Any]],
    query_profile: Dict[str, Any],
) -> Dict[str, List[str]]:
    query_sector = canonical_sector(str(query_profile.get("sector") or GENERAL_SECTOR))
    errors: List[str] = []
    warnings: List[str] = []
    if not sources:
        return {"errors": errors, "warnings": warnings}

    if is_document_driven(query_profile) and not query_profile.get('sector_restricted', True):
        from .evidence_scope import scope_completeness
        state = scope_completeness(query_profile.get('retrieval_query') or query_profile.get('normalized_query') or '', sources)
        if state['scope_gap']:
            warnings.append('The retrieved evidence does not establish the requested worker scope; special-sector passages cannot substitute for a general rule.')
        elif any(_source_sector(s) != GENERAL_SECTOR for s in sources):
            warnings.append('The question does not specify a sector; retain each source\'s applicability and do not generalize special-sector rules.')
        return {"errors": errors, "warnings": warnings}

    if is_document_driven(query_profile):
        from .evidence_scope import scope_completeness
        state = scope_completeness(query_profile.get('retrieval_query') or query_profile.get('normalized_query') or '', sources)
        if state['scope_gap']:
            warnings.append('The retrieved evidence does not establish the requested worker scope.')
        return {"errors": errors, "warnings": warnings}

    matching = [source for source in sources if sector_matches(query_sector, _source_sector(source))]
    if query_sector != GENERAL_SECTOR and not matching:
        errors.append("Retrieved sources do not match the requested employment sector.")

    if query_sector == GENERAL_SECTOR:
        special_sources = [
            source
            for source in sources
            if _source_sector(source) != GENERAL_SECTOR
            or str(source.get("rule_type") or "").strip() == "special_sector_rule"
        ]
        if special_sources and len(special_sources) == len(sources):
            errors.append("Only special-sector sources were retrieved for a general private-employment question.")
        elif special_sources:
            warnings.append("A special-sector source was retrieved and should not be generalized.")
    return {"errors": errors, "warnings": warnings}


def validate_source_status(sources: Sequence[Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []
    for source in sources or []:
        status = str(
            source.get("legal_status")
            or source.get("version_status")
            or source.get("status")
            or ""
        ).strip().lower()
        if status and status not in {"active", "current", "approved", "indexed"}:
            warnings.append(f"{source.get('label', 'A source')} is marked as {status}.")
    return warnings


def _source_has_article(source: Dict[str, Any], article: str) -> bool:
    article_number = str(source.get("article_number") or "").strip()
    if article_number == article:
        return True
    return bool(re.search(rf"\b(?:art\.?|article)\s*{re.escape(article)}\b", source_text(source), re.IGNORECASE))


def _source_has_canonical(source: Dict[str, Any], *canonical_ids: str) -> bool:
    canonical = str(source.get("canonical_document_id") or "").strip().lower()
    if canonical and canonical in {item.lower() for item in canonical_ids}:
        return True
    blob = source_text(source).lower()
    if "ra_10361" in canonical_ids and ("ra 10361" in blob or "republic act no. 10361" in blob or "batas kasambahay" in blob):
        return True
    if "ra_10361_irr" in canonical_ids and "10361" in blob and "irr" in blob:
        return True
    if "pd_851" in canonical_ids and ("pd 851" in blob or "presidential decree no. 851" in blob):
        return True
    return False


def _standard_validation_errors(answer: str, sources: Sequence[Dict[str, Any]]) -> tuple[List[str], List[str]]:
    errors = validate_source_ids(answer, sources)
    errors.extend(validate_metadata_urls(sources))
    errors.extend(validate_answer_urls(answer, sources))
    warnings = validate_source_status(sources)
    warnings.extend(validate_citation_hierarchy(sources))
    warnings.extend(_citation_validation_messages(answer, sources))
    if answer_has_raw_filename(answer):
        errors.append("The answer shows a raw uploaded filename.")
    if UNSUPPORTED_PLACEHOLDER_RE.search(answer or ""):
        errors.append("The answer contains an unsupported citation placeholder.")
    return errors, warnings


def _concept_supported(concept: str, source: Dict[str, Any]) -> bool:
    return concept in supported_passage_concepts(source, [concept])


def _supported_required_concepts(required: Sequence[str], sources: Sequence[Dict[str, Any]]) -> Set[str]:
    supported: Set[str] = set()
    for source in sources or []:
        for concept in required:
            if _concept_supported(concept, source):
                supported.add(concept)
    return supported


REGIONAL_WAGE_RATE_RE = re.compile(
    r"(?:₱|php|p)?\s*\d{3}(?:,\d{3})?(?:\.\d{2})?\b",
    re.IGNORECASE,
)
REGIONAL_WAGE_SECTOR_RE = re.compile(
    r"\b(non[-\s]?agricultur(?:e|al)|agricultur(?:e|al)|industrial|commercial|"
    r"retail|service|establishment|employ(?:ing|ment)|workers?)\b",
    re.IGNORECASE,
)
REGIONAL_WAGE_AUTHORITY_RE = re.compile(
    r"\b(dole|department\s+of\s+labor|rtwpb|regional\s+tripartite|"
    r"wage\s+order|rb(?:ncr|[ivx]+)[-\s]?(?:dw[-\s]?)?\d+)\b",
    re.IGNORECASE,
)
REGIONAL_WAGE_LOCATION_RE = re.compile(
    r"\b(region|ncr|car|barmm|manila|iloilo|cebu|davao|pampanga|bulacan|"
    r"laguna|cavite|batangas|province|city|municipalit(?:y|ies))\b",
    re.IGNORECASE,
)


def supported_regional_wage_table_concepts(
    sources: Sequence[Dict[str, Any]],
) -> Set[str]:
    """Recognize reliable wage-table evidence without requiring prose sentences.

    The exception is intentionally narrow: a source must be an official-looking
    DOLE/RTWPB or Wage Order source and must contain a location, a worker/sector
    classification, and a numeric rate. An effective date is credited only when
    the source actually supplies one in text or metadata.
    """

    supported: Set[str] = set()
    for source in sources or []:
        blob = source_text(source)
        category = str(source.get("category") or "").strip().lower().replace("_", " ")
        doc_type = str(source.get("doc_type") or "").strip().lower().replace("_", " ")
        official_wage_source = (
            category == "dole resources"
            or doc_type == "wage order"
            or bool(REGIONAL_WAGE_AUTHORITY_RE.search(blob))
        )
        if not official_wage_source:
            continue
        if not (
            REGIONAL_WAGE_AUTHORITY_RE.search(blob)
            and REGIONAL_WAGE_LOCATION_RE.search(blob)
            and REGIONAL_WAGE_SECTOR_RE.search(blob)
            and REGIONAL_WAGE_RATE_RE.search(blob)
        ):
            continue

        supported.update({"wage_order_or_region", "covered_workers"})
        if str(source.get("effective_date") or "").strip() or re.search(
            r"\b(effective\s+date|effectivity|effective\s+(?:on\s+)?\d{1,2}|"
            r"takes\s+effect)\b",
            blob,
            re.IGNORECASE,
        ):
            supported.add("effective_date")
    return supported


def validate_sector_consistency(
    answer: str,
    sources: Sequence[Dict[str, Any]],
    query_profile: Dict[str, Any],
) -> Dict[str, List[str]]:
    sector = canonical_sector(str(query_profile.get("sector") or GENERAL_SECTOR))
    topic = canonical_topic(query_profile.get("topic"))
    errors: List[str] = []
    warnings: List[str] = []

    if sector == DOMESTIC_SECTOR and topic == "overtime_pay":
        if any(pattern.search(answer or "") for pattern in FORBIDDEN_KASAMBAHAY_OVERTIME_PATTERNS):
            errors.append("The answer applies the general Labor Code overtime-rate table to a domestic-worker question.")
        has_ra10361 = any(_source_has_canonical(source, "ra_10361", "ra_10361_irr") for source in sources or [])
        has_article82 = any(_source_has_article(source, "82") for source in sources or [])
        if not has_ra10361:
            errors.append("No retrieved source supports RA 10361 or its IRR for the domestic-worker overtime question.")
        if not has_article82:
            warnings.append("Labor Code Article 82 was not retrieved to verify the domestic-helper hours-of-work exclusion.")
        if sources and all(
            "labor code" in source_text(source).lower()
            and re.search(r"\b(househelper|household\s+service|domestic\s+service)\b", source_text(source), re.IGNORECASE)
            and not _source_has_canonical(source, "ra_10361", "ra_10361_irr")
            for source in sources
        ):
            errors.append("Only old Labor Code househelper/domestic-service provisions were retrieved for a domestic-worker question.")

    if sector in {GOVERNMENT_SECTOR, MANAGERIAL_SECTOR, MANAGERIAL_STAFF_SECTOR, FIELD_SECTOR} and topic == "overtime_pay":
        if re.search(r"\b(125\s*%|169\s*%|195\s*%|260\s*%|338\s*%)\b", answer or "", re.IGNORECASE):
            warnings.append("A special-sector overtime question should first verify coverage before applying ordinary overtime rates.")

    if sector == HOSPITAL_SECTOR and topic == "overtime_pay":
        if "hospital" not in (answer or "").lower() and "clinic" not in (answer or "").lower():
            warnings.append("A hospital/clinic personnel question should acknowledge the special hospital personnel rule when relevant.")

    return {"errors": errors, "warnings": warnings}


def validate_holiday_pay_answer(
    answer: str,
    sources: Sequence[Dict[str, Any]],
    query_profile: Dict[str, Any],
) -> Dict[str, Any]:
    errors, warnings = _standard_validation_errors(answer, sources)
    scope = validate_source_scope(sources, query_profile)
    errors.extend(scope["errors"])
    warnings.extend(scope["warnings"])

    answer_text = answer or ""
    query_text = str(query_profile.get("normalized_query") or "")
    asks_regular_holiday = bool(re.search(r"\b(regular\s+holiday|holiday\s+pay|double\s+pay)\b", query_text, re.IGNORECASE))
    has_article_94 = any(_source_has_article(source, "94") for source in sources or [])
    has_article_93 = any(_source_has_article(source, "93") for source in sources or [])
    has_omnibus_holiday = any(
        "omnibus" in source_text(source).lower() and "holiday" in source_text(source).lower()
        for source in sources or []
    )

    forbidden_patterns = (
        re.compile(r"\bmust\s+work\b.{0,80}\bregular\s+holiday\b.{0,80}\b(receive|get|entitled)\b", re.IGNORECASE),
        re.compile(r"\bregular\s+holiday\b.{0,80}\bno\s+work[, ]+no\s+pay\b", re.IGNORECASE),
        re.compile(r"\bemployer'?s?\s+discretion\b.{0,80}\bholiday\s+pay\b", re.IGNORECASE),
    )
    if any(pattern.search(answer_text) for pattern in forbidden_patterns):
        errors.append("The answer contains an unsupported regular-holiday pay limitation.")

    if asks_regular_holiday and not (has_article_94 or has_omnibus_holiday):
        errors.append("No retrieved source supports Article 94 or an Omnibus regular-holiday pay rule.")
    if asks_regular_holiday and has_article_93 and not has_article_94:
        warnings.append("Article 93 may support rest-day or special-day pay but is not the main source for regular holiday pay.")

    query = str(query_profile.get("normalized_query") or "")
    task = str(query_profile.get("task") or "")
    missing = missing_required_concepts("holiday_pay", sources, query, task=task)
    supported = sorted(collect_supported_concepts("holiday_pay", sources, query, task=task))
    if not sources:
        errors.append("No source metadata was available for validation.")
    if errors:
        status = SOURCE_STATUS_INSUFFICIENT
    elif missing or warnings:
        status = SOURCE_STATUS_PARTIAL
    else:
        status = SOURCE_STATUS_DIRECT if query_profile.get("source_match") == SOURCE_STATUS_DIRECT else SOURCE_STATUS_PARTIAL
    return {
        "source_match": status,
        "errors": errors,
        "warnings": warnings,
        "missing_articles": [] if has_article_94 else ["94"],
        "missing_concepts": missing,
        "supported_concepts": supported,
        "required_concepts": required_concepts_for_topic("holiday_pay", query, task=task),
        "essential_concepts": essential_concepts_for_topic("holiday_pay", query, task=task),
    }


def validate_thirteenth_month_answer(
    answer: str,
    sources: Sequence[Dict[str, Any]],
    query_profile: Dict[str, Any],
) -> Dict[str, Any]:
    errors, warnings = _standard_validation_errors(answer, sources)
    scope = validate_source_scope(sources, query_profile)
    errors.extend(scope["errors"])
    warnings.extend(scope["warnings"])
    answer_text = answer or ""
    if re.search(r"\ball\s+regular\s+rank[-\s]?and[-\s]?file\s+(?:workers?|employees?)\b", answer_text, re.IGNORECASE):
        errors.append("The answer incorrectly limits 13th-month coverage wording to regular rank-and-file workers.")

    if re.search(r"(?:p|php|peso(?:s)?)\s*1[, ]?000", answer_text, re.IGNORECASE) and re.search(
        r"\b(ceiling|limit|not\s+more\s+than|only|entitled)\b",
        answer_text,
        re.IGNORECASE,
    ) and not re.search(r"\b(historical|original|old|superseded|no\s+longer)\b", answer_text, re.IGNORECASE):
        errors.append("The answer appears to present the old 13th-month salary ceiling as a current rule.")

    has_foundational_source = any(
        _source_has_canonical(source, "pd_851")
        or "13th" in source_text(source).lower()
        or "thirteenth" in source_text(source).lower()
        for source in sources or []
    )
    query = str(query_profile.get("normalized_query") or "")
    task = str(query_profile.get("task") or "")
    missing = missing_required_concepts("thirteenth_month_pay", sources, query, task=task)
    supported = sorted(collect_supported_concepts("thirteenth_month_pay", sources, query, task=task))

    if not sources:
        errors.append("No source metadata was available for validation.")
    if not has_foundational_source:
        errors.append("No retrieved source supports the 13th-month pay law or current guidance.")

    if errors:
        status = SOURCE_STATUS_INSUFFICIENT
    elif missing or warnings:
        status = SOURCE_STATUS_PARTIAL
    else:
        status = SOURCE_STATUS_DIRECT if query_profile.get("source_match") == SOURCE_STATUS_DIRECT else SOURCE_STATUS_PARTIAL
    return {
        "source_match": status,
        "errors": errors,
        "warnings": warnings,
        "missing_articles": [],
        "missing_concepts": missing,
        "supported_concepts": supported,
        "required_concepts": required_concepts_for_topic("thirteenth_month_pay", query, task=task),
        "essential_concepts": essential_concepts_for_topic("thirteenth_month_pay", query, task=task),
    }


def validate_kasambahay_answer(
    answer: str,
    sources: Sequence[Dict[str, Any]],
    query_profile: Dict[str, Any],
) -> Dict[str, Any]:
    errors, warnings = _standard_validation_errors(answer, sources)
    scope = validate_source_scope(sources, query_profile)
    errors.extend(scope["errors"])
    warnings.extend(scope["warnings"])
    consistency = validate_sector_consistency(answer, sources, query_profile)
    errors.extend(consistency["errors"])
    warnings.extend(consistency["warnings"])

    has_ra10361 = any(_source_has_canonical(source, "ra_10361", "ra_10361_irr") for source in sources or [])
    has_old_househelper_only = bool(sources) and all(
        "labor code" in source_text(source).lower()
        and re.search(r"\b(househelper|household\s+service|domestic\s+service)\b", source_text(source), re.IGNORECASE)
        and not _source_has_canonical(source, "ra_10361", "ra_10361_irr")
        for source in sources or []
    )
    if not has_ra10361:
        errors.append("No retrieved source supports RA 10361 or its IRR for the Kasambahay question.")
    if has_old_househelper_only:
        errors.append("Only old Labor Code househelper/domestic-service provisions were retrieved for a Kasambahay Act question.")
    if re.search(r"\b(fifteen|15)\s+days?'?\s+wages?\b|\bindemnity\b", answer or "", re.IGNORECASE):
        query_text = str(query_profile.get("normalized_query") or "")
        source_blob = " ".join(source_text(source) for source in sources or [])
        if not re.search(r"\b(termination|unjust|dismiss|pre[-\s]?termination)\b", f"{query_text} {source_blob}", re.IGNORECASE):
            errors.append("The answer mentions fifteen days' wages or indemnity without a clear termination context and supporting source.")

    query = str(query_profile.get("normalized_query") or "")
    task = str(query_profile.get("task") or "")
    required = list(query_profile.get("required_concepts") or required_concepts_for_topic("domestic_workers", query, task=task))
    essential = list(query_profile.get("essential_concepts") or essential_concepts_for_topic("domestic_workers", query, task=task))
    if query_profile.get("required_concepts"):
        supported_set = _supported_required_concepts(required, sources)
        missing = [concept for concept in required if concept not in supported_set]
        supported = sorted(supported_set)
    else:
        missing = missing_required_concepts("domestic_workers", sources, query, task=task)
        supported = sorted(collect_supported_concepts("domestic_workers", sources, query, task=task))
    essential_supported = any(concept in supported for concept in essential) if essential else bool(supported)

    if not sources:
        errors.append("No source metadata was available for validation.")
    if errors:
        status = SOURCE_STATUS_INSUFFICIENT
    elif required and not essential_supported:
        if supported:
            status = SOURCE_STATUS_PARTIAL
        else:
            status = SOURCE_STATUS_INSUFFICIENT
            errors.append("No source supports the essential domestic-worker legal concept for this question.")
    elif missing or warnings:
        status = SOURCE_STATUS_PARTIAL
    else:
        status = SOURCE_STATUS_DIRECT if query_profile.get("source_match") == SOURCE_STATUS_DIRECT else SOURCE_STATUS_PARTIAL
    return {
        "source_match": status,
        "errors": errors,
        "warnings": warnings,
        "missing_articles": [],
        "missing_concepts": missing,
        "supported_concepts": supported,
        "required_concepts": required,
        "essential_concepts": essential,
    }


def _validate_legal_answer_impl(
    answer: str,
    sources: Sequence[Dict[str, Any]],
    query_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate generated legal answers beyond overtime-specific checks.

    The validator is deliberately conservative. It does not decide the legal
    conclusion by itself; it checks whether cited source metadata is internally
    valid and whether retrieved sources cover the topic checklist.
    """

    if is_document_driven(query_profile):
        # New documents do not need a registered topic, fixed rate, article
        # allowlist or authority ID to support an answer. Provenance and scope
        # remain mandatory, and independent verification checks the prose.
        from routers.services.document_evidence import assessment_coverage
        coverage = assessment_coverage(query_profile.get('document_evidence'), list(sources),
                                       query_profile.get('retrieval_query') or query_profile.get('normalized_query') or '')
        errors = validate_source_ids(answer, sources) + validate_metadata_urls(sources) + validate_answer_urls(answer, sources)
        citation_warnings = _citation_validation_messages(answer, sources)
        warnings = citation_warnings + validate_source_status(sources) + validate_citation_hierarchy(sources)
        # Missing book/rule metadata is a citation-quality warning, not proof
        # that a checked substantive claim is false. Unsupported affirmative
        # authority claims remain fatal, as do invalid source IDs and URLs.
        errors.extend(_unsupported_citation_claim_errors(answer, citation_warnings))
        scope = validate_source_scope(sources, query_profile)
        errors.extend(scope['errors'])
        warnings.extend(scope['warnings'])
        final_checked = bool(query_profile.get('final_grounding_checked'))
        if not final_checked and query_profile.get('document_evidence') and (not coverage or not coverage['supported_points']):
            errors.append('Question-derived quotations did not survive source selection or prompt truncation.')
        if not sources:
            errors.append('No source metadata was available for validation.')
        if coverage is None and not final_checked:
            warnings.append('Structured question evidence assessment was unavailable; candidate relevance alone is not verification.')
        if answer_has_raw_filename(answer):
            errors.append('The answer shows a raw uploaded filename.')
        return {'source_match': SOURCE_STATUS_INSUFFICIENT if errors else SOURCE_STATUS_PARTIAL,
                'errors': errors, 'warnings': warnings, 'required_concepts': [],
                'missing_concepts': [], 'supported_concepts': [], 'document_evidence_coverage': coverage}

    topic = canonical_topic(query_profile.get("topic"))
    if is_domestic_worker_overtime_query(query_profile):
        return validate_kasambahay_answer(answer, sources, query_profile)
    if topic == "overtime_pay":
        validation = validate_overtime_answer(answer, sources, query_profile)
        validation.setdefault("missing_concepts", [])
        validation.setdefault("supported_concepts", [])
        validation.setdefault("warnings", [])
        return validation
    if topic == "holiday_pay":
        return validate_holiday_pay_answer(answer, sources, query_profile)
    if topic == "thirteenth_month_pay":
        return validate_thirteenth_month_answer(answer, sources, query_profile)
    if topic == "domestic_workers":
        return validate_kasambahay_answer(answer, sources, query_profile)

    if topic == "non_legal" or query_profile.get("intent") in {
        "greeting",
        "small_talk",
        "identity",
        "capability_question",
        "gratitude",
        "farewell",
    }:
        return {
            "source_match": SOURCE_STATUS_NOT_REQUIRED,
            "errors": [],
            "warnings": [],
            "missing_articles": [],
            "missing_concepts": [],
            "supported_concepts": [],
        }

    errors = validate_source_ids(answer, sources)
    citation_warnings = _citation_validation_messages(answer, sources)
    errors.extend(_unsupported_citation_claim_errors(answer, citation_warnings))
    errors.extend(validate_metadata_urls(sources))
    errors.extend(validate_answer_urls(answer, sources))
    warnings = validate_source_status(sources)
    hierarchy_warnings = validate_citation_hierarchy(sources)
    if answer_has_raw_filename(answer):
        errors.append("The answer shows a raw uploaded filename.")

    scope = validate_source_scope(sources, query_profile)
    errors.extend(scope["errors"])
    warnings.extend(scope["warnings"])

    query_text = str(query_profile.get("normalized_query") or "")
    task = str(query_profile.get("task") or "")
    required = list(query_profile.get("required_concepts") or required_concepts_for_topic(topic, query_text, task=task))
    essential = list(query_profile.get("essential_concepts") or essential_concepts_for_topic(topic, query_text, task=task))
    supported_set = _supported_required_concepts(required, sources)
    if topic == "minimum_wage":
        supported_set.update(supported_regional_wage_table_concepts(sources))
    supported = sorted(supported_set)
    missing = [concept for concept in required if concept not in supported_set]
    missing_authorities = missing_required_authorities(list(sources or []), query_profile)
    essential_supported = any(concept in supported for concept in essential) if essential else bool(supported)

    if topic == "jurisprudence_request":
        errors.extend(validate_jurisprudence_evidence(sources))

    if not sources:
        status = SOURCE_STATUS_INSUFFICIENT
        errors.append("No source metadata was available for validation.")
    elif errors:
        status = SOURCE_STATUS_INSUFFICIENT
    elif required and not essential_supported:
        if supported:
            status = SOURCE_STATUS_PARTIAL
        else:
            status = SOURCE_STATUS_INSUFFICIENT
            errors.append("No source supports the essential legal concept for this topic.")
    elif missing or missing_authorities:
        status = SOURCE_STATUS_PARTIAL
    elif hierarchy_warnings:
        status = SOURCE_STATUS_PARTIAL
    else:
        base_status = query_profile.get("source_match")
        status = SOURCE_STATUS_DIRECT if base_status == SOURCE_STATUS_DIRECT else SOURCE_STATUS_PARTIAL
        if not required and base_status == SOURCE_STATUS_DIRECT:
            status = SOURCE_STATUS_DIRECT

    return {
        "source_match": status,
        "errors": errors,
        "warnings": warnings + hierarchy_warnings + citation_warnings,
        "unsupported_citations": citation_warnings,
        "missing_articles": [],
        "missing_concepts": missing,
        "missing_authorities": missing_authorities,
        "supported_concepts": supported,
        "required_concepts": required,
        "essential_concepts": essential,
    }


def validate_legal_answer(
    answer: str,
    sources: Sequence[Dict[str, Any]],
    query_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate generated legal prose against the supplied source evidence."""

    with measure("validation_ms"):
        result = _validate_legal_answer_impl(answer, sources, query_profile)
        if is_document_driven(query_profile):
            # Keep universal citation/scope and high-risk claim guards, but do
            # not reinterpret a dynamic question with an old topic checklist.
            query_profile = {**query_profile, 'topic': 'general_labor',
                             'required_concepts': [], 'essential_concepts': [],
                             'task': 'document_evidence_answer'}
        # Specialized validators must still honor the question's complete
        # component list, using the excerpts actually sent by the provider.
        required = list(query_profile.get("required_concepts") or result.get('required_concepts') or
                        required_concepts_for_topic(query_profile.get('topic'), query_profile.get('normalized_query') or '', query_profile.get('task') or ''))
        supported = _supported_required_concepts(required, sources)
        if query_profile.get('topic') == 'minimum_wage':
            # Preserve the existing verified fragmented-table parser. Plain
            # prose patterns cannot reliably read PDF wage-table columns.
            supported.update(supported_regional_wage_table_concepts(sources))
        missing = [c for c in required if c not in supported]
        from routers.services.issue_evidence import incomplete_contract, unclassified_evidence_responsive
        passages = [source_passage_text(s.get('excerpt') or s.get('text') or '') for s in sources or []]
        query = query_profile.get('retrieval_query') or query_profile.get('normalized_query') or ''
        if not is_document_driven(query_profile) and incomplete_contract({**query_profile, 'required_concepts': required}) and result.get('source_match') != SOURCE_STATUS_NOT_REQUIRED:
            responsive = unclassified_evidence_responsive(query, passages)
            # Never override an existing scope/citation error with a lexical match.
            result['source_match'] = SOURCE_STATUS_PARTIAL if responsive and not result.get('errors') else SOURCE_STATUS_INSUFFICIENT
            result.setdefault('warnings', []).append('The question-specific evidence contract is unresolved; direct verification is not established.')
            if not responsive:
                result.setdefault('errors', []).append('No operative passage answers the issue asked about.')
        # Source coverage and answer coverage are separate. A source may contain
        # all needed rules while the generated prose answers a different issue.
        answer_concepts = _supported_required_concepts(required, [{'excerpt': answer}])
        if query_profile.get('topic') == 'minimum_wage':
            answer_concepts.update(supported_regional_wage_table_concepts([{'excerpt': answer}]))
        # Legacy topic contracts include optional background rules not asked
        # about (e.g. payment deadline in a formula question). Require the core
        # task concepts, or every explicitly extracted issue-specific point.
        answer_required = required if query_profile.get('requested_issues') else list(
            query_profile.get('essential_concepts') or essential_concepts_for_topic(
                query_profile.get('topic'), query_profile.get('normalized_query') or '', query_profile.get('task') or ''))
        omitted = [c for c in answer_required if c != 'legal_basis' and c not in answer_concepts]
        result['missing_answer_concepts'] = omitted
        if omitted and result.get('source_match') == SOURCE_STATUS_DIRECT:
            result['source_match'] = SOURCE_STATUS_PARTIAL
        if required and not (answer_concepts - {'legal_basis'}) and query_profile.get('answer_language', 'en') == 'en' and not re.search(
            r'cannot\s+(?:confirm|verify)|insufficient|not\s+(?:available|supported|verified)|'
            r'do\s+not\s+have\s+enough|sources?\s+(?:do\s+not|lack)', answer or '', re.I,
        ):
            result.setdefault('errors', []).append('The answer does not address a requested legal component.')
            result['source_match'] = SOURCE_STATUS_INSUFFICIENT
        elif omitted and query_profile.get('answer_language', 'en') != 'en':
            # Passage checks are English-source checks. Do not confuse an
            # untranslated lexical matcher with proof a Filipino/Hiligaynon
            # explanation is unrelated. Retain the conservative partial label.
            result.setdefault('warnings', []).append('Translated answer coverage is not fully verified by the source-language matcher.')
        result["missing_concepts"] = list(dict.fromkeys(list(result.get("missing_concepts") or []) + missing))
        if missing and result.get("source_match") == SOURCE_STATUS_DIRECT:
            result["source_match"] = SOURCE_STATUS_PARTIAL
        from routers.services.kb_calculation_support import grounded_holiday_calculation, calculation_answer_errors
        numeric_check = None if is_document_driven(query_profile) else grounded_holiday_calculation(
            query_profile.get("retrieval_query") or query_profile.get("normalized_query") or "", sources,
        )
        errors = unsupported_scenario_claims(
            answer, "\n".join(passages),
            query_profile.get("retrieval_query") or query_profile.get("normalized_query") or "",
        )
        from routers.services.evidence_limits import unsupported_safety_claims, unsupported_rule_quantities
        errors.extend(unsupported_safety_claims(answer, '\n'.join(passages), query))
        errors.extend(unsupported_rule_quantities(answer, '\n'.join(passages)))
        errors.extend(calculation_answer_errors(answer, numeric_check or query_profile.get("verified_calculation")))
        # Only check relational patterns here. A bare topic word such as
        # "leave" is not a legal claim, and an explicit evidence limitation
        # must not be mistaken for an assertion of entitlement.
        sentences = re.split(r"(?<=[.!?])\s+|\n+", answer or "")
        for concept in missing:
            if concept == "legal_basis":
                continue
            patterns = [p for p in CONCEPT_PATTERNS.get(concept, ()) if ".{0," in p]
            if any(
                not re.search(r"\b(?:not\s+(?:supported|verified|established|available|confirmed)|"
                              r"do(?:es)?\s+not\s+(?:establish|specify|confirm|cover)|"
                              r"cannot\s+(?:confirm|verify|establish)|insufficient\s+(?:source|evidence)|"
                              r"sources?\s+(?:do\s+not|lack|are\s+missing))\b", sentence, re.I)
                and any(re.search(p, sentence, re.I) for p in patterns)
                for sentence in sentences
            ):
                errors.append("A claimed legal component lacks supporting passage evidence: " + concept)
        from routers.services.scenario_retrieval import scenario_evidence_gap
        gap = None if is_document_driven(query_profile) else scenario_evidence_gap(
            query_profile.get("retrieval_query") or query_profile.get("normalized_query") or "",
            [str(s.get("excerpt") or s.get("text") or "") for s in sources or []],
        )
        if gap:
            errors.append(gap)
        if errors:
            result["errors"] = list(result.get("errors") or []) + errors
            result["source_match"] = SOURCE_STATUS_INSUFFICIENT
        return result
