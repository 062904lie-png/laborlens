"""Lightweight conversation memory for LaborLens.

The legal knowledge base stays separate from this memory. This file only keeps
facts the user already shared so follow-up questions can be understood without
asking the user to repeat themselves.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, Optional

from routers.services.rag_policy import infer_query_sector, infer_topic

CALCULATION_REQUEST_RE = re.compile(
    r"\b(how\s+much|compute|calculate|computation|estimate|amount|pila|magkano)\b",
    re.IGNORECASE,
)

EMPLOYMENT_STATUS_PATTERNS = (
    ("probationary", r"\bprobationary\b"),
    ("regular", r"\b(regular employee|regular worker|regularized|regular status|regular na|regular ako|regular\b(?!\s+(?:day|working\s+day|workday|holiday)))\b"),
    ("contractual", r"\b(contractual|contract worker|kontrata|contract-based)\b"),
    ("project_based", r"\b(project[-\s]?based|project employee)\b"),
    ("part_time", r"\b(part[-\s]?time)\b"),
    ("full_time", r"\b(full[-\s]?time)\b"),
)

REGION_PATTERNS = (
    ("Region I", r"\b(region\s*(?:i|1)|ilocos\s+region)\b"),
    ("Region II", r"\b(region\s*(?:ii|2)|cagayan\s+valley)\b"),
    ("Region III", r"\b(region\s*(?:iii|3)|central\s+luzon)\b"),
    ("Region IV-A", r"\b(region\s*(?:iv[-\s]?a|4[-\s]?a)|calabarzon)\b"),
    ("MIMAROPA", r"\b(region\s*(?:iv[-\s]?b|4[-\s]?b)|mimaropa)\b"),
    ("Region V", r"\b(region\s*(?:v|5)|bicol)\b"),
    ("Region VI", r"\b(region\s*(vi|6)|western visayas)\b"),
    ("NCR", r"\b(ncr|metro manila|national capital region)\b"),
    ("Region VII", r"\b(region\s*(vii|7)|central visayas)\b"),
    ("Region VIII", r"\b(region\s*(?:viii|8)|eastern\s+visayas)\b"),
    ("Region IX", r"\b(region\s*(?:ix|9)|zamboanga\s+peninsula)\b"),
    ("Region X", r"\b(region\s*(?:x|10)|northern\s+mindanao)\b"),
    ("Region XI", r"\b(region\s*(?:xi|11)|davao\s+region)\b"),
    ("Region XII", r"\b(region\s*(?:xii|12)|soccsksargen)\b"),
    ("Region XIII", r"\b(region\s*(?:xiii|13)|caraga)\b"),
    ("CAR", r"\b(car|cordillera|baguio)\b"),
    ("BARMM", r"\b(barmm|bangsamoro)\b"),
)

PROVINCE_CITY_PATTERNS = (
    ("Iloilo", r"\biloilo\b"),
    ("Manila", r"\bmanila\b"),
    ("Cebu", r"\bcebu\b"),
    ("Bacolod", r"\bbacolod\b"),
    ("Baguio", r"\bbaguio\b"),
    ("Pampanga", r"\bpampanga\b"),
    ("Bulacan", r"\bbulacan\b"),
    ("Laguna", r"\blaguna\b"),
    ("Cavite", r"\bcavite\b"),
    ("Batangas", r"\bbatangas\b"),
    ("Davao", r"\bdavao\b"),
)

WORKER_TYPE_PATTERNS = (
    ("domestic_worker", r"\b(kasambahay|domestic\s+worker|househelper|household\s+worker|yaya|maid)\b"),
    ("managerial_employee", r"\b(managerial\s+employee|manager)\b"),
    ("field_personnel", r"\bfield\s+personnel\b"),
    ("rank_and_file", r"\brank[-\s]?and[-\s]?file\b"),
)

INDUSTRY_PATTERNS = (
    ("retail", r"\bretail\b"),
    ("agriculture", r"\b(agriculture|agricultural|plantation|farm)\b"),
    ("non_agriculture", r"\bnon[-\s]?agricultur(?:e|al)\b"),
    ("manufacturing", r"\bmanufactur(?:e|ing)\b"),
    ("hospitality", r"\b(hotel|restaurant|hospitality)\b"),
)

AGENCY_PATTERNS = (
    ("SSS", r"\bsss|social\s+security\s+system\b"),
    ("PhilHealth", r"\bphilhealth\b"),
    ("Pag-IBIG", r"\bpag[-\s]?ibig|hdmf\b"),
    ("DOLE", r"\bdole|department\s+of\s+labor\b"),
)

DAY_TYPE_PATTERNS = (
    ("regular_holiday_rest_day", r"\b(regular holiday|legal holiday)\b.{0,40}\b(rest day|day off)\b|\b(rest day|day off)\b.{0,40}\b(regular holiday|legal holiday)\b"),
    ("special_day_rest_day", r"\b(special non[-\s]?working day|special day|special holiday)\b.{0,40}\b(rest day|day off)\b|\b(rest day|day off)\b.{0,40}\b(special non[-\s]?working day|special day|special holiday)\b"),
    ("ordinary_day", r"\b(ordinary (?:working )?day|regular (?:working )?day|regular workday|normal day|normal workday)\b"),
    ("rest_day", r"\b(rest day|day off)\b"),
    ("special_non_working_day", r"\b(special non[-\s]?working day|special day|special holiday)\b"),
    ("regular_holiday", r"\b(regular holiday|legal holiday)\b"),
)


def parse_state(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def state_json(state: Dict[str, Any]) -> str:
    return json.dumps(state or {}, ensure_ascii=False, sort_keys=True)


def _decimal_from_text(value: str) -> Optional[str]:
    clean = re.sub(r"[^0-9.]", "", value or "")
    if not clean:
        return None
    try:
        return str(Decimal(clean))
    except InvalidOperation:
        return None


def _first_match(text: str, patterns: Iterable[tuple[str, str]]) -> Optional[str]:
    for value, pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return value
    return None


def extract_money_facts(text: str) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    lowered = text or ""

    hourly = re.search(
        r"\b(?:hourly\s+rate|rate\s+is|per\s+hour)\s*(?:is|=|:)?\s*(?:php|p|₱)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        lowered,
        re.IGNORECASE,
    )
    if not hourly:
        hourly = re.search(
            r"\b(?:hourly\s+(?:rate|wage)|hour\s+(?:rate|wage)|per\s+hour\s+wage|"
            r"rate\s+per\s+hour|timely\s+wage|time\s+wage)\s*"
            r"(?:is|=|:)?\s*(?:php|p|\u20b1)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            lowered,
            re.IGNORECASE,
        )
    if not hourly:
        hourly = re.search(
            r"(?:php|p|\u20b1)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
            r"(?:per\s+hour|/hour|an\s+hour|hourly|kada\s+oras|kada\s+hour)",
            lowered,
            re.IGNORECASE,
        )
    if not hourly:
        hourly = re.search(
            r"(?:php|p|\u20b1)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
            r"(?:pay|wage|rate)?\s*"
            r"(?:for\s+(?:a\s+)?(?:regular|ordinary|normal)\s+(?:working\s+)?day\s*)?"
            r"(?:per\s+hour|/hour|an\s+hour|hourly|kada\s+oras|kada\s+hour)",
            lowered,
            re.IGNORECASE,
        )
    if not hourly:
        hourly = re.search(
            r"\b(?:regular\s+pay|ordinary\s+pay|normal\s+pay|pay)\s*"
            r"(?:is|=|:)?\s*(?:php|p|\u20b1)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
            r"(?:per\s+hour|/hour|an\s+hour|hourly)\b",
            lowered,
            re.IGNORECASE,
        )
    if not hourly:
        hourly = re.search(
            r"\b(?:sweldo|suhol|sahod|wage|pay)\s*(?:ko|akon|nako|is|ay|amo\s+ang)?\s*"
            r"(?:kada\s+(?:oras|hour)|per\s+hour|hourly)\s*"
            r"(?:is|ay|amo\s+ang|=|:)?\s*(?:php|p|\u20b1)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            lowered,
            re.IGNORECASE,
        )
    if (
        not hourly
        and re.search(r"\b(overtime|ot|extra\s+hours?|how\s+much)\b", lowered, re.IGNORECASE)
        and not re.search(r"\b(daily\s+(?:wage|rate)|per\s+day|day\s+rate)\b", lowered, re.IGNORECASE)
    ):
        hourly = re.search(
            r"\b(?:my\s+)?wage\s*(?:is|=|:)?\s*(?:php|p|\u20b1)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            lowered,
            re.IGNORECASE,
        )
    if hourly:
        value = _decimal_from_text(hourly.group(1))
        if value:
            facts["hourly_rate"] = value

    total_basic = re.search(
        r"\b(?:total\s+basic\s+salary(?:\s+earned)?|basic\s+salary\s+earned|total\s+earned\s+basic\s+salary)\b"
        r".{0,40}?(?:php|p|\u20b1)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)(?![0-9,]*(?:st|nd|rd|th)\b)",
        lowered,
        re.IGNORECASE,
    )
    if not total_basic:
        total_basic = re.search(
            r"(?:php|p|\u20b1)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)(?![0-9,]*(?:st|nd|rd|th)\b)\s+"
            r"(?:in\s+)?(?:total\s+basic\s+salary|basic\s+salary)",
            lowered,
            re.IGNORECASE,
        )
    if total_basic:
        value = _decimal_from_text(total_basic.group(1))
        if value:
            facts["total_basic_salary_earned"] = value

    daily = re.search(
        r"\b(?:daily\s+wage|daily\s+rate|per\s+day|day\s+rate)\s*(?:is|=|:)?\s*(?:php|p|₱)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        lowered,
        re.IGNORECASE,
    )
    if not daily:
        daily = re.search(
            r"(?:php|p|₱)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:per\s+day|a\s+day|daily)\b",
            lowered,
            re.IGNORECASE,
        )
    if daily:
        value = _decimal_from_text(daily.group(1))
        if value:
            facts["daily_wage"] = value

    normal_hours = re.search(
        r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:normal|regular|standard)\s*"
        r"(?:work(?:ing)?\s*)?(?:hours?|hrs?|shift)\b"
        r"|\b(?:normal|regular|standard)\s*(?:work(?:ing)?\s*)?"
        r"(?:hours?|hrs?|shift)\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\b"
        r"|\b(?:standard\s+shift|normal\s+shift)\s*(?:is|=|:)?\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:hours?|hrs?)\b",
        lowered,
        re.IGNORECASE,
    )
    if not normal_hours:
        normal_hours = re.search(
            r"\b(?:i\s+)?work\s+([0-9]+(?:\.[0-9]+)?)\s*(?:hours?|hrs?)\s+(?:a|per)\s+day\b",
            lowered,
            re.IGNORECASE,
        )
    if normal_hours:
        value = next((group for group in normal_hours.groups() if group), None)
        if value:
            facts["normal_work_hours"] = value

    monthly = re.search(
        r"\b(?:monthly\s+salary|basic\s+salary|salary)\s*(?:is|=|:)?\s*(?:php|p|₱)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        lowered,
        re.IGNORECASE,
    )
    if monthly:
        value = _decimal_from_text(monthly.group(1))
        if value:
            facts["monthly_salary"] = value

    return facts


def extract_hours(text: str) -> Dict[str, Any]:
    from routers.services.numeric_quantities import explicit_overtime_hours, hour_quantities, worked_total_hours
    hours = explicit_overtime_hours(text)
    facts = {"overtime_hours": hours} if hours is not None else {}
    # Keep total time and the separately stated overtime duration. The legal
    # threshold still comes from retrieved law when the user does not state it.
    total = worked_total_hours(text)
    # In phrases such as "I worked two hours and thirty minutes of overtime",
    # the quantity parser also recognizes a past-tense work verb. Do not save
    # that same duration a second time as the user's total hours worked.
    if total is not None and total != hours:
        facts["hours_worked"] = total
    elif hours is None and re.search(r"\b(?:i\s+)?(?:have\s+)?work(?:ed|ing)?\b", text, re.I):
        quantities = list(hour_quantities(text))
        if len(quantities) == 1:
            facts["hours_worked"] = quantities[0]["value"]
    return facts


def infer_issue(text: str, nlp: Optional[Dict[str, Any]] = None) -> Optional[str]:
    combined = f"{text or ''} {(nlp or {}).get('normalized', '')}"
    topic = infer_topic(combined)
    from routers.services.numeric_quantities import hour_quantities
    if (topic == "general_labor" and CALCULATION_REQUEST_RE.search(combined)
            and list(hour_quantities(combined))
            and re.search(r"\b(?:work(?:ed|ing)?|shift|hours?)\b", combined, re.I)):
        topic = "overtime_pay"
    if topic == "overtime_pay":
        if re.search(r"\b(unpaid|not\s+paid|wala\s+bayad|wala\s+ginbayad|kulang|underpaid)\b", combined, re.IGNORECASE):
            return "unpaid_overtime"
        return "overtime_pay"
    if topic in {"13th_month_pay", "wages", "leave_benefits", "termination", "domestic_workers", "occupational_safety"}:
        return topic
    return None


def extract_user_facts(
    message: str,
    nlp: Optional[Dict[str, Any]] = None,
    preferred_language: str = "",
) -> Dict[str, Any]:
    text = message or ""
    facts: Dict[str, Any] = {}

    from .question_understanding import employment_fact_text
    employment_text = employment_fact_text(text)
    employment_status = _first_match(employment_text, EMPLOYMENT_STATUS_PATTERNS)
    if employment_status:
        facts["employment_status"] = employment_status

    region = _first_match(text, REGION_PATTERNS)
    if region:
        facts["region"] = region

    place = _first_match(text, PROVINCE_CITY_PATTERNS)
    if place:
        facts["location"] = place
        location_regions = {
            "Iloilo": "Region VI",
            "Manila": "NCR",
            "Cebu": "Region VII",
            "Bacolod": "Region VI",
            "Baguio": "CAR",
            "Pampanga": "Region III",
            "Bulacan": "Region III",
            "Laguna": "Region IV-A",
            "Cavite": "Region IV-A",
            "Batangas": "Region IV-A",
            "Davao": "Region XI",
        }
        facts.setdefault("region", location_regions.get(place))

    day_type = _first_match(text, DAY_TYPE_PATTERNS)
    if day_type:
        facts["day_type"] = day_type

    worker_type = _first_match(employment_text, WORKER_TYPE_PATTERNS)
    if worker_type:
        facts["worker_type"] = worker_type

    industry = _first_match(text, INDUSTRY_PATTERNS)
    if industry:
        facts["industry"] = industry

    establishment_size = re.search(
        r"\b(?:with|has|employs?|employing)\s+(\d+)\s+(?:employees?|workers?)\b|"
        r"\b(\d+)\s+(?:employees?|workers?)\s+(?:in|at|for)\s+(?:the|my|our)\s+(?:shop|business|establishment|company)\b",
        text,
        re.IGNORECASE,
    )
    if establishment_size:
        facts["establishment_size"] = next(group for group in establishment_size.groups() if group)

    agencies = [name for name, pattern in AGENCY_PATTERNS if re.search(pattern, text, re.IGNORECASE)]
    if agencies:
        facts["requested_agencies"] = agencies
        facts["requested_benefit"] = "social_benefits"

    facts.update(extract_money_facts(text))
    facts.update(extract_hours(text))
    from routers.services.question_understanding import stated_service_facts, stated_separation_reason
    facts.update(stated_service_facts(text))
    reason = stated_separation_reason(text)
    if reason:
        facts["separation_reason"] = reason

    issue = infer_issue(text, nlp)
    if issue:
        facts["issue"] = issue
    if issue == "overtime_pay" and CALCULATION_REQUEST_RE.search(text):
        facts["calculation_topic"] = "overtime_pay"

    sector = infer_query_sector(employment_text)
    if sector:
        facts["sector"] = sector

    if preferred_language:
        facts["preferred_language"] = preferred_language

    return facts


def merge_conversation_state(
    existing_state: Dict[str, Any],
    message: str,
    nlp: Optional[Dict[str, Any]] = None,
    preferred_language: str = "",
) -> Dict[str, Any]:
    state = dict(existing_state or {})
    from .question_understanding import employment_context_parts, employment_fact_text
    positive = employment_fact_text(message)
    _, denied = employment_context_parts(message)
    # Retract a denied prior role/status; silence on a short follow-up preserves it.
    for field, patterns in (('employment_status', EMPLOYMENT_STATUS_PATTERNS),
                            ('worker_type', WORKER_TYPE_PATTERNS)):
        if state.get(field) and any(name == state[field] and re.search(pattern, denied, re.I)
                                    for name, pattern in patterns):
            state.pop(field, None)
    if denied:
        from .rag_policy import infer_sector
        if state.get('sector') == infer_sector(denied):
            state.pop('sector', None)
    facts = extract_user_facts(message, nlp, preferred_language)
    for key, value in facts.items():
        if value not in ("", None, [], {}):
            if key == "issue" and state.get("issue") == "unpaid_overtime" and value == "overtime_pay":
                continue
            if key == "sector" and value == "general_private_employment" and state.get("sector") not in {
                "", None, "general_private_employment", "unknown"
            }:
                # A short benefit follow-up often contains no worker label. Do
                # not erase an established special sector such as kasambahay.
                if not re.search(r'\bprivate\b', positive, re.I):
                    continue
            state[key] = value
    if "region" in facts and "location" not in facts:
        # A new explicit region supersedes an older city from a prior wage
        # question (for example Manila -> Region III).
        state.pop("location", None)
    return state


def state_context_text(state: Dict[str, Any]) -> str:
    if not state:
        return ""
    labels = {
        "employment_status": "employment status",
        "sector": "sector",
        "worker_type": "worker type",
        "region": "region",
        "location": "location",
        "daily_wage": "daily wage",
        "hourly_rate": "hourly rate",
        "normal_work_hours": "normal work hours",
        "monthly_salary": "monthly salary",
        "years_of_service": "years of service",
        "service_duration": "stated service duration",
        "separation_reason": "stated separation reason",
        "overtime_hours": "overtime hours",
        "hours_worked": "total hours worked",
        "day_type": "day type",
        "industry": "industry",
        "establishment_size": "establishment size",
        "requested_benefit": "requested benefit",
        "requested_agencies": "requested agencies",
        "last_topic": "last topic",
        "calculation_topic": "calculation topic",
        "issue": "main issue",
        "preferred_language": "preferred language",
    }
    parts = []
    for key, label in labels.items():
        value = state.get(key)
        if value not in ("", None):
            parts.append(f"{label}: {value}")
    return "; ".join(parts)


def build_standalone_search_query(
    current_message: str,
    base_search_text: str,
    history_text: str,
    state: Dict[str, Any],
    summary: str = "",
) -> str:
    parts = [base_search_text or current_message]
    state_text = state_context_text(state)
    if state_text:
        parts.append(f"Known user facts: {state_text}")
    if summary:
        parts.append(f"Conversation summary: {summary[:500]}")
    if history_text:
        parts.append(f"Recent conversation: {history_text[:900]}")
    return ". ".join(part.strip() for part in parts if part and part.strip())[:1800]


def title_from_state_or_message(message: str, state: Dict[str, Any]) -> str:
    issue = str((state or {}).get("issue") or "").replace("_", " ").strip()
    if issue:
        return issue.title()[:60]
    words = re.findall(r"[A-Za-z0-9]+", message or "")
    title = " ".join(words[:8]).strip()
    return title[:60] or "New Consultation"
