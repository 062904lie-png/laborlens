"""Shared knowledge base category taxonomy for LaborLens RAG."""

import re
from typing import Optional


OCCUPATIONAL_SAFETY_CATEGORY = "Occupational Safety & Health"

KB_CATEGORIES = [
    "Core Labor Laws",
    "Employment Benefits",
    "Leave Benefits",
    "Domestic Workers",
    "Anti-Harassment / Discrimination",
    "Social Benefits",
    OCCUPATIONAL_SAFETY_CATEGORY,
    "DOLE Resources",
    "Labor Relations",
    "Jurisprudence",
    "Labor Advisories",
    "Others",
    "Civil Service / Government Employment",
]

LEGACY_KB_CATEGORY_MAP = {
    "Domestic Workers / Kasambahay": "Domestic Workers",
    "Labor Code (PD 442)": "Core Labor Laws",
    "Kasambahay Act (RA 10361)": "Domestic Workers",
    "OSH Law (RA 11058)": OCCUPATIONAL_SAFETY_CATEGORY,
    "Expanded Maternity Leave (RA 11210)": "Leave Benefits",
    "Paternity Leave Act (RA 8187)": "Leave Benefits",
    "Solo Parents Act (RA 11861)": "Leave Benefits",
    "Magna Carta of Women (RA 9710)": "Anti-Harassment / Discrimination",
    "Safe Spaces Act (RA 11313)": "Anti-Harassment / Discrimination",
    "Retirement Pay Law (RA 7641)": "Employment Benefits",
    "SSS Laws": "Social Benefits",
    "PhilHealth Laws": "Social Benefits",
    "Pag-IBIG Laws": "Social Benefits",
    "DOLE Department Orders": "DOLE Resources",
    "Wage Orders": "Employment Benefits",
    "Supreme Court Labor Cases": "Jurisprudence",
}

DEFAULT_KB_CATEGORY = KB_CATEGORIES[0]

TOPIC_KB_CATEGORY_ROUTES = {
    "approved_leave_recall": (["Leave Benefits", "Core Labor Laws"], ["Labor Relations", "Jurisprudence", "DOLE Resources", "Labor Advisories"]),
    "minimum_wage": (["DOLE Resources", "Employment Benefits"], []),
    "cost_of_living_allowance": (["Employment Benefits", "DOLE Resources"], []),
    "payment_of_wages": (["Employment Benefits", "Core Labor Laws"], ["DOLE Resources"]),
    "wage_deductions": (["Core Labor Laws", "Employment Benefits"], []),
    "overtime_pay": (["Core Labor Laws", "Employment Benefits"], []),
    "holiday_pay": (["Employment Benefits", "Core Labor Laws"], ["Labor Advisories"]),
    "premium_pay": (["Employment Benefits", "Core Labor Laws"], ["Labor Advisories"]),
    "night_shift_differential": (["Core Labor Laws", "Employment Benefits"], []),
    "hours_of_work": (["Core Labor Laws"], ["Employment Benefits"]),
    "thirteenth_month_pay": (["Employment Benefits"], ["Labor Advisories"]),
    "service_incentive_leave": (["Core Labor Laws", "Leave Benefits"], []),
    "maternity_leave": (["Leave Benefits"], []),
    "paternity_leave": (["Leave Benefits"], []),
    "solo_parent_leave": (["Leave Benefits"], []),
    "special_leave_for_women": (["Leave Benefits"], ["Anti-Harassment / Discrimination"]),
    "domestic_workers": (["Domestic Workers"], ["DOLE Resources"]),
    "social_security": (["Social Benefits"], []),
    "philhealth": (["Social Benefits"], []),
    "pagibig": (["Social Benefits"], []),
    "employees_compensation": (["Social Benefits"], [OCCUPATIONAL_SAFETY_CATEGORY]),
    "sexual_harassment": (["Anti-Harassment / Discrimination"], []),
    "safe_spaces": (["Anti-Harassment / Discrimination"], []),
    "discrimination": (["Anti-Harassment / Discrimination"], ["Core Labor Laws"]),
    "workplace_safety": ([OCCUPATIONAL_SAFETY_CATEGORY], ["DOLE Resources"]),
    "workplace_injury": ([OCCUPATIONAL_SAFETY_CATEGORY, "Social Benefits"], []),
    "termination": (["Core Labor Laws"], ["Jurisprudence"]),
    "procedural_due_process": (["Core Labor Laws"], ["Jurisprudence"]),
    "separation_pay": (["Core Labor Laws", "Employment Benefits"], []),
    "retirement_pay": (["Core Labor Laws", "Employment Benefits"], []),
    "probationary_employment": (["Core Labor Laws"], ["Jurisprudence"]),
    "security_floating_status": (["Core Labor Laws"], ["DOLE Resources"]),
    "regular_employment": (["Core Labor Laws"], ["Jurisprudence"]),
    "project_employment": (["Core Labor Laws"], ["Jurisprudence"]),
    "fixed_term_employment": (["Core Labor Laws"], ["Jurisprudence"]),
    "union_rights": (["Labor Relations"], ["Core Labor Laws"]),
    "collective_bargaining": (["Labor Relations"], ["Core Labor Laws"]),
    "strike_lockout": (["Labor Relations"], ["Core Labor Laws"]),
    "labor_complaint": (["DOLE Resources"], ["Labor Relations"]),
    "service_charges": (["DOLE Resources", "Employment Benefits"], ["Core Labor Laws"]),
    "telecommuting": (["DOLE Resources", "Employment Benefits"], ["Core Labor Laws"]),
    "contracting_subcontracting": (["Employment Benefits", "Core Labor Laws"], ["DOLE Resources"]),
    "government_employment": (["Civil Service / Government Employment"], ["Jurisprudence"]),
    "ofw": (["Others"], []),
    "jurisprudence_request": (["Jurisprudence"], []),
    "general_labor": (["Core Labor Laws", "DOLE Resources"], ["Jurisprudence"]),
}

TOPIC_KB_CATEGORY_MAP = {
    topic: primary[0]
    for topic, (primary, _secondary) in TOPIC_KB_CATEGORY_ROUTES.items()
    if primary
}

# Conversation badges use the user-facing topic grouping, which is allowed to
# differ from the retrieval route's first category (for example, night-shift
# rules retrieve Labor Code first but belong under Employment Benefits in the
# sidebar). Keep this map separate so retrieval behavior is unchanged.
SIDEBAR_TOPIC_CATEGORY_MAP = {
    "government_employment": "Civil Service / Government Employment",
    "civil_service": "Civil Service / Government Employment",
    "administrative_case": "Civil Service / Government Employment",
    "procedural_due_process": "Core Labor Laws",
    **{topic: "Employment Benefits" for topic in (
        "overtime_pay", "ordinary_overtime", "rest_day_overtime", "holiday_overtime",
        "night_shift_differential", "holiday_pay", "regular_holiday_pay", "special_day_pay",
        "holiday_on_rest_day", "regular_holiday_on_rest_day", "rest_day_pay", "special_day_overtime",
        "regular_holiday_overtime", "holiday_rest_day_overtime", "thirteenth_month_pay",
        "thirteenth_month_coverage", "thirteenth_month_formula", "thirteenth_month_computation",
        "thirteenth_month_resigned_employee", "thirteenth_month_nonpayment", "thirteenth_month_payment_deadline",
        "minimum_wage", "minimum_wage_compliance", "regional_minimum_wage", "separation_pay",
        "separation_pay_eligibility", "separation_pay_computation", "retirement_pay", "retirement_pay_computation",
        "wage_deduction", "allowable_wage_deduction", "wage_nonpayment", "wage_theft", "wage_distortion",
        "service_charge", "service_charge_distribution", "workers_paid_by_results",
    )},
    **{topic: "Core Labor Laws" for topic in (
        "core_labor_law", "labor_code", "hours_of_work", "working_hours", "meal_period", "weekly_rest_period",
        "probationary_employment", "probationary_duration", "probationary_termination", "probationary_standards", "probationary_regularization",
        "termination", "termination_just_cause", "termination_authorized_cause",
        "termination_due_process", "illegal_dismissal", "illegal_dismissal_remedies", "security_of_tenure",
        "employment_contract", "employment_status", "fixed_term_employment", "project_employment", "seasonal_employment",
        "regular_employment", "regularization", "apprenticeship", "learnership", "special_workers",
        "recruitment_and_placement",
    )},
    **{topic: "Leave Benefits" for topic in (
        "maternity_leave", "maternity_standard_duration", "maternity_live_childbirth", "maternity_miscarriage",
        "maternity_emergency_termination", "maternity_solo_parent_additional", "maternity_optional_extension",
        "maternity_allocation", "paternity_leave", "solo_parent_leave", "special_leave_women", "gynecological_leave",
        "mcw_special_leave", "service_incentive_leave", "sil", "leave_commutation", "battered_woman_leave",
        "violence_against_women_leave",
    )},
    **{topic: "Domestic Workers" for topic in (
        "kasambahay", "kasambahay_rights", "kasambahay_definition", "kasambahay_coverage", "kasambahay_minimum_wage",
        "kasambahay_wages", "kasambahay_contract", "kasambahay_employment_contract", "kasambahay_benefits",
        "kasambahay_social_benefits", "kasambahay_sss", "kasambahay_philhealth", "kasambahay_pagibig",
        "kasambahay_rest_period", "kasambahay_working_hours", "kasambahay_board_lodging", "kasambahay_medical",
        "kasambahay_termination", "kasambahay_employment_certification", "kasambahay_registry", "domestic_worker",
        "domestic_workers", "domestic_worker_rights", "domestic_worker_wage", "domestic_worker_contract",
        "domestic_worker_benefits", "domestic_worker_termination", "household_worker", "household_helper",
        "batas_kasambahay", "ra10361", "ra_10361",
    )},
    **{topic: "Anti-Harassment / Discrimination" for topic in (
        "sexual_harassment", "sexual_harassment_definition", "sexual_harassment_reporting", "safe_spaces",
        "safe_spaces_workplace", "gender_based_sexual_harassment", "codi", "codi_definition", "codi_composition",
        "codi_procedure", "codi_reporting", "age_discrimination", "gender_discrimination", "sex_discrimination",
        "women_equal_opportunity", "workplace_dignity",
    )},
    **{topic: "Social Benefits" for topic in (
        "sss", "sss_coverage", "sss_registration", "sss_contribution", "sss_benefits", "sss_sickness", "sss_maternity",
        "sss_disability", "sss_death", "sss_loan", "philhealth", "philhealth_coverage", "philhealth_contribution",
        "philhealth_benefits", "philhealth_claim", "pagibig", "pagibig_coverage", "pagibig_contribution",
        "pagibig_benefits", "pagibig_housing_loan", "pagibig_short_term_loan", "social_benefits",
        "mandatory_contributions", "contribution_nonremittance",
    )},
    **{topic: OCCUPATIONAL_SAFETY_CATEGORY for topic in (
        "osh", "occupational_safety", "workplace_safety", "workplace_health", "right_to_refuse_unsafe_work",
        "imminent_danger", "safety_officer", "safety_officer_requirement", "safety_officer_training", "osh_committee",
        "health_and_safety_committee", "ppe", "personal_protective_equipment", "accident_reporting", "illness_reporting",
        "hazardous_workplace", "work_stoppage", "medical_services", "dental_services", "osh_penalty", "osh_penalties",
        "employer_osh_duties",
    )},
    **{topic: "Labor Relations" for topic in (
        "labor_relations", "union", "union_rights", "union_membership", "right_to_self_organization", "self_organization",
        "union_registration", "labor_organization", "collective_bargaining", "cba", "cba_definition", "cba_registration",
        "cba_negotiation", "certification_election", "unfair_labor_practice", "ulp", "strike", "strike_requirements",
        "strike_notice", "lockout", "lockout_requirements", "grievance", "grievance_machinery", "voluntary_arbitration",
        "labor_management_council", "tripartism", "sena", "single_entry_approach",
    )},
    **{topic: "Jurisprudence" for topic in (
        "jurisprudence", "case_law", "supreme_court_case", "supreme_court_decision", "labor_case", "case_lookup",
        "case_citation", "gr_number", "gr_case", "landmark_case", "nlrc_decision",
    )},
    **{topic: "DOLE Resources" for topic in (
        "dole", "dole_complaint", "dole_procedure", "dole_inspection", "labor_inspection", "labor_standards_enforcement",
        "regional_wage_order", "wage_order", "rtwpb", "regional_wage_board", "minimum_wage_region", "dole_regional_office",
        "dole_field_office", "dole_jurisdiction", "inspection", "regional_office", "peso", "peso_services",
        "employment_permit", "alien_employment_permit", "establishment_reporting", "compliance_reporting", "sena_procedure",
        "complaint_filing",
    )},
    **{topic: "Labor Advisories" for topic in (
        "labor_advisory", "department_advisory", "department_circular", "dole_advisory", "holiday_advisory",
        "holiday_pay_advisory", "flexible_work_arrangement", "fwa", "work_from_home", "telecommuting_guidance",
        "emergency_workplace_protocol", "seasonal_employment_advisory", "year_end_advisory", "compliance_advisory",
    )},
    # Broad intent values used by the NLP layer when no narrower topic exists.
    "wage_inquiry": "Employment Benefits", "payment_of_wages": "Employment Benefits",
    "leave_benefits": "Leave Benefits", "employment_status": "Core Labor Laws",
    "termination_rights": "Core Labor Laws", "occupational_safety": OCCUPATIONAL_SAFETY_CATEGORY,
    "union_rights": "Labor Relations", "labor_complaint": "DOLE Resources",
    "jurisprudence_request": "Jurisprudence", "13th_month": "Employment Benefits",
    "discrimination": "Anti-Harassment / Discrimination",
    "social_security": "Social Benefits", "employees_compensation": "Social Benefits",
    "workplace_injury": "Social Benefits", "telecommuting": "Labor Advisories",
}

# Sidebar topics may be more specific than the knowledge-base folders.
SIDEBAR_TOPIC_CATEGORY_MAP.update({topic: "Wages" for topic in (
    "wage_inquiry", "payment_of_wages", "wage_deductions", "minimum_wage",
    "regional_wage_order", "wage_order", "minimum_wage_region", "salary", "wages",
    "minimum_wage_compliance", "regional_minimum_wage", "wage_deduction",
    "allowable_wage_deduction", "wage_nonpayment", "wage_theft", "wage_distortion",
)})

SIDEBAR_MESSAGE_CATEGORY_RULES = (
    ("Civil Service / Government Employment", re.compile(r"\b(government\s+employee|civil\s+service|raccs|oraohra|kawani\s+ng\s+gobyerno)\b", re.I)),
    ("Domestic Workers", re.compile(r"\b(kasambahay|domestic\s+worker|household\s+(?:worker|helper)|househelper|yaya|maid)\b", re.I)),
    ("Anti-Harassment / Discrimination", re.compile(r"\b(sexual\s+harassment|harassment|discrimination|bullying|safe\s+spaces|codi|committee\s+on\s+decorum)\b", re.I)),
    ("Leave Benefits", re.compile(r"\b(maternity|paternity|solo\s+parent|sick\s+leave|vacation\s+leave|service\s+incentive\s+leave|gynecological|battered\s+woman)\b", re.I)),
    ("Occupational Safety & Health", re.compile(r"\b(osh|occupational\s+(?:safety|health)|workplace\s+safety|imminent\s+danger|unsafe\s+work|safety\s+officer|ppe|personal\s+protective\s+equipment|hazard)\b", re.I)),
    ("Labor Relations", re.compile(r"\b(union|unyon|collective\s+bargaining|\bcba\b|strike|lockout|grievance|sena|self[-\s]?organization)\b", re.I)),
    ("Jurisprudence", re.compile(r"\b(supreme\s+court|jurisprudence|case\s+law|landmark\s+case|g\.r\.\s*no\.?|gr\s+no\.?)\b", re.I)),
    ("DOLE Resources", re.compile(r"\b(dole\s+(?:complaint|procedure|inspection|regional|field\s+office)|labor\s+(?:standards\s+)?complaint|wage\s+order|rtwpb|regional\s+wage\s+board)\b", re.I)),
    ("Social Benefits", re.compile(r"\b(sss|philhealth|pag[-\s]?ibig|hdmf|social\s+security|mandatory\s+contribution)\b", re.I)),
    ("Employment Benefits", re.compile(r"\b(minimum\s+wage|regional\s+(?:minimum\s+)?wage|overtime|holiday\s+pay|rest\s+day\s+pay|night\s+(?:shift\s+)?differential|13(?:th|th-)?\s*month|separation\s+pay|retirement\s+pay|wage\s+(?:deduction|nonpayment|theft)|service\s+charge)\b", re.I)),
    ("Labor Advisories", re.compile(r"\b(labor\s+advisory|department\s+advisory|department\s+circular|compliance\s+advisory)\b", re.I)),
    ("Core Labor Laws", re.compile(r"\b(labor\s+code|pd\.?\s*442|article\s+(?:296|297|298|299)|termination|dismissal|fired|firing|sacked|tinanggal|natanggal|ginpahalin|probationary|security\s+of\s+tenure|employment\s+contract|working\s+hours?)\b", re.I)),
)

SIDEBAR_REFERENCE_CATEGORY_RULES = (
    ("Domestic Workers", re.compile(r"\b(?:r\.?a\.?\s*10361|republic\s+act\s+no\.?\s*10361|batas\s+kasambahay|kasambahay\s+act)\b", re.I)),
    ("Leave Benefits", re.compile(r"\b(?:r\.?a\.?\s*(?:11210|8187|11861|9710))\b", re.I)),
    ("Anti-Harassment / Discrimination", re.compile(r"\b(?:r\.?a\.?\s*(?:7877|10911|11313)|safe\s+spaces\s+act)\b", re.I)),
    ("Social Benefits", re.compile(r"\b(?:sss|r\.?a\.?\s*(?:11199|11223|9679)|philhealth|pag[-\s]?ibig|hdmf)\b", re.I)),
    ("Occupational Safety & Health", re.compile(r"\b(?:r\.?a\.?\s*11058|d\.?o\.?\s*(?:no\.?)?\s*198(?:-18)?|oshs?)\b", re.I)),
    ("Employment Benefits", re.compile(r"\b(?:p\.?d\.?\s*851)\b", re.I)),
    ("Core Labor Laws", re.compile(r"\b(?:labor\s+code|p\.?d\.?\s*442)\b", re.I)),
)


def normalize_sidebar_key(value: Optional[str]) -> str:
    """Normalize topic/intent values before looking them up."""

    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower())).strip("_")


def get_sidebar_category(
    *,
    topic: Optional[str] = None,
    sub_intent: Optional[str] = None,
    intent: Optional[str] = None,
    message: Optional[str] = None,
    existing_category: Optional[str] = None,
) -> str:
    """Return the single canonical category used by chat, storage, and UI.

    Specific sub-intents win over topics, then recognized intents, message
    signals/references, an existing saved category, and finally ``Others``.
    """

    from routers.services.scenario_retrieval import everyday_topic
    scenario = everyday_topic(str(message or ''))
    # Specific facts outrank a generic NLP intent such as wage_inquiry.
    if scenario in SIDEBAR_TOPIC_CATEGORY_MAP:
        return SIDEBAR_TOPIC_CATEGORY_MAP[scenario]
    for value in (sub_intent, topic):
        category = SIDEBAR_TOPIC_CATEGORY_MAP.get(normalize_sidebar_key(value))
        if category:
            return category
    text = str(message or "")
    for category, pattern in SIDEBAR_MESSAGE_CATEGORY_RULES:
        if pattern.search(text):
            return category
    for category, pattern in SIDEBAR_REFERENCE_CATEGORY_RULES:
        if pattern.search(text):
            return category
    category = SIDEBAR_TOPIC_CATEGORY_MAP.get(normalize_sidebar_key(intent))
    if category:
        return category
    if existing_category == "Wages":
        return "Wages"
    saved = canonical_kb_category(existing_category) if existing_category else ""
    return saved if saved in KB_CATEGORIES else "Others"


def canonical_kb_category(category: Optional[str]) -> str:
    value = (category or DEFAULT_KB_CATEGORY).strip()
    return LEGACY_KB_CATEGORY_MAP.get(value, value)


def normalize_kb_category(category: Optional[str]) -> str:
    value = canonical_kb_category(category)
    return value if value in KB_CATEGORIES else DEFAULT_KB_CATEGORY


def preferred_kb_category(topic: Optional[str]) -> Optional[str]:
    return TOPIC_KB_CATEGORY_MAP.get((topic or "").strip().lower())


def routed_kb_categories(topic: Optional[str], *, expanded: bool = False, limit: int = 3) -> list[str]:
    """Return a bounded category route; never return the full taxonomy."""

    primary, secondary = TOPIC_KB_CATEGORY_ROUTES.get(
        (topic or "general_labor").strip().lower(),
        TOPIC_KB_CATEGORY_ROUTES["general_labor"],
    )
    categories = [*primary, *(secondary if expanded else [])]
    return list(dict.fromkeys(categories))[:max(1, min(limit, len(KB_CATEGORIES)))]
