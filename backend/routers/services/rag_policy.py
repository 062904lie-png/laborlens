"""Scope-aware RAG helpers for LaborLens.

This module keeps legal-scope decisions outside the LLM so retrieved special
rules cannot be treated as general rules by accident.
"""

from __future__ import annotations

import logging
import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from routers.services.kb_categories import normalize_kb_category, preferred_kb_category
from routers.services.document_categories import (
    category_match_type,
    category_metadata,
    secondary_categories_from_metadata,
)

from routers.services.legal_topic_matrix import (
    CONCEPT_PATTERNS,
    authority_search_terms,
    canonical_topic,
    classify_topic as matrix_classify_topic,
    collect_supported_concepts,
    essential_concepts_for_topic,
    missing_required_concepts,
    source_has_indexed_metadata_header,
    source_passage_text,
    preferred_sources_for_topic,
    required_authorities_for_topic,
    required_concepts_for_topic,
)

logger = logging.getLogger(__name__)

SOURCE_STATUS_NOT_REQUIRED = "not_required"
SOURCE_STATUS_DIRECT = "direct"
SOURCE_STATUS_PARTIAL = "partial"
SOURCE_STATUS_INSUFFICIENT = "insufficient"
SOURCE_STATUS_CONFLICTING = "conflicting"
SOURCE_STATUS_PROVIDER_UNAVAILABLE = "provider_unavailable"

GENERAL_SECTOR = "general_private_employment"
HOSPITAL_SECTOR = "hospital_clinic_personnel"
GOVERNMENT_SECTOR = "government_employment"
DOMESTIC_SECTOR = "domestic_workers"
MANAGERIAL_SECTOR = "managerial_employees"
MANAGERIAL_STAFF_SECTOR = "managerial_staff"
FIELD_SECTOR = "field_personnel"
NIGHT_WORKERS_SECTOR = "night_workers"
MINOR_WORKERS_SECTOR = "minor_workers"
OVERSEAS_WORKERS_SECTOR = "overseas_workers"
UNKNOWN_SECTOR = "unknown"

GENERAL_RULE = "general_rule"
SPECIAL_SECTOR_RULE = "special_sector_rule"
COVERAGE_RULE = "coverage_rule"
EXCEPTION_RULE = "exception"
DEFINITION_RULE = "definition"
ELIGIBILITY_RULE = "eligibility_rule"
COMPUTATION_RULE = "computation_rule"
PROCEDURAL_RULE = "procedural_rule"
DEADLINE_RULE = "deadline_rule"
PENALTY_RULE = "penalty_rule"
JURISPRUDENTIAL_RULE = "jurisprudential_interpretation"

DIRECT_SCORE_THRESHOLD = 0.52
PARTIAL_SCORE_THRESHOLD = 0.25

# Fine-grained intents keep retrieval focused on the exact legal question while
# retaining the existing broad topic taxonomy for backwards compatibility.
SUB_INTENT_RULES = (
    ("security_floating_status", r"\b(?:security\s+guards?|security\s+agenc(?:y|ies)|private\s+security|security\s+service\s+contractors?|ssc\s*/\s*psa)\b.{0,140}\b(?:floating\s+status|reserved\s+status|work\s+pool|off[-\s]?detail)\b|\b(?:floating\s+status|reserved\s+status|work\s+pool|off[-\s]?detail)\b.{0,140}\b(?:security\s+guards?|security\s+agenc(?:y|ies)|private\s+security|security\s+service\s+contractors?|ssc\s*/\s*psa)\b"),
    ("thirteenth_month_resigned_employee", r"\b(resign(?:ed|ation)?|nag[-\s]?resign|nag[-\s]?quit|separat(?:ed|ion)|terminated|gintanggal|tinanggal|left\s+the\s+job|before\s+year[-\s]?end)\b.*\b(13th|thirteenth|ika[-\s]?13)\b|\b(13th|thirteenth|ika[-\s]?13)\b.*\b(resign|nag[-\s]?resign|separat|terminat|gintanggal|tinanggal|pro[-\s]?rata|proportionate)\b"),
    ("thirteenth_month_computation", r"\b(13th|thirteenth)\b.*\b(comput|amount|how\s+much|formula|proportionate)\b"),
    ("thirteenth_month_payment_deadline", r"\b(13th|thirteenth)\b.*\b(when|deadline|due|pay(?:ment)?\s+date|december)\b"),
    ("thirteenth_month_nonpayment", r"(?:\b(13th|thirteenth|ika[-\s]?13)\b.*\b(not\s+paid|didn['’]?t\s+give|withheld|nonpayment|unpaid|wala\s+(?:pa\s+)?(?:ang\s+)?(?:13th|ika[-\s]?13)|hindi\s+.{0,30}\b(?:bayad|nabayaran|bigay|binigay|nakuha)|indi\s+.{0,30}\b(?:bayad|nabayaran|hatag|ginhatag|nakuha)|what\s+can\s+i\s+do|ano\s+ang\s+(?:dapat|pwede))\b|\b(not\s+paid|didn['’]?t\s+give|withheld|nonpayment|unpaid|wala\s+(?:pa\s+)?(?:ang\s+)?(?:13th|ika[-\s]?13)|hindi\s+.{0,30}\b(?:bayad|nabayaran|bigay|binigay|nakuha)|indi\s+.{0,30}\b(?:bayad|nabayaran|hatag|ginhatag|nakuha))\b.*\b(13th|thirteenth|ika[-\s]?13)\b)"),
    ("thirteenth_month_exclusions", r"\b(13th|thirteenth|ika[-\s]?13)\b.*\b(exclud|not\s+entitled|who\s+is\s+not)\b"),
    ("thirteenth_month_coverage", r"\b(13th|thirteenth|ika[-\s]?13)\b.*\b(who|covered|entitled|eligible|sino|may\s+karapatan|sin-o|katungod)\b"),
    ("regular_holiday_on_rest_day", r"\b(regular\s+holiday|holiday)\b.*\b(rest\s+day|day\s+off)\b|\b(rest\s+day|day\s+off)\b.*\b(regular\s+holiday|holiday)\b"),
    ("regular_holiday_pay", r"\b(regular\s+holiday|holiday\s+pay|double\s+pay)\b"),
    ("special_day_pay", r"\b(special\s+(?:non[-\s]?working|day|holiday)|non[-\s]?working\s+holiday)\b"),
    ("rest_day_pay", r"\b(rest\s+day|day\s+off)\b.*\b(pay|work|rate|premium)\b|\b(pay|work|rate|premium)\b.*\b(rest\s+day|day\s+off)\b"),
    ("night_shift_differential", r"\b(night\s*(?:shift)?\s*differential|night\s*diff|graveyard)\b"),
    ("ordinary_overtime", r"\b(overtime|extra\s+hours?|beyond\s+eight|ot)\b.*\b(ordinary|normal\s+working\s+day)\b|\b(ordinary|normal\s+working\s+day)\b.*\b(overtime|extra\s+hours?|ot)\b"),
    ("rest_day_overtime", r"\b(rest\s+day|day\s+off)\b.*\b(overtime|extra\s+hours?|rate)\b"),
    ("regular_holiday_overtime", r"\b(regular\s+holiday|holiday)\b.*\b(overtime|extra\s+hours?)\b"),
    ("holiday_rest_day_overtime", r"\b(holiday|special\s+day)\b.*\b(rest\s+day|day\s+off)\b.*\b(overtime|pay|rate)\b"),
    ("minimum_wage_compliance", r"\b(minimum\s+wage|wage\s+rate)\b.*\b(compli|underpaid|below|lawful|legal|should\s+i\s+be\s+paid)\b"),
    ("regional_wage_order", r"\b(minimum\s+wage|wage\s+rate|wage\s+order)\b.*\b(region|ncr|manila|province|city|iloilo|cebu|davao|car|barmm)\b"),
    ("wage_order_exemption", r"\b(wage\s+order|minimum\s+wage)\b.*\b(exempt|exemption|excluded|exception)\b"),
    ("wage_nonpayment", r"\b(unpaid|not\s+paid|withheld|kulang|wala\s+(?:bayad|sweldo)|nonpayment)\b.*\b(wage|salary|pay|sahod|sweldo)\b"),
    ("maternity_solo_parent_additional", r"\b(maternity)\b.*\b(solo\s+parent|additional|extension)\b"),
    ("maternity_optional_extension", r"\b(maternity)\b.*\b(optional|extend|extension)\b"),
    ("maternity_standard_duration", r"\b(maternity|childbirth|pregnan|maternity\s+leave)\b.*\b(days?|weeks?|duration|how\s+long|entitled|ilang\s+araw|pila\s+(?:ka\s+)?adlaw)\b"),
    ("solo_parent_leave", r"\b(solo\s+parent|single\s+parent|solo\s+nanay|solo\s+tatay)\b.*\b(leave|days?|months?|service|araw|adlaw|pila)\b|\bparental\s+leave\b"),
    ("probationary_termination", r"\bprobationary\b.*\b(?:terminat\w*|dismiss\w*|fir\w*|end\w*|anytime)\b|\b(?:terminat\w*|dismiss\w*|fir\w*|end\w*|anytime)\b.*\bprobationary\b"),
    ("probationary_regularization", r"\b(probationary|probation)\b.*\b(regular|regulariz|standards?)\b"),
    ("probationary_duration", r"\b(probationary|probation)\b.*\b(how\s+long|months?|period|six)\b"),
    ("termination_just_cause", r"\b(just\s+causes?|serious\s+misconduct|willful\s+disobedience|gross\s+neglect|fraud|valid\s+grounds?)\b"),
    ("termination_authorized_cause", r"\b(authorized\s+cause|redundancy|retrenchment|closure|disease)\b"),
    ("termination_due_process", r"\b(due\s+process|notice\s+to\s+explain|twin\s+notice|written\s+notice|hearing|wala\s+notice|walang\s+notice|wala\s+sulat|walang\s+sulat)\b"),
    ("illegal_dismissal_remedies", r"\b(illegal\s+dismissal|wrongful\s+dismissal)\b.*\b(remed|complaint|reinstate|where\s+can|what\s+can\s+i\s+do)\b"),
    ("separation_pay_computation", r"\b(separation\s+pay)\b.*\b(comput|how\s+much|formula|amount)\b"),
    ("separation_pay_eligibility", r"\b(separation\s+pay)\b.*\b(entitled|eligible|receive|who|automatically)\b|\b(entitled|eligible|receive|who|automatically)\b.*\b(separation\s+pay)\b"),
    ("sexual_harassment_reporting", r"\b(sexual\s+harassment|harassment)\b.*\b(report|complaint|where|file)\b"),
    ("codi_definition", r"\b(codi|committee\s+on\s+decorum)\b.*\b(what|define|meaning|is\s+it|ano\s+(?:ang|ini|ni))\b|^\s*(?:what\s+is|ano\s+(?:ang|ini|ni))\s+codi\b"),
    ("safe_spaces_workplace", r"\b(safe\s+spaces|ra\s*11313|bawal\s+bastos)\b"),
    ("right_to_self_organization", r"\b(form|join|organize|self[-\s]?organization|union|unyon|sumali|bumuo|makasulod|mag[-\s]?organisa)\b.*\b(union|employees?|workers?|unyon|manggagawa|mamumugon)\b|\b(union|unyon)\b.*\b(form|join|organize|sumali|bumuo|makasulod)\b"),
    ("cba_definition", r"\b(cba|collective\s+bargaining\s+agreement)\b.*\b(what|define|meaning|is\s+it|ano\s+(?:ang|ini|ni))\b|\b(what|define|meaning|is\s+it|ano\s+(?:ang|ini|ni))\b.*\b(cba|collective\s+bargaining\s+agreement)\b"),
    ("strike_requirements", r"\b(strike|lockout)\b.*\b(require|notice|requirement|legal)\b"),
    ("sena", r"\b(sena|single[-\s]?entry|request\s+for\s+assistance)\b"),
    ("social_contribution_remittance", r"(?=.*\b(?:sss|social\s+security)\b)(?=.*\bphilhealth\b)(?=.*\b(?:pag[-\s]?ibig|hdmf)\b)(?=.*\b(?:remit\w*|proof|verify|posted|contribution\s+history)\b)"),
    ("sss_contribution", r"\b(?:sss|social\s+security)\b.{0,160}\b(?:contribution|rate|premium|deduct\w*|remit\w*|amount)\b|\b(?:contribution|rate|premium|deduct\w*|remit\w*|amount)\b.{0,160}\b(?:sss|social\s+security)\b"),
    ("philhealth_contribution", r"\bphilhealth\b.{0,160}\b(?:contribution|rate|premium|deduct\w*|remit\w*|amount)\b|\b(?:contribution|rate|premium|deduct\w*|remit\w*|amount)\b.{0,160}\bphilhealth\b"),
    ("pagibig_contribution", r"\b(?:pag[-\s]?ibig|hdmf)\b.{0,160}\b(?:contribution|rate|premium|deduct\w*|remit\w*|amount)\b|\b(?:contribution|rate|premium|deduct\w*|remit\w*|amount)\b.{0,160}\b(?:pag[-\s]?ibig|hdmf)\b"),
    ("right_to_refuse_unsafe_work", r"\b(refuse|decline|stop)\b.*\b(unsafe|danger|hazard)\b|\bright\s+to\s+refuse\b"),
    ("ppe_requirement", r"\b(ppe|personal\s+protective\s+equipment)\b.*\b(pay|pays|provide|cost|free|employer)\b|\bwho\s+pays?\b.*\b(ppe|personal\s+protective\s+equipment)\b"),
    ("safety_officer_requirement", r"\b(safety\s+officer)\b.*\b(require|required|number|must|need)\b|\b(require|required|must|need)\b.*\b(safety\s+officer)\b"),
    ("osh_program_requirement", r"\b(osh|occupational\s+safety\s+and\s+health|safety\s+and\s+health)\s+program\b"),
    ("osh_committee_requirement", r"\b(osh|occupational\s+safety\s+and\s+health|safety\s+and\s+health)\s+committee\b"),
    ("osh_employer_duties", r"\b(employer(?:'s|s)?|company)\b.{0,100}\b(dut(?:y|ies)|obligations?|responsibilit(?:y|ies)|must|shall)\b.{0,120}\b(osh|occupational\s+safety|safety|health|safe\s+workplace)\b|\b(osh|occupational\s+safety|workplace\s+safety)\b.{0,120}\b(employer(?:'s|s)?|company)\b.{0,80}\b(dut(?:y|ies)|obligations?|must|shall)\b"),
)

SUB_INTENT_SEARCH_TERMS = {
    key: key.replace("_", " ") for key, _pattern in SUB_INTENT_RULES
}
SUB_INTENT_SEARCH_TERMS.update({
    "social_contribution_remittance": (
        "employer remittance of SSS PhilHealth and Pag-IBIG contributions; member portal, "
        "online contribution history, posted payments, and how to verify each agency account"
    ),
    "sss_contribution": (
        "SSS employee contribution rate table monthly salary credit employee share "
        "effective contribution schedule employer deduction"
    ),
    "philhealth_contribution": (
        "PhilHealth employee premium contribution rate table salary base share "
        "effective contribution schedule employer deduction"
    ),
    "pagibig_contribution": (
        "Pag-IBIG HDMF employee contribution rate table monthly compensation share "
        "effective contribution schedule employer deduction"
    ),
})

SUB_INTENT_CONCEPT_OVERRIDES = {
    "security_floating_status": (
        ["security_reserved_status_rule", "six_month_security_status_rule", "security_reserved_status_conditions", "security_separation_entitlement"],
        ["security_reserved_status_rule", "six_month_security_status_rule"],
    ),
    "social_contribution_remittance": (
        ["sss_member_verification_process", "philhealth_member_verification_process", "pagibig_member_verification_process"],
        ["sss_member_verification_process", "philhealth_member_verification_process", "pagibig_member_verification_process"],
    ),
    "sss_contribution": (
        ["current_contribution_rate_schedule"],
        ["current_contribution_rate_schedule"],
    ),
    "philhealth_contribution": (
        ["current_contribution_rate_schedule"],
        ["current_contribution_rate_schedule"],
    ),
    "pagibig_contribution": (
        ["current_contribution_rate_schedule"],
        ["current_contribution_rate_schedule"],
    ),
    "thirteenth_month_resigned_employee": (
        ["employee_coverage", "resigned_employee_entitlement", "basic_salary_basis"],
        ["resigned_employee_entitlement"],
    ),
    "thirteenth_month_computation": (["basic_salary_basis", "calculation_formula"], ["basic_salary_basis"]),
    "termination_just_cause": (["just_cause"], ["just_cause"]),
    "termination_authorized_cause": (["authorized_cause"], ["authorized_cause"]),
    "termination_comparison": (
        ["just_cause", "authorized_cause", "notice", "opportunity_to_be_heard", "authorized_cause_notice", "legal_basis"],
        ["just_cause", "authorized_cause"],
    ),
    "probationary_termination": (["probationary_period", "standards_at_engagement", "termination_ground"], ["probationary_period"]),
    "regular_holiday_pay": (["holiday_type", "regular_holiday_worked_rate", "regular_holiday_unworked_rule"], ["holiday_type"]),
    "regular_holiday_on_rest_day": (["holiday_type", "holiday_rest_day_rule"], ["holiday_rest_day_rule"]),
    "special_day_pay": (["holiday_type", "special_non_working_worked_rate"], ["holiday_type"]),
    "rest_day_pay": (["day_type", "premium_rate"], ["day_type"]),
    "night_shift_differential": (["night_period", "rate", "covered_employee"], ["night_period"]),
    "maternity_standard_duration": (["leave_duration", "qualifying_event", "benefit_source"], ["leave_duration"]),
    "maternity_solo_parent_additional": (["leave_duration", "solo_parent_qualification"], ["leave_duration"]),
    "codi_definition": (["complaint_investigation_body"], ["complaint_investigation_body"]),
    "right_to_self_organization": (["self_organization"], ["self_organization"]),
}

NON_LEGAL_INTENTS = {
    "greeting",
    "small_talk",
    "identity",
    "capability_question",
    "gratitude",
    "farewell",
}

OVERTIME_RE = re.compile(
    r"\b(overtime|over\s*time|ot|extra\s+hours?|beyond\s+eight|beyond\s+8|"
    r"work\s+beyond\s+eight|hours?\s+of\s+work|forty[-\s]?hour|40[-\s]?hour|"
    r"labaw\s+sa\s+8|lampas\s+sa\s+8)\b",
    re.IGNORECASE,
)
PD_442_RE = re.compile(
    r"\b(pd\s*442|p\.?\s*d\.?\s*442|presidential\s+decree\s+no\.?\s*442)\b",
    re.IGNORECASE,
)
ARTICLE_87_RE = re.compile(r"\b(?:art\.?|article)\s*87\b", re.IGNORECASE)
OVERTIME_SUPPORT_ARTICLE_RE = re.compile(r"\b(?:art\.?|article)\s*(82|84|87|88|89|93|94)\b", re.IGNORECASE)
HOSPITAL_RE = re.compile(
    r"\b(hospital|clinic|nurse|nurses|nursing|health\s+personnel|medical\s+staff|"
    r"healthcare|health\s+care|rule\s+i-a|rule\s+ia)\b",
    re.IGNORECASE,
)
GOVERNMENT_RE = re.compile(
    r"\b(government|public\s+sector|civil\s+service|government\s+employee|"
    r"public\s+employee|lg?u|deped|doh|state\s+university)\b",
    re.IGNORECASE,
)
DOMESTIC_RE = re.compile(
    r"\b(kasambahay|kasambhay|domestic\s+workers?|domestics?\s+workers?|"
    r"house\s*helpers?|househelpers?|household\s+(?:helpers?|workers?|service)|"
    r"domestic\s+helpers?|persons?\s+in\s+personal\s+service|"
    r"yaya|maid|housemaid|house\s*maid|household\s+cook|cook\s+in\s+(?:the\s+)?household|"
    r"household\s+gardener|laundry\s+person\s+in\s+(?:the\s+)?household)\b",
    re.IGNORECASE,
)
MANAGERIAL_RE = re.compile(
    r"\b(managerial|manager|supervisory|member\s+of\s+managerial\s+staff)\b",
    re.IGNORECASE,
)
FIELD_RE = re.compile(
    r"\b(field\s+personnel|field\s+worker|outside\s+sales)\b",
    re.IGNORECASE,
)
NIGHT_WORKER_RE = re.compile(
    r"\b(night\s+worker|night\s+shift\s+worker|night\s+shift|graveyard)\b",
    re.IGNORECASE,
)
MINOR_WORKER_RE = re.compile(
    r"\b(minor\s+worker|working\s+child|child\s+labor|below\s+18|under\s+18)\b",
    re.IGNORECASE,
)
OVERSEAS_WORKER_RE = re.compile(
    r"\b(ofw|overseas\s+worker|migrant\s+worker|seafarer|overseas\s+employment)\b",
    re.IGNORECASE,
)
CALCULATION_RE = re.compile(
    r"\b(calculate|compute|computation|how\s+much|amount|estimate|formula|"
    r"hourly\s+rate|daily\s+rate|rate\s+is|worked\s+\d+|worked\s+two|"
    r"ordinary\s+day|holiday|rest\s+day|₱|php|pesos?)\b",
    re.IGNORECASE,
)
PERSONAL_FACT_RE = re.compile(
    r"\b(i\s+am|i'm|my|ako|akon|ko|hourly\s+rate|salary|wage|sweldo|suhol|"
    r"worked|nagtrabaho|nag-obra|ordinary\s+day|rest\s+day|holiday|from|to|"
    r"\d+)\b",
    re.IGNORECASE,
)
ARTICLE_RE = re.compile(r"\b(?:art\.?|article)\s*(\d+)\b", re.IGNORECASE)
ARTICLE_TITLE_RE = re.compile(
    r"\b(?:art\.?|article)\s*(?P<number>\d+)\s*\.?\s*(?P<title>[A-Z][A-Za-z][^.\n\r]{1,90})\.",
    re.IGNORECASE,
)
SECTION_RE = re.compile(r"\b(?:sec\.?|section)\s*(\d+)\b", re.IGNORECASE)
BOOK_RE = re.compile(r"\bbook\s+([ivxlcdm]+|\d+)\b", re.IGNORECASE)
RULE_RE = re.compile(r"\brule\s+([ivxlcdm]+(?:-[a-z])?|\d+(?:-[a-z])?)\b", re.IGNORECASE)
CHAPTER_RE = re.compile(r"\bchapter\s+([ivxlcdm]+|\d+)\b", re.IGNORECASE)
TITLE_RE = re.compile(r"\btitle\s+([ivxlcdm]+|\d+)\b", re.IGNORECASE)
PERSONAL_CALCULATION_RE = re.compile(
    r"\b(calculate|compute|how\s+much|amount|estimate|example\s+calculat(?:ion|ions)|"
    r"sample\s+calculat(?:ion|ions)|example\s+comput(?:ation|ations)|"
    r"sample\s+comput(?:ation|ations)|worked\s+example|my\s+(?:hourly|daily|monthly)|"
    r"hourly\s+(?:wage|salary|pay)|daily\s+(?:wage|salary|pay)|rate\s+is|"
    r"worked\s+\d+|worked\s+(?:one|two|three|four|five|six|seven|eight|nine|ten)|"
    r"â‚±|₱|php|pesos?)\b",
    re.IGNORECASE,
)
DEFINITION_TASK_RE = re.compile(r"\b(what\s+is|define|definition|meaning\s+of|ano\s+(?:ang|nga)|ano\s+ang\s+kahulugan)\b", re.IGNORECASE)
COVERAGE_TASK_RE = re.compile(r"\b(who\s+(?:is|are)|who\s+can|covered\s+by|covered\s+under|coverage|entitled\s+to|eligible\s+for|qualified\s+for|may\s+receive|can\s+receive|rights?|right\s+to)\b", re.IGNORECASE)
EXCLUSION_TASK_RE = re.compile(r"\b(excluded|exempt|not\s+covered|not\s+entitled|managerial|field\s+personnel|government\s+employee)\b", re.IGNORECASE)
RATE_TASK_RE = re.compile(r"\b(rate|percent|percentage|premium|125\s*%|130\s*%|150\s*%|169\s*%|195\s*%|200\s*%|260\s*%|338\s*%|ordinary[-\s]+day\s+overtime)\b", re.IGNORECASE)
FORMULA_TASK_RE = re.compile(r"\b(formula|computed|computation|how\s+(?:is|are).{0,40}(?:computed|compute)|basis\s+of\s+computation)\b", re.IGNORECASE)
AUTHORIZATION_TASK_RE = re.compile(r"\b(written\s+approval|approval|approve|authorized?|authorization|permission|permit|required\s+approval)\b", re.IGNORECASE)
UNDERTIME_TASK_RE = re.compile(r"\b(undertime|offset|set\s*off|make\s+up\s+for\s+late)\b", re.IGNORECASE)
COMPULSORY_TASK_RE = re.compile(r"\b(compulsory|mandatory|force|forced|require\s+me|require\s+overtime|required\s+overtime|emergency\s+overtime|can\s+employer\s+require)\b", re.IGNORECASE)
THRESHOLD_TASK_RE = re.compile(r"\b(eight\s+hours|8\s+hours|forty\s+hours|40\s+hours|per\s+day|per\s+week|daily\s+or\s+weekly|threshold|beyond\s+8)\b", re.IGNORECASE)
DEADLINE_TASK_RE = re.compile(r"\b(deadline|due|when\s+(?:should|must|is)|not\s+later\s+than|payment\s+date|december\s+24|within\s+\d+\s+days)\b", re.IGNORECASE)
COMPARISON_TASK_RE = re.compile(
    r"\b(difference|compare|comparison|versus|vs\.?|which\s+is\s+better|same\s+as|"
    r"pagkakaiba|kaibahan|pagkukumpara|pareho|kinalain|kaangay)\b",
    re.IGNORECASE,
)
FALSE_ASSUMPTION_TASK_RE = re.compile(r"\b(is\s+it\s+true|isn't\s+it|diba|di\s+ba|indi\s+bala|no\s+need|always|never)\b", re.IGNORECASE)
DOCUMENT_TASK_RE = re.compile(r"\b(document|contract|notice|certificate|permit|proof|record|requirement|required\s+to\s+submit)\b", re.IGNORECASE)
REMEDY_TASK_RE = re.compile(
    r"\b(complain|complaint|file\s+a\s+case|remedy|what\s+can\s+i\s+do|where\s+can\s+i\s+go|dole|nlrc|lawyer)\b|"
    # 'Finish a report' is a work task, not a request for complaint procedures.
    r"\b(?:can|could|should|may|do|does)\s+(?:i|we|he|she|they|you)\s+(?:(?:also|still)\s+)?report\b|"
    r"\b(?:want|wish|need)\s+to\s+report\b|\b(?:where|how)\s+to\s+report\b|"
    r"\breport\s+(?:this|it|that|these\s+(?:issues?|incidents?|violations?))\b",
    re.IGNORECASE)
CITATION_TASK_RE = re.compile(r"\b(cite|citation|legal\s+basis|article|art\.|section|sec\.|source\s+for|where\s+in\s+the\s+law)\b", re.IGNORECASE)
SOURCE_EXPLANATION_TASK_RE = re.compile(r"\b(source|sources|where\s+did\s+you|get\s+the\s+info|verify|link)\b", re.IGNORECASE)
FOLLOW_UP_TASK_RE = re.compile(r"^\s*(what\s+about|how\s+about|and\s+if|what\s+if|how\s+much\s+is\s+it|can\s+you\s+explain|translate\s+it|that|it|them)\b", re.IGNORECASE)
NORMAL_HOURS_TASK_RE = re.compile(r"\b(normal\s+hours?\s+of\s+work|how\s+many\s+hours?.{0,30}(?:work|day))\b", re.IGNORECASE)
MEAL_PERIOD_TASK_RE = re.compile(r"\b(meal\s+periods?|meal\s+breaks?|lunch\s+breaks?)\b", re.IGNORECASE)
SUCCESSIVE_HOLIDAY_TASK_RE = re.compile(r"\b(?:two\s*\(?2\)?\s+)?successive\s+regular\s+holidays?\b", re.IGNORECASE)
HOLIDAY_ELIGIBILITY_TASK_RE = re.compile(
    r"\b(condition|qualified|qualification|eligible|eligibility|entitled)\b.{0,100}\bholiday\s+pay\b|"
    r"\bholiday\s+pay\b.{0,100}\b(condition|qualified|qualification|eligible|eligibility|entitled)\b",
    re.IGNORECASE,
)

WAGE_ORDER_QUERY_RE = re.compile(
    r"\b(minimum\s+wage|wage\s+order|regional\s+wage|rtwpb|nwpc|"
    r"current\s+(?:daily|monthly)?\s*wage\s+rate)\b",
    re.IGNORECASE,
)
WAGE_QUERY_RE = re.compile(
    r"\b(minimum\s+wage|daily\s+wage|wage|salary|rate|pay|sweldo|sahod|suhol|magkano|pesos?)\b",
    re.IGNORECASE,
)

REGION_ALIASES: Tuple[Tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\b(?:ncr|national\s+capital\s+region|metro\s+manila|manila|quezon\s+city|makati|pasig|taguig|caloocan|mandaluyong|marikina|muntinlupa|paranaque|para[nñ]aque|pasay|valenzuela|las\s+pinas|las\s+pi[nñ]as|malabon|navotas|san\s+juan|pateros)\b", re.IGNORECASE), "NCR", "National Capital Region"),
    (re.compile(r"\b(?:car|cordillera\s+administrative\s+region|baguio(?:\s+city)?|benguet|abra|apayao|ifugao|kalinga|mountain\s+province)\b", re.IGNORECASE), "CAR", "Cordillera Administrative Region"),
    (re.compile(r"\b(?:barmm|bangsamoro)\b", re.IGNORECASE), "BARMM", "Bangsamoro Autonomous Region in Muslim Mindanao"),
    (re.compile(r"\bregion\s*(?:iv[-\s]?a|4[-\s]?a)\b|\bcalabarzon\b|\b(?:cavite|laguna|batangas|rizal|quezon\s+province|lucena)\b", re.IGNORECASE), "Region IV-A", "CALABARZON"),
    (re.compile(r"\bregion\s*(?:iv[-\s]?b|4[-\s]?b)\b|\bmimaropa\b", re.IGNORECASE), "Region IV-B", "MIMAROPA"),
    (re.compile(r"\bregion\s*(?:xiii|13)\b|\bcaraga\b", re.IGNORECASE), "Region XIII", "Caraga"),
    (re.compile(r"\bregion\s*(?:xii|12)\b|\bsoccsksargen\b", re.IGNORECASE), "Region XII", "SOCCSKSARGEN"),
    (re.compile(r"\bregion\s*(?:xi|11)\b|\b(?:davao(?:\s+region|\s+city|\s+del\s+norte|\s+del\s+sur|\s+occidental|\s+oriental)?|davao\s+de\s+oro)\b", re.IGNORECASE), "Region XI", "Davao Region"),
    (re.compile(r"\bregion\s*(?:x|10)\b|\bnorthern\s+mindanao\b", re.IGNORECASE), "Region X", "Northern Mindanao"),
    (re.compile(r"\bregion\s*(?:ix|9)\b|\bzamboanga\s+peninsula\b", re.IGNORECASE), "Region IX", "Zamboanga Peninsula"),
    (re.compile(r"\bregion\s*(?:viii|8)\b|\beastern\s+visayas\b", re.IGNORECASE), "Region VIII", "Eastern Visayas"),
    (re.compile(r"\bregion\s*(?:vii|7)\b|\bcentral\s+visayas\b|\b(?:cebu|bohol|negros\s+oriental|siquijor)\b", re.IGNORECASE), "Region VII", "Central Visayas"),
    (re.compile(r"\bregion\s*(?:vi|6)\b|\bwestern\s+visayas\b|\b(?:iloilo|bacolod|aklan|antique|capiz|guimaras|negros\s+occidental)\b", re.IGNORECASE), "Region VI", "Western Visayas"),
    (re.compile(r"\bregion\s*(?:v|5)\b|\bbicol\s+region\b", re.IGNORECASE), "Region V", "Bicol Region"),
    (re.compile(r"\bregion\s*(?:iv|4)\b", re.IGNORECASE), "Region IV", "Region IV"),
    (re.compile(r"\bregion\s*(?:iii|3)\b|\bcentral\s+luzon\b|\b(?:pampanga|bulacan|bataan|nueva\s+ecija|tarlac|zambales|aurora|angeles(?:\s+city)?|olongapo)\b", re.IGNORECASE), "Region III", "Central Luzon"),
    (re.compile(r"\bregion\s*(?:ii|2)\b|\bcagayan\s+valley\b", re.IGNORECASE), "Region II", "Cagayan Valley"),
    (re.compile(r"\bregion\s*(?:i|1)\b|\bilocos\s+region\b", re.IGNORECASE), "Region I", "Ilocos Region"),
    (re.compile(r"\bmindanao\b", re.IGNORECASE), "Mindanao", "Mindanao"),
    (re.compile(r"\bvisayas\b", re.IGNORECASE), "Visayas", "Visayas"),
    (re.compile(r"\bluzon\b", re.IGNORECASE), "Luzon", "Luzon"),
)


def wage_order_query_details(text: str) -> Dict[str, Any]:
    """Identify wage-order questions and normalize any named Philippine region."""

    normalized = _norm(text)
    region_code = ""
    region_name = ""
    for pattern, code, name in REGION_ALIASES:
        if pattern.search(normalized):
            region_code, region_name = code, name
            break
    if region_code.startswith("Region "):
        region_slug = "region-" + region_code.removeprefix("Region ").lower().replace("-", "")
    elif region_code == "NCR":
        region_slug = "ncr"
    elif region_code == "CAR":
        region_slug = "car"
    elif region_code == "BARMM":
        region_slug = "barmm"
    else:
        region_slug = ""
    strong_wage_intent = bool(WAGE_ORDER_QUERY_RE.search(normalized))
    regional_wage_intent = strong_wage_intent or bool(
        region_code and WAGE_QUERY_RE.search(normalized)
    )
    return {
        "is_wage_order_query": regional_wage_intent,
        "is_regional_wage_query": bool(region_code and WAGE_QUERY_RE.search(normalized)),
        "region_code": region_code,
        "region_name": region_name,
        "region_source_url": f"https://nwpc.dole.gov.ph/{region_slug}/" if region_slug else "",
    }


def is_regional_wage_query(query: str) -> bool:
    """Return whether a query should take the category-first wage-order route."""

    return bool(wage_order_query_details(query).get("is_regional_wage_query"))


def infer_document_year(text: str = "", metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Derive a stable document year from explicit metadata, then title/content."""

    meta = metadata or {}
    explicit = meta.get("document_year")
    try:
        explicit_year = int(explicit)
    except (TypeError, ValueError):
        explicit_year = 0
    if 1900 <= explicit_year <= 2099:
        return explicit_year

    date_candidates = [
        meta.get("effective_date"),
        meta.get("date_issued"),
        meta.get("publication_date"),
        meta.get("law_version"),
    ]
    for candidate in date_candidates:
        years = [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", _norm(candidate))]
        if years:
            return max(years)

    descriptor_candidates = [
        meta.get("document_title"),
        meta.get("original_filename"),
        meta.get("source"),
        meta.get("source_url"),
        meta.get("official_url"),
    ]
    for candidate in descriptor_candidates:
        years = [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", _norm(candidate))]
        if years:
            return max(years)

    content = _norm(text)
    for label in ("issued", "published", "updated", "effective"):
        labeled_years = [
            int(year)
            for year in re.findall(
                rf"\b{label}(?:\s+on)?\b[^.\n]{{0,80}}\b((?:19|20)\d{{2}})\b",
                content,
                re.IGNORECASE,
            )
        ]
        if labeled_years:
            return max(labeled_years)
    # Incidental cited years and download timestamps do not date a provision.
    return None


def document_recency_factor(document_year: Optional[int]) -> float:
    if document_year is None or document_year < 2015:
        return 1.00
    if document_year <= 2019:
        return 1.05
    if document_year <= 2023:
        return 1.10
    return 1.15


def chunk_matches_query_region(
    text: str,
    query_profile: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Match the requested region without confusing Region VI with Region VII/VIII."""

    region_code = _norm(query_profile.get("region_code"))
    region_name = _norm(query_profile.get("region_name"))
    if not region_code and not region_name:
        return False
    meta = metadata or {}
    from .regional_wage_evidence import document_regions
    owning_regions = document_regions(meta)
    if owning_regions:
        return region_code in owning_regions
    expected_url = _lower(query_profile.get("region_source_url"))
    actual_url = _lower(meta.get("source_url") or meta.get("official_url"))
    if expected_url and actual_url == expected_url:
        return True
    descriptor = " ".join(
        _norm(meta.get(key))
        for key in ("document_title", "source_name", "source")
        if meta.get(key)
    )
    for pattern, code, name in REGION_ALIASES:
        if code == region_code or name == region_name:
            if descriptor and pattern.search(descriptor):
                return True
            if re.search(r"\b(?:region|rtwpb)\b", descriptor, re.IGNORECASE):
                return False
            return bool(pattern.search(text or ""))
    return False


def is_region_specific_wage_chunk(chunk: Dict[str, Any]) -> bool:
    """Return true for a chunk tied to one RTWPB/NWPC regional wage page or order."""

    meta = chunk.get("metadata") or {}
    descriptor = " ".join(
        _norm(meta.get(key))
        for key in ("document_title", "source_name", "source", "source_url", "official_url")
        if meta.get(key)
    )
    return bool(
        re.search(r"\b(?:rtwpb|regional\s+wage\s+orders?|region\s+[ivx\d])\b", descriptor, re.IGNORECASE)
        or re.search(r"nwpc\.dole\.gov\.ph/(?:region-|national-capital-region|ncr|car|barmm)", descriptor, re.IGNORECASE)
    )


def is_wage_order_chunk(chunk: Dict[str, Any]) -> bool:
    """Identify Wage Order material so it cannot leak into unrelated legal answers."""

    meta = enrich_chunk_metadata(chunk.get("text", ""), chunk.get("metadata") or {}, infer_concepts=False)
    if _lower(meta.get("doc_type")) == "wage_order":
        return True
    descriptor = " ".join(
        _norm(meta.get(key))
        for key in ("document_title", "source_name", "source", "source_url", "official_url")
        if meta.get(key)
    )
    return bool(
        normalize_kb_category(meta.get("category")) == "DOLE Resources"
        and re.search(
            r"\b(?:wage\s+order|rtwpb|regional\s+wage\s+orders?|minimum\s+wage\s+rates?)\b",
            f"{descriptor} {chunk.get('text', '')[:500]}",
            re.IGNORECASE,
        )
    )


def wage_order_detail_score(chunk: Dict[str, Any]) -> int:
    """Favor excerpts that can support order, effectivity, and rate in one citation."""

    meta = chunk.get("metadata") or {}
    blob = f"{chunk.get('text', '')} {_metadata_text(meta)}"
    return sum(
        (
            bool(re.search(r"\b(?:wage\s+order|w\.?s*o\.?)\b", blob, re.IGNORECASE)),
            bool(meta.get("effective_date") or re.search(r"\beffective\s*:?\s*\d{1,2}\s+[A-Za-z]+\s+\d{4}\b", blob, re.IGNORECASE)),
            bool(re.search(r"(?:₱|php|p)\s*\d[\d,]*(?:\.\d{2})?|\bminimum\s+wage\s+rates?\b", blob, re.IGNORECASE)),
        )
    )


def build_wage_order_search_text(query: str, query_profile: Dict[str, Any]) -> str:
    """Expand wage searches with the terms used by DOLE/NWPC regional pages."""

    parts = [
        _norm(query),
        "current minimum wage rate DOLE NWPC RTWPB regional wage order",
    ]
    if canonical_sector(query_profile.get("sector")) == DOMESTIC_SECTOR:
        parts.append("domestic workers kasambahay current monthly minimum wage")
    if query_profile.get("region_code"):
        parts.append(_norm(query_profile.get("region_code")))
    if query_profile.get("region_name"):
        parts.append(_norm(query_profile.get("region_name")))
    return " ".join(part for part in parts if part)

SECTOR_RULE_PRECEDENCE: Dict[str, Dict[str, Any]] = {
    DOMESTIC_SECTOR: {
        "primary_law": "ra_10361",
        "supporting_laws": ["ra_10361_irr", "labor_code_article_82"],
        "controlling_rule": "special_sector_rule",
        "excluded_general_topics": [
            "ordinary_article_87_overtime_multiplier",
            "general_rest_day_overtime_table",
            "general_holiday_overtime_table",
        ],
    },
    GOVERNMENT_SECTOR: {
        "primary_law": "civil_service_rules",
        "supporting_laws": ["labor_code_article_82"],
        "controlling_rule": "special_sector_rule",
        "excluded_general_topics": ["ordinary_article_87_overtime_multiplier"],
    },
    HOSPITAL_SECTOR: {
        "primary_law": "labor_code_hospital_and_clinic_personnel_rule",
        "supporting_laws": ["labor_code_article_83"],
        "controlling_rule": "special_sector_rule",
        "excluded_general_topics": ["ordinary_hours_without_hospital_exception"],
    },
    MANAGERIAL_SECTOR: {
        "primary_law": "labor_code_article_82",
        "supporting_laws": [],
        "controlling_rule": "exclusion_rule",
        "excluded_general_topics": ["ordinary_article_87_overtime_multiplier"],
    },
    MANAGERIAL_STAFF_SECTOR: {
        "primary_law": "labor_code_article_82",
        "supporting_laws": [],
        "controlling_rule": "exclusion_rule",
        "excluded_general_topics": ["ordinary_article_87_overtime_multiplier"],
    },
    FIELD_SECTOR: {
        "primary_law": "labor_code_article_82",
        "supporting_laws": [],
        "controlling_rule": "exclusion_rule",
        "excluded_general_topics": ["ordinary_article_87_overtime_multiplier"],
    },
}


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _norm(value).lower()


def _metadata_text(metadata: Dict[str, Any]) -> str:
    keys = (
        "document_title",
        "source_name",
        "law_number",
        "article_section",
        "issuing_agency",
        "book",
        "title",
        "chapter",
        "rule",
        "section",
        "rule_title",
        "section_title",
        "article",
        "article_title",
        "topic",
        "sector",
        "coverage",
        "exclusions",
        "rule_type",
        "citation",
        "category",
        "primary_category",
        "secondary_categories",
        "topics",
        "sub_intents",
        "implements",
        "document_type",
        "legal_role",
        "canonical_document_id",
        "legal_status",
        "official_url",
        "source_url",
        "document_year",
        "doc_type",
    )
    return " ".join(_norm(metadata.get(key)) for key in keys if metadata.get(key))


def _concept_supported(concept: str, source: Dict[str, Any]) -> bool:
    from routers.services.legal_topic_matrix import supported_passage_concepts
    return concept in supported_passage_concepts(source, [concept])


def _supported_profile_concepts(required: List[str], sources: Iterable[Dict[str, Any]]) -> set[str]:
    supported: set[str] = set()
    for source in sources or []:
        for concept in required:
            if _concept_supported(concept, source):
                supported.add(concept)
    return supported


def canonical_sector(value: str) -> str:
    text = _lower(value).replace("-", "_").replace(" ", "_")
    if text in {
        HOSPITAL_SECTOR,
        "hospital",
        "clinic",
        "hospital_clinic",
        "hospital_and_clinic_personnel",
    }:
        return HOSPITAL_SECTOR
    if text in {GOVERNMENT_SECTOR, "government", "public_sector", "civil_service"}:
        return GOVERNMENT_SECTOR
    if text in {
        DOMESTIC_SECTOR,
        "kasambahay",
        "kasambhay",
        "domestic_worker",
        "domestic_workers",
        "domestics_worker",
        "domestics_workers",
        "domestic_helper",
        "domestic_helpers",
        "househelper",
        "househelpers",
        "house_helper",
        "house_helpers",
        "household_helper",
        "household_helpers",
        "household_worker",
        "household_workers",
        "household_service",
        "personal_service",
        "yaya",
        "maid",
        "housemaid",
        "house_maid",
    }:
        return DOMESTIC_SECTOR
    if text in {MANAGERIAL_SECTOR, "managerial", "managerial_employees"}:
        return MANAGERIAL_SECTOR
    if text in {MANAGERIAL_STAFF_SECTOR, "managerial_staff", "member_of_managerial_staff"}:
        return MANAGERIAL_STAFF_SECTOR
    if text in {FIELD_SECTOR, "field", "field_workers"}:
        return FIELD_SECTOR
    if text in {NIGHT_WORKERS_SECTOR, "night_worker", "night_workers", "night_shift_worker"}:
        return NIGHT_WORKERS_SECTOR
    if text in {MINOR_WORKERS_SECTOR, "minor_worker", "minor_workers", "working_child"}:
        return MINOR_WORKERS_SECTOR
    if text in {OVERSEAS_WORKERS_SECTOR, "ofw", "overseas_worker", "overseas_workers", "migrant_worker"}:
        return OVERSEAS_WORKERS_SECTOR
    if text in {UNKNOWN_SECTOR, "unknown_sector"}:
        return UNKNOWN_SECTOR
    return text or GENERAL_SECTOR


def reviewed_document_sector(metadata: Optional[Dict[str, Any]]) -> str:
    value = _norm((metadata or {}).get('document_sector'))
    normalized = canonical_sector(value)
    known = {GENERAL_SECTOR, GOVERNMENT_SECTOR, HOSPITAL_SECTOR, DOMESTIC_SECTOR,
             MANAGERIAL_SECTOR, MANAGERIAL_STAFF_SECTOR, FIELD_SECTOR,
             NIGHT_WORKERS_SECTOR, MINOR_WORKERS_SECTOR, OVERSEAS_WORKERS_SECTOR}
    return normalized if value and normalized in known else ''


def infer_sector(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    # Reviewed owner coverage is stronger than incidental words in a fragment.
    # Only accept known labels; this is metadata, not an invented legal rule.
    owner_sector = reviewed_document_sector(metadata)
    if owner_sector:
        return owner_sector
    combined = f"{text or ''} {_metadata_text(metadata or {})}"
    existing = _norm((metadata or {}).get("sector"))
    if (metadata or {}).get("primary_category", (metadata or {}).get("category")) == "Civil Service / Government Employment":
        return GOVERNMENT_SECTOR
    if existing:
        return canonical_sector(existing)
    canonical_doc = _norm((metadata or {}).get("canonical_document_id")) or infer_canonical_document_id(text, metadata)
    article_number = _norm((metadata or {}).get("article_number")) or _extract_article_number_from_text(combined)
    if canonical_doc == "labor_code_pd_442" and article_number in {"82", "83", "84", "85", "86", "87", "88", "89", "93", "94", "95", "298", "302"}:
        return GENERAL_SECTOR
    if GOVERNMENT_RE.search(combined):
        return GOVERNMENT_SECTOR
    if HOSPITAL_RE.search(combined):
        return HOSPITAL_SECTOR
    if DOMESTIC_RE.search(combined):
        return DOMESTIC_SECTOR
    if MANAGERIAL_RE.search(combined):
        return MANAGERIAL_SECTOR
    if FIELD_RE.search(combined):
        return FIELD_SECTOR
    if NIGHT_WORKER_RE.search(combined):
        return NIGHT_WORKERS_SECTOR
    if MINOR_WORKER_RE.search(combined):
        return MINOR_WORKERS_SECTOR
    if OVERSEAS_WORKER_RE.search(combined):
        return OVERSEAS_WORKERS_SECTOR
    return GENERAL_SECTOR


def infer_query_sector(text: str) -> str:
    """Resolve employment scope without treating a regulator as the employer.

    Document tagging intentionally examines all coverage words; a question must
    instead prioritize the employment relationship being described. Ambiguous
    mentions must not activate the government-only retrieval guard.
    """
    from .question_understanding import employment_context_parts
    text, _ = employment_context_parts(text)
    if re.search(r'\bgovernment\s+contractor\b', text, re.I):
        return UNKNOWN_SECTOR
    # The worker is asking whether a classification applies, not asserting it.
    text = re.sub(r'\b(?:am|are)\s+(?:i|we)\b[^.!?]*', '', text, flags=re.I)
    # Stop at a new clause: a destination, supervisor, relative or comparison
    # elsewhere in the question is not part of this worker's employment facts.
    employment = re.compile(
        r"\b(?:i\s+(?:now\s+)?(?:am|work|worked)|i'm|we\s+(?:are|work|worked)|"
        r"my\s+employer\s+is|(?:he|she)\s+(?:is|works)|"
        r"[A-Z][a-z]+\s+works)\b"
        r"(?P<facts>[^.!?;\n,]{0,180})", re.I,
    )
    scopes = set()
    speaker_scopes = set()
    for match in employment.finditer(text):
        facts = re.split(
            r"\b(?:but|and|can|should|because|who|which|report|complain|contact|visit|consult)\b",
            match['facts'], maxsplit=1, flags=re.I,
        )[0]
        # Negated descriptions are not positive evidence of sector membership.
        if re.search(r"\b(?:not|never|no\s+longer)\b", facts, re.I):
            continue
        if re.search(r"\bprivate\b", facts, re.I) or any(pattern.search(facts) for pattern in (
            GOVERNMENT_RE, HOSPITAL_RE, DOMESTIC_RE, MANAGERIAL_RE,
            FIELD_RE, NIGHT_WORKER_RE, MINOR_WORKER_RE, OVERSEAS_WORKER_RE,
        )):
            scope = infer_sector(facts)
            scopes.add(scope)
            if re.match(r"(?:i\b|i'm\b|we\b|my\s+employer\b)", match[0], re.I):
                speaker_scopes.add(scope)
    if speaker_scopes:
        scopes = speaker_scopes
    if len(scopes) == 1:
        return scopes.pop()
    if len(scopes) > 1:
        # Multiple workers/scopes need broad evidence, not a single-sector veto.
        return UNKNOWN_SECTOR

    # For general questions, require a government employment/coverage phrase.
    # Bare agency names and "government office" can be complaint destinations.
    government_employment = re.search(
        r"\b(?:government|public(?:[-\s]+sector)?|civil[-\s]+service)\s+"
        r"(?:employees?|workers?|personnel|nurses?|teachers?|employment|benefits|leave|overtime)\b|"
        r"\b(?:public[-\s]+sector|civil[-\s]+service)\b|"
        r"\b(?:employed|working|work)\s+(?:at|in|for|by)\s+(?:a\s+|the\s+)?"
        r"(?:government|lgu|deped|doh|state\s+university)\b",
        text, re.I,
    )
    if government_employment:
        prefix = text[max(0, government_employment.start() - 20):government_employment.start()]
        if not re.search(r"\b(?:not|never)\s+(?:(?:a|an)\s+)?$", prefix, re.I):
            return GOVERNMENT_SECTOR
    return infer_sector(GOVERNMENT_RE.sub("", text))


def infer_topic(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    # A supplied fact (e.g. "regular employee") must not replace the user's
    # actual question (e.g. paid-leave eligibility) in a clarification chain.
    parts = re.split(r'\n(?:User clarification|Clarification from user|Clarified leave type)\s*[^:]*:', text or '', maxsplit=1, flags=re.I)
    if len(parts) == 2 and not _norm((metadata or {}).get('topic')):
        original_topic = infer_topic(parts[0], metadata)
        if original_topic not in {'general_labor', 'non_legal'}:
            # An explicit leave choice can narrow generic leave entitlement.
            if original_topic == 'service_incentive_leave':
                choice_topic = infer_topic(parts[1])
                if choice_topic in {'maternity_leave', 'paternity_leave', 'solo_parent_leave', 'service_incentive_leave'}:
                    return choice_topic
            return original_topic
    combined = f"{text or ''} {_metadata_text(metadata or {})}"
    existing = _norm((metadata or {}).get("topic"))
    if existing:
        return canonical_topic(existing)
    # A specific extra-hours calculation can also mention a daily salary and
    # an underpayment. Preserve overtime as the governing issue before the
    # broader everyday wage-payment classifier claims the query.
    from routers.services.numeric_quantities import explicit_overtime_hours
    if (explicit_overtime_hours(text) is not None
            and re.search(r"\b(?:work(?:ed|ing)?|stay(?:ed|ing)?|pay|paid|wage|salary)\b", text, re.I)):
        return "overtime_pay"
    from routers.services.scenario_retrieval import everyday_topic
    scenario_topic = everyday_topic(text)
    if scenario_topic:
        return scenario_topic
    if re.search(r"\b(?:supreme\s+court|jurisprudence|case\s+law|court\s+decision|what\s+did\s+the\s+court\s+decide)\b", combined, re.I):
        return "jurisprudence_request"
    if re.search(r'\bquitclaim\b', text or '', re.I):
        return 'jurisprudence_request'
    if re.search(r'\bfinal[-\s]+pay\b', text or '', re.I):
        if re.search(r'\bdole\b|complain|reklamo|tulong|assistance', text, re.I):
            return 'labor_complaint'
        # Final pay can contain a specifically requested benefit. Preserve
        # that benefit's evidence requirements instead of routing to wages.
        benefit_topic = matrix_classify_topic(combined, metadata)
        if benefit_topic not in {'general_labor', 'non_legal', 'payment_of_wages', 'termination'}:
            return benefit_topic
        return 'payment_of_wages'
    if (re.search(r'binayaran|unpaid|sahod|wages?|salary', combined, re.I)
            and re.search(r'saan.{0,65}tulong|where.{0,50}(?:help|complain)|humingi ng tulong', combined, re.I)):
        return 'labor_complaint'
    matrix_topic = matrix_classify_topic(combined, metadata)
    if matrix_topic != "general_labor":
        return matrix_topic
    if re.search(r"\bdiscriminat\w*\b", combined, re.I):
        return "discrimination"
    if re.search(r"\bworked?\s+(?:on|during)\s+(?:a\s+)?(?:regular\s+)?holiday\b", combined, re.I):
        return "holiday_pay"
    if DOMESTIC_RE.search(combined):
        return "domestic_workers"
    if re.search(r"\b(13th[-\s]*month|thirteenth[-\s]*month|pd\s*851)\b", combined, re.IGNORECASE):
        return "thirteenth_month_pay"
    if OVERTIME_RE.search(combined) or re.search(r"\b(article|art\.?)\s*(82|84|87|88|89)\b", combined, re.IGNORECASE):
        return "overtime_pay"
    if re.search(r"\b(minimum\s+wage|wage\s+order|salary|sahod|sweldo|suhol)\b", combined, re.IGNORECASE):
        return "payment_of_wages"
    if re.search(r"\b(leave|service\s+incentive|maternity|paternity)\b", combined, re.IGNORECASE):
        return "service_incentive_leave"
    if re.search(r"\b(termination|dismissal|separation\s+pay|resign|resignation)\b", combined, re.IGNORECASE):
        return "termination"
    if re.search(r"\b(ginpahalin|ginpahalinan|gintanggal|tinanggal|natanggal)\b", combined, re.IGNORECASE):
        return "termination"
    if re.search(
        r"\b(workplace\s+safety|osh|oshs|occupational\s+(?:safety|health)|"
        r"safety\s+and\s+health|hazards?|safety\s+committee|safety\s+officer|"
        r"ra\s*11058|do\s*198[-\s]?18)\b",
        combined,
        re.IGNORECASE,
    ):
        return "workplace_safety"
    return "general_labor"


def infer_sub_intent(text: str, topic: str = "") -> str:
    """Return the most specific supported legal task for targeted retrieval.

    Matching is deliberately ordered from narrow/compound questions to broad
    ones.  The broad ``topic`` value remains unchanged so existing callers and
    stored chat records stay backwards compatible.
    """
    normalized = _norm(text)
    # Compound termination comparisons must win over single-cause rules.
    termination_terms = re.search(r"\b(termination|terminate|dismiss|fired|just\s+cause|authorized\s+cause)\b", normalized)
    comparison_terms = re.search(r"\b(difference|compare|comparison|versus|vs\.?|both|including)\b", normalized)
    both_causes = re.search(r"\bjust\s+cause", normalized) and re.search(r"\bauthorized\s+cause", normalized)
    requested_complements = re.search(r"\b(notice|due\s+process|separation\s+pay|grounds|examples?)\b", normalized)
    if termination_terms and ((both_causes and comparison_terms) or (both_causes and requested_complements)):
        return "termination_comparison"
    # Premium/overtime combinations must win over the broader holiday or
    # rest-day labels because their controlling multiplier is different.
    if re.search(r"\b(overtime|extra\s+hours?|ot)\b", normalized, re.IGNORECASE):
        if re.search(r"\b(holiday|special\s+day)\b", normalized, re.IGNORECASE) and re.search(r"\b(rest\s+day|day\s+off)\b", normalized, re.IGNORECASE):
            return "holiday_rest_day_overtime"
        if re.search(r"\b(regular\s+holiday|holiday)\b", normalized, re.IGNORECASE):
            return "regular_holiday_overtime"
        if re.search(r"\b(rest\s+day|day\s+off)\b", normalized, re.IGNORECASE):
            return "rest_day_overtime"
        if re.search(r"\b(ordinary|normal\s+working\s+day)\b", normalized, re.IGNORECASE):
            return "ordinary_overtime"
    for sub_intent, pattern in SUB_INTENT_RULES:
        if re.search(pattern, normalized, re.IGNORECASE | re.DOTALL):
            return sub_intent
    canonical = canonical_topic(topic)
    return {
        "thirteenth_month_pay": "thirteenth_month_coverage",
        "maternity_leave": "maternity_standard_duration",
        "solo_parent_leave": "solo_parent_leave",
        "night_shift_differential": "night_shift_differential",
        "minimum_wage": "minimum_wage",
        "workplace_safety": "osh_employer_duties",
    }.get(canonical, canonical or "general_labor")


def infer_task(
    text: str,
    topic: str,
    sector: str = GENERAL_SECTOR,
    intent: str = "",
    calculation_requested: bool = False,
) -> str:
    normalized = _norm(text)
    if intent in NON_LEGAL_INTENTS:
        return intent
    if calculation_requested:
        return "calculation_request"
    canonical = canonical_topic(topic)
    if canonical == "holiday_pay" and re.search(
        r"\b(rest\s+day|day\s+off)\b",
        normalized,
        re.IGNORECASE,
    ):
        return "rest_day_question"
    if canonical == "workplace_safety" and re.search(
        r"\b(refuse|decline|stop)\b.{0,100}\b(unsafe|danger|hazard|work)\b|\bright\s+to\s+refuse\b",
        normalized,
        re.IGNORECASE,
    ):
        return "refusal_of_unsafe_work"
    if canonical == "workplace_safety":
        from routers.services.scenario_retrieval import EXIT_QUERY
        if re.search(EXIT_QUERY, normalized, re.I):
            return 'hazard_reporting'
        sub_intent = infer_sub_intent(normalized, canonical)
        return {
            "ppe_requirement": "ppe_requirement",
            "safety_officer_requirement": "safety_officer_requirement",
            "osh_program_requirement": "osh_program_requirement",
            "osh_committee_requirement": "osh_committee_requirement",
            "osh_employer_duties": "employer_duties",
        }.get(sub_intent, "general_explanation")
    if canonical in {"sexual_harassment", "safe_spaces"} and re.search(
        r"\b(employer\s+dut(?:y|ies)|what\s+must\s+(?:an?\s+)?employer|policy|codi|committee|investigat|prevent)\b",
        normalized,
        re.IGNORECASE,
    ):
        return "employer_duties"
    if FOLLOW_UP_TASK_RE.search(normalized):
        return "follow_up_question"
    if SOURCE_EXPLANATION_TASK_RE.search(normalized) and not CITATION_TASK_RE.search(normalized):
        return "source_explanation"
    if canonical == "hours_of_work":
        if MEAL_PERIOD_TASK_RE.search(normalized):
            return "meal_period_question"
        return "normal_hours_question"
    if canonical == "cost_of_living_allowance":
        return "eligibility_question" if COVERAGE_TASK_RE.search(normalized) or re.search(
            r"\b(basis|condition|entitled|eligible|qualification)\b", normalized, re.IGNORECASE
        ) else "general_explanation"
    if canonical == "holiday_pay":
        if SUCCESSIVE_HOLIDAY_TASK_RE.search(normalized):
            return "successive_holidays_question"
        if HOLIDAY_ELIGIBILITY_TASK_RE.search(normalized):
            return "eligibility_question"
    if (
        canonical == "overtime_pay"
        and sector != GENERAL_SECTOR
        and (
            COVERAGE_TASK_RE.search(normalized)
            or EXCLUSION_TASK_RE.search(normalized)
            or OVERTIME_RE.search(normalized)
            or re.search(r"\b(?:article|art\.?)\s*(82|87)\b", normalized, re.IGNORECASE)
        )
    ):
        return "sector_coverage_question"
    if CITATION_TASK_RE.search(normalized):
        return "citation_request"
    if REMEDY_TASK_RE.search(normalized):
        return "legal_remedy"
    if DOCUMENT_TASK_RE.search(normalized):
        return "document_requirement"
    if FALSE_ASSUMPTION_TASK_RE.search(normalized):
        return "false_assumption_question"
    if COMPARISON_TASK_RE.search(normalized):
        return "comparison_question"
    if canonical == "overtime_pay":
        if sector != GENERAL_SECTOR:
            return "sector_coverage_question"
        if UNDERTIME_TASK_RE.search(normalized):
            return "undertime_offset_question"
        if COMPULSORY_TASK_RE.search(normalized):
            return "compulsory_overtime_question"
        if AUTHORIZATION_TASK_RE.search(normalized):
            return "authorization_question"
        if EXCLUSION_TASK_RE.search(normalized):
            return "exclusion_question"
        if COVERAGE_TASK_RE.search(normalized):
            return "coverage_question"
        if THRESHOLD_TASK_RE.search(normalized):
            return "threshold_question"
        if RATE_TASK_RE.search(normalized):
            return "rate_question"
        if FORMULA_TASK_RE.search(normalized):
            return "formula_question"
        return "general_explanation"

    if canonical == "domestic_workers" and OVERTIME_RE.search(normalized):
        return "overtime_question"

    if DEADLINE_TASK_RE.search(normalized):
        return "deadline_question"
    if EXCLUSION_TASK_RE.search(normalized):
        return "exclusion_question"
    if COVERAGE_TASK_RE.search(normalized):
        return "coverage_question"
    if RATE_TASK_RE.search(normalized):
        return "rate_question"
    if FORMULA_TASK_RE.search(normalized):
        return "formula_question"
    if DEFINITION_TASK_RE.search(normalized):
        return "definition_question"
    return "general_explanation"


def sector_rule_precedence(sector: str) -> Dict[str, Any]:
    return dict(SECTOR_RULE_PRECEDENCE.get(canonical_sector(sector), {}))


def is_special_sector_overtime_query(query_profile: Dict[str, Any]) -> bool:
    if canonical_topic(query_profile.get("topic")) != "overtime_pay":
        return False
    return canonical_sector(_norm(query_profile.get("sector"))) in SECTOR_RULE_PRECEDENCE


def is_domestic_worker_overtime_query(query_profile: Dict[str, Any]) -> bool:
    sector = canonical_sector(_norm(query_profile.get("sector")))
    if sector != DOMESTIC_SECTOR:
        return False
    topic = canonical_topic(query_profile.get("topic"))
    task = _norm(query_profile.get("task"))
    query = _norm(query_profile.get("normalized_query"))
    return (
        (topic == "overtime_pay" and task == "sector_coverage_question")
        or (topic == "domestic_workers" and task == "overtime_question")
        or bool(OVERTIME_RE.search(query))
    )


def infer_rule_type(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    combined = f"{text or ''} {_metadata_text(metadata or {})}"
    existing = _norm((metadata or {}).get("rule_type"))
    if existing:
        return existing
    if HOSPITAL_RE.search(combined) or "rule i-a" in combined.lower():
        return SPECIAL_SECTOR_RULE
    if re.search(r"\b(coverage|covered|applies?|excluded|exempt|article\s*82|art\.?\s*82)\b", combined, re.IGNORECASE):
        return COVERAGE_RULE
    if re.search(r"\b(except|exception|excluded|shall\s+not\s+apply|not\s+covered)\b", combined, re.IGNORECASE):
        return EXCEPTION_RULE
    if re.search(r"\b(means|defined|definition|refers\s+to)\b", combined, re.IGNORECASE):
        return DEFINITION_RULE
    if re.search(r"\b(eligible|eligibility|qualified|qualification|entitled|coverage)\b", combined, re.IGNORECASE):
        return ELIGIBILITY_RULE
    if re.search(r"\b(rate|plus\s+at\s+least|125%|130%|twenty-five|thirty\s+percent|formula|computed)\b", combined, re.IGNORECASE):
        return COMPUTATION_RULE
    if re.search(r"\b(deadline|not\s+later\s+than|within\s+\d+\s+days|december\s+24|prescriptive\s+period)\b", combined, re.IGNORECASE):
        return DEADLINE_RULE
    if re.search(r"\b(procedure|filing|notice|hearing|submission|permit)\b", combined, re.IGNORECASE):
        return PROCEDURAL_RULE
    if re.search(r"\b(penalty|penalties|fine|imprisonment|sanction)\b", combined, re.IGNORECASE):
        return PENALTY_RULE
    if re.search(r"\b(supreme\s+court|jurisprudence|g\.r\.\s+no\.|case\s+law)\b", combined, re.IGNORECASE):
        return JURISPRUDENTIAL_RULE
    return GENERAL_RULE


def infer_coverage(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    existing = _norm((metadata or {}).get("coverage"))
    if existing:
        return existing
    combined = f"{text or ''} {_metadata_text(metadata or {})}"
    if HOSPITAL_RE.search(combined):
        return "Hospital and clinic personnel."
    if re.search(r"\b(article|art\.?)\s*82\b", combined, re.IGNORECASE):
        return "Coverage and exclusions for hours-of-work provisions."
    if OVERTIME_RE.search(combined):
        return "General private-sector employees covered by hours-of-work rules, subject to statutory exclusions."
    return ""


def infer_scope_flags(metadata: Dict[str, Any]) -> Dict[str, Any]:
    sector = canonical_sector(_norm(metadata.get("sector")))
    rule_type = _norm(metadata.get("rule_type")) or GENERAL_RULE
    is_special = rule_type == SPECIAL_SECTOR_RULE or sector not in {"", GENERAL_SECTOR}
    scope = _norm(metadata.get("coverage"))
    if not scope:
        if sector == HOSPITAL_SECTOR:
            scope = "Hospital and clinic personnel."
        elif sector == GOVERNMENT_SECTOR:
            scope = "Government employment or civil-service context."
        elif sector == DOMESTIC_SECTOR:
            scope = "Domestic workers or kasambahay."
        elif sector == MANAGERIAL_SECTOR:
            scope = "Managerial employees or qualifying managerial staff."
        elif sector == MANAGERIAL_STAFF_SECTOR:
            scope = "Members of managerial staff."
        elif sector == FIELD_SECTOR:
            scope = "Field personnel."
        elif sector == NIGHT_WORKERS_SECTOR:
            scope = "Night workers or night-shift work."
        elif sector == MINOR_WORKERS_SECTOR:
            scope = "Minor workers or working children."
        elif sector == OVERSEAS_WORKERS_SECTOR:
            scope = "Overseas Filipino or migrant workers."
        elif sector == UNKNOWN_SECTOR:
            scope = "Unknown or unclassified employment sector."
        else:
            scope = "General private employment."
    return {
        "source_scope": scope,
        "is_general_rule": not is_special,
        "is_special_rule": is_special,
    }


def classify_legal_query(message: str, nlp: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    text = message or ""
    normalized = _norm((nlp or {}).get("normalized")) or text
    intent = _norm((nlp or {}).get("intent")) or "general_inquiry"
    sector = infer_query_sector(text)
    topic = infer_topic(text if re.search(r'\n(?:User clarification|Clarification from user|Clarified leave type)', text, re.I) else normalized)
    if topic == 'government_employment' and sector != GOVERNMENT_SECTOR:
        # The topic classifier must not reintroduce the same incidental-sector
        # routing mistake after the employment relationship has been resolved.
        topic = infer_topic(GOVERNMENT_RE.sub('', text))
    wage_details = wage_order_query_details(normalized)
    if wage_details["is_wage_order_query"]:
        topic = "minimum_wage"
    overtime_like = bool(
        OVERTIME_RE.search(normalized)
        or re.search(r"\b(?:article|art\.?)\s*(82|84|87|88|89|93|94)\b", normalized, re.IGNORECASE)
        or re.search(r"\b(hours?\s+of\s+work|forty[-\s]?hour|40[-\s]?hour|work\s+beyond\s+eight)\b", normalized, re.IGNORECASE)
    )
    if overtime_like and sector != GENERAL_SECTOR:
        topic = "overtime_pay"
    elif sector == DOMESTIC_SECTOR and topic not in {"minimum_wage", "payment_of_wages", "wage_deductions"}:
        topic = "domestic_workers"

    from routers.services.clarification_intake import broad_benefits_query
    broad_benefits = broad_benefits_query(text)
    benefit_categories = []
    if broad_benefits:
        intent = "general_inquiry"
        topic = "general_labor"
        if sector == GOVERNMENT_SECTOR:
            benefit_categories = ["Civil Service / Government Employment"]
        elif sector == DOMESTIC_SECTOR:
            topic = "domestic_workers"
            benefit_categories = ["Domestic Workers", "Social Benefits"]
        else:
            benefit_categories = ["Employment Benefits", "Leave Benefits", "Social Benefits"]

    calculation_requested = bool(PERSONAL_CALCULATION_RE.search(normalized))
    # Explicit total hours plus a personal pay question is an overtime scenario
    # even when the user has not called those hours "overtime" yet.
    if (topic == "general_labor" and calculation_requested
            and re.search(r"\b(?:worked|working|work\s+for)\b", normalized, re.I)
            and re.search(r"\b\d+(?:\.\d+)?\s*(?:hours?|hrs?)\b", normalized, re.I)
            and re.search(r"\b(?:pay|paid|wage|salary|much|amount)\b", normalized, re.I)):
        topic = "overtime_pay"

    # An unscoped legal question can concern either private employment or
    # government service. Preserve that ambiguity instead of treating the
    # default general-private classifier value as an explicit user choice.
    from routers.services.evidence_scope import requested_scope
    both_sector_comparison = bool(re.search(
        r"\b(?:government|public(?:[-\s]+sector)?|civil[-\s]+service)\b.{0,60}\b(?:and|versus|vs\.?|or)\b.{0,40}\bprivate(?:[-\s]+sector)?\b|"
        r"\bprivate(?:[-\s]+sector)?\b.{0,60}\b(?:and|versus|vs\.?|or)\b.{0,40}\b(?:government|public(?:[-\s]+sector)?|civil[-\s]+service)\b",
        text, re.I,
    ))
    compare_government_private = bool(
        intent not in NON_LEGAL_INTENTS
        and (sector == GENERAL_SECTOR or both_sector_comparison)
        and topic not in {"government_employment", "domestic_workers"}
        and (requested_scope(text) is None or both_sector_comparison)
    )

    if intent in NON_LEGAL_INTENTS:
        topic = "non_legal"
        task = intent
        source_match = SOURCE_STATUS_NOT_REQUIRED
    else:
        task = infer_task(
            normalized,
            topic,
            sector=sector,
            intent=intent,
            calculation_requested=calculation_requested,
        )
        source_match = SOURCE_STATUS_INSUFFICIENT

    sub_intent = infer_sub_intent(normalized, topic)
    # A comparison needs the task-level requirements for every concept in the
    # question. A narrower sub-intent would otherwise drop one side.
    if task == "comparison_question" and sub_intent != "termination_comparison":
        sub_intent = ""
    if sub_intent == "termination_comparison":
        # "Authorized cause" also appears in separation-pay questions. When the
        # user compares both causes, retain the termination route so both
        # governing provisions are mandatory.
        topic = "termination"
    # Some concise questions do not contain the broad topic keywords used by
    # the legacy classifier. Promote them to the correct retrieval route while
    # retaining the same public topic names for existing integrations.
    if sub_intent in {"rest_day_pay", "special_day_pay", "regular_holiday_on_rest_day"} and topic == "general_labor":
        topic = "premium_pay" if sub_intent != "regular_holiday_on_rest_day" else "holiday_pay"
    elif sub_intent == "codi_definition" and topic == "general_labor":
        topic = "safe_spaces"
    elif sub_intent == "cba_definition" and topic == "general_labor":
        topic = "collective_bargaining"
    elif sub_intent.startswith("termination_") and topic == "general_labor":
        topic = "termination"
    elif sub_intent == "right_to_self_organization":
        # Employment status (including probationary status) is a coverage
        # fact here; the legal issue being asked is union self-organization.
        # Route the query to Labor Relations so a probationary-employment
        # passage cannot displace the controlling union-rights source.
        topic = "union_rights"
    elif sub_intent == "probationary_termination":
        topic = "probationary_employment"
    elif sub_intent in {
        "right_to_refuse_unsafe_work", "osh_employer_duties", "ppe_requirement",
        "safety_officer_requirement", "osh_program_requirement", "osh_committee_requirement",
    } and topic == "general_labor":
        topic = "workplace_safety"
        task = infer_task(normalized, topic, sector, intent, calculation_requested)

    article_match = ARTICLE_RE.search(normalized)
    issue_topic = topic
    if sector == GOVERNMENT_SECTOR:
        # Keep the issue in the original query but never force private-sector
        # statutory entitlements onto government personnel questions.
        topic = "government_employment"
        sub_intent = "government_employment"
    from routers.services.scenario_retrieval import is_leave_recall_query
    if is_leave_recall_query(text):
        topic = "approved_leave_recall"
        sub_intent = "approved_leave_recall"
        task = "general_explanation"
    section_match = SECTION_RE.search(normalized)
    required_concepts = required_concepts_for_topic(topic, normalized, task=task)
    essential_concepts = essential_concepts_for_topic(topic, normalized, task=task)
    override = SUB_INTENT_CONCEPT_OVERRIDES.get(sub_intent)
    if override:
        required_concepts, essential_concepts = list(override[0]), list(override[1])
        if task in {"calculation_request", "comparison_question"}:
            required_concepts = list(dict.fromkeys([
                *required_concepts, *required_concepts_for_topic(topic, normalized, task=task),
            ]))
    if topic in {"termination", "procedural_due_process", "separation_pay"} and re.search(r"\bseparation\s+pay\b", normalized, re.I):
        required_concepts = list(dict.fromkeys([*required_concepts, "separation_pay_rule"]))
    if sub_intent == "security_floating_status" and re.search(
        r"\b(?:unpaid|without\s+pay|no\s+pay|wages?|salary|compensation|payment)\b",
        normalized,
        re.IGNORECASE,
    ):
        # Wage entitlement during reserved status needs direct evidence; the
        # general rule for a suspended business operation is not equivalent.
        required_concepts.append("security_reserved_status_pay_rule")
        essential_concepts.append("security_reserved_status_pay_rule")
        required_concepts = list(dict.fromkeys(required_concepts))
        essential_concepts = list(dict.fromkeys(essential_concepts))
    preferred_sources = preferred_sources_for_topic(topic, task=task)
    required_authorities = required_authorities_for_topic(topic, task=task)
    if sector == GOVERNMENT_SECTOR:
        required_concepts, essential_concepts = [], []
        preferred_sources = []
        required_authorities = {"essential": [], "supporting": []}
    if topic == "approved_leave_recall":
        # Entitlement does not establish cancellation/recall. Do not force SIL
        # concepts or a single statute onto an issue with unspecified leave type.
        required_concepts, essential_concepts = [], []
        preferred_sources = []
        required_authorities = {"essential": [], "supporting": []}
    from routers.services.issue_evidence import safety_issue_contract
    issue_contract = safety_issue_contract(text) if sector != GOVERNMENT_SECTOR else {}
    if issue_contract and intent not in NON_LEGAL_INTENTS:
        topic, task = issue_contract['topic'], issue_contract['task']
        sub_intent = 'issue_specific_safety'
        required_concepts = issue_contract['required']
        essential_concepts = issue_contract['essential']
        preferred_sources = ['RA 11058', 'DOLE Department Order No. 252-25', 'Occupational Safety and Health Standards']
        # The statute establishes core duties. Implementing and technical rules
        # supplement it; do not demand every rule from a single mandatory IRR.
        required_authorities = {'essential': ['ra_11058'],
                                'supporting': ['dole_do_252_25', 'occupational_safety_health_standards']}
        if 'ec_eligibility_conditions' in required_concepts:
            benefit_categories = ['Occupational Safety & Health', 'Social Benefits', 'DOLE Resources']
    authority_override = {
        "social_contribution_remittance": {
            "essential": ["ra_11199", "ra_11223", "ra_9679"],
            "supporting": [],
        },
        "sss_contribution": {
            "essential": ["sss_contribution_schedule_2025"],
            "supporting": ["ra_11199"],
        },
        "philhealth_contribution": {"essential": ["ra_11223"], "supporting": []},
        "pagibig_contribution": {"essential": ["ra_9679"], "supporting": []},
        "termination_just_cause": {"essential": ["labor_code_article_297"], "supporting": []},
        "termination_authorized_cause": {"essential": ["labor_code_article_298"], "supporting": []},
        "termination_comparison": {
            "essential": ["labor_code_article_297", "labor_code_article_298"],
            "supporting": ["omnibus_rules_labor_code"],
        },
        "probationary_termination": {
            "essential": ["labor_code_article_296"],
            # Once probation has expired, distinguish just cause from
            # authorized cause rather than implying only one route is lawful.
            "supporting": ["labor_code_article_297", "labor_code_article_298"],
        },
        "regular_holiday_pay": {"essential": ["labor_code_article_94"], "supporting": ["omnibus_rules_labor_code"]},
        "regular_holiday_on_rest_day": {"essential": ["labor_code_article_94"], "supporting": ["labor_code_article_93"]},
        "night_shift_differential": {"essential": ["labor_code_article_86"], "supporting": []},
    }.get(sub_intent)
    if authority_override:
        required_authorities = authority_override
    if sub_intent == "termination_authorized_cause" and re.search(r"\bdisease|illness|medical\b", normalized, re.IGNORECASE):
        required_authorities = {
            "essential": ["labor_code_article_298", "labor_code_article_299"],
            "supporting": [],
        }
    if (
        canonical_topic(topic) == "overtime_pay"
        and canonical_sector(sector) == DOMESTIC_SECTOR
        and task == "sector_coverage_question"
    ):
        essential_concepts = ["labor_code_domestic_exclusion", "ra10361_contract_additional_payment"]
        required_concepts = [
            "labor_code_domestic_exclusion",
            "ra10361_contract_additional_payment",
            "kasambahay_daily_rest_period",
            "kasambahay_weekly_rest_period",
        ]
        preferred_sources = [
            "Labor Code Article 82",
            "RA 10361 Section 11",
            "RA 10361 Section 20",
            "RA 10361 Section 21",
            "IRR of RA 10361",
        ]
        required_authorities = required_authorities_for_topic(topic, task=task)
    from routers.services.kb_categories import (
        KB_CATEGORIES, SIDEBAR_TOPIC_CATEGORY_MAP, TOPIC_KB_CATEGORY_MAP, TOPIC_KB_CATEGORY_ROUTES,
    )
    category = SIDEBAR_TOPIC_CATEGORY_MAP.get(topic)
    if category not in KB_CATEGORIES:
        category = TOPIC_KB_CATEGORY_MAP.get(topic)
    category = category if category in KB_CATEGORIES else "Others"
    primary, secondary = TOPIC_KB_CATEGORY_ROUTES.get(topic, ([category], []))
    if sector == DOMESTIC_SECTOR and topic in {"overtime_pay", "holiday_pay", "payment_of_wages"}:
        # Preserve the worker's special-sector identity while still allowing
        # the issue's ordinary benefits and core-law sources to be discovered.
        category = "Domestic Workers"
        primary = ["Domestic Workers", *primary]
    if sector == GOVERNMENT_SECTOR and issue_topic == "overtime_pay":
        primary = ["Civil Service / Government Employment", "Core Labor Laws", "Employment Benefits"]
    category_hints = list(dict.fromkeys(value for value in [*primary, *secondary]
                                        if value in KB_CATEGORIES))
    from routers.services.conversation_memory import extract_user_facts
    scenario = extract_user_facts(text, nlp)
    if (topic == "overtime_pay" and task in {"calculation_request", "personalized_computation"}
            and scenario.get("day_type") == "ordinary_day"):
        # A stated ordinary workday does not need holiday/rest-day rates to
        # establish the requested calculation. Keep the ordinary overtime
        # rule as the source requirement and leave independent claim checking
        # in place for the salary-inclusion question.
        required_concepts = ["ordinary_overtime_rule"]
        essential_concepts = ["ordinary_overtime_rule"]
        required_authorities = {"essential": ["labor_code_article_87"], "supporting": [], "context": []}
    return {
        "topic": topic,
        "issue_topic": issue_topic if issue_topic != topic else "",
        "category": category,
        "category_hints": category_hints or [category],
        "scope": sector,
        "scenario": scenario,
        "sector": sector,
        "compare_government_private": compare_government_private,
        "task": task,
        "sub_intent": sub_intent,
        "sub_intent_terms": issue_contract.get('search_terms') or SUB_INTENT_SEARCH_TERMS.get(sub_intent, sub_intent.replace("_", " ")),
        "requested_issues": issue_contract.get('issues', []),
        "independent_evidence_components": ['ec_eligibility_conditions'] if 'ec_eligibility_conditions' in required_concepts else [],
        "evidence_contract_unresolved": topic != 'non_legal' and not required_concepts,
        "intent": intent,
        "calculation_requested": calculation_requested,
        "personal_facts_available": bool(PERSONAL_FACT_RE.search(normalized)),
        "source_match": source_match,
        "normalized_query": normalized,
        "article_number": article_match.group(1) if article_match else "",
        "section_number": section_match.group(1) if section_match else "",
        "required_concepts": required_concepts,
        "essential_concepts": essential_concepts,
        "preferred_sources": preferred_sources,
        "required_authorities": required_authorities,
        "sector_precedence": sector_rule_precedence(sector),
        **wage_details,
        "missing_facts": [],
        "category_override": benefit_categories,
    }


def _extract_hierarchy(text: str) -> Dict[str, str]:
    hierarchy: Dict[str, str] = {}
    for key, regex in (
        ("book", BOOK_RE),
        ("title", TITLE_RE),
        ("chapter", CHAPTER_RE),
        ("rule", RULE_RE),
        ("section", SECTION_RE),
    ):
        match = regex.search(text or "")
        if match:
            value = match.group(1).upper()
            if key == "section":
                value = f"Section {value}"
            else:
                value = f"{key.title()} {value}"
            hierarchy[key] = value
    article = ARTICLE_RE.search(text or "")
    if article:
        hierarchy["article"] = f"ART. {article.group(1)}"
    return hierarchy


def _normalized_blob(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _hash_text(text: str, length: int = 16) -> str:
    normalized = _normalized_blob(text)
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()[:length]


def infer_canonical_document_id(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    meta = metadata or {}
    from routers.services.named_issuance import requested_issuance
    # Prefer the document's own identity to statutes cited in its body.
    for field in ("document_title", "source", "original_filename"):
        descriptor = str(meta.get(field) or "").replace("_", " ")
        advisory = re.search(
            r"labor[\s-]+advisory[\s-]+(?:no\.?\s*)?(\d+)\s*(?:,?\s*(?:s\.?|series\s+of)\s*(\d{4})|[-\s]+(\d{2,4}))",
            descriptor, re.I,
        )
        if advisory:
            year = advisory.group(2) or advisory.group(3)
            year = "20" + year if len(year) == 2 else year
            return f"dole_labor_advisory_{int(advisory.group(1))}_{year}"
    issuance = requested_issuance(str(meta.get("law_number") or "")) or requested_issuance(str(meta.get("document_title") or ""))
    if issuance:
        return f"dole_do_{issuance[0]}_{issuance[1]}"
    # This current SSS issuance is the indexed employer/employee contribution
    # schedule. Keep it distinct from RA 11199 so schedule answers cite the
    # table itself rather than relying on the Act's general contribution rule.
    sss_descriptor = " ".join(
        str(meta.get(field) or "")
        for field in ("document_title", "source", "original_filename", "official_url", "source_url")
    )
    if re.search(r"\bsss\s+circular\s+2024[-\s]?006\b|\bci[-_/]2024[-_/]006\b", sss_descriptor, re.I):
        return "sss_contribution_schedule_2025"
    # The SSS Act cites other statutes in its body; those are not its identity.
    if re.search(r"^(?:document:\s*)?(?:ra|republic\s+act(?:\s+no\.?)?)[\s.-]*11199\b", str(meta.get("document_title") or ""), re.I):
        return "ra_11199"
    for act_number in (9679, 11223):
        if re.search(
            rf"(?:^|\b)(?:document:\s*)?(?:ra|republic\s+act(?:\s+no\.?)?)[\s.-]*{act_number}\b",
            " ".join(str(meta.get(key) or "") for key in ("document_title", "law_number", "source")),
            re.I,
        ):
            return f"ra_{act_number}"
    descriptor_blob = _normalized_blob(
        " ".join(
            _norm(meta.get(key))
            for key in (
                "document_title",
                "source",
                "original_filename",
                "law_number",
                "official_url",
                "source_url",
            )
            if meta.get(key)
        )
    )
    if re.search(r"\b(?:department\s+order|dole\s+do|do)\s*(?:no\.?)?\s*198[-\s]?18\b", descriptor_blob):
        return "dole_do_198_18"
    if re.search(r"\b(?:department\s+order|dole\s+do|do)\s*(?:no\.?)?\s*252[-\s]?25\b", descriptor_blob):
        return "dole_do_252_25"
    if re.search(r"\b(?:ra|republic\s+act(?:\s+no\.?)?)\s*11058\b", descriptor_blob) or "republic-act-no-11058" in descriptor_blob:
        return "ra_11058"
    if re.search(r"\boccupational\s+safety\s+and\s+health\s+standards\b|\bosh\s+standards\b", descriptor_blob):
        return "occupational_safety_health_standards"
    # Explicit owning identifiers take precedence over Acts cited in a body.
    case_match = re.search(r"\bG\.?\s*R\.?\s*(?:Nos?\.?\s*)?(\d{4,8})\b",
                           " ".join(str(meta.get(k) or "") for k in ("document_title", "source")), re.I)
    if case_match:
        return "jurisprudence_gr_" + case_match[1]
    own_act = re.search(
        r"\b(?:ra|republic\s+act(?:\s+no\.?)?)[\s.-]*(\d{4,6})\b",
        " ".join(str(meta.get(k) or "") for k in ("document_title", "source", "original_filename")), re.I,
    )
    if own_act:
        return f"ra_{own_act[1]}"
    identity_blob = _normalized_blob(
        " ".join(
            [
                text or "",
                _metadata_text(meta),
                _norm(meta.get("original_filename")),
                _norm(meta.get("source_url")),
                _norm(meta.get("official_url")),
            ]
        )
    )
    if re.search(r"faqs?\s+on\s+wage\s+and\s+wage\s+related\s+benefits", identity_blob):
        return "dole_wage_benefits_faq"
    if re.search(
        r"\b(?:republic\s+act(?:\s+no\.?)?|ra)[_\s.-]*11861\b|expanded\s+solo\s+parents?\s+welfare\s+act",
        identity_blob,
        re.IGNORECASE,
    ):
        return "ra_11861"
    # Use a document's own descriptor before trusting legacy metadata. Older
    # chunks that merely mention PD 851 were incorrectly labeled as the decree,
    # even when they came from a later advisory, FAQ, or a court decision.
    if "labor advisory" in identity_blob and ("13th" in identity_blob or "thirteenth" in identity_blob):
        advisory = re.search(
            r"labor\s+advisory\s+(?:no\.?\s*)?(\d+)(?:\s*,?\s*(?:s\.?|series\s+of)\s*(\d{4}))?",
            identity_blob,
            re.IGNORECASE,
        )
        if advisory:
            return f"dole_labor_advisory_{advisory.group(1)}_{advisory.group(2) or ''}".rstrip("_")
        return "dole_13th_month_advisory"
    if "faq" in descriptor_blob and ("13th" in descriptor_blob or "thirteenth" in descriptor_blob):
        return "dole_13th_month_faq"
    existing = _norm(meta.get("canonical_document_id"))
    if existing:
        return existing.lower().replace(" ", "_")

    blob = _normalized_blob(
        " ".join(
            [
                text or "",
                _metadata_text(meta),
                _norm(meta.get("original_filename")),
                _norm(meta.get("source_url")),
                _norm(meta.get("official_url")),
            ]
        )
    )
    if not blob:
        return ""

    if "10361" in blob and re.search(r"\b(irr|implementing\s+rules|rules\s+and\s+regulations)\b", blob):
        return "ra_10361_irr"
    if "10361" in blob or "batas kasambahay" in blob or "domestic workers act" in blob:
        return "ra_10361"
    if "7641" in blob and re.search(r"\b(retirement|retire|republic\s+act|ra[_\s.-]*7641)\b", blob):
        return "ra_7641"
    if re.search(r"\b(pd|p\.d\.|presidential decree)\s*851\b", blob) or "13th month" in blob and "decree" in blob:
        return "pd_851"
    if "labor advisory" in blob and ("13th" in blob or "thirteenth" in blob):
        advisory = re.search(r"(?:labor\s+advisory\s+(?:no\.?\s*)?)(\d+)(?:\s*,?\s*series\s+of\s+(\d{4}))?", blob)
        if advisory:
            number = advisory.group(1)
            year = advisory.group(2) or ""
            return f"dole_labor_advisory_{number}_{year}".rstrip("_")
        return "dole_13th_month_advisory"
    if "faq" in blob and ("13th" in blob or "thirteenth" in blob):
        return "dole_13th_month_faq"
    if re.search(r"\b(?:department\s+order|d\.?o\.?)\s*(?:no\.?)?\s*249[-–\s]?25\b", blob):
        return "dole_do_249_25"
    if re.search(r"\b(?:department\s+order|d\.?o\.?)\s*(?:no\.?)?\s*252[-–\s]?25\b", blob):
        return "dole_do_252_25"
    if re.search(r"\b(?:arms|assistance\s+request\s+monitoring\s+system)\b", blob) and re.search(
        r"\b(sena|single[-\s]+entry|request\s+for\s+assistance)\b",
        blob,
    ):
        return "dole_arms_sena"
    if "omnibus" in blob and "labor code" in blob:
        return "omnibus_rules_labor_code"
    if "labor code" in blob or re.search(r"\b(pd|p\.d\.|presidential decree)\s*442\b", blob):
        return "labor_code_pd_442"
    law = re.search(r"\b(?:ra|republic\s+act)\s*no?\.?\s*(\d{4,6})\b", blob)
    if law:
        return f"ra_{law.group(1)}"
    return _hash_text(blob, length=12)


def _extract_article_number_from_text(text: str) -> str:
    match = ARTICLE_RE.search(text or "")
    return match.group(1) if match else ""


def _extract_article_title_from_text(text: str) -> str:
    match = ARTICLE_TITLE_RE.search(text or "")
    if not match:
        return ""
    title = re.sub(r"\s+", " ", match.group("title") or "").strip()
    title = re.sub(r"\b(the|this)\s+provisions?\b.*$", "", title, flags=re.IGNORECASE).strip()
    return title.strip(" .:-")


def infer_normalized_authority_id(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    meta = metadata or {}
    from routers.services.named_issuance import requested_issuance
    for field in ("law_number", "document_title", "source", "original_filename", "official_url", "source_url"):
        issuance = requested_issuance(str(meta.get(field) or ""))
        if issuance:
            return f"dole_do_{issuance[0]}_{issuance[1]}"
    canonical_doc = _norm(meta.get("canonical_document_id")) or infer_canonical_document_id(text, meta)
    if canonical_doc.startswith("jurisprudence_gr_") or re.fullmatch(r"(?:ra|pd|eo)_\d+", canonical_doc):
        return canonical_doc
    if canonical_doc.startswith("dole_do_") or canonical_doc.startswith("dole_labor_advisory_"):
        return canonical_doc
    article_number = (
        _norm(meta.get("article_number"))
        or _extract_article_number_from_text(_norm(meta.get("article")) or "")
        or _extract_article_number_from_text(_norm(meta.get("citation")) or "")
        or _extract_article_number_from_text(text or "")
    )
    if canonical_doc == "labor_code_pd_442" and article_number:
        return f"labor_code_article_{article_number}"
    if canonical_doc in {
        "pd_851",
        "dole_13th_month_faq",
        "dole_13th_month_advisory",
        "ra_10361",
        "ra_10361_irr",
        "ra_11861",
        "ra_8972",
        "ra_11210",
        "ra_7641",
        "ra_11058",
        "ra_11199",
        "sss_contribution_schedule_2025",
        "ra_9679",
        "ra_11223",
        "omnibus_rules_labor_code",
        "dole_wage_benefits_faq",
        "dole_do_249_25",
        "dole_do_252_25",
        "dole_do_198_18",
        "occupational_safety_health_standards",
        "dole_arms_sena",
    }:
        return canonical_doc
    category = normalize_kb_category(meta.get("category"))
    combined = f"{text or ''} {_metadata_text(meta)}"
    if category == "Jurisprudence" and re.search(
        r"\b(supreme\s+court|g\.?\s*r\.?\s*(?:no\.?)?)\b",
        combined,
        re.IGNORECASE,
    ):
        return "supreme_court_decision"
    return _norm(meta.get("normalized_authority_id"))


def infer_chunk_concepts(text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
    meta = metadata or {}
    existing = meta.get("concepts")
    concepts: List[str] = []
    if isinstance(existing, str):
        concepts.extend(part.strip() for part in re.split(r"[,;|]", existing) if part.strip())
    elif isinstance(existing, (list, tuple, set)):
        concepts.extend(str(part).strip() for part in existing if str(part).strip())

    blob = f"{text or ''} {_metadata_text(meta)}"
    for concept, patterns in CONCEPT_PATTERNS.items():
        if concept == "legal_basis":
            continue
        if any(re.search(pattern, blob, re.IGNORECASE) for pattern in patterns):
            concepts.append(concept)
    return list(dict.fromkeys(concepts))


def normalize_legal_status(metadata: Optional[Dict[str, Any]] = None) -> str:
    meta = metadata or {}
    raw = _lower(
        meta.get("legal_status")
        or meta.get("version_status")
        or meta.get("status")
        or meta.get("document_status")
    )
    if raw in {"", "approved", "indexed", "active", "effective", "valid"}:
        return "active"
    if raw == "current":
        return "current"
    if raw in {"inactive", "disabled", "archived"}:
        return "inactive"
    if raw in {"superseded", "replaced", "outdated", "obsolete"}:
        return "superseded"
    if raw == "historical":
        return "historical"
    if raw in {"processing", "pending", "failed"}:
        return raw
    return raw


def enrich_chunk_metadata(text: str, metadata: Optional[Dict[str, Any]] = None, *, infer_concepts: bool = True) -> Dict[str, Any]:
    meta = dict(metadata or {})
    # Re-identify indexed DOLE orders from the document's own metadata before
    # trusting legacy canonical IDs, which may have been inferred from cited
    # Labor Code articles or unrelated OCR headers.
    from routers.services.named_issuance import requested_issuance
    for field in ("law_number", "document_title", "source", "original_filename", "official_url", "source_url"):
        issuance = requested_issuance(str(meta.get(field) or ""))
        if issuance:
            canonical_order = f"dole_do_{issuance[0]}_{issuance[1]}"
            meta["canonical_document_id"] = canonical_order
            meta["normalized_authority_id"] = canonical_order
            break
    hierarchy = _extract_hierarchy(f"{text or ''} {_metadata_text(meta)}")
    for key, value in hierarchy.items():
        meta.setdefault(key, value)

    if "article" in meta and "article_number" not in meta:
        article_number = _extract_article_number_from_text(_norm(meta.get("article")))
        if article_number:
            meta["article_number"] = article_number
    if "citation" in meta and "article_number" not in meta:
        article_number = _extract_article_number_from_text(_norm(meta.get("citation")))
        if article_number:
            meta["article_number"] = article_number
    if "article_number" not in meta:
        article_number = _extract_article_number_from_text(text or "")
        if article_number:
            meta["article_number"] = article_number
    if "article_number" in meta and "article" not in meta:
        meta["article"] = f"ART. {meta['article_number']}"
    if not _norm(meta.get("article_title")):
        article_title = _extract_article_title_from_text(text or "")
        if article_title:
            meta["article_title"] = article_title

    combined_text = f"{text or ''} {_metadata_text(meta)}"
    category = normalize_kb_category(meta.get("primary_category") or meta.get("category"))
    meta.update(category_metadata(category, secondary_categories_from_metadata(meta)))
    if not _norm(meta.get("doc_type")) and _norm(meta.get("document_type")):
        meta["doc_type"] = _norm(meta.get("document_type"))
    if not _norm(meta.get("doc_type")) and category == "DOLE Resources" and re.search(
        r"\b(wage\s+order|rtwpb|regional\s+wage|minimum\s+wage\s+rates?)\b",
        combined_text,
        re.IGNORECASE,
    ):
        meta["doc_type"] = "wage_order"
    if _norm(meta.get("doc_type")):
        meta.setdefault("document_type", _norm(meta.get("doc_type")))
    region_details = wage_order_query_details(combined_text)
    if region_details.get("region_code"):
        meta.setdefault("region", region_details["region_code"])
        meta.setdefault("region_code", region_details["region_code"])
        meta.setdefault("region_name", region_details.get("region_name") or "")
    if not _norm(meta.get("effective_date")):
        effective_match = re.search(
            r"\bEffective\s*:?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
            combined_text,
            re.IGNORECASE,
        )
        if effective_match:
            meta["effective_date"] = effective_match.group(1)
    document_year = infer_document_year(combined_text, meta)
    if document_year is not None:
        meta["document_year"] = document_year

    meta["topic"] = infer_topic(text, meta)
    meta["sector"] = infer_sector(text, meta)
    # Mentioning competent government agencies as recipients of a hazard
    # report does not make the reporting worker a government employee.
    operative_body = str(text or '').split('\nText:\n')[-1]
    if (re.search(r'workers.{0,150}right to report.{0,200}hazards', operative_body, re.I | re.S)
            and re.search(r'government agencies', operative_body, re.I)
            and not re.search(r'government employees|public sector employees', operative_body, re.I)):
        meta['sector'] = GENERAL_SECTOR
        meta['source_scope'] = 'General private employment.'
        meta['coverage'] = 'General private employment.'
    meta["rule_type"] = infer_rule_type(text, meta)
    coverage = infer_coverage(text, meta)
    if coverage:
        meta["coverage"] = coverage
    inferred_canonical_document = infer_canonical_document_id(text, meta)
    if (inferred_canonical_document.startswith("jurisprudence_gr_")
            or inferred_canonical_document.startswith("ra_")
            or inferred_canonical_document.startswith("dole_do_")
            or inferred_canonical_document.startswith("dole_labor_advisory_")
            or inferred_canonical_document in {"dole_13th_month_faq", "dole_13th_month_advisory"}
            or inferred_canonical_document in {
        "ra_11199",
        "sss_contribution_schedule_2025",
        "ra_9679",
        "ra_11223",
        "dole_wage_benefits_faq",
        "ra_11861",
        "ra_11058",
        "dole_do_198_18",
        "dole_do_252_25",
        "occupational_safety_health_standards",
    }):
        # Older FAQ chunks were indexed with content-hash IDs. Normalize them
        # at read time. RA 11861 was also previously mislabeled as RA 8972 when
        # an amendatory clause mentioned the older law before the document title.
        meta["canonical_document_id"] = inferred_canonical_document
    else:
        meta.setdefault("canonical_document_id", inferred_canonical_document)
    canonical_doc = _norm(meta.get("canonical_document_id"))
    if canonical_doc in {"dole_13th_month_faq", "dole_13th_month_advisory"}:
        # The FAQ/advisory applies to general private employment. Older OCR
        # ingestion treated a sentence defining managerial employees or an
        # exception as the scope of the entire source, which hid its operative
        # coverage clause from ordinary employee questions.
        meta["topic"] = "thirteenth_month_pay"
        meta["sector"] = GENERAL_SECTOR
        meta["source_scope"] = "General private employment."
        if _norm(meta.get("rule_type")) == SPECIAL_SECTOR_RULE:
            meta["rule_type"] = GENERAL_RULE
    if canonical_doc.startswith("dole_do_"):
        # A parent Labor Code article mentioned by an order is not that
        # order's own article number. Older ingestion mixed these identities.
        for key in ("article", "article_number", "article_title"):
            meta.pop(key, None)
        for key in ("citation", "article_section"):
            if re.search(r'\b(?:art\.?|article)\s*\d', str(meta.get(key) or ''), re.I):
                meta.pop(key, None)
    if canonical_doc in {
        "ra_11058",
        "dole_do_198_18",
        "dole_do_252_25",
        "occupational_safety_health_standards",
    }:
        meta["topic"] = "workplace_safety"
        # Legacy ingestion inferred government/hospital sectors from phrases
        # such as "government inspector" and "occupational health personnel."
        # These authorities state generally applicable covered-workplace rules,
        # so restore their intended general employment scope at read time.
        meta["sector"] = GENERAL_SECTOR
        if _norm(meta.get("rule_type")) == SPECIAL_SECTOR_RULE:
            meta["rule_type"] = GENERAL_RULE
    if canonical_doc == "ra_11861":
        meta["topic"] = "solo_parent_leave"
        amendatory_match = re.search(
            r"\bsection\s*8\s+of\s+(?:republic\s+act(?:\s+no\.?)?|ra)\s*8972\b.{0,120}\bamended\b",
            text or "",
            re.IGNORECASE,
        )
        if amendatory_match or re.search(
            r"\bsection\s*7\b.{0,180}\bsection\s*8\b.{0,120}\b(?:ra|republic\s+act)\s*(?:no\.?)?\s*8972\b",
            text or "",
            re.IGNORECASE,
        ):
            # Preserve both sides of the amendment.  The provision belongs to
            # Section 7 of RA 11861 while the provision being amended remains
            # Section 8 of RA 8972.
            meta["amending_act_section"] = "RA 11861 Section 7"
            meta["amended_law_section"] = "RA 8972 Section 8"
            meta["section"] = "Section 7"
            meta["article_section"] = "Section 7"
            meta["citation"] = "RA 11861 Section 7 (amending RA 8972 Section 8)"
        if re.search(
            r"\bgovernment\s+and\s+(?:the\s+)?private\s+sector\b|\bprivate\s+sector\b",
            text or "",
            re.IGNORECASE,
        ):
            # A provision expressly covering both sectors is valid general
            # employment context, even if its first keyword is "government".
            meta["sector"] = GENERAL_SECTOR
        if re.search(
            r"\b(?:seven\s*\(7\)|seven|7)\s+working\s+days?\s+with\s+pay\b|"
            r"\bat\s+least\s+(?:six\s*\(6\)|six|6)\s+months?\b",
            text or "",
            re.IGNORECASE,
        ):
            meta["rule_type"] = ELIGIBILITY_RULE
    article_number = _norm(meta.get("article_number"))
    if canonical_doc == "labor_code_pd_442" and article_number in {"82", "83", "84", "85", "86", "87", "88", "89", "93", "94", "95", "298", "302"}:
        meta["sector"] = GENERAL_SECTOR
        if article_number in {"83", "85"}:
            meta["topic"] = "hours_of_work"
        elif article_number == "86":
            meta["topic"] = "night_shift_differential"
        elif article_number in {"82", "84", "87", "88", "89"}:
            meta["topic"] = "overtime_pay"
        elif article_number == "93":
            meta["topic"] = "premium_pay"
        elif article_number == "94":
            meta["topic"] = "holiday_pay"
        elif article_number == "95":
            meta["topic"] = "service_incentive_leave"
        elif article_number == "298":
            meta["topic"] = "separation_pay"
        elif article_number == "302":
            meta["topic"] = "retirement_pay"
        if article_number == "82":
            meta["rule_type"] = COVERAGE_RULE
    if canonical_doc in {"ra_10361", "ra_10361_irr"}:
        meta["topic"] = "domestic_workers"
        meta["sector"] = DOMESTIC_SECTOR
        meta["rule_type"] = SPECIAL_SECTOR_RULE
        meta.setdefault("controls_over", "labor_code_househelper_provisions,general_hours_of_work_rules")
    if canonical_doc == "labor_code_pd_442" and _is_old_househelper_labor_code(
        f"{text or ''} {_metadata_text(meta)}",
        meta,
    ):
        meta["legal_status"] = "superseded"
        meta["superseded_by"] = "ra_10361"
        meta["rule_type"] = SPECIAL_SECTOR_RULE
        meta["sector"] = DOMESTIC_SECTOR
    legal_status = normalize_legal_status(meta)
    if canonical_doc in {"ra_10361", "ra_10361_irr"}:
        legal_status = "active"
    meta["legal_status"] = legal_status
    meta.setdefault("status", legal_status)
    authority_id = infer_normalized_authority_id(text, meta)
    if authority_id:
        meta["normalized_authority_id"] = authority_id
    if infer_concepts:
        concepts = infer_chunk_concepts(text, meta)
        if concepts:
            meta["concepts"] = ",".join(concepts)
    meta.setdefault("content_hash", _hash_text(text))
    if not _norm(meta.get("document_title")):
        meta["document_title"] = _norm(meta.get("source_name") or meta.get("source") or meta.get("original_filename"))
    if meta.get("source_url") and not meta.get("official_url"):
        meta["official_url"] = meta.get("source_url")
    if meta.get("official_url") and not meta.get("source_url"):
        meta["source_url"] = meta.get("official_url")
    if meta.get("date_issued") and not meta.get("publication_date"):
        meta["publication_date"] = meta.get("date_issued")
    if meta.get("law_version") and not meta.get("edition"):
        meta["edition"] = meta.get("law_version")
    meta.setdefault("official_source", "yes" if _norm(meta.get("source_url")) else "")
    if "page" in meta and "page_number" not in meta:
        meta["page_number"] = meta.get("page")
    meta.update(infer_scope_flags(meta))
    from routers.services.legal_version_ranking import infer_amendment_metadata
    meta = infer_amendment_metadata(text, meta)
    return meta


def build_chunk_context_text(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    meta = metadata or {}
    lines: List[str] = []
    labels = (
        ("Document", meta.get("document_title") or meta.get("source")),
        ("Canonical Document", meta.get("canonical_document_id")),
        ("Law/Issuance", meta.get("law_number")),
        ("Issuing Agency", meta.get("issuing_agency")),
        ("Book", meta.get("book")),
        ("Title", meta.get("title")),
        ("Chapter", meta.get("chapter")),
        ("Rule", meta.get("rule")),
        ("Rule Title", meta.get("rule_title")),
        ("Section", meta.get("section") or meta.get("article_section")),
        ("Amending Act Section", meta.get("amending_act_section")),
        ("Amended Law Section", meta.get("amended_law_section")),
        ("Section Title", meta.get("section_title")),
        ("Article", meta.get("citation") or meta.get("article")),
        ("Authority ID", meta.get("normalized_authority_id")),
        ("Concepts", meta.get("concepts")),
        ("Scope", meta.get("source_scope") or meta.get("coverage")),
        ("Rule Type", meta.get("rule_type")),
        ("Sector", meta.get("sector")),
        ("Legal Status", meta.get("legal_status") or meta.get("status") or meta.get("version_status")),
        ("Publication Date", meta.get("publication_date") or meta.get("date_issued")),
        ("Effective Date", meta.get("effective_date")),
        ("Document Year", meta.get("document_year")),
        ("Document Type", meta.get("doc_type")),
        ("Primary Category", meta.get("primary_category") or meta.get("category")),
        ("Secondary Categories", meta.get("secondary_categories")),
        ("Topics", meta.get("topics")),
        ("Sub-intents", meta.get("sub_intents")),
        ("Legal Role", meta.get("legal_role")),
        ("Implements", meta.get("implements")),
        ("Edition", meta.get("edition") or meta.get("law_version")),
        ("Official URL", meta.get("official_url")),
        ("Source URL", meta.get("source_url")),
    )
    for label, value in labels:
        if value not in (None, "", 0, "0"):
            lines.append(f"{label}: {value}")
    if not lines:
        return (text or "").strip()
    return "\n".join(lines) + "\n\nText:\n" + (text or "").strip()


def prepare_chunks_for_indexing(
    chunks: Iterable[Tuple[str, Dict[str, Any]]],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for text, metadata in chunks:
        merged = {**(metadata or {}), **(extra_metadata or {})}
        if extra_metadata and (extra_metadata.get('primary_category') or extra_metadata.get('category')):
            # Extraction defaults must not override the category saved on upload.
            merged.update(category_metadata(
                extra_metadata.get('primary_category') or extra_metadata['category'],
                secondary_categories_from_metadata(extra_metadata),
            ))
        enriched = enrich_chunk_metadata(text, merged)
        texts.append(build_chunk_context_text(text, enriched))
        metadatas.append(enriched)
    return texts, metadatas


def sector_matches(query_sector: str, chunk_sector: str) -> bool:
    query_sector = canonical_sector(query_sector)
    chunk_sector = canonical_sector(chunk_sector)
    if query_sector == GOVERNMENT_SECTOR:
        return chunk_sector == GOVERNMENT_SECTOR
    if query_sector == HOSPITAL_SECTOR:
        return chunk_sector in {HOSPITAL_SECTOR, GENERAL_SECTOR}
    if query_sector == DOMESTIC_SECTOR:
        return chunk_sector in {DOMESTIC_SECTOR, GENERAL_SECTOR}
    if query_sector in {
        MANAGERIAL_SECTOR,
        MANAGERIAL_STAFF_SECTOR,
        FIELD_SECTOR,
        NIGHT_WORKERS_SECTOR,
        MINOR_WORKERS_SECTOR,
        OVERSEAS_WORKERS_SECTOR,
    }:
        return chunk_sector in {query_sector, GENERAL_SECTOR}
    if query_sector == UNKNOWN_SECTOR:
        return chunk_sector == GENERAL_SECTOR
    if query_sector == GENERAL_SECTOR:
        return chunk_sector == GENERAL_SECTOR
    return chunk_sector in {query_sector, GENERAL_SECTOR}


def _is_labor_code_authority(text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    meta = metadata or {}
    authority_blob = " ".join(
        _norm(meta.get(key))
        for key in (
            "document_title",
            "source",
            "source_name",
            "law_number",
            "citation",
            "original_filename",
        )
        if meta.get(key)
    )
    combined = f"{authority_blob} {text or ''}"
    if PD_442_RE.search(combined):
        return True

    title_blob = authority_blob.lower()
    if "labor code" not in title_blob:
        return False
    if "omnibus" in title_blob or "implementing" in title_blob:
        return False
    return True


def _has_article_reference(combined: str, article_number: str) -> bool:
    return bool(re.search(rf"\b(?:art\.?|article)\s*{re.escape(article_number)}\b", combined, re.IGNORECASE))


def _is_omnibus_authority(combined: str) -> bool:
    return bool(re.search(r"\bomnibus\b", combined, re.IGNORECASE) and re.search(r"\blabor\s+code\b", combined, re.IGNORECASE))


def authority_ids_for_chunk(chunk: Dict[str, Any]) -> set[str]:
    if not chunk:
        return set()
    text = _norm(chunk.get("text") or chunk.get("excerpt"))
    raw_meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else chunk
    meta = enrich_chunk_metadata(text, raw_meta or {}, infer_concepts=False)
    authority_ids: set[str] = set()
    authority_id = _norm(meta.get("normalized_authority_id"))
    if authority_id:
        authority_ids.add(authority_id)
    canonical_doc = _norm(meta.get("canonical_document_id"))
    if canonical_doc:
        authority_ids.add(canonical_doc)
    article_number = _norm(meta.get("article_number"))
    if canonical_doc == "labor_code_pd_442" and article_number:
        authority_ids.add(f"labor_code_article_{article_number}")
    return authority_ids


def required_authorities_for_query(query_profile: Dict[str, Any]) -> Dict[str, List[str]]:
    configured = query_profile.get("required_authorities")
    if isinstance(configured, dict):
        return {
            "essential": list(configured.get("essential") or []),
            "supporting": list(configured.get("supporting") or []),
            "context": list(configured.get("context") or []),
        }
    return required_authorities_for_topic(
        query_profile.get("topic"),
        _norm(query_profile.get("task")),
    )


def missing_required_authorities(
    chunks: List[Dict[str, Any]],
    query_profile: Dict[str, Any],
    include_supporting: bool = False,
) -> List[str]:
    requirements = required_authorities_for_query(query_profile)
    required = list(requirements.get("essential") or [])
    if include_supporting:
        required.extend(requirements.get("supporting") or [])
    required = list(dict.fromkeys(required))
    if not required:
        return []
    present: set[str] = set()
    for chunk in chunks or []:
        present.update(authority_ids_for_chunk(chunk))
    return [authority_id for authority_id in required if authority_id not in present]


def authority_query_terms_for_profile(query_profile: Dict[str, Any]) -> List[str]:
    requirements = required_authorities_for_query(query_profile)
    authority_ids = list(dict.fromkeys(
        list(requirements.get("essential") or []) + list(requirements.get("supporting") or [])
    ))
    return [authority_search_terms(authority_id) for authority_id in authority_ids]


def _source_body_text(text: str) -> str:
    """Ignore generated metadata headers when judging whether a passage answers."""

    return re.sub(r"\s+", " ", source_passage_text(text))


def osh_chunk_supports_task(chunk: Dict[str, Any], query_profile: Dict[str, Any]) -> bool:
    """Require the actual OSH passage—not declared keywords—to state the proposition."""

    body = _source_body_text(chunk.get("text") or chunk.get("excerpt") or "")
    task = _norm(query_profile.get("task"))
    sub_intent = _norm(query_profile.get("sub_intent"))
    if task == 'issue_specific_safety':
        from routers.services.legal_topic_matrix import supported_passage_concepts
        # A complementary rule need not prove every requested point on its own.
        return bool(supported_passage_concepts({'text': body}, query_profile.get('required_concepts') or []))
    if task == 'hazard_reporting':
        return bool(re.search(r'right.{0,100}report.{0,220}hazards?|right.{0,70}refus.{0,80}(?:work|unsafe)|imminent danger', body, re.I | re.S))
    if task == "definition_question":
        return bool(re.search(r"\b(means|refers\s+to|defined\s+as)\b", body, re.IGNORECASE))
    if task == "refusal_of_unsafe_work" or sub_intent == "right_to_refuse_unsafe_work":
        return bool(re.search(r"\bright\s+(?:of\s+)?refusal|right\s+to\s+refuse|refuse\s+unsafe\s+work", body, re.IGNORECASE))
    if task == "ppe_requirement" or sub_intent == "ppe_requirement":
        return bool(re.search(
            r"\b(personal\s+protective\s+equipment|ppe)\b.{0,220}\b(provide|furnish|free\s+of\s+charge|cost|expense|employer|shall)\b|"
            r"\b(provide|furnish|free\s+of\s+charge)\b.{0,220}\b(personal\s+protective\s+equipment|ppe)\b",
            body,
            re.IGNORECASE | re.DOTALL,
        ))
    if task == "safety_officer_requirement" or sub_intent == "safety_officer_requirement":
        return bool(re.search(
            r"\bcovered\s+workplaces?\s+shall\s+have\s+safety\s+officers?\b|"
            r"\bminimum\s+(?:classification\s+and\s+)?number\s+of\s+safety\s+officers?\b|"
            r"\bnumber\s+and\s+qualification\s+of\s+safety\s+officers?\s+shall\b|"
            r"\bat\s+least\s+one\s*\(?1\)?\s+safety\s+officer\s+must\s+be\s+deployed\b|"
            r"\bsection\s*14\b.{0,100}\bsafety\s+officer\b",
            body,
            re.IGNORECASE | re.DOTALL,
        ))
    if task == "osh_program_requirement" or sub_intent == "osh_program_requirement":
        return bool(re.search(r"\b(?:occupational\s+safety\s+and\s+health|osh|safety\s+and\s+health)\s+program\b", body, re.IGNORECASE))
    if task == "osh_committee_requirement" or sub_intent == "osh_committee_requirement":
        return bool(re.search(r"\b(?:occupational\s+safety\s+and\s+health|osh|safety\s+and\s+health)\s+committee\b", body, re.IGNORECASE))
    return bool(re.search(
        r"\b(dut(?:y|ies)\s+of\s+(?:the\s+)?employer|duties\s+of\s+employers|every\s+employer\b.{0,100}\b(?:shall|furnish|give|inform|provide|comply)|"
        r"place\s+of\s+employment\s+free\s+from\s+hazardous\s+conditions|safe\s+and\s+healthful\s+workplace)",
        body,
        re.IGNORECASE | re.DOTALL,
    ))


def osh_evidence_is_responsive(chunks: List[Dict[str, Any]], query_profile: Dict[str, Any]) -> bool:
    return any(osh_chunk_supports_task(chunk, query_profile) for chunk in chunks or [])


def _looks_historical_13th_limit(combined: str) -> bool:
    return bool(
        re.search(r"(?:₱|p|php|peso(?:s)?)\s*[1lI][, ]?000", combined, re.IGNORECASE)
        and re.search(r"\b(ceiling|limit|not\s+more\s+than|only|entitled)\b", combined, re.IGNORECASE)
    )


def _is_old_househelper_labor_code(combined: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    meta = metadata or {}
    canonical = _norm(meta.get("canonical_document_id"))
    if canonical != "labor_code_pd_442":
        return False
    article_number = _norm(meta.get("article_number")) or _extract_article_number_from_text(combined)
    if article_number == "82":
        return False
    return bool(re.search(r"\b(househelper|household\s+service|domestic\s+service)\b", combined, re.IGNORECASE))


def score_chunk(chunk: Dict[str, Any], query_profile: Dict[str, Any]) -> Dict[str, Any]:
    # Requested components are checked against the body below. Scanning every
    # topic's tag patterns here adds no proof and blocks unrelated requests.
    meta = enrich_chunk_metadata(chunk.get("text", ""), chunk.get("metadata") or {}, infer_concepts=False)
    original_score = float(chunk.get("score", 0) or 0)
    document_year = infer_document_year(chunk.get("text", ""), meta)
    recency_factor = document_recency_factor(document_year)
    score = original_score * recency_factor
    reasons: List[str] = []
    authority_score = 0.0
    concept_score = 0.0
    if recency_factor > 1.0:
        reasons.append(f"recency_x{recency_factor:.2f}")
    topic = _norm(query_profile.get("topic"))
    task = _norm(query_profile.get("task"))
    sub_intent = _norm(query_profile.get("sub_intent"))
    query_sector = canonical_sector(_norm(query_profile.get("sector")))
    chunk_topic = _norm(meta.get("topic"))
    chunk_sector = canonical_sector(_norm(meta.get("sector")))
    rule_type = _norm(meta.get("rule_type"))
    combined = f"{chunk.get('text', '')} {_metadata_text(meta)}"
    canonical_doc = _norm(meta.get("canonical_document_id"))
    legal_status = _lower(meta.get("legal_status") or meta.get("version_status") or meta.get("status"))
    authority_id = _norm(meta.get("normalized_authority_id"))
    authority_requirements = required_authorities_for_query(query_profile)
    essential_authorities = set(authority_requirements.get("essential") or [])
    supporting_authorities = set(authority_requirements.get("supporting") or [])
    article_number = _norm(meta.get("article_number"))
    chunk_authority_ids = {authority_id, canonical_doc}
    if canonical_doc == "labor_code_pd_442" and article_number:
        chunk_authority_ids.add(f"labor_code_article_{article_number}")

    if topic and topic == chunk_topic:
        score += 0.18
        reasons.append("topic_match")
    if sub_intent:
        chunk_sub_intent = _norm(meta.get("sub_intent") or meta.get("subtopic") or meta.get("task"))
        sub_terms = str(query_profile.get("sub_intent_terms") or SUB_INTENT_SEARCH_TERMS.get(sub_intent, ""))
        if chunk_sub_intent == sub_intent:
            score += 0.42
            reasons.append("sub_intent_match")
        else:
            sub_pattern = re.escape(sub_terms).replace(r"\ ", r"\s+")
            name_pattern = re.escape(sub_intent.replace("_", " ")).replace(r"\ ", r"\s+")
            if sub_pattern and re.search(rf"\b(?:{sub_pattern}|{name_pattern})\b", combined, re.IGNORECASE):
                score += 0.16
                reasons.append("sub_intent_evidence")

    preferred_rule_types = {
        "calculation": {"computation_rule", "rate_rule", "main_rule"},
        "formula": {"computation_rule", "rate_rule", "main_rule"},
        "coverage": {"coverage_rule", "eligibility_rule", "main_rule"},
        "eligibility": {"eligibility_rule", "coverage_rule", "main_rule"},
        "deadline": {"deadline_rule", "procedural_rule", "main_rule"},
        "definition": {"definition_rule", "main_rule"},
    }
    task_key = next((key for key in preferred_rule_types if key in task), "")
    if not task_key and ("rate" in task or "computation" in task):
        task_key = "calculation"
    if task_key and rule_type in preferred_rule_types[task_key]:
        score += 0.20
        reasons.append(f"{task_key}_rule_priority")
    elif rule_type in {"optional_rule", "historical_rule", "secondary_guidance"}:
        score -= 0.15
        reasons.append("optional_or_historical_penalty")
    preferred_category = preferred_kb_category(topic)
    chunk_category = normalize_kb_category(meta.get("primary_category") or meta.get("category"))
    requested_categories = list(query_profile.get("requested_categories") or [])
    match_type = category_match_type(meta, requested_categories or ([preferred_category] if preferred_category else []))
    meta["category_match_type"] = match_type
    category_score = 0.0
    if preferred_category and chunk_category == preferred_category:
        score += 0.30
        category_score += 0.30
        reasons.append("preferred_category_boost")
    elif preferred_category and preferred_category in secondary_categories_from_metadata(meta):
        score += 0.19
        category_score += 0.19
        reasons.append("preferred_secondary_category_boost")
    elif match_type == "primary":
        score += 0.16
        category_score += 0.16
        reasons.append("routed_primary_category_boost")
    elif match_type == "secondary":
        score += 0.10
        category_score += 0.10
        reasons.append("routed_secondary_category_boost")
    compare_government_private = bool(query_profile.get("compare_government_private"))
    sector_match = sector_matches(query_sector, chunk_sector) or (
        compare_government_private and chunk_sector in {GENERAL_SECTOR, GOVERNMENT_SECTOR}
    )
    if sector_match:
        score += 0.16
        reasons.append("sector_match")
    else:
        score -= 0.35
        reasons.append("sector_mismatch")

    if essential_authorities.intersection(chunk_authority_ids):
        score += 0.72
        authority_score += 0.72
        reasons.append("essential_authority_boost")
    elif supporting_authorities.intersection(chunk_authority_ids):
        score += 0.32
        authority_score += 0.32
        reasons.append("supporting_authority_boost")

    required_concepts = list(query_profile.get("required_concepts") or [])
    supported_required = [
        concept for concept in required_concepts
        if _concept_supported(concept, {**meta, "text": chunk.get("text", "")})
    ]
    if supported_required:
        matched_concept_score = min(0.40, 0.08 * len(supported_required))
        score += matched_concept_score
        concept_score += matched_concept_score
        reasons.append(f"required_concepts_{len(supported_required)}")
    elif required_concepts:
        score -= 0.12
        reasons.append("no_required_concept_penalty")

    if query_sector == GENERAL_SECTOR:
        if (rule_type == SPECIAL_SECTOR_RULE
            and not (compare_government_private and chunk_sector == GOVERNMENT_SECTOR)) or (
            chunk_sector != GENERAL_SECTOR
            and not (compare_government_private and chunk_sector == GOVERNMENT_SECTOR)
        ):
            score -= 0.45
            reasons.append("special_scope_penalty")
        if topic == "overtime_pay" and _is_labor_code_authority(combined, meta):
            score += 0.18
            reasons.append("labor_code_authority_boost")
        if topic == "overtime_pay" and ARTICLE_87_RE.search(combined) and task not in {
            "coverage_question",
            "exclusion_question",
            "authorization_question",
            "undertime_offset_question",
            "compulsory_overtime_question",
        }:
            score += 0.45
            reasons.append("article_87_boost")
        if topic == "overtime_pay" and re.search(r"\b(section|sec\.?)\s*(8|9)\b", combined, re.IGNORECASE):
            if not HOSPITAL_RE.search(combined):
                score += 0.16
                reasons.append("general_rule_section_boost")
        if topic == "overtime_pay" and re.search(r"\b(article|art\.?)\s*82\b", combined, re.IGNORECASE):
            if task in {"coverage_question", "exclusion_question"}:
                score += 0.50
                reasons.append("task_coverage_article_boost")
            else:
                score += 0.10
                reasons.append("coverage_article_boost")
        if topic == "overtime_pay":
            support_articles = set(OVERTIME_SUPPORT_ARTICLE_RE.findall(combined))
            target_articles = set(_overtime_articles_for_task(task))
            if support_articles.intersection(target_articles):
                score += 0.34
                reasons.append("task_relevant_overtime_article_boost")
            elif task in {
                "coverage_question",
                "exclusion_question",
                "authorization_question",
                "undertime_offset_question",
                "compulsory_overtime_question",
                "threshold_question",
            } and support_articles:
                score -= 0.12
                reasons.append("non_task_overtime_article_penalty")
            elif support_articles.intersection({"84", "93", "94"}):
                score += 0.18
                reasons.append("overtime_support_article_boost")
            elif support_articles.intersection({"82", "88"}):
                score += 0.10
                reasons.append("overtime_related_article_boost")

    if query_sector == HOSPITAL_SECTOR and chunk_sector == HOSPITAL_SECTOR:
        score += 0.30
        reasons.append("hospital_scope_boost")

    if query_sector == GOVERNMENT_SECTOR and chunk_sector != GOVERNMENT_SECTOR:
        score -= 0.30
        reasons.append("government_scope_guard")

    if legal_status in {"active", "current"}:
        score += 0.04
        reasons.append("active_source_boost")
    if legal_status == "current":
        score += 0.10
        reasons.append("current_source_boost")
    if _norm(meta.get("source_url") or meta.get("official_url")):
        score += 0.04
        reasons.append("official_url_boost")

    status = legal_status
    if status and status not in {"active", "current"}:
        score -= 0.25
        reasons.append("inactive_or_superseded")

    if topic == "holiday_pay":
        regular_holiday_query = bool(re.search(r"\b(regular\s+holiday|holiday\s+pay|double\s+pay)\b", _norm(query_profile.get("normalized_query")), re.IGNORECASE))
        if _has_article_reference(combined, "94"):
            score += 0.45
            reasons.append("article_94_holiday_boost")
        if _is_omnibus_authority(combined) and re.search(r"\b(holiday\s+pay|regular\s+holidays?|section\s+(4|5|6|7|8|9|10))\b", combined, re.IGNORECASE):
            score += 0.22
            reasons.append("omnibus_holiday_boost")
        if regular_holiday_query and _has_article_reference(combined, "93") and not _has_article_reference(combined, "94"):
            score -= 0.18
            reasons.append("article_93_not_primary_for_regular_holiday")

    if topic == "night_shift_differential":
        if _has_article_reference(combined, "86"):
            score += 0.55
            reasons.append("article_86_night_shift_boost")
        if is_wage_order_chunk({**chunk, "metadata": meta}):
            score -= 1.10
            reasons.append("regional_wage_order_not_night_shift_authority")

    if topic == "separation_pay":
        if _has_article_reference(combined, "298"):
            score += 0.60
            reasons.append("article_298_separation_pay_boost")
        elif re.search(r"\b(handbook|workers'?\s+rights)\b", combined, re.IGNORECASE):
            score -= 0.35
            reasons.append("general_handbook_not_controlling_article")

    if topic == "workplace_safety":
        body = _source_body_text(chunk.get("text", ""))
        responsive = osh_chunk_supports_task({**chunk, "text": body}, query_profile)
        if task == "refusal_of_unsafe_work" and canonical_doc == "ra_11058" and re.search(
            r"\bsection\s*6\b|\bright\s+(?:of\s+)?refusal|right\s+to\s+refuse\b",
            body,
            re.IGNORECASE,
        ):
            score += 0.70
            authority_score += 0.70
            reasons.append("ra11058_section6_boost")
        elif task in {"safety_officer_requirement", "osh_program_requirement", "osh_committee_requirement"}:
            hierarchy_boost = {
                "dole_do_198_18": 0.12,
                "ra_11058": 0.42,
                "occupational_safety_health_standards": 0.30,
                "dole_do_252_25": 0.64 if legal_status == "current" else 0.38,
            }.get(canonical_doc, 0.0)
            score += hierarchy_boost
            authority_score += hierarchy_boost
            if hierarchy_boost:
                reasons.append("osh_specific_authority_hierarchy")
        else:
            hierarchy_boost = {
                "ra_11058": 0.58,
                "dole_do_198_18": 0.10,
                "occupational_safety_health_standards": 0.28,
                "dole_do_252_25": 0.54 if legal_status == "current" else 0.32,
            }.get(canonical_doc, 0.0)
            score += hierarchy_boost
            authority_score += hierarchy_boost
            if hierarchy_boost:
                reasons.append("osh_authority_hierarchy")
        if responsive:
            score += 0.58
            concept_score += 0.58
            reasons.append("osh_task_proposition_match")
        else:
            # Legacy OSH metadata sometimes declares concepts that are not in
            # the passage itself. Do not let those labels outrank a direct
            # Section 4/6, PPE, program, committee, or safety-officer rule.
            if concept_score:
                score -= concept_score
                concept_score = 0.0
                reasons.append("osh_declared_concepts_ignored")
            if re.search(r"\b(means|refers\s+to|definition)\b", body, re.IGNORECASE):
                score -= 0.65
                reasons.append("osh_definition_only_penalty")
            if re.search(r"\b(accident|incident)\b.{0,120}\b(classification|collapse|burst|failure|reporting\s+form)\b", body, re.IGNORECASE | re.DOTALL):
                score -= 0.70
                reasons.append("osh_incident_list_penalty")
            if canonical_doc == "dole_do_252_25" and task in {"general_explanation", "employer_duties"}:
                score -= 0.45
                reasons.append("tangential_newer_osh_issuance_penalty")

    if topic in {"sexual_harassment", "safe_spaces"} and task == "employer_duties":
        if canonical_doc == "ra_7877" and re.search(r"\bsection\s*4\b|\b(duty|duties)\s+of\s+the\s+employer\b", combined, re.IGNORECASE):
            score += 0.62
            reasons.append("ra7877_section4_employer_duties_boost")
        if canonical_doc == "ra_11313" and re.search(r"\b(workplace|employer|codi|committee|investigat)\b", combined, re.IGNORECASE):
            score += 0.34
            reasons.append("ra11313_workplace_duties_boost")
        if re.search(r"\bremed(?:y|ies)\b", combined, re.IGNORECASE) and not re.search(
            r"\b(policy|prevent|committee|codi|investigat|employer\s+dut)\b",
            combined,
            re.IGNORECASE,
        ):
            score -= 0.45
            reasons.append("general_remedies_only_penalty")

    if topic == "labor_complaint" and re.search(r"\b(sena|single[-\s]+entry|request\s+for\s+assistance)\b", _norm(query_profile.get("normalized_query")), re.IGNORECASE):
        if re.search(r"\b(dileep|tupad|livelihood\s+program|integrated\s+livelihood)\b", combined, re.IGNORECASE):
            score -= 2.00
            reasons.append("unrelated_dole_program_excluded")
        if canonical_doc in {"ra_10396", "dole_do_249_25", "dole_arms_sena"}:
            score += 0.62
            reasons.append("sena_authority_boost")

    if topic == "jurisprudence_request":
        is_case = chunk_category == "Jurisprudence" and bool(re.search(
            r"\b(supreme\s+court|g\.?\s*r\.?\s*(?:no\.?)?|promulgated|decision)\b",
            combined,
            re.IGNORECASE,
        ))
        if is_case:
            score += 0.90
            reasons.append("actual_jurisprudence_boost")
        else:
            score -= 1.50
            reasons.append("non_jurisprudence_source_penalty")

    if topic == "contracting_subcontracting":
        if re.search(r"\bdepartment\s+order\b", combined, re.IGNORECASE):
            score += 0.30
            reasons.append("contracting_department_order_boost")
        if legal_status in {"active", "current"}:
            score += 0.20
            reasons.append("current_contracting_source_boost")
        if canonical_doc == "omnibus_rules_labor_code" and not re.search(r"\bcurrent|effective\b", combined, re.IGNORECASE):
            score -= 0.25
            reasons.append("generic_omnibus_contracting_penalty")

    if topic == "wage_deductions" and re.search(r"\bhomeworkers?\b", combined, re.IGNORECASE) and not re.search(
        r"\bhomeworkers?\b",
        _norm(query_profile.get("normalized_query")),
        re.IGNORECASE,
    ):
        score -= 0.65
        reasons.append("homeworker_only_scope_penalty")

    if topic == "thirteenth_month_pay":
        if canonical_doc in {"dole_13th_month_faq", "dole_13th_month_advisory"} or canonical_doc.startswith("dole_labor_advisory"):
            score += 0.35
            reasons.append("current_dole_13th_month_boost")
        if canonical_doc == "pd_851":
            score += 0.16
            reasons.append("pd_851_foundational_boost")
        if _looks_historical_13th_limit(combined):
            score -= 0.45
            reasons.append("historical_13th_month_ceiling_penalty")

    if topic == "domestic_workers" or query_sector == DOMESTIC_SECTOR:
        if canonical_doc == "ra_10361":
            score += 0.38
            reasons.append("ra_10361_boost")
        if canonical_doc == "ra_10361_irr":
            score += 0.34
            reasons.append("ra_10361_irr_boost")
        if is_domestic_worker_overtime_query(query_profile):
            if canonical_doc in {"ra_10361", "ra_10361_irr"} and re.search(
                r"\b(section|sec\.?)\s*11\b|proportionate\s+additional\s+payment|"
                r"hours?\s+of\s+work|rest\s+days?|allowable\s+leaves?|employment\s+contract",
                combined,
                re.IGNORECASE,
            ):
                score += 0.36
                reasons.append("kasambahay_contract_additional_payment_boost")
            if canonical_doc in {"ra_10361", "ra_10361_irr"} and re.search(
                r"\b(section|sec\.?)\s*(20|21)\b|rest\s+period|daily\s+rest|weekly\s+rest|"
                r"24\s+consecutive|twenty[-\s]?four\s+consecutive|eight\s+hours|8\s+hours",
                combined,
                re.IGNORECASE,
            ):
                score += 0.34
                reasons.append("kasambahay_rest_period_boost")
            if _has_article_reference(combined, "82") and re.search(
                r"\b(domestic\s+helpers?|persons?\s+in\s+personal\s+service|household\s+service)\b",
                combined,
                re.IGNORECASE,
            ):
                score += 0.28
                reasons.append("domestic_worker_article_82_exclusion_boost")
            if chunk_topic == "overtime_pay" and canonical_doc not in {"ra_10361", "ra_10361_irr"}:
                score -= 0.22
                reasons.append("general_overtime_table_not_primary_for_domestic_worker")
            if canonical_doc in {"labor_code_pd_442", "omnibus_rules_labor_code"} and re.search(
                r"\b(125\s*%|169\s*%|195\s*%|260\s*%|338\s*%|ordinary\s+working\s+day|"
                r"rest\s+day\s+overtime|special\s+non[-\s]?working)\b",
                combined,
                re.IGNORECASE,
            ):
                score -= 0.42
                reasons.append("generic_overtime_rates_excluded_for_domestic_worker")
        if _is_old_househelper_labor_code(combined, meta):
            score -= 0.35
            reasons.append("old_househelper_labor_code_penalty")

    if query_profile.get("is_wage_order_query"):
        if chunk_category == "DOLE Resources":
            score += 0.48
            reasons.append("dole_wage_resource_boost")
        if chunk_topic == "minimum_wage":
            score += 0.24
            reasons.append("minimum_wage_topic_boost")
        if _lower(meta.get("doc_type")) == "wage_order" or re.search(
            r"\b(wage\s+order|rtwpb|nwpc|regional\s+wage)\b",
            combined,
            re.IGNORECASE,
        ):
            score += 0.10
            reasons.append("wage_order_plus_0.10")
        if chunk_matches_query_region(combined, query_profile, meta):
            score += 0.40
            reasons.append("exact_region_boost")
        elif is_region_specific_wage_chunk({**chunk, "metadata": meta}):
            score -= 0.90
            reasons.append("wrong_region_wage_order_penalty")
        if canonical_doc in {"ra_10361", "ra_10361_irr"}:
            score -= 0.42
            reasons.append("generic_kasambahay_baseline_penalty")
    elif is_wage_order_chunk({**chunk, "metadata": meta}):
        score -= 0.90
        reasons.append("wage_order_excluded_for_non_wage_query")

    if topic == "retirement_pay":
        if canonical_doc == "ra_7641":
            score += 0.38
            reasons.append("ra_7641_retirement_boost")
        if canonical_doc == "labor_code_pd_442" and _has_article_reference(combined, "302"):
            score += 0.45
            reasons.append("article_302_retirement_boost")
        if re.search(
            r"\b(sixty|60).*\b(sixty[-\s]?five|65)\b|"
            r"\bfive\s*\(?(?:5)?\)?\s+years?\b|"
            r"\b(?:one[-\s]?half|1\s*/\s*2)\s+month\s+salary\b",
            combined,
            re.IGNORECASE,
        ):
            score += 0.22
            reasons.append("retirement_operational_details_boost")

    article_number = _norm(query_profile.get("article_number"))
    if article_number and (
        _norm(meta.get("article_number")) == article_number
        or re.search(rf"\b(?:art\.?|article)\s*{re.escape(article_number)}\b", combined, re.IGNORECASE)
    ):
        score += 0.35
        reasons.append("exact_article_match")

    updated = dict(chunk)
    # A chunk can be reprocessed after an LDA model rebuild. LDA metadata is
    # never part of legal policy scoring and must not survive into selection.
    for key in ("retrieval_rank_score", "lda_topic_similarity", "lda_topic_boost"):
        updated.pop(key, None)
    updated["metadata"] = meta
    updated["rerank_score"] = round(score, 4)
    updated["rerank_reasons"] = reasons
    updated["authority_score"] = round(authority_score, 4)
    updated["concept_score"] = round(concept_score, 4)
    updated["category_score"] = round(category_score, 4)
    updated["category_match_type"] = match_type
    return updated


def _chunk_identity(chunk: Dict[str, Any]) -> str:
    meta = chunk.get("metadata") or {}
    return str(meta.get("chunk_id") or chunk.get("id") or chunk.get("text", "")[:180])


def _source_identity(chunk: Dict[str, Any], include_content: bool = True) -> str:
    meta = enrich_chunk_metadata(chunk.get("text", ""), chunk.get("metadata") or {}, infer_concepts=False)
    canonical = _norm(meta.get("canonical_document_id"))
    document_key = canonical
    if not document_key:
        for key in ("document_id", "source_url", "official_url", "document_title", "source_name", "source"):
            value = _norm(meta.get(key))
            if value:
                document_key = value
                break
    provision_key = (
        _norm(meta.get("article_number"))
        or _norm(meta.get("article"))
        or _norm(meta.get("section"))
        or _norm(meta.get("citation"))
        or _norm(meta.get("page_number"))
        or _norm(meta.get("page"))
    )
    content_key = _norm(meta.get("content_hash")) or _hash_text(chunk.get("text", ""))
    if provision_key and not include_content:
        content_key = ""
    if document_key:
        return "|".join(part.lower() for part in (document_key, provision_key, content_key[:12]) if part)
    return _chunk_identity(chunk)


def _is_labor_code_article_87(chunk: Dict[str, Any]) -> bool:
    meta = enrich_chunk_metadata(chunk.get("text", ""), chunk.get("metadata") or {}, infer_concepts=False)
    combined = f"{chunk.get('text', '')} {_metadata_text(meta)}"
    return bool(ARTICLE_87_RE.search(combined) and _is_labor_code_authority(combined, meta))


def _labor_code_overtime_article_number(chunk: Dict[str, Any]) -> str:
    meta = enrich_chunk_metadata(chunk.get("text", ""), chunk.get("metadata") or {}, infer_concepts=False)
    combined = f"{chunk.get('text', '')} {_metadata_text(meta)}"
    if not _is_labor_code_authority(combined, meta):
        return ""
    explicit = _norm(meta.get("article_number"))
    if explicit in {"82", "84", "87", "88", "89", "93", "94"}:
        return explicit
    match = OVERTIME_SUPPORT_ARTICLE_RE.search(combined)
    return match.group(1) if match else ""


def _overtime_articles_for_task(task: str) -> List[str]:
    task = _norm(task)
    if task == "coverage_question":
        return ["82", "87"]
    if task == "exclusion_question":
        return ["82"]
    if task == "authorization_question":
        return ["84"]
    if task == "undertime_offset_question":
        return ["88"]
    if task == "compulsory_overtime_question":
        return ["89"]
    if task == "threshold_question":
        return ["87", "84"]
    if task in {"rate_question", "formula_question", "calculation_request", "personalized_computation"}:
        return ["87", "93", "94", "84"]
    return ["82", "84", "87", "93", "94", "88"]


def _is_allowed_general_chunk(chunk: Dict[str, Any]) -> bool:
    meta = chunk.get("metadata") or {}
    chunk_sector = canonical_sector(_norm(meta.get("sector")))
    rule_type = _norm(meta.get("rule_type"))
    return chunk_sector == GENERAL_SECTOR and rule_type != SPECIAL_SECTOR_RULE


def _ensure_overtime_labor_code_basis(
    selected: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    limit: int,
    task: str = "",
) -> List[Dict[str, Any]]:
    if _norm(task) in {
        "coverage_question",
        "exclusion_question",
        "authorization_question",
        "undertime_offset_question",
        "compulsory_overtime_question",
    }:
        return selected[:limit]
    if any(_is_labor_code_article_87(chunk) for chunk in selected[:3]):
        return selected[:limit]

    article_chunk = next(
        (
            chunk
            for chunk in ranked
            if _is_allowed_general_chunk(chunk) and _is_labor_code_article_87(chunk)
        ),
        None,
    )
    if not article_chunk:
        return selected[:limit]

    reordered: List[Dict[str, Any]] = [article_chunk]
    seen = {_chunk_identity(article_chunk)}
    for chunk in selected:
        key = _chunk_identity(chunk)
        if key in seen:
            continue
        reordered.append(chunk)
        seen.add(key)
        if len(reordered) >= limit:
            break
    return reordered[:limit]


def _ensure_overtime_support_articles(
    selected: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    limit: int,
    task: str = "",
) -> List[Dict[str, Any]]:
    required_order = _overtime_articles_for_task(task)
    chosen_by_article: Dict[str, Dict[str, Any]] = {}
    for chunk in ranked:
        if not _is_allowed_general_chunk(chunk):
            continue
        article = _labor_code_overtime_article_number(chunk)
        if article in required_order and article not in chosen_by_article:
            chosen_by_article[article] = chunk

    if not chosen_by_article:
        return selected[:limit]

    reordered: List[Dict[str, Any]] = []
    seen = set()
    for article in required_order:
        chunk = chosen_by_article.get(article)
        if not chunk:
            continue
        key = _chunk_identity(chunk)
        if key in seen:
            continue
        reordered.append(chunk)
        seen.add(key)
        if len(reordered) >= limit:
            return reordered

    for chunk in selected:
        key = _chunk_identity(chunk)
        if key in seen:
            continue
        reordered.append(chunk)
        seen.add(key)
        if len(reordered) >= limit:
            break
    return reordered[:limit]


def _topic_priority_key(topic: str, chunk: Dict[str, Any]) -> str:
    meta = enrich_chunk_metadata(chunk.get("text", ""), chunk.get("metadata") or {}, infer_concepts=False)
    combined = f"{chunk.get('text', '')} {_metadata_text(meta)}"
    canonical_doc = _norm(meta.get("canonical_document_id"))
    if topic == "holiday_pay":
        if _has_article_reference(combined, "94"):
            return "labor_code_article_94"
        if _is_omnibus_authority(combined) and re.search(r"\bholiday\b", combined, re.IGNORECASE):
            return "omnibus_holiday"
    if topic in {"hours_of_work", "cost_of_living_allowance"}:
        if canonical_doc == "dole_wage_benefits_faq":
            return "dole_wage_benefits_faq"
        if topic == "hours_of_work" and _has_article_reference(combined, "83"):
            return "labor_code_article_83"
        if topic == "hours_of_work" and _has_article_reference(combined, "85"):
            return "labor_code_article_85"
        if topic == "hours_of_work" and _is_omnibus_authority(combined) and re.search(r"\bmeal\b", combined, re.IGNORECASE):
            return "omnibus_meal_period"
    if topic == "thirteenth_month_pay":
        if canonical_doc.startswith("dole_labor_advisory") or canonical_doc in {"dole_13th_month_faq", "dole_13th_month_advisory"}:
            return "current_dole_13th_month"
        if canonical_doc == "pd_851":
            return "pd_851"
    if topic == "domestic_workers":
        if canonical_doc == "ra_10361":
            return "ra_10361"
        if canonical_doc == "ra_10361_irr":
            return "ra_10361_irr"
    if topic == "maternity_leave" and canonical_doc == "ra_11210":
        return "ra_11210"
    if topic == "night_shift_differential" and _has_article_reference(combined, "86"):
        return "labor_code_article_86"
    if topic == "separation_pay" and _has_article_reference(combined, "298"):
        return "labor_code_article_298"
    if topic == "workplace_safety":
        if canonical_doc == "ra_11058" and re.search(r"\bsection\s*6\b|\bright\s+to\s+refuse\b", combined, re.IGNORECASE):
            return "ra_11058_section_6"
        if canonical_doc == "dole_do_252_25":
            return "current_osh_irr"
    if topic in {"sexual_harassment", "safe_spaces"}:
        if canonical_doc == "ra_7877" and re.search(r"\bsection\s*4\b|\bemployer\s+dut", combined, re.IGNORECASE):
            return "ra_7877_section_4"
        if canonical_doc == "ra_11313":
            return "ra_11313_workplace"
    if topic == "jurisprudence_request" and normalize_kb_category(meta.get("category")) == "Jurisprudence":
        return "supreme_court_decision"
    return ""


def _ensure_topic_authorities(
    selected: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    topic: str,
    limit: int,
) -> List[Dict[str, Any]]:
    priority_orders = {
        "holiday_pay": ["labor_code_article_94", "omnibus_holiday"],
        "hours_of_work": ["dole_wage_benefits_faq", "labor_code_article_83", "labor_code_article_85", "omnibus_meal_period"],
        "cost_of_living_allowance": ["dole_wage_benefits_faq"],
        "thirteenth_month_pay": ["current_dole_13th_month", "pd_851"],
        "domestic_workers": ["ra_10361", "ra_10361_irr"],
        "maternity_leave": ["ra_11210"],
        "night_shift_differential": ["labor_code_article_86"],
        "separation_pay": ["labor_code_article_298"],
        "workplace_safety": ["ra_11058_section_6", "current_osh_irr"],
        "sexual_harassment": ["ra_7877_section_4", "ra_11313_workplace"],
        "safe_spaces": ["ra_11313_workplace", "ra_7877_section_4"],
        "jurisprudence_request": ["supreme_court_decision"],
    }
    priority_order = priority_orders.get(topic)
    if not priority_order:
        return selected[:limit]

    chosen: List[Dict[str, Any]] = []
    seen_chunks = set()
    seen_sources = set()

    for priority in priority_order:
        candidate = next(
            (
                chunk
                for chunk in ranked
                if _topic_priority_key(topic, chunk) == priority
                and _chunk_identity(chunk) not in seen_chunks
                and _source_identity(chunk) not in seen_sources
            ),
            None,
        )
        if candidate:
            chosen.append(candidate)
            seen_chunks.add(_chunk_identity(candidate))
            seen_sources.add(_source_identity(candidate))

    for chunk in selected:
        chunk_key = _chunk_identity(chunk)
        source_key = _source_identity(chunk)
        if chunk_key in seen_chunks or source_key in seen_sources:
            continue
        chosen.append(chunk)
        seen_chunks.add(chunk_key)
        seen_sources.add(source_key)
        if len(chosen) >= limit:
            break

    for chunk in ranked:
        chunk_key = _chunk_identity(chunk)
        source_key = _source_identity(chunk)
        if chunk_key in seen_chunks or source_key in seen_sources:
            continue
        chosen.append(chunk)
        seen_chunks.add(chunk_key)
        seen_sources.add(source_key)
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def _allowed_for_query_scope(
    chunk: Dict[str, Any],
    query_sector: str,
    query_profile: Optional[Dict[str, Any]] = None,
) -> bool:
    meta = chunk.get("metadata") or {}
    chunk_sector = canonical_sector(_norm(meta.get("sector")))
    rule_type = _norm(meta.get("rule_type"))
    if not (query_profile or {}).get("is_wage_order_query") and is_wage_order_chunk(chunk):
        return False
    if (query_profile or {}).get("compare_government_private"):
        from .evidence_scope import source_scope
        return source_scope(chunk) in {GENERAL_SECTOR, GOVERNMENT_SECTOR}
    if query_sector == GENERAL_SECTOR and (
        rule_type == SPECIAL_SECTOR_RULE or chunk_sector != GENERAL_SECTOR
    ):
        return False
    return sector_matches(query_sector, chunk_sector)


def _ensure_document_diversity(
    selected: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    query_sector: str,
    min_score: float,
    limit: int,
    query_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if len(selected) >= limit:
        current = selected[:limit]
    else:
        current = list(selected)

    selected_ids = {_chunk_identity(chunk) for chunk in current}
    source_counts: Dict[str, int] = {}
    for chunk in current:
        group_key = _source_identity(chunk, include_content=False)
        source_counts[group_key] = source_counts.get(group_key, 0) + 1

    for candidate in ranked:
        if len(current) >= limit and len(source_counts) >= min(limit, 4):
            break
        if _chunk_identity(candidate) in selected_ids:
            continue
        if not _allowed_for_query_scope(candidate, query_sector, query_profile):
            continue
        score = float(candidate.get("rerank_score", candidate.get("score", 0)) or 0)
        if score < min_score:
            continue
        source_key = _source_identity(candidate, include_content=False)
        if source_key in source_counts:
            continue

        if len(current) < limit:
            current.append(candidate)
            selected_ids.add(_chunk_identity(candidate))
            source_counts[source_key] = 1
            continue

        replace_index = next(
            (
                index
                for index in range(len(current) - 1, -1, -1)
                if source_counts.get(_source_identity(current[index], include_content=False), 0) > 1
            ),
            None,
        )
        if replace_index is None:
            continue
        removed = current[replace_index]
        removed_source = _source_identity(removed, include_content=False)
        source_counts[removed_source] = max(0, source_counts.get(removed_source, 1) - 1)
        if source_counts[removed_source] == 0:
            source_counts.pop(removed_source, None)
        selected_ids.discard(_chunk_identity(removed))
        current[replace_index] = candidate
        selected_ids.add(_chunk_identity(candidate))
        source_counts[source_key] = 1

    return current[:limit]


def _ensure_osh_required_authorities(
    selected: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    query_profile: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    """Keep direct passages from each essential OSH authority in context."""

    essential = list(
        required_authorities_for_query(query_profile).get("essential") or []
    )
    if not essential:
        return selected[:limit]

    required_matches: List[Dict[str, Any]] = []
    required_ids: set[str] = set()
    for authority_id in essential:
        candidate = next((
            chunk
            for chunk in ranked
            if authority_id in authority_ids_for_chunk(chunk)
            and osh_chunk_supports_task(chunk, query_profile)
            and _allowed_for_query_scope(
                chunk,
                canonical_sector(_norm(query_profile.get("sector"))),
                query_profile,
            )
        ), None)
        if candidate is not None:
            identity = _chunk_identity(candidate)
            if identity not in required_ids:
                required_matches.append(candidate)
                required_ids.add(identity)

    output = list(required_matches)
    output_ids = {_chunk_identity(chunk) for chunk in output}
    for chunk in [*selected, *ranked]:
        identity = _chunk_identity(chunk)
        if identity in output_ids:
            continue
        output.append(chunk)
        output_ids.add(identity)
        if len(output) >= limit:
            break
    return output[:limit]


def rerank_chunks(
    chunks: List[Dict[str, Any]],
    query_profile: Dict[str, Any],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    scored = [score_chunk(chunk, query_profile) for chunk in chunks or []]
    # Candidate selection, deduplication, authority recovery, and source
    # status are governed only by legal-policy and vector scores. LDA is
    # intentionally applied later, after this membership has been approved.
    for chunk in scored:
        for key in ("retrieval_rank_score", "lda_topic_similarity", "lda_topic_boost"):
            chunk.pop(key, None)
    scored.sort(
        key=lambda item: (
            item.get("rerank_score", 0),
            item.get("score", 0),
        ),
        reverse=True,
    )

    query = query_profile.get("retrieval_query") or query_profile.get("normalized_query") or ""
    from .evidence_scope import source_scope, requested_scope
    wanted = requested_scope(query)
    if wanted in {None, GENERAL_SECTOR}:
        general = [c for c in scored if source_scope(c) == GENERAL_SECTOR]
        special = [c for c in scored if source_scope(c) != GENERAL_SECTOR]
        scored = general + special
    return scored if limit is None else scored[:max(0, int(limit))]


def coverage_ranked_chunks(chunks: List[Dict[str, Any]], query_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prefer complementary passages, not repeated matches for one point.

    Call only after relevance/scope filtering. This changes order, never
    manufactures evidence or upgrades a source's similarity score.
    """
    from routers.services.legal_topic_matrix import supported_passage_concepts, source_passage_text
    from routers.services.legal_version_ranking import apply_version_precedence, version_precedence
    query = query_profile.get("retrieval_query") or query_profile.get("normalized_query") or ""
    chunks = apply_version_precedence(query, chunks)
    required = set(query_profile.get("required_concepts") or [])
    essential = set(query_profile.get("essential_concepts") or [])
    authorities = set(required_authorities_for_query(query_profile).get("essential") or [])
    prepared = [{**c, "text": source_passage_text(c.get("text") or c.get("excerpt") or "")} for c in chunks]
    support = [supported_passage_concepts(c, required) for c in prepared]
    owned = [set(authority_ids_for_chunk(c)) for c in chunks]
    constraints = version_precedence(query, chunks)
    remaining = list(range(len(chunks)))
    result = []
    while remaining:
        if not required and not authorities and not constraints:
            result.extend(prepared[i] for i in remaining)
            break
        available = [i for i in remaining if not any(old == i and new in remaining for new, old in constraints)]
        if not available:  # Conflicting version metadata: preserve relevance order.
            available = remaining
        index = max(available, key=lambda i: (
            sum(3 if c in essential else 0.25 if c == "legal_basis" else 1 for c in support[i] & required)
            + 2 * len(owned[i] & authorities),
            -i,
        ))
        if not support[index].intersection(required) and not owned[index].intersection(authorities):
            result.extend(prepared[i] for i in remaining)
            break
        result.append(prepared[index])
        required.difference_update(support[index])
        authorities.difference_update(owned[index])
        remaining.remove(index)
    return result


def select_generator_chunks(
    chunks: List[Dict[str, Any]],
    query_profile: Dict[str, Any],
    min_score: float = PARTIAL_SCORE_THRESHOLD,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    effective_limit = len(chunks or []) if limit is None else max(0, int(limit))
    ranked = rerank_chunks(chunks, query_profile)
    topic = _norm(query_profile.get("topic"))
    if topic == "workplace_safety":
        responsive = [c for c in ranked if osh_chunk_supports_task(c, query_profile)]
        # With no operative rule, do not keep unrelated passages merely to
        # populate the response/source list.
        ranked = responsive
    if topic == "thirteenth_month_pay":
        non_historical = [
            chunk for chunk in ranked
            if not _looks_historical_13th_limit(
                f"{chunk.get('text', '')} {_metadata_text(chunk.get('metadata') or {})}"
            )
        ]
        # Exclude obsolete ceiling passages when the KB also has responsive,
        # non-historical material. This is source filtering, not an answer rule.
        if non_historical:
            ranked = non_historical
    from routers.services.scenario_retrieval import scenario_evidence_gap
    scenario_query = query_profile.get("retrieval_query") or query_profile.get("normalized_query") or ""
    # Prefer operative issue evidence over generic topic mentions, without
    # raising scores or bypassing approval/sector checks. With no scenario
    # constraint every key is equal and ranking remains unchanged.
    ranked.sort(key=lambda c: not scenario_evidence_gap(scenario_query, [c.get("text", "")]), reverse=True)
    selected: List[Dict[str, Any]] = []
    selected_source_ids: set[str] = set()
    query_sector = canonical_sector(_norm(query_profile.get("sector")))
    query = query_profile.get("retrieval_query") or query_profile.get("normalized_query") or ""
    from .evidence_scope import requested_scope, scope_compatible, source_scope
    wanted_scope = requested_scope(query)
    eligible = []
    for chunk in ranked:
        meta = chunk.get("metadata") or {}
        score = float(chunk.get("rerank_score", chunk.get("score", 0)) or 0)
        if score < min_score:
            continue
        if not query_profile.get("is_wage_order_query") and is_wage_order_chunk(chunk):
            continue
        source_key = _source_identity(chunk)
        if source_key in selected_source_ids:
            continue
        eligible.append(chunk)
        selected_source_ids.add(source_key)
    # Applicable sector is a ranking signal. Keep other populations available
    # as supplementary evidence, while ordering responsive applicable sources
    # first; scope_completeness remains the publication gate.
    if query_profile.get("compare_government_private"):
        applicable = [c for c in eligible if source_scope(c) in {GENERAL_SECTOR, GOVERNMENT_SECTOR}]
    else:
        applicable = [c for c in eligible if scope_compatible(
            wanted_scope or GENERAL_SECTOR, source_scope(c))]
    supplementary = [c for c in eligible if c not in applicable]
    eligible = applicable + supplementary
    ranked = eligible
    from routers.services.legal_topic_matrix import supported_passage_concepts
    # Preserve complete retrieved passages; provider-safe context packing is
    # applied after ranking, not through a passage-count or character cutoff.
    selected = coverage_ranked_chunks(ranked, query_profile)
    if limit is not None:
        selected = selected[:effective_limit]
    chosen = {_chunk_identity(c): c for c in selected}
    ranked = [chosen.get(_chunk_identity(c)) or c for c in ranked]
    coverage_selected = list(selected)
    if query_sector == GENERAL_SECTOR and topic == "overtime_pay":
        selected = _ensure_overtime_labor_code_basis(selected, ranked, effective_limit, task=_norm(query_profile.get("task")))
        selected = _ensure_overtime_support_articles(selected, ranked, effective_limit, task=_norm(query_profile.get("task")))
    if is_domestic_worker_overtime_query(query_profile):
        selected = _ensure_topic_authorities(selected, ranked, "domestic_workers", effective_limit)
    selected = _ensure_topic_authorities(selected, ranked, topic, effective_limit)
    selected = _ensure_document_diversity(
        selected,
        ranked,
        query_sector,
        min_score,
        effective_limit,
        query_profile=query_profile,
    )
    if topic == "workplace_safety":
        selected = _ensure_osh_required_authorities(
            selected,
            ranked,
            query_profile,
            effective_limit,
        )
    # Diversity/authority backfilling must not reintroduce Regional Wage
    # Orders that the primary selector intentionally excluded for a non-wage
    # question (for example, night-shift differential).
    if not query_profile.get("is_wage_order_query"):
        selected = [chunk for chunk in selected if not is_wage_order_chunk(chunk)]
    if query_profile.get("is_wage_order_query"):
        matching_region_chunks = [
            chunk
            for chunk in selected
            if chunk_matches_query_region(
                chunk.get("text", ""),
                query_profile,
                chunk.get("metadata") or {},
            )
        ]
        if matching_region_chunks:
            selected = [
                chunk
                for chunk in selected
                if chunk in matching_region_chunks or not is_region_specific_wage_chunk(chunk)
            ]
        selected.sort(
            key=lambda chunk: (
                chunk_matches_query_region(
                    f"{chunk.get('text', '')} {_metadata_text(chunk.get('metadata') or {})}",
                    query_profile,
                    chunk.get("metadata") or {},
                ),
                canonical_sector(_norm((chunk.get("metadata") or {}).get("sector"))) == query_sector,
                normalize_kb_category((chunk.get("metadata") or {}).get("category")) == "DOLE Resources",
                wage_order_detail_score(chunk),
                float(chunk.get("rerank_score", chunk.get("score", 0)) or 0),
            ),
            reverse=True,
        )
    if query_profile.get("is_wage_order_query") and query_sector == DOMESTIC_SECTOR:
        from routers.services.scenario_retrieval import domestic_wage_passage
        selected = [chunk for chunk in selected if not is_wage_order_chunk(chunk)
                    or domestic_wage_passage(chunk.get("text", ""))]
    from routers.services.scenario_retrieval import is_employee_departure_query, property_remedy_evidence
    if is_employee_departure_query(scenario_query):
        selected = [chunk for chunk in selected
                    if not scenario_evidence_gap(scenario_query, [chunk.get("text", "")])
                    or property_remedy_evidence([chunk.get("text", "")])]
    required = query_profile.get("required_concepts") or []
    before = set().union(*(supported_passage_concepts(c, required) for c in coverage_selected))
    after = set().union(*(supported_passage_concepts(c, required) for c in selected))
    if before - after:
        # Document variety is secondary to keeping a uniquely supported rule.
        selected = coverage_selected
    if query_profile.get("compare_government_private"):
        # Keep at least one passage for each employment system when both are
        # present; fill remaining slots with the existing relevance order.
        from .evidence_scope import source_scope
        private = next((c for c in ranked if source_scope(c) == GENERAL_SECTOR), None)
        government = next((c for c in ranked if source_scope(c) == GOVERNMENT_SECTOR), None)
        if private is not None and government is not None and effective_limit >= 2:
            paired = [private, government]
            paired_ids = {_chunk_identity(c) for c in paired}
            paired.extend(c for c in selected if _chunk_identity(c) not in paired_ids)
            selected = paired[:effective_limit]
    return selected if limit is None else selected[:effective_limit]


def source_match_status(
    chunks: List[Dict[str, Any]],
    query_profile: Dict[str, Any],
    min_relevance_score: float = 0.45,
) -> Tuple[str, str]:
    if query_profile.get("intent") in NON_LEGAL_INTENTS or query_profile.get("topic") == "non_legal":
        return (
            SOURCE_STATUS_NOT_REQUIRED,
            "This message is small talk, identity, gratitude, or a capability question and does not require legal retrieval.",
        )
    if not chunks:
        return (
            SOURCE_STATUS_INSUFFICIENT,
            "No approved legal source passed scope-aware retrieval for this question.",
        )

    from routers.services.scenario_retrieval import scenario_evidence_gap
    gap = scenario_evidence_gap(
        query_profile.get("retrieval_query") or query_profile.get("normalized_query") or "",
        [chunk.get("text", "") for chunk in chunks],
    )
    if gap:
        return SOURCE_STATUS_INSUFFICIENT, gap

    from routers.services.scenario_retrieval import is_employee_departure_query, UNRETURNED_PROPERTY_QUERY, property_remedy_evidence
    scenario_query = query_profile.get("retrieval_query") or query_profile.get("normalized_query") or ""
    if (is_employee_departure_query(scenario_query)
            and re.search(UNRETURNED_PROPERTY_QUERY, scenario_query, re.I)
            and not property_remedy_evidence([chunk.get("text", "") for chunk in chunks])):
        return SOURCE_STATUS_PARTIAL, (
            "Retrieved evidence addresses employee departure, but does not establish a remedy for unreturned company property. "
            "Explain only supported employment rules with their conditions; identify the property-remedy evidence gap."
        )

    query_sector = canonical_sector(_norm(query_profile.get("sector")))
    direct = []
    partial = []
    mismatches = []
    special_used_for_general = False
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        chunk_sector = canonical_sector(_norm(meta.get("sector")))
        score = float(chunk.get("rerank_score", chunk.get("score", 0)) or 0)
        authority_grounded = (
            str(chunk.get("retrieval_method") or meta.get("retrieval_method") or "")
            == "required_authority_lookup"
            and bool(authority_ids_for_chunk(chunk))
        )
        if query_sector == GENERAL_SECTOR and chunk_sector != GENERAL_SECTOR:
            special_used_for_general = True
            mismatches.append(chunk)
            continue
        if sector_matches(query_sector, chunk_sector) and (
            score >= DIRECT_SCORE_THRESHOLD or authority_grounded
        ):
            direct.append(chunk)
        elif sector_matches(query_sector, chunk_sector) and score >= min_relevance_score:
            partial.append(chunk)
        else:
            mismatches.append(chunk)

    topic = canonical_topic(query_profile.get("topic"))
    if topic == "workplace_safety" and not osh_evidence_is_responsive(
        [*direct, *partial],
        query_profile,
    ):
        return (
            SOURCE_STATUS_INSUFFICIENT,
            "Retrieved OSH passages are topically related but do not state the legal proposition required for this OSH sub-intent.",
        )
    concept_sources: List[Dict[str, Any]] = []
    for chunk in [*direct, *partial]:
        meta = dict(chunk.get("metadata") or {})
        meta["text"] = chunk.get("text", "")
        meta["excerpt"] = chunk.get("text", "")
        concept_sources.append(meta)
    missing_essential_authorities = missing_required_authorities([*direct, *partial], query_profile)
    essential = list(query_profile.get("essential_concepts") or essential_concepts_for_topic(
        topic,
        _norm(query_profile.get("normalized_query")),
        task=_norm(query_profile.get("task")),
    ))
    required_concepts = list(query_profile.get("required_concepts") or required_concepts_for_topic(
        topic,
        _norm(query_profile.get("normalized_query")),
        task=_norm(query_profile.get("task")),
    ))
    if query_profile.get("required_concepts"):
        supported = _supported_profile_concepts(required_concepts, concept_sources)
        missing = [concept for concept in required_concepts if concept not in supported]
    else:
        supported = collect_supported_concepts(
            topic,
            concept_sources,
            _norm(query_profile.get("normalized_query")),
            task=_norm(query_profile.get("task")),
        )
        missing = missing_required_concepts(
            topic,
            concept_sources,
            _norm(query_profile.get("normalized_query")),
            task=_norm(query_profile.get("task")),
        )
    if required_concepts:
        essential_supported = any(concept in supported for concept in essential) if essential else bool(supported)
        logger.debug(
            "Source-match coverage | topic=%s | sector=%s | retrieved=%s | essential=%s | supported=%s | missing=%s",
            topic,
            query_sector,
            [
                _norm((chunk.get("metadata") or {}).get("chunk_id"))
                or _norm(chunk.get("id"))
                or _norm((chunk.get("metadata") or {}).get("citation"))
                for chunk in chunks
            ],
            essential,
            sorted(supported),
            missing,
        )
        if direct and not missing and not missing_essential_authorities and not special_used_for_general:
            return (
                SOURCE_STATUS_DIRECT,
                "Retrieved legal sources directly match the detected topic and cover the required source concepts: "
                + ", ".join(required_concepts)
                + ".",
            )
        if (direct or partial) and essential_supported:
            detail = (
                " Missing source concepts: " + ", ".join(missing) + "."
                if missing
                else " Missing required authorities: " + ", ".join(missing_essential_authorities) + "."
                if missing_essential_authorities
                else " Supported source concepts: " + ", ".join(sorted(supported)) + "."
            )
            return (
                SOURCE_STATUS_PARTIAL,
                "Relevant legal sources were found, but the source set is not complete enough for a direct match."
                + detail,
            )
        if direct or partial:
            if supported:
                return (
                    SOURCE_STATUS_PARTIAL,
                    "Relevant legal sources support a useful answer, but the source set is missing essential concept support: "
                    + ", ".join(missing or essential)
                    + ".",
                )
            return (
                SOURCE_STATUS_INSUFFICIENT,
                "Retrieved passages are related, but none supports the topic's essential legal concept.",
            )

    if not required_concepts:
        from routers.services.issue_evidence import unclassified_evidence_responsive
        responsive = unclassified_evidence_responsive(scenario_query, [c.get('text', '') for c in [*direct, *partial]])
        return (
            SOURCE_STATUS_PARTIAL if responsive else SOURCE_STATUS_INSUFFICIENT,
            "The question's evidence requirements are unresolved. Similarity alone cannot establish a direct source match."
            if responsive else "No operative passage establishes the issue asked about; an empty evidence checklist is not a complete match.",
        )

    if missing_essential_authorities and (direct or partial):
        return (
            SOURCE_STATUS_PARTIAL,
            "Relevant legal sources were found, but an essential authority was not retrieved: "
            + ", ".join(missing_essential_authorities)
            + ".",
        )

    if direct and not special_used_for_general:
        return (
            SOURCE_STATUS_DIRECT,
            "At least one authoritative retrieved passage directly answers the question and matches the detected legal scope.",
        )
    if direct:
        return (
            SOURCE_STATUS_PARTIAL,
            "Relevant sources were found, but a special-sector passage was also retrieved and must not be generalized.",
        )
    if partial:
        return (
            SOURCE_STATUS_PARTIAL,
            "Relevant legal sources were found, but additional facts or a narrower scope may be needed.",
        )
    return (
        SOURCE_STATUS_INSUFFICIENT,
        "Retrieved passages did not adequately match the question's legal scope.",
    )


def should_enqueue_review(source_status: str) -> bool:
    return source_status in {SOURCE_STATUS_INSUFFICIENT, SOURCE_STATUS_CONFLICTING, 'unverified'}


def frontend_confidence(status: str) -> str:
    return {
        'unverified': 'Unverified',
        SOURCE_STATUS_NOT_REQUIRED: "Not Required",
        SOURCE_STATUS_DIRECT: "Direct",
        SOURCE_STATUS_PARTIAL: "Partial",
        SOURCE_STATUS_CONFLICTING: "Conflicting",
        SOURCE_STATUS_PROVIDER_UNAVAILABLE: "Unavailable",
        SOURCE_STATUS_INSUFFICIENT: "Insufficient",
    }.get(status, "Insufficient")


def legal_scope_prompt(query_profile: Dict[str, Any], source_status: str) -> str:
    from routers.services.document_evidence import is_document_driven
    if is_document_driven(query_profile):
        return (
            'Use the question-derived evidence map and the supplied operative passages. '
            'Topic/category labels are routing hints, not laws or eligibility rules. '
            'Retain every relevant scope restriction, condition and exception. '
            'Distinguish user facts, source rules and conditional conclusions. '
            'Use only explicit amendment relationships and applicable effective dates; '
            'upload date alone never establishes precedence. Do not treat an indexed '
            'document as independent confirmation of current law. Explain missing '
            'evidence and conflicts without guessing. Ask only for a missing user fact '
            'that materially changes the requested conclusion, not facts already supplied.'
        )
    coverage = query_profile.get("required_evidence_coverage") or {}
    missing_concepts = [
        str(concept).replace("_", " ")
        for concept in coverage.get("missing_concepts") or []
    ]
    profile = {
        "topic": query_profile.get("topic"),
        "sector": query_profile.get("sector"),
        "task": query_profile.get("task"),
        "calculation_requested": query_profile.get("calculation_requested"),
        "personal_facts_available": query_profile.get("personal_facts_available"),
        "source_match": source_status,
        "missing_required_source_support": missing_concepts,
    }
    guidance = (
        "Backend legal-scope classification:\n"
        f"{profile}\n\n"
        "Rules for using the retrieved sources:\n"
        "- Legal-version metadata is evidence context, not a substitute for the operative rule. "
        "Do not present an explicitly superseded provision as current law. Apply amendments only "
        "to their identified provisions; keep unaffected rules and exceptions. Publication year "
        "or upload date alone does not establish repeal. For dated situations use the version "
        "effective then, and for comparisons distinguish the versions. If effectivity or an "
        "amendment relationship cannot be established, explain the uncertainty rather than guess.\n"
        "- Never generalize a provision that applies to a named occupation, sector, employee class, workplace type, or special rule.\n"
        "- If a source is marked special-sector, explicitly state its limited scope or omit it from the main answer.\n"
        "- Do not present hospital/clinic personnel weekly-hour rules as the general overtime rule.\n"
        "- For general overtime, use the eight-hours-per-day rule and supported ordinary-day/rest-day/holiday rates only.\n"
        "- Personalized computation is allowed only when the user supplies enough values, hours, and day type.\n"
    )
    if source_status == SOURCE_STATUS_PARTIAL and missing_concepts:
        guidance += (
            "- Required evidence was not retrieved for these points: "
            + ", ".join(missing_concepts)
            + ". Do not infer their legal outcome from a related rule, contract label, or model knowledge. "
            "Answer only the supported parts and clearly say which conclusion the current excerpts do not establish.\n"
        )
    return guidance


def log_rag_debug(
    query: str,
    intent: str,
    search_text: str,
    query_profile: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    source_status: str,
) -> None:
    logger.info(
        "RAG debug | normalized_query=%s | intent=%s | detected_sector=%s | topic=%s | "
        "rewritten_query=%s | source_match=%s",
        query,
        intent,
        query_profile.get("sector"),
        query_profile.get("topic"),
        search_text,
        source_status,
    )
    for index, chunk in enumerate(chunks or [], 1):
        meta = chunk.get("metadata") or {}
        logger.info(
            "RAG debug chunk %s | chunk_id=%s | similarity=%s | rerank=%s | "
            "document=%s | article=%s | rule=%s | sector=%s | scope=%s | reasons=%s",
            index,
            meta.get("chunk_id") or chunk.get("id"),
            chunk.get("score"),
            chunk.get("rerank_score"),
            meta.get("document_title") or meta.get("source"),
            meta.get("citation") or meta.get("article"),
            meta.get("rule"),
            meta.get("sector"),
            meta.get("source_scope"),
            ",".join(chunk.get("rerank_reasons") or []),
        )
