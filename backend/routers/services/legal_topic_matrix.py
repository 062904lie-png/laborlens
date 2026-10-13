"""Legal topic and source-requirement matrix for LaborLens.

The matrix gives the RAG layer a deterministic checklist before an answer is
marked as directly supported. It is intentionally heuristic: the knowledge base
still supplies the law, while this file tells the system what kinds of source
support to look for per topic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from routers.services.scenario_retrieval import WORK_INJURY_QUERY


GENERAL_LABOR_TOPIC = "general_labor"
NON_LEGAL_TOPIC = "non_legal"

SUPPORTED_TOPICS = {
    "minimum_wage",
    "cost_of_living_allowance",
    "payment_of_wages",
    "wage_deductions",
    "overtime_pay",
    "holiday_pay",
    "premium_pay",
    "night_shift_differential",
    "hours_of_work",
    "service_incentive_leave",
    "service_charges",
    "thirteenth_month_pay",
    "maternity_leave",
    "paternity_leave",
    "solo_parent_leave",
    "special_leave_for_women",
    "probationary_employment",
    "security_floating_status",
    "regular_employment",
    "project_employment",
    "fixed_term_employment",
    "termination",
    "procedural_due_process",
    "separation_pay",
    "retirement_pay",
    "domestic_workers",
    "workplace_safety",
    "workplace_injury",
    "sexual_harassment",
    "safe_spaces",
    "discrimination",
    "union_rights",
    "collective_bargaining",
    "strike_lockout",
    "contracting_subcontracting",
    "ofw",
    "jurisprudence_request",
    "government_employment",
    "social_security",
    "philhealth",
    "pagibig",
    "employees_compensation",
    "telecommuting",
    "labor_complaint",
    GENERAL_LABOR_TOPIC,
    NON_LEGAL_TOPIC,
}

TOPIC_ALIASES = {
    "13th_month": "thirteenth_month_pay",
    "13th_month_pay": "thirteenth_month_pay",
    "thirteenth_month": "thirteenth_month_pay",
    "wages": "payment_of_wages",
    "wage": "payment_of_wages",
    "wage_inquiry": "payment_of_wages",
    "leave_benefits": "service_incentive_leave",
    "termination_rights": "termination",
    "illegal_dismissal": "termination",
    "occupational_safety": "workplace_safety",
    "osh": "workplace_safety",
    "employment_status": "regular_employment",
    "kasambahay": "domestic_workers",
    "sss": "social_security",
    "pag_ibig": "pagibig",
    "pag-ibig": "pagibig",
    "cba": "collective_bargaining",
    "sena": "labor_complaint",
    "strike": "strike_lockout",
    "lockout": "strike_lockout",
}


def _clean_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def canonical_topic(topic: Any) -> str:
    key = _clean_key(topic)
    return TOPIC_ALIASES.get(key, key or GENERAL_LABOR_TOPIC)


def _metadata_text(metadata: Optional[Dict[str, Any]]) -> str:
    meta = metadata or {}
    keys = (
        "topic",
        "category",
        "document_title",
        "source",
        "source_name",
        "law_number",
        "citation",
        "article_number",
        "article_title",
        "article_section",
        "book",
        "title",
        "chapter",
        "rule",
        "section",
        "issuing_agency",
    )
    return " ".join(str(meta.get(key) or "") for key in keys)


TOPIC_PATTERNS: Sequence[tuple[str, Sequence[str]]] = (
    ("cost_of_living_allowance", (
        r"\b(COLA|cost[-\s]+of[-\s]+living\s+allowance)\b",
    )),
    ("hours_of_work", (
        r"\b(normal\s+hours?\s+of\s+work|meal\s+periods?|meal\s+breaks?)\b",
        r"\b(?:art\.?|article)\s*(83|85)\b",
    )),
    ("thirteenth_month_pay", (
        r"\b(13th(?:[-\s]*month)?|thirteenth(?:[-\s]*month)?|p\.?\s*d\.?\s*851|pd\s*851|month\s+pay)\b",
    )),
    ("overtime_pay", (
        r"\b(overtime|over\s*time|\bot\b|extra\s+hours?|beyond\s+eight|beyond\s+8|"
        r"hours?\s+after\s+(?:my\s+|the\s+)?normal\s+shift|after\s+(?:my\s+|the\s+)?normal\s+shift|"
        r"labaw\s+sa\s+8|lampas\s+sa\s+8)\b",
        r"\b(?:art\.?|article)\s*(82|84|87|88|89)\b",
    )),
    ("night_shift_differential", (
        r"\b(night\s*(shift)?\s*differential|night\s*diff|graveyard|between\s+ten\s+p\.?m\.?\s+and\s+six\s+a\.?m\.?)\b",
    )),
    ("holiday_pay", (
        r"\b(holiday\s+pay|regular\s+holidays?|double\s+pay)\b",
        r"\b(?:art\.?|article)\s*94\b",
    )),
    ("premium_pay", (
        r"\b(premium\s+pay|rest\s+day\s+pay|special\s+day\s+pay|worked\s+on\s+rest\s+day|"
        r"special\s+(?:non[-\s]?)?working\s+(?:day|holiday)|non[-\s]?working\s+holiday)\b",
        r"\b(?:art\.?|article)\s*93\b",
    )),
    ("minimum_wage", (
        r"\b(minimum\s+wage|wage\s+order|regional\s+tripartite|rtwpb|nwpc)\b",
    )),
    ("wage_deductions", (
        r"\b(deduct|deduction|withhold|salary\s+deduction|cash\s+bond|deposit|shortage|damage\s+deduction)\b",
    )),
    ("payment_of_wages", (
        r"\b(unpaid\s+(salary|wage|sweldo|suhol)|salary|wage|payday|late\s+pay|payment\s+of\s+wages|sahod|sweldo|suhol)\b",
    )),
    ("maternity_leave", (
        r"\b(maternity|pregnan|childbirth|ra\s*11210|11210)\b",
    )),
    ("paternity_leave", (
        r"\b(paternity|father\s+leave|ra\s*8187|8187)\b",
    )),
    ("solo_parent_leave", (
        # "Parental leave" is the statutory term used by RA 11861 for the
        # paid leave benefit of qualified solo parents.  Treating it as a
        # generic leave question sends retrieval to service-incentive leave
        # instead of the controlling solo-parent materials.
        r"\b(solo\s+parent|single\s+parent|parental\s+leave|ra\s*11861|ra\s*8972|11861|8972)\b",
    )),
    ("special_leave_for_women", (
        r"\b(special\s+leave\s+for\s+women|gynecological|gynaecological|ra\s*9710|magna\s+carta\s+of\s+women)\b",
    )),
    ("service_incentive_leave", (
        r"\b(service\s+incentive\s+leave|\bsil\b|unused\s+leave|leave\s+credits?|sick\s+leave|vacation\s+leave)\b",
    )),
    ("service_charges", (
        r"\b(service\s+charges?|service\s+charge\s+distribution|"
        r"service\s+charge\s+collected|article\s*96|do\s*242[-\s]?24)\b",
    )),
    ("probationary_employment", (
        r"\b(probationary|probation|new\s+hire|newly\s+hired|new\s+employee|six\s+months?|6\s+months?|standards\s+made\s+known)\b",
        r"\b(?:art\.?|article)\s*296\b",
    )),
    ("project_employment", (
        r"\b(project[-\s]?based|project\s+employee|project\s+employment)\b",
    )),
    ("fixed_term_employment", (
        r"\b(fixed[-\s]?term|fixed\s+period|term\s+employment|contractual\s+period)\b",
    )),
    ("security_floating_status", (
        r"\b(?:security\s+guards?|security\s+agenc(?:y|ies)|private\s+security|security\s+service\s+contractors?|ssc\s*/\s*psa)\b.{0,140}\b(?:floating\s+status|reserved\s+status|work\s+pool|off[-\s]?detail)\b",
        r"\b(?:floating\s+status|reserved\s+status|work\s+pool|off[-\s]?detail)\b.{0,140}\b(?:security\s+guards?|security\s+agenc(?:y|ies)|private\s+security|security\s+service\s+contractors?|ssc\s*/\s*psa)\b",
    )),
    ("regular_employment", (
        r"\b(regular\s+employee|regularization|regular\s+employment|casual\s+employee|employment\s+status)\b",
        r"\b(?:art\.?|article)\s*295\b",
    )),
    ("procedural_due_process", (
        r"\b(due\s+process|notice\s+to\s+explain|nte|hearing|twin\s+notice|procedural)\b",
    )),
    ("separation_pay", (
        r"\b(separation\s+pay|authorized\s+cause|redundancy|retrenchment|closure|disease)\b",
    )),
    ("termination", (
        r"\b(termination|terminate|terminated|terminating|dismissal|dismissed|fired|sacked|"
        r"illegal\s+dismissal|just\s+cause|serious\s+misconduct|willful\s+disobedience|"
        r"resign|resignation|tinanggal|natanggal|gintanggal|tanggalon)\b",
    )),
    ("retirement_pay", (
        r"\b(retirement|retiree|ra\s*7641|7641)\b",
    )),
    ("domestic_workers", (
        r"\b(kasambahay|kasambhay|domestic\s+workers?|domestics?\s+workers?|house\s*helpers?|househelpers?|"
        r"household\s+(?:helpers?|workers?|service)|domestic\s+helpers?|yaya|maid|housemaid|"
        r"ra\s*10361|10361|batas\s+kasambahay)\b",
    )),
    ("workplace_injury", (
        WORK_INJURY_QUERY,
        r"\bemployees'? compensation\b",
    )),
    ("workplace_safety", (
        r"\b(fire\s+exit|blocked\s+exit|nakaharang.{0,30}daanan)\b",
        r"\b(workplace\s+safety|osh|oshs|occupational\s+(?:safety|health)|"
        r"safety\s+and\s+health|unsafe\s+work|hazards?|ppe|safety\s+committee|"
        r"safety\s+officer|ra\s*11058|11058|do\s*198[-\s]?18)\b",
    )),
    ("sexual_harassment", (
        r"\b(sexual\s+harassment|ra\s*7877|7877)\b",
        r"\b(malalaswa\w*|malaswa\w*|lewd|sexual)\b.{0,60}\b(chat|messages?|mensahe)\b",
    )),
    ("safe_spaces", (
        r"\b(safe\s+spaces|bawal\s+bastos|gender[-\s]?based\s+sexual\s+harassment|ra\s*11313|11313)\b",
    )),
    ("discrimination", (
        r"(?:hindi.{0,15}tinanggap|not hired|rejected).{0,100}(?:babae|lalaki|gender|sex|woman|man)",
        r"\b(discrimination|discriminate|gender\s+discrimination|equal\s+pay|anti[-\s]?discrimination)\b",
    )),
    ("collective_bargaining", (
        r"\b(collective\s+bargaining|\bcba\b|bargaining\s+unit)\b",
    )),
    ("strike_lockout", (
        r"\b(strike|lockout|notice\s+of\s+strike|picket|work\s+stoppage)\b",
    )),
    ("union_rights", (
        r"\b(union|organize|self[-\s]?organization|labor\s+organization|concerted\s+activities|"
        r"unyon|paghiusa\s+sang\s+mamumugon)\b",
    )),
    ("government_employment", (
        r"\b(government\s+employee|public\s+sector|civil\s+service|state\s+university|lgu|deped|doh|government\s+agency)\b",
    )),
    ("social_security", (
        r"\b(sss|social\s+security|social\s+security\s+system)\b",
    )),
    ("philhealth", (
        r"\b(philhealth|national\s+health\s+insurance)\b",
    )),
    ("pagibig", (
        r"\b(pag[-\s]?ibig|hdmf|home\s+development\s+mutual\s+fund)\b",
    )),
    ("employees_compensation", (
        r"\b(employees'? compensation|ecc|ec\s+benefit|work\s+contingency)\b",
    )),
    ("telecommuting", (
        r"\b(telecommut(?:e|ing)|work[-\s]+from[-\s]+home|remote\s+work|"
        r"ra\s*11165|11165|do\s*202[-\s]?19|telecommuting\s+act)\b",
    )),
    ("labor_complaint", (
        r"\b(?:file|filing|submit|lodge)\s+(?:a\s+)?complain(?:t)?\b",
        r"\bcomplain(?:t)?\s+against\s+(?:my\s+|the\s+)?(?:employer|company|boss)\b",
        r"\b(file|filing|submit|lodge|make)\s+(?:a\s+)?(?:labor\s+)?complaint\s+(?:with|to|at)\s+dole\b",
        r"\b(dole\s+complaint|labor\s+complaint|request\s+for\s+assistance|\bsena\b|single[-\s]+entry\s+approach)\b",
    )),
    ("contracting_subcontracting", (
        r"\b(contracting|subcontracting|labor[-\s]?only\s+contracting|contractor|subcontractor|trilateral)\b",
    )),
    ("ofw", (
        r"\b(ofw|overseas\s+filipino\s+worker|migrant\s+worker|overseas\s+employment|recruitment\s+agency)\b",
    )),
    ("jurisprudence_request", (
        r"\b(supreme\s+court\s+(?:case|ruling|decision)|jurisprudence|g\.??r\.??\s*no\.?|case\s+law)\b",
    )),
)


TOPIC_SOURCE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "overtime_pay": {
        "essential": ["ordinary_overtime_rule"],
        "supporting": [
            "coverage_and_exclusions",
            "hours_worked",
            "rest_day_and_special_day_rates",
            "regular_holiday_rate",
            "holiday_and_rest_day_overtime",
            "omnibus_overtime_rules",
        ],
        "preferred": ["Labor Code Article 82", "Labor Code Article 84", "Labor Code Article 87", "Labor Code Article 93", "Labor Code Article 94", "Omnibus Rules Book III"],
    },
    "thirteenth_month_pay": {
        "essential": ["employee_coverage", "basic_salary_definition", "calculation_formula"],
        "supporting": ["payment_deadline", "resigned_employee_entitlement", "excluded_payments"],
        "preferred": ["PD 851", "DOLE 13th-month pay guidelines", "Labor Code wage-benefit sources"],
    },
    "minimum_wage": {
        "required": ["wage_order_or_region", "covered_workers", "effective_date"],
        "preferred": ["NWPC or RTWPB wage order", "DOLE wage advisory"],
    },
    "cost_of_living_allowance": {
        "required": ["cola_entitlement", "cola_definition"],
        "preferred": ["DOLE FAQs on Wage and Wage-Related Benefits"],
    },
    "payment_of_wages": {
        "required": ["wage_payment_rule", "legal_basis"],
        "preferred": ["Labor Code wage provisions", "DOLE wage rules"],
    },
    "wage_deductions": {
        "essential": ["deduction_rule", "employee_responsibility_for_loss"],
        "supporting": ["loss_or_damage_proof", "deduction_fairness_safeguards", "legal_basis"],
        "preferred": ["Labor Code wage deductions", "DOLE rules"],
    },
    "holiday_pay": {
        "required": ["holiday_type", "coverage", "first_8_rate"],
        "preferred": ["Labor Code Article 94", "Omnibus Rules holiday pay"],
    },
    "premium_pay": {
        "required": ["day_type", "coverage", "premium_rate"],
        "preferred": ["Labor Code Article 93", "Omnibus Rules premium pay"],
    },
    "night_shift_differential": {
        "required": ["night_period", "rate", "coverage"],
        "preferred": ["Labor Code Article 86", "Omnibus Rules night shift differential"],
    },
    "hours_of_work": {
        "required": ["normal_daily_hours", "coverage"],
        "preferred": ["Labor Code Article 83", "Labor Code Article 85", "Omnibus Rules Book III"],
    },
    "service_incentive_leave": {
        "required": ["one_year_service", "five_day_minimum", "coverage"],
        "preferred": ["Labor Code Article 95"],
    },
    "service_charges": {
        "required": ["service_charge_coverage", "service_charge_distribution", "legal_basis"],
        "preferred": ["Labor Code Article 96", "DOLE Department Order No. 242-24"],
    },
    "maternity_leave": {
        "required": ["leave_duration", "qualifying_event", "benefit_source"],
        "preferred": ["RA 11210", "SSS maternity benefit rules"],
    },
    "paternity_leave": {
        "required": ["covered_employee", "qualifying_delivery", "leave_duration"],
        "preferred": ["RA 8187"],
    },
    "solo_parent_leave": {
        "required": [
            "solo_parent_qualification",
            "solo_parent_leave_days",
            "solo_parent_service_requirement",
            "legal_basis",
        ],
        "preferred": ["RA 11861", "RA 8972"],
    },
    "special_leave_for_women": {
        "required": ["gynecological_condition", "leave_duration", "legal_basis"],
        "preferred": ["RA 9710"],
    },
    "probationary_employment": {
        "required": ["probationary_period", "standards_at_engagement", "regularization"],
        "preferred": ["Labor Code Article 296"],
    },
    "security_floating_status": {
        "essential": ["security_reserved_status_rule", "six_month_security_status_rule"],
        "supporting": ["security_reserved_status_conditions", "security_separation_entitlement"],
        "preferred": [
            "DOLE Department Order No. 150-16, Sections 10.3 and 10.4",
        ],
    },
    "regular_employment": {
        "required": ["regular_status_test", "legal_basis"],
        "preferred": ["Labor Code Article 295"],
    },
    "project_employment": {
        "required": ["project_duration_or_scope", "completion_condition", "legal_basis"],
        "preferred": ["Labor Code Article 295", "DOLE employment status rules"],
    },
    "fixed_term_employment": {
        "required": ["fixed_term_condition", "legal_basis"],
        "preferred": ["Labor Code Article 295", "jurisprudence on fixed-term employment"],
    },
    "termination": {
        "required": ["just_cause", "authorized_cause", "legal_basis"],
        "preferred": ["Labor Code Article 297", "Labor Code Article 298"],
    },
    "procedural_due_process": {
        "required": ["notice", "opportunity_to_be_heard", "decision_notice"],
        "preferred": ["Labor Code termination provisions", "DOLE due-process rules"],
    },
    "separation_pay": {
        "essential": ["authorized_cause", "rate_or_formula"],
        "supporting": ["authorized_cause_notice", "separation_pay_rounding", "coverage_and_exceptions", "legal_basis"],
        "preferred": ["Labor Code Article 298"],
    },
    "retirement_pay": {
        "required": [
            "retirement_age",
            "retirement_service_requirement",
            "retirement_pay_formula",
            "retirement_exemption",
            "legal_basis",
        ],
        "preferred": ["Labor Code Article 302", "RA 7641"],
    },
    "domestic_workers": {
        "required": ["domestic_worker_definition", "employment_contract", "minimum_benefits", "legal_basis"],
        "preferred": ["RA 10361", "IRR of RA 10361"],
    },
    "workplace_safety": {
        "required": ["employer_duties", "worker_rights", "legal_basis"],
        "preferred": ["RA 11058", "DOLE Department Order No. 198-18", "Occupational Safety and Health Standards"],
    },
    "workplace_injury": {
        "required": ["work_related_injury", "reporting_or_claim", "benefit_source"],
        "preferred": ["Employees Compensation rules", "ECC/SSS/GSIS sources"],
    },
    "sexual_harassment": {
        "required": ["prohibited_act", "covered_relationship", "remedies_or_duties"],
        "preferred": ["RA 7877 Section 4", "RA 11313 workplace provisions", "current DOLE workplace-harassment issuance"],
    },
    "telecommuting": {
        "required": ["fair_treatment", "minimum_labor_standards", "legal_basis"],
        "preferred": ["RA 11165", "DOLE Department Order No. 202-19"],
    },
    "labor_complaint": {
        "required": ["complaint_channel", "conciliation_process", "legal_basis"],
        "preferred": ["RA 10396", "current SEnA guidelines", "official SEnA forms", "DOLE ARMS"],
    },
    "safe_spaces": {
        "required": ["gender_based_harassment", "covered_place", "employer_duties"],
        "preferred": ["RA 11313"],
    },
    "discrimination": {
        "required": ["protected_ground", "prohibited_act", "legal_basis"],
        "preferred": ["Labor Code", "Magna Carta of Women", "relevant anti-discrimination law"],
    },
    "union_rights": {
        "required": ["self_organization", "covered_workers", "legal_basis"],
        "preferred": ["Labor Code Book V"],
    },
    "collective_bargaining": {
        "required": ["bargaining_unit_or_cba", "duty_to_bargain", "legal_basis"],
        "preferred": ["Labor Code Book V"],
    },
    "strike_lockout": {
        "required": ["strike_or_lockout_rule", "filing_or_notice_requirement", "legal_basis"],
        "preferred": ["Labor Code Book V", "NCMB rules"],
    },
    "contracting_subcontracting": {
        "essential": ["contracting_rule", "prohibited_labor_only_contracting"],
        "supporting": ["current_controlling_issuance", "legal_basis"],
        "preferred": ["current applicable DOLE contracting Department Order", "relevant current jurisprudence"],
    },
    "ofw": {
        "required": ["overseas_worker_scope", "applicable_protection", "legal_basis"],
        "preferred": ["Migrant Workers law", "DMW or POEA rules"],
    },
    "jurisprudence_request": {
        "essential": ["case_holding", "case_citation"],
        "supporting": ["case_name", "decision_date", "legal_basis"],
        "preferred": ["Supreme Court decision"],
    },
    "government_employment": {
        "required": ["civil_service_scope", "legal_basis"],
        "preferred": ["Civil Service rules", "special law for government employment"],
    },
    "social_security": {
        "required": ["covered_member", "contribution_or_benefit", "legal_basis"],
        "preferred": ["SSS law and regulations"],
    },
    "philhealth": {
        "required": ["covered_member", "contribution_or_benefit", "legal_basis"],
        "preferred": ["PhilHealth law and regulations"],
    },
    "pagibig": {
        "required": ["covered_member", "contribution_or_benefit", "legal_basis"],
        "preferred": ["Pag-IBIG/HDMF law and regulations"],
    },
    "employees_compensation": {
        "required": ["covered_contingency", "claim_or_benefit", "legal_basis"],
        "preferred": ["Employees Compensation Commission sources"],
    },
}

TASK_SOURCE_REQUIREMENTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "overtime_pay": {
        "coverage_question": {
            "essential": ["coverage_and_exclusions", "excluded_employee_categories"],
            "supporting": ["ordinary_overtime_rule"],
            "preferred": ["Labor Code Article 82", "Labor Code Article 87"],
        },
        "eligibility_question": {
            "essential": ["coverage_and_exclusions", "excluded_employee_categories"],
            "supporting": ["ordinary_overtime_rule"],
            "preferred": ["Labor Code Article 82", "Labor Code Article 87"],
        },
        "exclusion_question": {
            "essential": ["coverage_and_exclusions", "excluded_employee_categories"],
            "supporting": [],
            "preferred": ["Labor Code Article 82"],
        },
        "authorization_question": {
            "essential": ["hours_worked", "employer_required_suffered_permitted"],
            "supporting": [],
            "preferred": ["Labor Code Article 84"],
        },
        "undertime_offset_question": {
            "essential": ["undertime_offset_rule"],
            "supporting": [],
            "preferred": ["Labor Code Article 88"],
        },
        "compulsory_overtime_question": {
            "essential": ["emergency_overtime_rule"],
            "supporting": [],
            "preferred": ["Labor Code Article 89"],
        },
        "threshold_question": {
            "essential": ["ordinary_overtime_rule"],
            "supporting": ["hours_worked"],
            "preferred": ["Labor Code Article 87", "Labor Code Article 84"],
        },
        "rate_question": {
            "essential": ["ordinary_overtime_rule"],
            "supporting": ["rest_day_and_special_day_rates", "regular_holiday_rate"],
            "preferred": ["Labor Code Article 87", "Labor Code Article 93", "Labor Code Article 94"],
        },
        "formula_question": {
            "essential": ["ordinary_overtime_rule"],
            "supporting": ["hours_worked"],
            "preferred": ["Labor Code Article 87", "Labor Code Article 84"],
        },
        "calculation_request": {
            "essential": ["ordinary_overtime_rule"],
            "supporting": ["rest_day_and_special_day_rates", "regular_holiday_rate"],
            "preferred": ["Labor Code Article 87", "Labor Code Article 93", "Labor Code Article 94"],
        },
    },
    "holiday_pay": {
        "comparison_question": {
            "essential": ["regular_holiday_unworked_rule", "regular_holiday_worked_rate"],
            "supporting": [
                "special_non_working_unworked_rule",
                "special_non_working_worked_rate",
                "coverage_and_qualification",
            ],
            "preferred": ["Labor Code Article 94", "Labor Code Article 93", "Omnibus Rules holiday pay"],
        },
        "coverage_question": {
            "essential": ["coverage"],
            "supporting": ["holiday_type"],
            "preferred": ["Labor Code Article 94"],
        },
        "eligibility_question": {
            "essential": ["holiday_preceding_day_condition"],
            "supporting": ["coverage_and_qualification"],
            "preferred": ["Omnibus Rules Book III Rule IV Section 6", "DOLE wage-benefits FAQ"],
        },
        "successive_holidays_question": {
            "essential": ["successive_holiday_condition"],
            "supporting": ["holiday_preceding_day_condition"],
            "preferred": ["Omnibus Rules Book III Rule IV Section 10", "DOLE wage-benefits FAQ"],
        },
        "rate_question": {
            "essential": ["holiday_type", "first_8_rate"],
            "supporting": ["coverage"],
            "preferred": ["Labor Code Article 94", "Omnibus Rules holiday pay"],
        },
        "formula_question": {
            "essential": ["holiday_type", "first_8_rate"],
            "supporting": [],
            "preferred": ["Labor Code Article 94"],
        },
        "rest_day_question": {
            "essential": ["holiday_type", "first_8_rate", "holiday_rest_day_rule"],
            "supporting": ["coverage_and_qualification", "coverage_and_exceptions"],
            "preferred": ["Labor Code Article 94", "Labor Code Article 93", "Omnibus Rules Book III Rule IV"],
        },
    },
    "hours_of_work": {
        "normal_hours_question": {
            "essential": ["normal_daily_hours"],
            "supporting": ["coverage"],
            "preferred": ["Labor Code Article 83", "DOLE wage-benefits FAQ"],
        },
        "meal_period_question": {
            "essential": ["meal_period_duration"],
            "supporting": ["meal_period_compensability"],
            "preferred": ["Labor Code Article 85", "Omnibus Rules Book III Rule I Section 7", "DOLE wage-benefits FAQ"],
        },
    },
    "cost_of_living_allowance": {
        "eligibility_question": {
            "essential": ["cola_entitlement"],
            "supporting": ["cola_definition"],
            "preferred": ["DOLE FAQs on Wage and Wage-Related Benefits"],
        },
        "general_explanation": {
            "essential": ["cola_entitlement"],
            "supporting": ["cola_definition"],
            "preferred": ["DOLE FAQs on Wage and Wage-Related Benefits"],
        },
    },
    "thirteenth_month_pay": {
        "coverage_question": {
            "essential": ["employee_coverage"],
            "supporting": ["rank_and_file_coverage", "one_month_service", "regardless_nature_employment", "thirteenth_month_exclusions"],
            "preferred": ["PD 851", "DOLE 13th-month pay guidelines"],
        },
        "eligibility_question": {
            "essential": ["employee_coverage"],
            "supporting": ["rank_and_file_coverage", "one_month_service", "regardless_nature_employment", "resigned_employee_entitlement"],
            "preferred": ["PD 851", "DOLE 13th-month pay guidelines"],
        },
        "rate_question": {
            "essential": ["calculation_formula", "basic_salary_definition"],
            "supporting": ["excluded_payments"],
            "preferred": ["PD 851", "DOLE 13th-month pay guidelines"],
        },
        "formula_question": {
            "essential": ["calculation_formula", "basic_salary_definition"],
            "supporting": ["excluded_payments"],
            "preferred": ["PD 851", "DOLE 13th-month pay guidelines"],
        },
        "deadline_question": {
            "essential": ["payment_deadline"],
            "supporting": ["employee_coverage"],
            "preferred": ["PD 851", "DOLE 13th-month pay guidelines"],
        },
        "calculation_request": {
            "essential": ["calculation_formula", "basic_salary_definition"],
            "supporting": ["excluded_payments"],
            "preferred": ["PD 851", "DOLE 13th-month pay guidelines"],
        },
    },
    "domestic_workers": {
        "general_explanation": {
            "essential": ["domestic_worker_definition", "employment_contract", "minimum_benefits"],
            "supporting": [
                "humane_treatment",
                "board_lodging_medical",
                "privacy",
                "outside_communication",
                "education_training",
                "kasambahay_daily_rest_period",
                "kasambahay_weekly_rest_period",
                "kasambahay_wages",
                "kasambahay_13th_month",
                "kasambahay_social_benefits",
                "personal_documents",
                "termination_protections",
            ],
            "preferred": ["RA 10361", "IRR of RA 10361"],
        },
        "definition_question": {
            "essential": ["domestic_worker_definition"],
            "supporting": ["legal_basis"],
            "preferred": ["RA 10361"],
        },
        "coverage_question": {
            "essential": ["domestic_worker_definition", "minimum_benefits"],
            "supporting": [
                "employment_contract",
                "humane_treatment",
                "board_lodging_medical",
                "privacy",
                "outside_communication",
                "education_training",
                "kasambahay_daily_rest_period",
                "kasambahay_weekly_rest_period",
                "kasambahay_wages",
                "kasambahay_13th_month",
                "kasambahay_social_benefits",
                "personal_documents",
                "termination_protections",
            ],
            "preferred": ["RA 10361", "IRR of RA 10361"],
        },
        "document_requirement": {
            "essential": ["employment_contract"],
            "supporting": ["legal_basis"],
            "preferred": ["RA 10361", "IRR of RA 10361"],
        },
        "legal_remedy": {
            "essential": ["remedies_or_duties"],
            "supporting": ["legal_basis"],
            "preferred": ["RA 10361", "IRR of RA 10361"],
        },
        "overtime_question": {
            "essential": ["domestic_worker_definition", "kasambahay_rest_period"],
            "supporting": ["labor_code_domestic_exclusion", "legal_basis"],
            "preferred": ["RA 10361", "IRR of RA 10361", "Labor Code Article 82"],
        },
        "sector_coverage_question": {
            "essential": ["labor_code_domestic_exclusion", "ra10361_contract_additional_payment"],
            "supporting": ["kasambahay_daily_rest_period", "kasambahay_weekly_rest_period", "legal_basis"],
            "preferred": ["Labor Code Article 82", "RA 10361 Section 11", "RA 10361 Section 20", "RA 10361 Section 21"],
        },
    },
    "regular_employment": {
        "general_explanation": {
            "essential": ["regular_status_test"],
            "supporting": ["legal_basis"],
            "preferred": ["Labor Code Article 295"],
        },
        "definition_question": {
            "essential": ["regular_status_test"],
            "supporting": ["legal_basis"],
            "preferred": ["Labor Code Article 295"],
        },
        "coverage_question": {
            "essential": ["regular_status_test"],
            "supporting": ["legal_basis"],
            "preferred": ["Labor Code Article 295"],
        },
    },
    "probationary_employment": {
        "definition_question": {
            "essential": ["probationary_period", "standards_at_engagement"],
            "supporting": ["regularization"],
            "preferred": ["Labor Code Article 296"],
        },
        "coverage_question": {
            "essential": ["probationary_period", "standards_at_engagement"],
            "supporting": ["regularization"],
            "preferred": ["Labor Code Article 296"],
        },
        "eligibility_question": {
            "essential": ["probationary_period", "standards_at_engagement"],
            "supporting": ["regularization"],
            "preferred": ["Labor Code Article 296"],
        },
    },
    "termination": {
        "general_explanation": {
            "essential": ["just_cause", "authorized_cause"],
            "supporting": ["legal_basis"],
            "preferred": ["Labor Code Article 297", "Labor Code Article 298"],
        },
        "legal_remedy": {
            "essential": ["just_cause", "authorized_cause"],
            "supporting": ["notice", "opportunity_to_be_heard"],
            "preferred": ["Labor Code Article 297", "Labor Code Article 298"],
        },
        "document_requirement": {
            "essential": ["notice"],
            "supporting": ["opportunity_to_be_heard", "decision_notice"],
            "preferred": ["Labor Code termination provisions", "DOLE due-process rules"],
        },
        "coverage_question": {
            "essential": ["just_cause", "authorized_cause"],
            "supporting": ["legal_basis"],
            "preferred": ["Labor Code Article 297", "Labor Code Article 298"],
        },
    },
    "separation_pay": {
        task: {
            "essential": ["authorized_cause", "rate_or_formula"],
            "supporting": ["authorized_cause_notice", "separation_pay_rounding", "coverage_and_exceptions", "legal_basis"],
            "preferred": ["Labor Code Article 298"],
        }
        for task in (
            "general_explanation",
            "coverage_question",
            "rate_question",
            "formula_question",
            "calculation_request",
            "legal_remedy",
        )
    },
    "workplace_safety": {
        "general_explanation": {
            "essential": ["employer_duties"],
            "supporting": ["worker_rights", "safe_and_healthful_workplace", "legal_basis"],
            "preferred": ["RA 11058", "DOLE Department Order No. 252-25", "Occupational Safety and Health Standards"],
        },
        "hazard_reporting": {
            "essential": ["worker_rights"],
            "supporting": ["imminent_danger", "dole_determination", "corrective_action_not_taken", "legal_basis"],
            "preferred": ["RA 11058 Section 7", "RA 11058 Section 6", "DOLE Department Order No. 252-25"],
        },
        "employer_duties": {
            "essential": ["employer_duties", "safe_and_healthful_workplace"],
            "supporting": ["osh_information_training", "ppe_provision", "osh_program", "osh_committee", "safety_officer_requirement", "legal_basis"],
            "preferred": ["RA 11058 Section 4", "DOLE Department Order No. 252-25", "Occupational Safety and Health Standards"],
        },
        "refusal_of_unsafe_work": {
            "essential": ["right_to_refuse_unsafe_work", "imminent_danger"],
            "supporting": ["dole_determination", "corrective_action_not_taken", "reprisal_protection", "legal_basis"],
            "preferred": ["RA 11058 Section 6", "DOLE Department Order No. 252-25"],
        },
        "ppe_requirement": {
            "essential": ["ppe_provision"],
            "supporting": ["employer_duties", "safe_and_healthful_workplace", "legal_basis"],
            "preferred": ["RA 11058", "DOLE Department Order No. 252-25", "Occupational Safety and Health Standards"],
        },
        "safety_officer_requirement": {
            "essential": ["safety_officer_requirement"],
            "supporting": ["osh_program", "osh_committee", "legal_basis"],
            "preferred": ["DOLE Department Order No. 252-25", "RA 11058", "Occupational Safety and Health Standards"],
        },
        "osh_program_requirement": {
            "essential": ["osh_program"],
            "supporting": ["osh_committee", "safety_officer_requirement", "legal_basis"],
            "preferred": ["DOLE Department Order No. 252-25", "RA 11058", "Occupational Safety and Health Standards"],
        },
        "osh_committee_requirement": {
            "essential": ["osh_committee"],
            "supporting": ["osh_program", "safety_officer_requirement", "legal_basis"],
            "preferred": ["DOLE Department Order No. 252-25", "RA 11058", "Occupational Safety and Health Standards"],
        },
    },
    "sexual_harassment": {
        "employer_duties": {
            "essential": ["harassment_prevention_policy", "complaint_investigation_body"],
            "supporting": ["confidentiality", "retaliation_protection", "enforcement_responsibility", "legal_basis"],
            "preferred": ["RA 7877 Section 4", "RA 11313 workplace provisions", "current DOLE issuance"],
        },
    },
    "labor_complaint": {
        "document_requirement": {
            "essential": ["sena_request_for_assistance", "sena_document_requirements"],
            "supporting": ["conciliation_process", "complaint_channel", "legal_basis"],
            "preferred": ["RA 10396", "current SEnA guidelines", "official SEnA forms", "DOLE ARMS"],
        },
    },
    "retirement_pay": {
        "general_explanation": {
            "essential": ["retirement_age", "retirement_service_requirement", "retirement_pay_formula"],
            "supporting": ["retirement_exemption", "legal_basis"],
            "preferred": ["Labor Code Article 302", "RA 7641"],
        },
        "definition_question": {
            "essential": ["retirement_age", "retirement_service_requirement", "retirement_pay_formula"],
            "supporting": ["retirement_exemption", "legal_basis"],
            "preferred": ["Labor Code Article 302", "RA 7641"],
        },
        "coverage_question": {
            "essential": ["retirement_service_requirement", "retirement_exemption"],
            "supporting": ["retirement_age", "retirement_pay_formula", "legal_basis"],
            "preferred": ["Labor Code Article 302", "RA 7641"],
        },
        "rate_question": {
            "essential": ["retirement_pay_formula"],
            "supporting": ["retirement_age", "retirement_service_requirement", "legal_basis"],
            "preferred": ["Labor Code Article 302", "RA 7641"],
        },
        "formula_question": {
            "essential": ["retirement_pay_formula"],
            "supporting": ["retirement_age", "retirement_service_requirement", "legal_basis"],
            "preferred": ["Labor Code Article 302", "RA 7641"],
        },
    },
}

REQUIRED_AUTHORITIES: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "social_security": {
        "default": {"essential": ["ra_11199"], "supporting": []},
    },
    "philhealth": {
        "default": {"essential": ["ra_11223"], "supporting": []},
    },
    "pagibig": {
        "default": {"essential": ["ra_9679"], "supporting": []},
    },
    "security_floating_status": {
        "default": {
            "essential": ["dole_do_150_16"],
            "supporting": [],
        },
    },
    "overtime_pay": {
        "coverage_question": {
            "essential": ["labor_code_article_82"],
            "supporting": ["labor_code_article_87"],
        },
        "eligibility_question": {
            "essential": ["labor_code_article_82"],
            "supporting": ["labor_code_article_87"],
        },
        "exclusion_question": {
            "essential": ["labor_code_article_82"],
            "supporting": [],
        },
        "sector_coverage_question": {
            "essential": ["labor_code_article_82"],
            "supporting": ["ra_10361", "ra_10361_irr"],
        },
        "authorization_question": {
            "essential": ["labor_code_article_84"],
            "supporting": [],
        },
        "undertime_offset_question": {
            "essential": ["labor_code_article_88"],
            "supporting": [],
        },
        "compulsory_overtime_question": {
            "essential": ["labor_code_article_89"],
            "supporting": [],
        },
        "threshold_question": {
            "essential": ["labor_code_article_87"],
            "supporting": ["labor_code_article_84"],
        },
        "rate_question": {
            "essential": ["labor_code_article_87"],
            "supporting": ["labor_code_article_93", "labor_code_article_94"],
        },
        "formula_question": {
            "essential": ["labor_code_article_87"],
            "supporting": ["labor_code_article_84"],
        },
        "calculation_request": {
            "essential": ["labor_code_article_87"],
            "supporting": ["labor_code_article_93", "labor_code_article_94"],
        },
        "personalized_computation": {
            "essential": ["labor_code_article_87"],
            "supporting": ["labor_code_article_93", "labor_code_article_94"],
        },
    },
    "holiday_pay": {
        "comparison_question": {
            "essential": ["labor_code_article_94", "labor_code_article_93"],
            "supporting": ["omnibus_rules_labor_code"],
        },
        "coverage_question": {
            "essential": ["labor_code_article_94"],
            "supporting": [],
        },
        "eligibility_question": {
            "essential": ["omnibus_rules_labor_code"],
            "supporting": ["dole_wage_benefits_faq", "labor_code_article_94"],
        },
        "successive_holidays_question": {
            "essential": ["omnibus_rules_labor_code"],
            "supporting": ["dole_wage_benefits_faq", "labor_code_article_94"],
        },
        "rate_question": {
            "essential": ["labor_code_article_94"],
            "supporting": ["labor_code_article_93", "omnibus_rules_labor_code"],
        },
        "formula_question": {
            "essential": ["labor_code_article_94"],
            "supporting": ["labor_code_article_93"],
        },
        "rest_day_question": {
            "essential": ["labor_code_article_94", "labor_code_article_93"],
            "supporting": ["omnibus_rules_labor_code"],
        },
    },
    "hours_of_work": {
        "normal_hours_question": {
            # Article 83 is the controlling provision for the normal daily
            # work-hour rule. A FAQ may explain it but cannot replace it.
            "essential": ["labor_code_article_83"],
            "supporting": ["dole_wage_benefits_faq"],
        },
        "meal_period_question": {
            "essential": ["labor_code_article_85"],
            "supporting": ["omnibus_rules_labor_code", "dole_wage_benefits_faq"],
        },
    },
    "night_shift_differential": {
        task: {
            "essential": ["labor_code_article_86"],
            "supporting": ["omnibus_rules_labor_code"],
        }
        for task in (
            "general_explanation",
            "coverage_question",
            "rate_question",
            "formula_question",
            "calculation_request",
        )
    },
    "service_incentive_leave": {
        task: {
            "essential": ["labor_code_article_95"],
            "supporting": [],
        }
        for task in (
            "general_explanation",
            "coverage_question",
            "eligibility_question",
            "rate_question",
            "formula_question",
        )
    },
    "service_charges": {
        task: {
            "essential": ["labor_code_article_96"],
            "supporting": [],
        }
        for task in (
            "general_explanation",
            "coverage_question",
            "definition_question",
            "rate_question",
            "legal_remedy",
        )
    },
    "cost_of_living_allowance": {
        "eligibility_question": {
            "essential": ["dole_wage_benefits_faq"],
            "supporting": [],
        },
        "general_explanation": {
            "essential": ["dole_wage_benefits_faq"],
            "supporting": [],
        },
    },
    "thirteenth_month_pay": {
        "coverage_question": {
            # Eligibility is verified from the current DOLE FAQ; PD 851 remains
            # the foundational authority but its historical ceiling text cannot
            # replace current guidance.
            "essential": ["dole_13th_month_faq"],
            "supporting": ["pd_851"],
            # Keep the generated answer grounded in current DOLE coverage guidance.
            # Historical PD 851 material remains available for tasks it governs.
            "context": ["dole_13th_month_faq"],
        },
        "eligibility_question": {
            "essential": ["dole_13th_month_faq"],
            "supporting": ["pd_851"],
            "context": ["dole_13th_month_faq"],
        },
        "rate_question": {
            "essential": ["dole_13th_month_faq"],
            "supporting": ["pd_851", "dole_13th_month_advisory"],
            "context": ["dole_13th_month_faq", "dole_13th_month_advisory"],
        },
        "formula_question": {
            "essential": ["dole_13th_month_faq"],
            "supporting": ["pd_851", "dole_13th_month_advisory"],
            "context": ["dole_13th_month_faq", "dole_13th_month_advisory"],
        },
        "calculation_request": {
            "essential": ["dole_13th_month_faq"],
            "supporting": ["pd_851", "dole_13th_month_advisory"],
            "context": ["dole_13th_month_faq", "dole_13th_month_advisory"],
        },
    },
    "domestic_workers": {
        "general_explanation": {
            "essential": ["ra_10361"],
            "supporting": ["ra_10361_irr"],
        },
        "definition_question": {
            "essential": ["ra_10361"],
            "supporting": ["ra_10361_irr"],
        },
        "coverage_question": {
            "essential": ["ra_10361"],
            "supporting": ["ra_10361_irr"],
        },
        "document_requirement": {
            "essential": ["ra_10361"],
            "supporting": ["ra_10361_irr"],
        },
        "legal_remedy": {
            "essential": ["ra_10361"],
            "supporting": ["ra_10361_irr"],
        },
        "overtime_question": {
            "essential": ["ra_10361", "labor_code_article_82"],
            "supporting": ["ra_10361_irr"],
        },
        "sector_coverage_question": {
            "essential": ["ra_10361", "labor_code_article_82"],
            "supporting": ["ra_10361_irr"],
        },
    },
    "regular_employment": {
        "general_explanation": {
            "essential": ["labor_code_article_295"],
            "supporting": [],
        },
        "definition_question": {
            "essential": ["labor_code_article_295"],
            "supporting": [],
        },
        "coverage_question": {
            "essential": ["labor_code_article_295"],
            "supporting": [],
        },
    },
    "probationary_employment": {
        "general_explanation": {
            "essential": ["labor_code_article_296"],
            "supporting": ["labor_code_article_295"],
        },
        "definition_question": {
            "essential": ["labor_code_article_296"],
            "supporting": ["labor_code_article_295"],
        },
        "coverage_question": {
            "essential": ["labor_code_article_296"],
            "supporting": ["labor_code_article_295"],
        },
        "eligibility_question": {
            "essential": ["labor_code_article_296"],
            "supporting": ["labor_code_article_295"],
        },
    },
    "project_employment": {
        "general_explanation": {
            "essential": ["labor_code_article_295"],
            "supporting": [],
        },
        "definition_question": {
            "essential": ["labor_code_article_295"],
            "supporting": [],
        },
    },
    "fixed_term_employment": {
        "general_explanation": {
            "essential": ["labor_code_article_295"],
            "supporting": [],
        },
        "definition_question": {
            "essential": ["labor_code_article_295"],
            "supporting": [],
        },
    },
    "termination": {
        "general_explanation": {
            "essential": ["labor_code_article_297", "labor_code_article_298"],
            "supporting": [],
        },
        "legal_remedy": {
            "essential": ["labor_code_article_297", "labor_code_article_298"],
            "supporting": [],
        },
        "coverage_question": {
            "essential": ["labor_code_article_297", "labor_code_article_298"],
            "supporting": [],
        },
        "follow_up_question": {
            "essential": ["labor_code_article_297", "labor_code_article_298"],
            "supporting": [],
        },
    },
    "separation_pay": {
        task: {
            "essential": ["labor_code_article_298"],
            "supporting": [],
        }
        for task in (
            "general_explanation",
            "coverage_question",
            "rate_question",
            "formula_question",
            "calculation_request",
            "legal_remedy",
        )
    },
    "workplace_safety": {
        "general_explanation": {
            "essential": ["ra_11058"],
            "supporting": ["dole_do_252_25", "occupational_safety_health_standards"],
        },
        "employer_duties": {
            "essential": ["ra_11058"],
            "supporting": ["dole_do_252_25", "occupational_safety_health_standards"],
        },
        "refusal_of_unsafe_work": {
            "essential": ["ra_11058"],
            "supporting": ["dole_do_252_25"],
        },
        "ppe_requirement": {
            "essential": ["ra_11058"],
            "supporting": ["dole_do_252_25", "occupational_safety_health_standards"],
        },
        "safety_officer_requirement": {
            "essential": ["dole_do_252_25", "ra_11058"],
            "supporting": ["occupational_safety_health_standards"],
        },
        "osh_program_requirement": {
            "essential": ["dole_do_252_25", "ra_11058"],
            "supporting": ["occupational_safety_health_standards"],
        },
        "osh_committee_requirement": {
            "essential": ["dole_do_252_25", "ra_11058"],
            "supporting": ["occupational_safety_health_standards"],
        },
        "coverage_question": {"essential": ["ra_11058"], "supporting": ["dole_do_252_25"]},
        "eligibility_question": {"essential": ["ra_11058"], "supporting": ["dole_do_252_25"]},
        "legal_remedy": {"essential": ["ra_11058"], "supporting": ["dole_do_252_25"]},
        "definition_question": {"essential": ["ra_11058"], "supporting": []},
    },
    "sexual_harassment": {
        task: {
            "essential": ["ra_7877"],
            "supporting": ["ra_11313"],
        }
        for task in (
            "general_explanation",
            "definition_question",
            "coverage_question",
            "legal_remedy",
            "employer_duties",
        )
    },
    "safe_spaces": {
        task: {
            "essential": ["ra_11313"],
            "supporting": ["ra_7877"],
        }
        for task in (
            "general_explanation",
            "definition_question",
            "coverage_question",
            "legal_remedy",
        )
    },
    "telecommuting": {
        task: {
            "essential": ["ra_11165"],
            "supporting": [],
        }
        for task in (
            "general_explanation",
            "definition_question",
            "coverage_question",
            "legal_remedy",
        )
    },
    "labor_complaint": {
        "document_requirement": {
            "essential": ["ra_10396"],
            "supporting": ["dole_do_249_25", "dole_arms_sena"],
        },
        "general_explanation": {
            "essential": ["ra_10396"],
            "supporting": ["dole_do_249_25", "dole_arms_sena"],
        },
    },
    "jurisprudence_request": {
        task: {
            "essential": ["supreme_court_decision"],
            "supporting": [],
        }
        for task in (
            "general_explanation",
            "citation_request",
            "legal_remedy",
            "definition_question",
        )
    },
    "retirement_pay": {
        "general_explanation": {
            "essential": ["labor_code_article_302"],
            "supporting": ["ra_7641"],
        },
        "definition_question": {
            "essential": ["labor_code_article_302"],
            "supporting": ["ra_7641"],
        },
        "coverage_question": {
            "essential": ["labor_code_article_302"],
            "supporting": ["ra_7641"],
        },
        "rate_question": {
            "essential": ["labor_code_article_302"],
            "supporting": ["ra_7641"],
        },
        "formula_question": {
            "essential": ["labor_code_article_302"],
            "supporting": ["ra_7641"],
        },
    },
    "maternity_leave": {
        task: {
            "essential": ["ra_11210"],
            "supporting": [],
        }
        for task in (
            "general_explanation",
            "definition_question",
            "coverage_question",
            "eligibility_question",
            "rate_question",
            "follow_up_question",
        )
    },
    "solo_parent_leave": {
        "general_explanation": {
            "essential": ["ra_11861"],
            "supporting": ["ra_8972"],
        },
        "coverage_question": {
            "essential": ["ra_11861"],
            "supporting": ["ra_8972"],
        },
        "eligibility_question": {
            "essential": ["ra_11861"],
            "supporting": ["ra_8972"],
        },
    },
}

AUTHORITY_SEARCH_TERMS: Dict[str, str] = {
    "sss_contribution_schedule_2025": (
        "SSS Circular 2024-006 Schedule of SSS Contributions effective January 2025 "
        "business employers employees compensation range monthly salary credit employee share employer share"
    ),
    "dole_do_150_16": (
        "DOLE Department Order No. 150-16 revised guidelines security guards private security "
        "Section 10.3 reserved status work pool no service assignment Section 10.4 six months separation pay"
    ),
    "dole_do_215_20": (
        "DOLE Department Order No. 215-20 Rule I Section 12 suspension employment relationship "
        "payment wages while suspended subject to existing laws collective agreement employer policy"
    ),
    "ra_11210": (
        "Republic Act No. 11210 Expanded Maternity Leave Law one hundred five 105 days "
        "additional fifteen 15 days solo parent sixty 60 days miscarriage emergency termination"
    ),
    "labor_code_article_83": (
        "Labor Code Article 83 Normal Hours of Work shall not exceed eight hours a day"
    ),
    "labor_code_article_85": (
        "Labor Code Article 85 Meal Periods not less than sixty minutes regular meals"
    ),
    "labor_code_article_86": (
        "Labor Code Article 86 Night Shift Differential at least ten percent work between "
        "ten o'clock in the evening and six o'clock in the morning"
    ),
    "labor_code_article_295": (
        "Labor Code Article 295 Regular and Casual Employment regular employee "
        "necessary or desirable usual business project seasonal casual one year service"
    ),
    "labor_code_article_296": (
        "Labor Code Article 296 Probationary Employment six months reasonable standards "
        "made known at engagement regular employee"
    ),
    "labor_code_article_82": (
        "Labor Code Article 82 Coverage hours-of-work provisions covered employees "
        "exclusions government employees managerial employees field personnel domestic helpers "
        "persons in personal service dependent family members workers paid by results"
    ),
    "labor_code_article_84": (
        "Labor Code Article 84 Hours worked employee required suffered or permitted to work employer"
    ),
    "labor_code_article_87": (
        "Labor Code Article 87 Overtime Work beyond eight hours regular wage plus at least twenty-five percent"
    ),
    "labor_code_article_88": "Labor Code Article 88 Undertime not offset by overtime",
    "labor_code_article_89": "Labor Code Article 89 Emergency overtime work compulsory overtime",
    "labor_code_article_93": (
        "Labor Code Article 93 Compensation for rest day Sunday or holiday work rest day special day premium"
    ),
    "labor_code_article_94": (
        "Labor Code Article 94 Right to holiday pay regular holiday worked unworked holiday pay"
    ),
    "labor_code_article_95": (
        "Labor Code Article 95 Service Incentive Leave five days with pay after one year of service coverage exclusions"
    ),
    "labor_code_article_96": (
        "Labor Code Article 96 service charges hotels restaurants covered employees distribution Department Order 242-24"
    ),
    "labor_code_article_297": (
        "Labor Code Article 297 termination by employer just causes employee fault serious misconduct "
        "willful disobedience gross neglect fraud breach of trust crime analogous causes"
    ),
    "labor_code_article_298": (
        "Labor Code Article 298 authorized causes installation of labor-saving devices redundancy "
        "retrenchment closure business economic reasons separation pay"
    ),
    "labor_code_article_299": (
        "Labor Code Article 299 termination due to disease medical condition separation pay notice"
    ),
    "labor_code_article_302": (
        "Labor Code Article 302 retirement age sixty 60 compulsory sixty-five 65 "
        "five years service one-half month salary formula exemptions"
    ),
    "ra_7641": (
        "Republic Act No. 7641 Retirement Pay Law qualified private-sector employees "
        "age 60 65 five years service one-half month salary"
    ),
    "omnibus_rules_labor_code": (
        "Omnibus Rules Implementing the Labor Code Book III Rule I overtime holiday pay rest day special day"
    ),
    "dole_wage_benefits_faq": (
        "DOLE FAQs on Wage and Wage Related Benefits COLA normal hours meal period holiday pay successive regular holidays"
    ),
    "pd_851": "Presidential Decree No. 851 13th month pay rank-and-file employees one twelfth basic salary",
    "dole_13th_month_faq": "DOLE 13th month pay FAQ rank-and-file at least one month regardless employment status",
    "dole_13th_month_advisory": "DOLE Labor Advisory 13th month pay covered rank-and-file employees",
    "ra_10361": "Republic Act No. 10361 Domestic Workers Act Batas Kasambahay rights benefits",
    "ra_10361_irr": "Implementing Rules and Regulations RA 10361 Batas Kasambahay rest periods contract benefits",
    "ra_11861": (
        "Republic Act No. 11861 Section 7 amending Section 8 RA 8972 parental leave "
        "seven working days with pay at least six months service"
    ),
    "ra_8972": "Republic Act No. 8972 Solo Parents Welfare Act parental leave",
    "ra_11058": (
        "Republic Act No. 11058 employer duties safe and healthful workplace PPE safety information "
        "OSH program committee safety officer Section 6 right to refuse unsafe work imminent danger"
    ),
    "dole_do_198_18": (
        "DOLE Department Order No. 198-18 implementing rules RA 11058 duties of employers PPE "
        "OSH program safety and health committee safety officer"
    ),
    "occupational_safety_health_standards": (
        "Occupational Safety and Health Standards employer duties PPE safety officer committee program"
    ),
    "dole_do_252_25": "DOLE Department Order No. 252-25 current OSH IRR RA 11058 right to refuse unsafe work",
    "ra_7877": (
        "Republic Act No. 7877 Section 4 employer duties prevent deter rules complaint investigation committee "
        "sexual harassment workplace confidentiality retaliation"
    ),
    "ra_11313": (
        "Republic Act No. 11313 Safe Spaces Act gender-based sexual harassment workplace employer duties"
    ),
    "ra_11165": (
        "Republic Act No. 11165 Telecommuting Act Department Order 202-19 fair treatment minimum labor standards "
        "hours workload performance training equipment data protection"
    ),
    "ra_10396": "Republic Act No. 10396 Single Entry Approach SEnA request for assistance conciliation mediation",
    "dole_do_249_25": "DOLE Department Order No. 249-25 current SEnA guidelines request for assistance documents",
    "dole_arms_sena": "DOLE ARMS SEnA Request for Assistance official form documentary requirements",
    "supreme_court_decision": "Supreme Court decision case name G.R. No. decision date ruling doctrine",
}


CONCEPT_PATTERNS: Dict[str, Sequence[str]] = {
    "legal_basis": (r"\b(labor code|article|art\.|section|sec\.|republic act|ra\s*\d+|department order|wage order|rule)\b",),
    "coverage": (r"\b(coverage|covered|covers?|applies?|excluded|exempt|covered\s+employee|covered\s+worker)\b", r"\b(?:art\.?|article)\s*82\b"),
    "coverage_and_exclusions": (r"\b(coverage|covered|covers?|applies?|excluded|exempt|shall\s+not\s+apply|managerial|field\s+personnel|government\s+employees?|domestic\s+helpers?|personal\s+service)\b", r"\b(?:art\.?|article)\s*82\b"),
    "excluded_employee_categories": (r"\b(managerial|field\s+personnel|government|domestic|family\s+members?|personal\s+service|task\s+basis|piece\s+rate|non[-\s]?agricultural)\b", r"\b(?:art\.?|article)\s*82\b"),
    "hours_worked": (r"\b(hours?\s+worked|required|suffered|permitted|work\s+beyond|eight\s+hours|8\s+hours)\b", r"\b(?:art\.?|article)\s*84\b"),
    "employer_required_suffered_permitted": (r"\b(required|suffered|permitted|allowed|authorized|approval|employer)\b", r"\b(?:art\.?|article)\s*84\b"),
    "ordinary_overtime": (r"\b(overtime|beyond\s+eight|beyond\s+8|twenty[-\s]?five|25\s*%|125\s*%)\b", r"\b(?:art\.?|article)\s*87\b"),
    "ordinary_overtime_rule": (r"\b(overtime|beyond\s+eight|beyond\s+8|twenty[-\s]?five|25\s*%|125\s*%)\b", r"\b(?:art\.?|article)\s*87\b"),
    "undertime_offset_rule": (r"\b(undertime|offset|not\s+offset|compensate\s+by\s+overtime)\b", r"\b(?:art\.?|article)\s*88\b"),
    "emergency_overtime_rule": (r"\b(emergency|urgent|compulsory|required\s+overtime|force\s+overtime|require\s+overtime|article\s*89|art\.?\s*89)\b", r"\b(?:art\.?|article)\s*89\b"),
    "holiday_rest_day_overtime": (r"\b(rest\s+day|special\s+day|regular\s+holiday|holiday|thirty\s+percent|30\s*%|169\s*%|195\s*%|260\s*%|338\s*%)\b", r"\b(?:art\.?|article)\s*(93|94)\b"),
    "rest_day_special_day_rates": (r"\b(rest\s+day|special\s+day|special\s+non[-\s]?working|thirty\s+percent|30\s*%|130\s*%|150\s*%|169\s*%|195\s*%)\b", r"\b(?:art\.?|article)\s*93\b"),
    "rest_day_and_special_day_rates": (r"\b(rest\s+day|special\s+day|special\s+non[-\s]?working|thirty\s+percent|30\s*%|130\s*%|150\s*%|169\s*%|195\s*%)\b", r"\b(?:art\.?|article)\s*93\b"),
    "regular_holiday_rate": (r"\b(regular\s+holiday|holiday\s+pay|two\s+hundred|200\s*%|260\s*%|338\s*%)\b", r"\b(?:art\.?|article)\s*94\b"),
    "holiday_and_rest_day_overtime": (r"\b(rest\s+day|regular\s+holiday|holiday|special\s+day|30\s*%|169\s*%|195\s*%|260\s*%|338\s*%)\b", r"\b(?:art\.?|article)\s*(93|94)\b"),
    "omnibus_overtime_rules": (r"\bomnibus\b.*\b(book\s+iii|rule\s+i|section\s+(8|9|10))\b", r"\b(book\s+iii|rule\s+i)\b.*\bsection\s+(8|9|10)\b"),
    "employee_coverage": (r"\b(coverage|covered|covers?|employee|employer|rank[-\s]?and[-\s]?file|worker)\b",),
    "rank_and_file_coverage": (r"\b(rank[-\s]?and[-\s]?file|rank and file)\b",),
    "one_month_service": (r"\b(at\s+least\s+(?:one|1)(?:\s*\(\s*1\s*\))?\s+month|(?:one|1)(?:\s*\(\s*1\s*\))?\s+month|worked\s+for\s+at\s+least\s+(?:one|1)(?:\s*\(\s*1\s*\))?\s+month|not\s+less\s+than\s+(?:one|1)(?:\s*\(\s*1\s*\))?\s+month)\b",),
    "regardless_nature_employment": (r"\b(regardless\s+of\s+(?:their\s+)?(?:employment\s+)?status|regardless\s+of\s+the\s+nature\s+of\s+(?:their\s+)?employment|regular\s+or\s+probationary|probationary)\b",),
    "basic_salary_basis": (r"\b(basic\s+salary|basic\s+wage|salary\s+earned)\b",),
    "basic_salary_definition": (r"\b(basic\s+salary|basic\s+wage|salary\s+earned|remuneration)\b",),
    "formula": (r"\b(formula|computed|computation|one[-\s]?twelfth|1/12|divided\s+by\s+12)\b",),
    "calculation_formula": (r"\b(formula|computed|computation|one[-\s]?twelfth|1/12|divided\s+by\s+12)\b",),
    "payment_deadline": (r"\b(december\s+24|not\s+later\s+than|deadline|payment\s+date)\b",),
    "resigned_employee_entitlement": (r"\b(resigned|resignation|separated|proportionate|pro[-\s]?rated)\b",),
    "excluded_payments": (r"\b(excluded|not\s+included|allowances|bonus|profit[-\s]?sharing|cost[-\s]?of[-\s]?living|purely\s+commission|boundary|task\s+basis|piece[-\s]?rate|fixed\s+amount)\b",),
    "thirteenth_month_exclusions": (r"\b(purely\s+(?:on\s+)?commission|boundary|task\s+basis|fixed\s+amount|piece[-\s]?rate)\b",),
    "wage_order_or_region": (r"\b(wage\s+order|region|rtwpb|nwpc|regional\s+tripartite)\b",),
    "covered_workers": (r"\b(covered\s+workers?|covered\s+employees?|coverage|private\s+sector)\b",),
    "effective_date": (r"\b(effective\s+date|effectivity|takes\s+effect)\b",),
    "wage_payment_rule": (r"\b(payment\s+of\s+wages|paid\s+directly|payday|time\s+of\s+payment|place\s+of\s+payment|wage)\b",),
    "cola_definition": (r"\b(COLA|cost[-\s]+of[-\s]+living\s+allowance)\b",),
    "cola_entitlement": (
        r"\bpaid\s+(?:the\s+)?basic\s+pay\b.{0,140}\bentitled\s+to\s+(?:the\s+)?COLA\b",
        r"\bentitled\s+to\s+(?:the\s+)?COLA\b.{0,140}\b(?:worked\s+or\s+not|basic\s+pay)\b",
    ),
    "normal_daily_hours": (
        r"\bnormal\s+hours?\s+of\s+work\b.{0,100}\b(?:eight|8)\s*(?:\([^)]+\)\s*)?hours?\b.{0,40}\b(?:a|per)\s+day\b",
        r"\bshall\s+not\s+exceed\s+(?:eight|8)\s*(?:\([^)]+\)\s*)?hours?\s+(?:a|per)\s+day\b",
        r"\bnormal\s+hours?\s+of\s+work\s+in\s+a\s+day\b.{0,50}\b(?:is|are)\s+(?:eight|8)\b",
    ),
    "meal_period_duration": (
        r"\bmeal\s+periods?\b.{0,140}\b(?:sixty|60)\s*(?:\([^)]+\)\s*)?minutes?\b",
        r"\bnot\s+less\s+than\s+(?:one\s*\(1\)|one|1)\s+hour\b.{0,80}\b(?:meal|regular\s+meals?)\b",
    ),
    "meal_period_compensability": (
        r"\bmeal\s+periods?\b.{0,180}\bnon[-\s]?compensable\b",
        r"\bshorter\s+meal\s+period\b.{0,180}\bcompensable\s+hours?\s+worked\b",
        r"\b(?:5|five)\s+to\s+(?:20|twenty)\s+minutes?\b.{0,100}\bcompensable\b",
    ),
    "deduction_rule": (r"\b(deduction|deduct|withhold|deposit|cash\s+bond|authorization)\b",),
    "employee_responsibility_for_loss": (
        r"\b(employee|worker)\b.{0,160}\b(responsible|fault|negligence)\b.{0,160}\b(loss|damage|equipment|tools?)\b",
        r"\b(loss|damage|equipment|tools?)\b.{0,160}\b(employee|worker)\b.{0,160}\b(responsible|fault|negligence)\b",
    ),
    "loss_or_damage_proof": (r"\b(loss|damage)\b.{0,160}\b(proof|prove|actual|established|shown)\b",),
    "deduction_fairness_safeguards": (r"\b(deduction|withhold)\b.{0,220}\b(reasonable|fair|opportunity\s+to\s+explain|authorization|written|not\s+exceed|actual\s+loss)\b",),
    "holiday_type": (r"\b(regular\s+holiday|special\s+non[-\s]?working|special\s+day|holiday)\b",),
    "first_8_rate": (r"\b(100\s*%|130\s*%|150\s*%|200\s*%|260\s*%|first\s+eight|first\s+8|ordinary\s+rate)\b",),
    "regular_holiday_unworked_rule": (r"\b(regular\s+holiday).{0,120}\b(unworked|no\s+work|even\s+if\s+not\s+worked|regular\s+daily\s+wage|100\s*%)\b", r"\b(?:art\.?|article)\s*94\b"),
    "regular_holiday_worked_rate": (r"\b(regular\s+holiday).{0,120}\b(worked|work\s+on|200\s*%|double\s+pay)\b",),
    "special_non_working_unworked_rule": (r"\b(special\s+non[-\s]?working|special\s+day).{0,140}\b(no\s+work[, ]+no\s+pay|unworked|not\s+paid|unless\s+.*(?:policy|cba|practice))\b",),
    "special_non_working_worked_rate": (r"\b(special\s+non[-\s]?working|special\s+day).{0,120}\b(worked|work\s+on|130\s*%|additional\s+30\s*%)\b", r"\b(?:art\.?|article)\s*93\b"),
    "coverage_and_qualification": (r"\b(covered|qualified|coverage|employee|worker|condition|qualification|policy|cba)\b",),
    "coverage_and_exceptions": (r"\b(coverage|covered|except|exception|exempt|unless|provided\s+that|shall\s+not\s+apply|excluded)\b",),
    "holiday_rest_day_rule": (
        r"\b(regular\s+holiday|holiday)\b.{0,180}\b(rest\s+day|day\s+off)\b|\b(rest\s+day|day\s+off)\b.{0,180}\b(regular\s+holiday|holiday)\b",
        r"\b(?:art\.?|article)\s*93\b",
    ),
    "holiday_preceding_day_condition": (
        r"\b(?:present|worked|leave\s+of\s+absence\s+with\s+pay)\b.{0,180}\b(?:work\s*)?day\s+immediately\s+preceding\b.{0,100}\b(?:regular\s+)?holiday\b",
        r"\b(?:work\s*)?day\s+immediately\s+preceding\b.{0,180}\b(?:worked|leave\s+of\s+absence\s+with\s+pay|entitled\s+to\s+(?:the\s+)?holiday\s+pay)\b",
    ),
    "successive_holiday_condition": (
        r"\bsuccessive\s+regular\s+holidays?\b.{0,260}\bday\s+immediately\s+preceding\s+the\s+first\s+holiday\b",
        r"\btwo\s*\(?(?:2|two)?\)?\s+successive\s+regular\s+holidays?\b.{0,260}\b(?:works?\s+on\s+the\s+first\s+holiday|leave\s+of\s+absence\s+with\s+pay)\b",
    ),
    "day_type": (r"\b(ordinary\s+day|rest\s+day|special\s+day|regular\s+holiday|holiday)\b",),
    "premium_rate": (r"\b(130\s*%|150\s*%|200\s*%|premium|additional\s+compensation)\b",),
    "night_period": (r"\b(ten\s+p\.?m\.?|10\s*p\.?m\.?|six\s+a\.?m\.?|6\s*a\.?m\.?|night\s+shift)\b",),
    "rate": (r"\b(rate|percent|10\s*%|premium|differential)\b",),
    "one_year_service": (r"\b(one\s+year|at\s+least\s+one\s+year|12\s+months)\b",),
    "five_day_minimum": (r"\b(five\s+days?|5\s+days?|service\s+incentive\s+leave)\b",),
    "service_charge_coverage": (r"\b(service\s+charges?|hotels?|restaurants?|covered\s+establishments?|covered\s+employees?)\b",),
    "service_charge_distribution": (
        r"\bservice\s+charges?\b.{0,300}\b(?:distribut\w*|shar\w*|allocat\w*)\b",
        r"\b(?:distribut\w*|allocat\w*)\b.{0,200}\b(?:covered\s+)?employees?\b",
    ),
    "leave_duration": (r"\b(days?|leave\s+period|duration|105|120|7\s+days?|seven\s+days?)\b",),
    "qualifying_event": (r"\b(childbirth|delivery|miscarriage|emergency\s+termination|pregnancy|birth)\b",),
    "benefit_source": (r"\b(sss|philhealth|ecc|employees'? compensation|benefit)\b",),
    "covered_employee": (r"\b(married\s+male|employee|spouse|covered)\b",),
    "qualifying_delivery": (r"\b(delivery|childbirth|miscarriage|spouse)\b",),
    "solo_parent_qualification": (r"\b(solo\s+parent|single\s+parent|solo\s+parent\s+id|qualification)\b",),
    "leave_or_benefit": (r"\b(leave|benefit|discount|support)\b",),
    "solo_parent_leave_days": (
        r"\b(?:not\s+more\s+than\s+)?(?:seven\s*\(7\)|seven|7)\s+working\s+days?\s+with\s+pay\b",
    ),
    "solo_parent_service_requirement": (
        r"\b(?:rendered\s+)?service\b.{0,100}\bat\s+least\s+(?:six\s*\(6\)|six|6)\s+months?\b",
        r"\bat\s+least\s+(?:six\s*\(6\)|six|6)\s+months?\b.{0,100}\bservice\b",
    ),
    "gynecological_condition": (r"\b(gynecological|gynaecological|surgery|women)\b",),
    "probationary_period": (r"\b(probationary|six\s+months?|6\s+months?|period)\b",),
    "standards_at_engagement": (r"\b(standards?|made\s+known|reasonable\s+standards|engagement)\b",),
    "regularization": (r"\b(regular|regularized|regularization|continue\s+after)\b",),
    "regular_status_test": (r"\b(regular\s+employee|usual\s+business|necessary\s+or\s+desirable|one\s+year)\b",),
    "security_reserved_status_rule": (r"\b(reserved\s+status|work\s+pool)\b",),
    "six_month_security_status_rule": (r"\b(?:after\s+a\s+period\s+of\s+)?six\s*\(\s*6\s*\)\s*months?\b|\b(?:after\s+a\s+period\s+of\s+)?six\s+months?\b|\b6\s+months?\b",),
    "security_reserved_status_conditions": (r"\b(other\s+principals|constructive(?:ly)?\s+dismiss|retaliat|lack\s+of\s+service\s+assignment|no\s+work\s+assignment)\b",),
    "security_separation_entitlement": (r"\b(separation\s+pay|separated\s+from\s+service)\b",),
    "suspended_relationship_wage_rule": (r"\bpayment\s+of\s+wages\b[\s\S]{0,350}\bsuspended\s+employment\b[\s\S]{0,250}\bsubject\s+to\b",),
    # A general rule for business suspension does not establish wage entitlement
    # during a security guard's reserved status. Require a passage that expressly
    # connects reserved status/work pool with wages or compensation.
    "security_reserved_status_pay_rule": (
        r"\b(?:reserved\s+status|work\s+pool)\b[\s\S]{0,300}\b(?:wages?|salary|compensation)\b|"
        r"\b(?:wages?|salary|compensation)\b[\s\S]{0,300}\b(?:reserved\s+status|work\s+pool)\b",
    ),
    "project_duration_or_scope": (r"\b(project|scope|duration|undertaking)\b",),
    "completion_condition": (r"\b(completion|termination\s+of\s+project|coterminous|known\s+at\s+engagement)\b",),
    "fixed_term_condition": (r"\b(fixed\s+term|fixed\s+period|knowingly|voluntarily|contract\s+period)\b",),
    "termination_ground": (r"\b(just\s+cause|authorized\s+cause|serious\s+misconduct|willful\s+disobedience|gross\s+neglect|fraud|redundancy|retrenchment|closure|disease)\b",),
    "just_cause": (r"\b(just\s+causes?|article\s*297|art\.?\s*297|serious\s+misconduct|willful\s+disobedience|gross\s+neglect|fraud|breach\s+of\s+trust|commission\s+of\s+a\s+crime|analogous\s+causes?)\b",),
    "notice": (r"\b(notice|written\s+notice|notice\s+to\s+explain|twin\s+notice)\b",),
    "opportunity_to_be_heard": (
        r"\b(?:opportunity|chance)\s+(?:to|of)\s+(?:be\s+heard|explain|defend|respond)\b",
        r"\b(?:afford\w*|giv\w*|provid\w*|allow\w*|entitled|required?|shall|must|due\s+process)\b.{0,140}\b(?:hearing|conference|explain|defend)\b",
        r"\b(?:hearing|conference)\b.{0,120}\b(?:shall|must|due\s+process|not\s+mandatory)\b",
    ),
    "decision_notice": (r"\b(decision|notice\s+of\s+termination|final\s+notice)\b",),
    "authorized_cause": (r"\b(authorized\s+causes?|article\s*298|art\.?\s*298|redundancy|retrenchment|closure|labor[-\s]?saving\s+devices?|business\s+or\s+economic\s+reasons?)\b",),
    "rate_or_formula": (r"\b(rate|formula|one\s+month|one[-\s]?half|half[-\s]?month|computed|pay)\b",),
    "authorized_cause_notice": (r"\b(written\s+notice|notice)\b.{0,180}\b(employee|worker|dole|department\s+of\s+labor|termination|retrenchment)\b",),
    "separation_pay_rounding": (r"\b(fraction|at\s+least\s+six\s+months|six\s+months|one\s+whole\s+year|round)\b",),
    "retirement_age": (r"\b(retirement\s+age|sixty|60|65|compulsory|optional)\b",),
    "retirement_service_requirement": (
        r"\b(served|service)\s+(?:for\s+)?at\s+least\s+(?:five\s*\(5\)|five|5)\s+years?\b",
    ),
    "retirement_pay_formula": (
        r"\b(?:one[-\s]?half|1\s*/\s*2)\s+month\s+salary\b",
        r"\bfifteen\s*\(?(?:15)?\)?\s+days?.*one[-\s]?twelfth|\b1\s*/\s*12\b.*13th\s+month",
    ),
    "retirement_exemption": (
        r"\b(retail|service|agricultural)\s+establishments?.*not\s+more\s+than\s+(?:ten|10)\b",
        r"\bexempted?\s+from\s+(?:the\s+)?coverage\b",
    ),
    "domestic_worker_definition": (r"\b(domestic\s+workers?|kasambahay|kasambhay|house\s*helpers?|househelpers?|household\s+(?:workers?|helpers?|service)|yaya|maid|housemaid)\b",),
    "employment_contract": (r"\b(employment\s+contract|written\s+contract|kasulatan|agreement)\b",),
    "minimum_benefits": (r"\b(minimum\s+wage|sss|philhealth|pag[-\s]?ibig|rest\s+period|leave|13th\s+month)\b",),
    "humane_treatment": (r"\b(humane\s+treatment|respect|dignity|physical\s+violence|abuse)\b",),
    "board_lodging_medical": (r"\b(board|lodging|food|medical\s+attendance|medical\s+assistance)\b",),
    "privacy": (r"\b(privacy|private\s+communication)\b",),
    "outside_communication": (r"\b(outside\s+communication|communicate\s+outside|access\s+to\s+communication)\b",),
    "education_training": (r"\b(education|training|alternative\s+learning|skills\s+training)\b",),
    "kasambahay_wages": (r"\b(wages?|minimum\s+wage|monthly\s+wage|salary)\b",),
    "kasambahay_13th_month": (r"\b(13th[-\s]*month|thirteenth[-\s]*month)\b",),
    "kasambahay_social_benefits": (r"\b(sss|philhealth|pag[-\s]?ibig|social\s+security)\b",),
    "personal_documents": (r"\b(personal\s+documents?|passport|birth\s+certificate|retain\s+documents?)\b",),
    "termination_protections": (r"\b(termination|terminate|pre[-\s]?termination|unjust|notice)\b",),
    "kasambahay_rest_period": (r"\b(rest\s+period|daily\s+rest|weekly\s+rest|24\s+consecutive|twenty[-\s]?four\s+consecutive|eight\s+hours|8\s+hours)\b",),
    "ra10361_contract_additional_payment": (
        r"\b(section|sec\.?)\s*11\b",
        r"\b(employment\s+contract|written\s+contract|hours?\s+of\s+work|rest\s+days?|allowable\s+leaves?|proportionate\s+additional\s+payment|additional\s+payment|agreement)\b",
    ),
    "kasambahay_daily_rest_period": (
        r"\b(section|sec\.?)\s*20\b",
        r"\b(aggregate\s+daily\s+rest|daily\s+rest\s+period|eight\s+hours|8\s+hours)\b",
    ),
    "kasambahay_weekly_rest_period": (
        r"\b(section|sec\.?)\s*21\b",
        r"\b(weekly\s+rest\s+period|24\s+consecutive|twenty[-\s]?four\s+consecutive)\b",
    ),
    "labor_code_domestic_exclusion": (r"\b(?:art\.?|article)\s*82\b", r"\b(domestic\s+helpers?|persons?\s+in\s+personal\s+service|household\s+service)\b"),
    "employer_duties": (
        r"\b(dut(?:y|ies)\s+of\s+(?:the\s+)?employer|employer(?:s|'s)?\s+(?:shall|must|dut(?:y|ies)|obligations?|responsibilit(?:y|ies)))\b",
        r"\bevery\s+employer\b.{0,120}\b(shall|furnish|give|inform|provide|comply|maintain)\b",
    ),
    "safe_and_healthful_workplace": (r"\b(safe\s+and\s+healthful\s+workplace|safe\s+working\s+environment|safe\s+workplace|place\s+of\s+employment\s+free\s+from\s+hazardous\s+conditions)\b",),
    "osh_information_training": (r"\b(job\s+safety\s+instructions?|safety\s+orientation|inform\s+(?:the\s+)?workers?.{0,80}hazards?|safety\s+and\s+health\s+information|osh\s+training)\b",),
    "ppe_provision": (
        r"\b(personal\s+protective\s+equipment|ppe|protective\s+gear|safety\s+gear)\b.{0,160}\b(provide|furnish|free\s+of\s+charge|cost|expense|pay|employer)\b|\b(provide|furnish|free\s+of\s+charge)\b.{0,160}\b(personal\s+protective\s+equipment|ppe|protective\s+gear|safety\s+gear)\b",
        r"\b(?:employer|boss)\b.{0,80}\b(?:provide|give|supply)\w*\b.{0,80}\b(?:protective\s+gear|safety\s+gear|gloves|masks|respirators)\b",
    ),
    "osh_establishment_coverage": (
        r"\b(?:all|every)\s+(?:establishments?|workplaces?|projects?|sites?)\b.{0,220}\b(?:apply|covered|coverage|comply|safety|health|regardless)\b",
        r"\b(?:apply|covered|coverage|comply)\b.{0,180}\b(?:all\s+(?:establishments|workplaces|projects|sites)|regardless\s+of\s+(?:size|number))\b",
        r"\b(?:micro|small)\s+(?:enterprises?|establishments?)\b.{0,180}\b(?:shall|must|required|comply)\b",
    ),
    "ppe_cost": (
        r"\b(?:personal\s+protective\s+equipment|ppe|protective\s+equipment|protective\s+gear|safety\s+gear|gloves|masks|respirators)\b.{0,180}\b(?:free\s+of\s+charge|for\s+free|without\s+cost|at\s+no\s+cost|no\s+cost\s+to|employer.s\s+expense)\b",
        r"\b(?:free\s+of\s+charge|for\s+free|without\s+cost|at\s+no\s+cost|no\s+cost\s+to|employer.s\s+expense)\b.{0,180}\b(?:personal\s+protective\s+equipment|ppe|protective\s+equipment|protective\s+gear|safety\s+gear)\b",
    ),
    "emergency_preparedness": (
        r"\b(?:emergency\s+(?:preparedness|response|procedures?|plans?)|evacuation\s+(?:procedures?|plans?))\b.{0,180}\b(?:shall|must|provide|include|develop|implement|communicat|required|program)\w*\b",
        r"\b(?:shall|must|provide|include|develop|implement|communicat|required|program)\w*\b.{0,180}\b(?:emergency\s+(?:preparedness|response|procedures?|plans?)|evacuation\s+(?:procedures?|plans?))\b",
        r"\bsteps\s+to\s+be\s+taken\s+in\s+case\s+of\s+(?:an?\s+)?emergency\b",
    ),
    "fire_drill_requirement": (
        r"\b(?:fire(?:\s+and\s+earthquake)?|evacuation|emergency)\s+drills?\b.{0,220}\b(?:shall|must|required|conduct|held|periodic|annual|year|month|frequency)\w*\b",
        r"\b(?:shall|must|required|conduct|held|institute)\w*\b.{0,180}\b(?:fire(?:\s+and\s+earthquake)?|evacuation|emergency)\s+drills?\b",
    ),
    "safety_notice_requirement": (
        r"\b(?:safety\s+(?:signage|signs|notices)|warning\s+(?:signage|signs)|emergency\s+procedures|evacuation\s+plans)\b.{0,180}\b(?:post|display|conspicuous|prominent|strategic|required|shall)\w*\b",
        r"\b(?:post|display|conspicuous|prominent|strategic)\w*\b.{0,180}\b(?:safety\s+(?:signage|signs|notices)|warning\s+(?:signage|signs)|emergency\s+procedures|evacuation\s+plans)\b",
    ),
    "hazard_assessment": (
        r"\b(?:hazard\s+(?:identification|evaluation|assessment)|risk\s+assessment|evaluate\s+the\s+hazards?|assessment\s+of\s+(?:the\s+)?hazards?)\b",
        r"\b(?:identify|assess|evaluat)\w*\b.{0,100}\b(?:workplace\s+)?(?:hazards?|risks?)\b",
    ),
    "chemical_safety_information": (
        r"\b(?:safety\s+data\s+sheets?|chemical\s+safety|chemical\s+hazards?)\b.{0,180}\b(?:provide|available|accessible|inform|train|shall|must|right)\w*\b",
        r"\b(?:provide|inform|train|right|shall|must)\w*\b.{0,180}\b(?:safety\s+data\s+sheets?|chemical\s+safety|chemical\s+hazards?|hazardous\s+chemicals?)\b",
    ),
    "occupational_health_protection": (
        r"\b(?:health\s+surveillance|medical\s+(?:examinations?|monitoring)|occupational\s+health\s+(?:services|program))\b.{0,180}\b(?:shall|must|required|provide|expos|hazard|worker)\w*\b",
        r"\b(?:shall|must|required|provide|undergo|worker)\w*\b.{0,180}\b(?:health\s+surveillance|medical\s+(?:examinations?|monitoring)|occupational\s+health\s+(?:services|program))\b",
    ),
    "workplace_lighting": (
        r"\b(?:lighting|illumination)\b.{0,180}\b(?:adequate|sufficient|suitable|minimum|required|shall|must|lux)\b",
        r"\b(?:adequate|sufficient|suitable|minimum|required|shall|must)\b.{0,180}\b(?:lighting|illumination)\b",
    ),
    "osh_compliance_liability": (
        r"\b(?:joint|solidar)\w*\b.{0,200}\b(?:compliance|safety|health|this\s+act)\b",
        r"\b(?:compliance|safety|health)\b.{0,200}\b(?:joint|solidar)\w*\b",
    ),
    "ec_eligibility_conditions": (
        r"\b(?:injur|accident|disease)\w*\b.{0,220}\b(?:arising\s+out\s+of|course\s+of\s+employment|increased\s+(?:the\s+)?risk|work[-\s]connection|conditions\s+(?:are|must))\b",
    ),
    "osh_program": (r"\b(occupational\s+safety\s+and\s+health|osh)\s+program\b|\bsafety\s+and\s+health\s+program\b",),
    "osh_committee": (r"\b(occupational\s+safety\s+and\s+health|osh|safety\s+and\s+health)\s+committee\b",),
    "safety_officer_requirement": (r"\bsafety\s+officer\b.{0,180}\b(require|required|shall|number|class|type|ratio|worker|risk|establishment)\b|\b(require|required|shall|number)\b.{0,100}\bsafety\s+officer\b",),
    "worker_rights": (r"\b(worker\s+rights|right\s+to\s+refuse|unsafe|report|protection)\b",),
    "right_to_refuse_unsafe_work": (r"\bright\s+to\s+refuse\b.{0,120}\b(unsafe|work|danger)\b",),
    "imminent_danger": (r"\bimminent\s+danger\b",),
    "dole_determination": (r"\b(dole|department\s+of\s+labor)\b.{0,140}\b(determine|determination|finds?|found)\b",),
    "corrective_action_not_taken": (r"\b(corrective\s+actions?|abatement)\b.{0,140}\b(not\s+(?:been\s+)?(?:taken|undertaken)|failed|failure)\b",),
    "separation_pay_rule": (
        r"\bseparation\s+pay\b.{0,200}\b(?:equivalent|month|year\s+of\s+service|not\s+entitled|no\s+right)\b",
    ),
    "reprisal_protection": (r"\b(reprisal|retaliation|discrimination|dismiss|terminate|protection)\b",),
    "work_related_injury": (r"\b(work[-\s]?related|injury|accident|occupational\s+disease)\b",),
    "reporting_or_claim": (r"\b(report|claim|file|notice|ecc|sss|gsis)\b",),
    "prohibited_act": (r"\b(prohibited|harassment|demand|request|sexual\s+favor|unwelcome)\b",),
    "covered_relationship": (r"\b(employer|employee|supervisor|teacher|trainer|relationship)\b",),
    "remedies_or_duties": (r"\b(remedy|complaint|committee|investigate|duty|penalty)\b",),
    "harassment_prevention_policy": (r"\b(prevent|deter|policy|rules|regulations|code\s+of\s+conduct)\b.{0,180}\b(sexual\s+harassment|harassment)\b",),
    "complaint_investigation_body": (r"\b(committee\s+on\s+decorum\s+and\s+investigation|codi|committee|complaint|investigat)\b",),
    "confidentiality": (r"\b(confidential|confidentiality|privacy)\b",),
    "retaliation_protection": (r"\b(retaliation|retaliate|reprisal|adverse\s+action)\b",),
    "enforcement_responsibility": (r"\b(enforce|implementation|employer\s+responsib|duty\s+of\s+employer|penalty|liability)\b",),
    "fair_treatment": (r"\b(fair\s+treatment|comparable|same\s+treatment|telecommuting|remote\s+worker)\b",),
    "minimum_labor_standards": (r"\b(minimum\s+labor\s+standards|hours\s+of\s+work|overtime|rest\s+day|leave\s+benefits?|social\s+welfare)\b",),
    "complaint_channel": (r"\b(dole|regional\s+office|field\s+office|provincial\s+office|request\s+for\s+assistance|sena)\b",),
    "conciliation_process": (r"\b(conciliation|mediation|single[-\s]+entry\s+approach|sena|grievance\s+machinery)\b",),
    "sena_request_for_assistance": (r"\b(sena|single[-\s]+entry\s+approach|request\s+for\s+assistance|rfa)\b",),
    "sena_document_requirements": (r"\b(document|requirements?|form|request\s+for\s+assistance|proof|attachment|identification)\b",),
    "gender_based_harassment": (r"\b(gender[-\s]?based|sexual\s+harassment|bastos|misogynistic|sexist)\b",),
    "covered_place": (r"\b(workplace|public\s+space|online|street|school|training)\b",),
    "protected_ground": (r"\b(sex|gender|age|disability|religion|union|pregnancy|marital\s+status)\b",),
    "self_organization": (r"\b(self[-\s]?organization|organize|union|labor\s+organization)\b",),
    "bargaining_unit_or_cba": (r"\b(bargaining\s+unit|cba|collective\s+bargaining\s+agreement)\b",),
    "duty_to_bargain": (r"\b(duty\s+to\s+bargain|collective\s+bargaining|negotiate)\b",),
    "strike_or_lockout_rule": (r"\b(strike|lockout|work\s+stoppage|picket)\b",),
    "filing_or_notice_requirement": (r"\b(notice|file|filing|cooling[-\s]?off|strike\s+vote|report)\b",),
    "contracting_rule": (r"\b(contracting|subcontracting|contractor|subcontractor|trilateral)\b",),
    "prohibited_labor_only_contracting": (r"\b(labor[-\s]?only\s+contracting|prohibited|substantial\s+capital|control)\b",),
    "current_controlling_issuance": (r"\b(current|effective|applicable|department\s+order|supersed|repeal|amend)\b",),
    "overseas_worker_scope": (r"\b(ofw|overseas\s+filipino|migrant\s+worker|overseas\s+employment)\b",),
    "applicable_protection": (r"\b(protection|rights?|contract|recruitment|claim|assistance)\b",),
    "case_holding": (r"\b(held|ruled|decision|ruling|wherefore|doctrine)\b",),
    "case_citation": (r"\b(g\.??r\.??\s*no\.?|supreme\s+court|case)\b",),
    "case_name": (r"\b[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)*\s+v(?:s\.?|\.)\s+[A-Z][A-Za-z.'-]+", r"\bcase\s+name\b"),
    "decision_date": (r"\b(decided|decision\s+date|promulgated)\b.{0,80}\b(?:19|20)\d{2}\b", r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+(?:19|20)\d{2}\b"),
    "civil_service_scope": (r"\b(civil\s+service|government\s+employee|public\s+sector|government\s+agency)\b",),
    "covered_member": (r"\b(member|coverage|covered|employee|employer|contribution)\b",),
    "contribution_or_benefit": (r"\b(contribution|premium|benefit|claim|loan|insurance)\b",),
    "current_contribution_rate_schedule": (
        r"\b(?:contribution|premium)\s+(?:rate\s+)?(?:table|schedule|matrix)\b|"
        r"\b(?:schedule\s+of\s+sss\s+contributions|sss\s+contribution\s+schedule)\b|"
        r"\bsss_contribution_schedule_2025\b|"
        r"\bmonthly\s+salary\s+credit\b[\s\S]{0,250}\b(?:contribution|rate|employee\s+share)\b",
    ),
    "sss_member_verification_process": (
        r"\b(?:my\.?sss|sss\s+(?:online\s+)?account|member\s+contribution\s+record)\b[\s\S]{0,180}\b(?:view|check|verify|history|posted|contribution)\b|"
        r"\b(?:view|check|verify|history|posted|contribution)\b[\s\S]{0,180}\b(?:my\.?sss|sss\s+(?:online\s+)?account|member\s+contribution\s+record)\b",
    ),
    "philhealth_member_verification_process": (
        r"\b(?:philhealth|member\s+portal|member\s+account)\b[\s\S]{0,180}\b(?:view|check|verify|history|posted|contribution)\b|"
        r"\b(?:view|check|verify|history|posted|contribution)\b[\s\S]{0,180}\b(?:philhealth|member\s+portal|member\s+account)\b",
    ),
    "pagibig_member_verification_process": (
        r"\b(?:virtual\s+pag[-\s]?ibig|pag[-\s]?ibig\s+(?:online\s+)?account|member\s+contribution\s+record)\b[\s\S]{0,180}\b(?:view|check|verify|history|posted|contribution)\b|"
        r"\b(?:view|check|verify|history|posted|contribution)\b[\s\S]{0,180}\b(?:virtual\s+pag[-\s]?ibig|pag[-\s]?ibig\s+(?:online\s+)?account|member\s+contribution\s+record)\b",
    ),
    "covered_contingency": (r"\b(sickness|injury|disability|death|contingency|work[-\s]?connected)\b",),
    "claim_or_benefit": (r"\b(claim|benefit|compensation|file|report)\b",),
}


def classify_topic(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    existing = canonical_topic((metadata or {}).get("topic"))
    if existing in SUPPORTED_TOPICS and existing not in {GENERAL_LABOR_TOPIC, NON_LEGAL_TOPIC}:
        return existing

    from routers.services.scenario_retrieval import everyday_topic
    specific = everyday_topic(text)
    if specific:
        return specific

    combined = f"{text or ''} {_metadata_text(metadata)}"
    if any(
        re.search(pattern, combined, re.IGNORECASE)
        for topic, patterns in TOPIC_PATTERNS
        if topic == "jurisprudence_request"
        for pattern in patterns
    ):
        return "jurisprudence_request"
    for topic, patterns in TOPIC_PATTERNS:
        if any(re.search(pattern, combined, re.IGNORECASE) for pattern in patterns):
            return topic
    return GENERAL_LABOR_TOPIC


def _task_config_for_topic(topic: Any, task: str = "") -> Dict[str, Any]:
    canonical = canonical_topic(topic)
    task_key = _clean_key(task)
    if task_key:
        task_config = (TASK_SOURCE_REQUIREMENTS.get(canonical) or {}).get(task_key)
        if task_config:
            return task_config
    return TOPIC_SOURCE_REQUIREMENTS.get(canonical) or {}


def essential_concepts_for_topic(topic: Any, query_text: str = "", task: str = "") -> List[str]:
    canonical = canonical_topic(topic)
    config = _task_config_for_topic(canonical, task)
    essential = list(config.get("essential") or [])
    if not essential:
        legacy_required = list(config.get("required") or [])
        essential = legacy_required[:1]
    text = query_text or ""
    if canonical == "payment_of_wages" and re.search(r"\b(minimum\s+wage|wage\s+order)\b", text, re.IGNORECASE):
        return essential_concepts_for_topic("minimum_wage", query_text)
    return essential


def supporting_concepts_for_topic(topic: Any, query_text: str = "", task: str = "") -> List[str]:
    canonical = canonical_topic(topic)
    config = _task_config_for_topic(canonical, task)
    supporting = list(config.get("supporting") or [])
    if not supporting:
        legacy_required = list(config.get("required") or [])
        supporting = legacy_required[1:]
    text = query_text or ""

    if canonical == "termination" and re.search(r"\b(due\s+process|notice|hearing|procedure)\b", text, re.IGNORECASE):
        return ["notice", "opportunity_to_be_heard", "decision_notice"]
    if canonical in {"holiday_pay", "premium_pay"} and re.search(r"\b(overtime|beyond\s+eight|beyond\s+8)\b", text, re.IGNORECASE):
        supporting = list(dict.fromkeys(supporting + ["ordinary_overtime", "holiday_rest_day_overtime"]))
    if canonical == "payment_of_wages" and re.search(r"\b(minimum\s+wage|wage\s+order)\b", text, re.IGNORECASE):
        return supporting_concepts_for_topic("minimum_wage", query_text)
    return supporting


def required_concepts_for_topic(topic: Any, query_text: str = "", task: str = "") -> List[str]:
    required = essential_concepts_for_topic(topic, query_text, task) + supporting_concepts_for_topic(topic, query_text, task)
    return list(dict.fromkeys(required))


def preferred_sources_for_topic(topic: Any, task: str = "") -> List[str]:
    canonical = canonical_topic(topic)
    config = _task_config_for_topic(canonical, task)
    return list(config.get("preferred") or [])


def required_authorities_for_topic(topic: Any, task: str = "") -> Dict[str, List[str]]:
    canonical = canonical_topic(topic)
    task_key = _clean_key(task)
    topic_config = REQUIRED_AUTHORITIES.get(canonical) or {}
    config = topic_config.get(task_key) or topic_config.get("default") or {}
    essential = list(config.get("essential") or [])
    supporting = list(config.get("supporting") or [])
    context = list(config.get("context") or [])
    return {
        "essential": list(dict.fromkeys(essential)),
        "supporting": list(dict.fromkeys(supporting)),
        "context": list(dict.fromkeys(context)),
    }


def authority_search_terms(authority_id: str) -> str:
    return AUTHORITY_SEARCH_TERMS.get(_clean_key(authority_id), authority_id)


def authority_search_terms_for_topic(topic: Any, task: str = "") -> List[str]:
    requirements = required_authorities_for_topic(topic, task)
    authority_ids = requirements["essential"] + requirements["supporting"]
    return [authority_search_terms(authority_id) for authority_id in authority_ids]


_INDEXED_CHUNK_METADATA_LINE_RE = re.compile(
    r"(?im)^\s*(?:Document|Canonical Document|Authority ID|Concepts|Scope|Rule Type|Source URL):"
)
_INDEXED_CHUNK_TEXT_MARKER_RE = re.compile(r"(?im)^\s*Text:\s*\r?\n")


def source_passage_text(value: Any) -> str:
    """Remove the ingestion header from a stored chunk, if present.

    Ingestion prepends document-level tags such as ``Concepts`` and ``Scope``
    before ``Text:``. Those tags describe a whole source and must not be read
    as proof that this individual passage contains every tagged rule.
    """

    text = str(value or "")
    marker = _INDEXED_CHUNK_TEXT_MARKER_RE.search(text)
    if marker and _INDEXED_CHUNK_METADATA_LINE_RE.search(text[:marker.start()]):
        return text[marker.end():]
    return text


MAX_EVIDENCE_PASSAGE_CHARS = 6000


def bounded_evidence_passage(value: Any, concepts: Iterable[str] = (), limit: int = MAX_EVIDENCE_PASSAGE_CHARS) -> str:
    """Choose one contiguous source window, not a fabricated rule summary.

    Large case chunks can bury the operative paragraph after the prompt's
    first-page cutoff. Prefer a window containing requested clauses, with
    surrounding context and sentence boundaries. Never stitch distant rules
    together or credit clauses outside the resulting excerpt.
    """
    body = re.sub(r"\s+", " ", source_passage_text(value)).strip()
    if len(body) <= limit:
        return body
    budget = max(1, limit - 4)  # Reserve space for both omission markers.
    required = tuple(dict.fromkeys(concepts))
    boundaries = [0, *(m.end() for m in re.finditer(r'(?<=[.!?;])\s+(?=[A-Z(“\"])', body))]
    starts = {0}
    from bisect import bisect_right
    for concept in required:
        if concept == "legal_basis":
            continue
        for pattern in CONCEPT_PATTERNS.get(concept, ()):
            for count, match in enumerate(re.finditer(pattern, body, re.I)):
                # Include preceding qualifications, not only the matched word.
                target = max(0, match.start() - 600)
                starts.add(boundaries[max(0, bisect_right(boundaries, target) - 1)])
                if count >= 2:
                    break
    def window(start):
        end = min(len(body), start + budget)
        if end < len(body):
            boundary = boundaries[max(0, bisect_right(boundaries, end) - 1)]
            if boundary > start + budget // 2:
                end = boundary
            else:
                # One oversized sentence: mark the cutoff, don't complete it.
                end = body.rfind(" ", start, end)
        return body[start:end].strip()
    candidates = [(start, window(start)) for start in sorted(starts)]
    start, passage = max(candidates, key=lambda pair: (
        len(supported_passage_concepts({"text": pair[1]}, required)), -pair[0],
    ))
    # Omission markers are visible to the generator and source reader.
    prefix = "… " if start else ""
    suffix = " …" if start + len(passage) < len(body) else ""
    return prefix + passage + suffix


def source_has_indexed_metadata_header(source: Dict[str, Any]) -> bool:
    meta = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    for key in ("text", "excerpt"):
        value = source.get(key) or meta.get(key) or ""
        if source_passage_text(value) != str(value or ""):
            return True
    return False

def _source_blob(source: Dict[str, Any]) -> str:
    meta = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    keys = (
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
        "excerpt",
        "text",
        "source_url",
        "topic",
        "category",
    )
    parts = []
    for key in keys:
        value = source.get(key) or meta.get(key) or ""
        if key in {"excerpt", "text"}:
            value = source_passage_text(value)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def supported_passage_concepts(source: Dict[str, Any], concepts: Iterable[str]) -> Set[str]:
    """Match operative evidence, never document-level topic/concept tags.

    Collapse PDF line breaks, but do not repair garbled columns by guessing
    missing words. Structural identity may establish legal_basis only when
    an actual passage accompanies it.
    """
    meta = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    body = source.get("text") or source.get("excerpt") or meta.get("text") or meta.get("excerpt") or ""
    blob = re.sub(r"\s+", " ", source_passage_text(body)).strip()
    supported: Set[str] = set()
    if not blob:
        return supported
    for concept in concepts:
        patterns = CONCEPT_PATTERNS.get(concept, ())
        evidence = _source_blob(source) if concept == "legal_basis" else blob
        if any(re.search(pattern, evidence, re.IGNORECASE) for pattern in patterns):
            supported.add(concept)
    return supported


def detect_supported_concepts(topic: Any, source: Dict[str, Any], query_text: str = "", task: str = "") -> Set[str]:
    required = required_concepts_for_topic(canonical_topic(topic), query_text, task)
    return supported_passage_concepts(source, [*required, "legal_basis"] if required else [])


def collect_supported_concepts(topic: Any, sources: Iterable[Dict[str, Any]], query_text: str = "", task: str = "") -> Set[str]:
    supported: Set[str] = set()
    for source in sources or []:
        supported.update(detect_supported_concepts(topic, source, query_text, task))
    return supported


def missing_required_concepts(topic: Any, sources: Iterable[Dict[str, Any]], query_text: str = "", task: str = "") -> List[str]:
    required = required_concepts_for_topic(topic, query_text, task)
    if not required:
        return []
    supported = collect_supported_concepts(topic, sources, query_text, task)
    return [concept for concept in required if concept not in supported]


def topic_clarification_questions(topic: Any, known_state: Optional[Dict[str, Any]] = None) -> List[str]:
    state = known_state or {}
    canonical = canonical_topic(topic)
    if canonical in {"minimum_wage", "payment_of_wages"}:
        questions = [
            "What city or region does the worker work in?",
            "Is the concern minimum wage, unpaid salary, deduction, or delayed payment?",
        ]
    elif canonical in {"overtime_pay", "holiday_pay", "premium_pay", "night_shift_differential"}:
        questions = [
            "What day type is involved: ordinary workday, rest day, special day, or regular holiday?",
            "How many hours were worked beyond the regular schedule?",
        ]
    elif canonical in {"termination", "procedural_due_process", "separation_pay"}:
        questions = [
            "Was the worker terminated, suspended, forced to resign, or retrenched?",
            "What reason did the employer give, and was written notice given?",
        ]
    elif canonical in {"maternity_leave", "paternity_leave", "solo_parent_leave", "service_incentive_leave", "special_leave_for_women"}:
        questions = [
            "Which leave benefit are you asking about?",
            "How long has the employee worked and what is the employment status?",
        ]
    else:
        questions = [
            "What specific workplace issue do you need help with?",
            "Are you asking as an employee, employer, applicant, or resigned worker?",
        ]

    if state.get("region") or state.get("location"):
        questions = [q for q in questions if "city or region" not in q.lower()]
    if state.get("employment_status"):
        questions = [q for q in questions if "employment status" not in q.lower()]
    return questions[:3]
