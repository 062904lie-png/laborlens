"""Reusable pending-calculation state helpers for LaborLens."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from routers.services.calculators import CALCULATION_SPECS


PENDING_CALCULATION_KEY = "pending_calculation"
CALCULATION_TASKS = {"calculation_request", "personalized_computation"}
CALCULATION_STATE_FIELDS = {
    "calculation_topic",
    "hourly_rate",
    "daily_wage",
    "daily_rate",
    "normal_work_hours",
    "overtime_hours",
    "hours_worked",
    "night_shift_hours",
    "monthly_salary",
    "total_basic_salary_earned",
    "months_worked",
    "unused_sil_days",
    "years_of_service",
    "service_duration",
    "separation_reason",
    "day_type",
}
CALCULATION_NUMERIC_FIELDS = {
    "hourly_rate",
    "daily_wage",
    "daily_rate",
    "normal_work_hours",
    "overtime_hours",
    "hours_worked",
    "night_shift_hours",
    "monthly_salary",
    "total_basic_salary_earned",
    "months_worked",
    "unused_sil_days",
    "years_of_service",
}
SHORT_VALUE_REPLY_RE = re.compile(
    r"^\s*(?:php|p|\u20b1)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:php|pesos?)?\s*$",
    re.IGNORECASE,
)


def active_pending_calculation(state: Dict[str, Any]) -> Dict[str, Any]:
    pending = (state or {}).get(PENDING_CALCULATION_KEY)
    if isinstance(pending, str):
        try:
            pending = json.loads(pending)
        except json.JSONDecodeError:
            pending = {}
    if isinstance(pending, dict) and pending.get("status") == "awaiting_input":
        return pending
    return {}


def stored_calculation_state(state: Dict[str, Any]) -> Dict[str, Any]:
    pending = (state or {}).get(PENDING_CALCULATION_KEY)
    if isinstance(pending, str):
        try:
            pending = json.loads(pending)
        except json.JSONDecodeError:
            pending = {}
    return pending if isinstance(pending, dict) else {}


def is_completed_calculation_state(state: Dict[str, Any]) -> bool:
    return stored_calculation_state(state).get("status") == "completed"


def supersede_calculation_state(state: Dict[str, Any], reason: str = "new_task") -> Dict[str, Any]:
    updated = dict(state or {})
    previous = stored_calculation_state(updated)
    if previous or any(updated.get(field) not in ("", None) for field in CALCULATION_STATE_FIELDS):
        updated[PENDING_CALCULATION_KEY] = {
            **previous,
            "status": "superseded",
            "pending_field": None,
            "last_question_asked": None,
            "missing_fields": [],
            "superseded_reason": reason,
        }
    for field in CALCULATION_STATE_FIELDS:
        updated.pop(field, None)
    return updated


def parse_numeric_reply(message: str) -> Optional[str]:
    match = SHORT_VALUE_REPLY_RE.search(message or "")
    if not match:
        from routers.services.numeric_quantities import parse_number
        value = parse_number(message)
        return format(value.normalize(), "f") if value is not None else None
    try:
        return str(Decimal(match.group(1).replace(",", "")))
    except InvalidOperation:
        return None


def parse_day_type_reply(message: str) -> Optional[str]:
    text = (message or "").strip().lower()
    if not text:
        return None
    if re.search(r"\bnot\s+sure|di\s+sure|indi\s+sure|hindi\s+sure\b", text, re.IGNORECASE):
        return None
    day_aliases = (
        ("regular_holiday_rest_day", r"\b(regular holiday|legal holiday|holiday)\b.{0,40}\b(rest day|day off)\b|\b(rest day|day off)\b.{0,40}\b(regular holiday|legal holiday|holiday)\b"),
        ("special_day_rest_day", r"\b(special non[-\s]?working day|special day|special holiday)\b.{0,40}\b(rest day|day off)\b|\b(rest day|day off)\b.{0,40}\b(special non[-\s]?working day|special day|special holiday)\b"),
        ("ordinary_day", r"\b(ordinary (?:working )?day|regular (?:working )?day|regular workday|normal day|normal workday)\b"),
        ("rest_day", r"\b(rest day|day off)\b"),
        ("special_non_working_day", r"\b(special non[-\s]?working day|special day|special holiday)\b"),
        ("regular_holiday", r"\b(regular holiday|legal holiday|holiday)\b"),
    )
    for value, pattern in day_aliases:
        if re.search(pattern, text, re.IGNORECASE):
            return value
    return None


def parse_calculation_reply_value(message: str, field: str) -> Optional[str]:
    if field == "day_type":
        return parse_day_type_reply(message)
    if field in CALCULATION_NUMERIC_FIELDS:
        if field in {"overtime_hours", "normal_work_hours", "hours_worked", "night_shift_hours"}:
            from routers.services.numeric_quantities import hour_quantities
            quantities = list(hour_quantities(message))
            if len(quantities) == 1:
                return quantities[0]["value"]
        return parse_numeric_reply(message)
    return None


def infer_pending_field_from_answer(answer: str) -> Optional[str]:
    text = (answer or "").lower()
    if "hourly wage" in text or "hourly rate" in text:
        return "hourly_rate"
    if "overtime hours" in text or "how many overtime" in text:
        return "overtime_hours"
    if "normal working hours" in text or "normal work hours" in text or "convert your daily wage" in text:
        return "normal_work_hours"
    if "ordinary working day" in text and "rest day" in text and "regular holiday" in text:
        return "day_type"
    return None


def copy_pending_inputs_into_state(state: Dict[str, Any], pending: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(state or {})
    for field in pending.get("fields") or []:
        if merged.get(field) in ("", None) and pending.get(field) not in ("", None):
            merged[field] = pending[field]
    if pending.get("topic"):
        merged.setdefault("calculation_topic", pending["topic"])
    return merged


def apply_pending_calculation_reply(state: Dict[str, Any], message: str) -> Dict[str, Any]:
    pending = active_pending_calculation(state)
    if not pending:
        return dict(state or {})

    merged = copy_pending_inputs_into_state(state, pending)
    field = str(pending.get("last_question_asked") or "")
    value = parse_calculation_reply_value(message, field)
    if field == "hourly_rate" and value is None and re.search(r"\b(?:daily|per\s+day|a\s+day)\b", message or "", re.I):
        from routers.services.conversation_memory import extract_money_facts
        value = (extract_money_facts(message) or {}).get("daily_wage")
        if value is not None:
            field = "daily_wage"
    if value is None:
        return merged

    merged[field] = value
    pending = dict(pending)
    pending[field] = value
    pending["last_reply_field"] = field
    pending["last_reply_value"] = value
    merged[PENDING_CALCULATION_KEY] = pending
    return merged


def first_actionable_missing_field(missing_fields: List[str]) -> Optional[str]:
    for field in missing_fields or []:
        if field in CALCULATION_NUMERIC_FIELDS or field == "day_type":
            return field
    return None


def update_pending_calculation_state(
    state: Dict[str, Any],
    calculation: Dict[str, Any],
    topic: str = "overtime_pay",
) -> Dict[str, Any]:
    updated = dict(state or {})
    missing = list(calculation.get("missing_fields") or [])
    fields = list(
        dict.fromkeys(
            (CALCULATION_SPECS.get(topic, {}).get("required_inputs") or [])
            + (CALCULATION_SPECS.get(topic, {}).get("optional_inputs") or [])
            + missing
        )
    )
    last_question = first_actionable_missing_field(missing)
    if updated.get("hours_worked") and not updated.get("day_type") and "day_type" in missing:
        last_question = "day_type"
    elif (updated.get("hours_worked") and updated.get("day_type")
          and not updated.get("hourly_rate") and not updated.get("daily_wage")
          and "hourly_rate" in missing):
        last_question = "hourly_rate"
    if last_question:
        pending: Dict[str, Any] = {
            "status": "awaiting_input",
            "topic": topic,
            "fields": fields,
            "missing_fields": missing,
            "last_question_asked": last_question,
        }
        for field in fields:
            pending[field] = updated.get(field)
        updated[PENDING_CALCULATION_KEY] = pending
    else:
        updated[PENDING_CALCULATION_KEY] = {
            "status": "completed",
            "topic": topic,
            "missing_fields": [],
            "pending_field": None,
            "last_question_asked": None,
        }
    return updated


def calculation_verification_query(message: str, state: Dict[str, Any], calculation: Dict[str, Any]) -> str:
    inputs = calculation.get("inputs") or {}
    day_label = inputs.get("day_type_label") or inputs.get("day_type") or state.get("day_type") or "ordinary working day"
    overtime_hours = inputs.get("overtime_hours") or state.get("overtime_hours") or ""
    hours_worked = inputs.get("hours_worked") or state.get("hours_worked") or ""
    hourly_rate = inputs.get("hourly_rate") or state.get("hourly_rate") or ""
    daily_wage = inputs.get("daily_wage") or state.get("daily_wage") or ""
    return (
        "Worker scenario: calculate applicable overtime compensation for work beyond the normal daily hours. "
        "Philippine Labor Code overtime pay computation and applicable multiplier. "
        f"Day type: {day_label}. "
        f"Hourly rate: {hourly_rate}. "
        f"Daily wage: {daily_wage}. "
        f"Overtime hours: {overtime_hours}. "
        f"Total hours worked: {hours_worked}. "
        "Verify overtime multiplier, coverage, and legal basis."
    ).strip()
