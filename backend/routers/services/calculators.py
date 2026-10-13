"""Centralized benefit and wage calculators for LaborLens."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional


MONEY_QUANT = Decimal("0.01")
PESO = "\u20b1"
MAX_REASONABLE_OVERTIME_HOURS = Decimal("24")

OVERTIME_RATES = {
    "ordinary_day": {
        "label": "Ordinary working day",
        "first_8_hours": Decimal("1.00"),
        "overtime": Decimal("1.25"),
    },
    "rest_day": {
        "label": "Rest day",
        "first_8_hours": Decimal("1.30"),
        "overtime": Decimal("1.69"),
    },
    "special_non_working_day": {
        "label": "Special non-working day",
        "first_8_hours": Decimal("1.30"),
        "overtime": Decimal("1.69"),
    },
    "special_day_rest_day": {
        "label": "Special non-working day falling on a rest day",
        "first_8_hours": Decimal("1.50"),
        "overtime": Decimal("1.95"),
    },
    "regular_holiday": {
        "label": "Regular holiday",
        "first_8_hours": Decimal("2.00"),
        "overtime": Decimal("2.60"),
    },
    "regular_holiday_rest_day": {
        "label": "Regular holiday falling on a rest day",
        "first_8_hours": Decimal("2.60"),
        "overtime": Decimal("3.38"),
    },
}

OVERTIME_MULTIPLIERS = {
    key: value["overtime"]
    for key, value in OVERTIME_RATES.items()
}

DAY_TYPE_QUICK_REPLIES = [
    "Ordinary working day",
    "Rest day",
    "Special non-working day",
    "Regular holiday",
    "Holiday on rest day",
    "Not sure",
]

CALCULATION_SPECS = {
    "overtime_pay": {
        "required_inputs": ["hourly_rate", "overtime_hours"],
        "optional_inputs": ["day_type", "daily_wage", "normal_work_hours"],
        "legal_assumptions": ["covered by the general private-sector overtime rule"],
        "formula": "hourly rate x overtime multiplier x overtime hours",
        "rounding": "ROUND_HALF_UP to two decimal places",
        "legal_source_concepts": ["ordinary_overtime_rule", "holiday_and_rest_day_overtime"],
    },
    "holiday_pay": {
        "required_inputs": ["daily_rate", "holiday_type", "worked_hours_or_day"],
        "optional_inputs": ["rest_day"],
        "legal_assumptions": ["covered by holiday-pay rules"],
        "formula": "daily/hourly rate x applicable holiday multiplier",
        "rounding": "ROUND_HALF_UP to two decimal places",
        "legal_source_concepts": ["holiday_type", "coverage", "first_8_rate"],
    },
    "premium_pay": {
        "required_inputs": ["hourly_rate_or_daily_rate", "day_type", "hours_worked"],
        "optional_inputs": ["rest_day"],
        "legal_assumptions": ["covered by premium-pay rules"],
        "formula": "base rate x applicable premium multiplier",
        "rounding": "ROUND_HALF_UP to two decimal places",
        "legal_source_concepts": ["day_type", "coverage", "premium_rate"],
    },
    "night_shift_differential": {
        "required_inputs": ["hourly_rate", "night_shift_hours"],
        "optional_inputs": ["day_type"],
        "legal_assumptions": ["work was performed during the statutory night period"],
        "formula": "hourly rate x night shift differential multiplier x hours",
        "rounding": "ROUND_HALF_UP to two decimal places",
        "legal_source_concepts": ["night_period", "rate", "coverage"],
    },
    "thirteenth_month_pay": {
        "required_inputs": ["total_basic_salary_earned"],
        "optional_inputs": ["months_worked"],
        "legal_assumptions": ["amount includes only basic salary"],
        "formula": "total basic salary earned during the year / 12",
        "rounding": "ROUND_HALF_UP to two decimal places",
        "legal_source_concepts": ["employee_coverage", "basic_salary_definition", "calculation_formula"],
    },
    "service_incentive_leave_conversion": {
        "required_inputs": ["daily_rate", "unused_sil_days"],
        "optional_inputs": [],
        "legal_assumptions": ["worker is entitled to service incentive leave conversion"],
        "formula": "daily rate x unused SIL days",
        "rounding": "ROUND_HALF_UP to two decimal places",
        "legal_source_concepts": ["leave_entitlement", "conversion_rule"],
    },
    "separation_pay_estimate": {
        "required_inputs": ["monthly_salary", "years_of_service", "authorized_cause"],
        "optional_inputs": [],
        "legal_assumptions": ["separation pay applies only when the legal cause qualifies"],
        "formula": "monthly salary x years of service x applicable fraction",
        "rounding": "ROUND_HALF_UP to two decimal places",
        "legal_source_concepts": ["authorized_cause", "separation_pay_formula"],
    },
    "retirement_pay_estimate": {
        "required_inputs": ["monthly_salary", "years_of_service"],
        "optional_inputs": ["cba_or_company_policy"],
        "legal_assumptions": ["statutory retirement pay applies absent a better plan"],
        "formula": "monthly salary x applicable retirement factor x years of service",
        "rounding": "ROUND_HALF_UP to two decimal places",
        "legal_source_concepts": ["retirement_eligibility", "retirement_pay_formula"],
    },
}


@dataclass
class CalculationResult:
    topic: str
    inputs: Dict[str, Any]
    missing_fields: List[str]
    assumptions: List[str]
    formula: str
    steps: List[str]
    amount: Optional[Decimal]
    currency: str = "PHP"
    is_estimate: bool = False
    clarification_question: Optional[str] = None
    quick_replies: Optional[List[str]] = None

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.amount is not None:
            payload["amount"] = str(self.amount.quantize(MONEY_QUANT, rounding=ROUND_HALF_UP))
        return payload


def money(value: Any) -> Decimal:
    return Decimal(str(value)).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)


def fmt_money(value: Decimal) -> str:
    return f"{PESO}{value.quantize(MONEY_QUANT, rounding=ROUND_HALF_UP):,.2f}"


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value).replace(",", "")).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)


def _plain_decimal(value: Any) -> Decimal:
    return Decimal(str(value).replace(",", ""))


def percent(multiplier: Decimal) -> str:
    value = (multiplier * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{format(value.normalize(), 'f')}%"


def calculate_worked_holiday(daily_rate: Any, worked_hours: Any, rest_day: bool = False) -> Dict[str, Decimal]:
    """One regular holiday, eight-hour basic daily rate; no COLA/NSD added.

    Preserve precision until each payable component is rounded. In particular,
    the first-eight-hours daily amount must never be used as an hourly rate.
    """
    daily, hours = _plain_decimal(daily_rate), _plain_decimal(worked_hours)
    if not daily.is_finite() or not hours.is_finite() or daily <= 0 or not 0 < hours <= 24:
        raise ValueError('A positive daily rate and 0 < worked hours <= 24 are required.')
    rates = OVERTIME_RATES['regular_holiday_rest_day' if rest_day else 'regular_holiday']
    hourly = daily / Decimal('8')
    base = money(hourly * min(hours, Decimal('8')) * rates['first_8_hours'])
    overtime = money(hourly * max(hours - Decimal('8'), Decimal('0')) * rates['overtime'])
    return {'hourly_rate': hourly, 'base_pay': base, 'overtime_pay': overtime,
            'total': base + overtime, 'hours': hours,
            'base_multiplier': rates['first_8_hours'], 'ot_multiplier': rates['overtime']}


def calculate_overtime(hourly_rate: Any, hours: Any, day_type: str = "ordinary_day") -> Decimal:
    return calculate_overtime_pay(money(hourly_rate), _plain_decimal(hours), day_type)


def calculate_overtime_pay(
    hourly_rate: Decimal,
    overtime_hours: Decimal,
    day_type: str,
) -> Decimal:
    hourly_rate = _decimal(hourly_rate)
    overtime_hours = _plain_decimal(overtime_hours)
    if hourly_rate <= 0:
        raise ValueError("Hourly rate must be greater than zero.")
    if overtime_hours <= 0:
        raise ValueError("Overtime hours must be greater than zero.")
    if overtime_hours > MAX_REASONABLE_OVERTIME_HOURS:
        raise ValueError("Overtime hours must be within a reasonable range.")
    if day_type not in OVERTIME_MULTIPLIERS:
        raise ValueError("Unsupported overtime day type.")
    total = hourly_rate * overtime_hours * OVERTIME_MULTIPLIERS[day_type]
    return total.quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)


def overtime_rate_table() -> List[Dict[str, str]]:
    return [
        {
            "day_type": key,
            "label": str(value["label"]),
            "first_8_hours": str(value["first_8_hours"]),
            "first_8_hours_percent": percent(value["first_8_hours"]),
            "overtime": str(value["overtime"]),
            "overtime_percent": percent(value["overtime"]),
        }
        for key, value in OVERTIME_RATES.items()
    ]


def ordinary_overtime_pay(hourly_rate: Any, overtime_hours: Any) -> Dict[str, Any]:
    return overtime_pay(hourly_rate, overtime_hours, "ordinary_day")


def overtime_pay(hourly_rate: Any, overtime_hours: Any, day_type: str = "ordinary_day") -> Dict[str, Any]:
    if day_type not in OVERTIME_RATES:
        raise ValueError(f"Unsupported overtime day type: {day_type}")
    hourly = money(hourly_rate)
    hours = Decimal(str(overtime_hours))
    rate = OVERTIME_RATES[day_type]
    multiplier = rate["overtime"]
    result = calculate_overtime(hourly, hours, day_type)
    label = str(rate["label"])
    return {
        "calculator": "overtime_pay",
        "formula": f"hourly rate x overtime hours x {multiplier}",
        "inputs": {
            "hourly_rate": str(hourly),
            "overtime_hours": str(hours),
            "day_type": day_type,
            "day_type_label": label,
            "first_8_hours_multiplier": str(rate["first_8_hours"]),
            "multiplier": str(multiplier),
        },
        "steps": [
            f"{fmt_money(hourly)} x {hours} hour(s) x {multiplier}",
            f"Estimated {label.lower()} overtime pay = {fmt_money(result)}",
        ],
        "estimated_result": str(result),
        "assumptions": [
            f"The work was beyond eight hours on a {label.lower()}.",
            "The worker is covered by the general private-sector overtime rule.",
        ],
    }


def build_overtime_calculation(state: Dict[str, Any]) -> CalculationResult:
    state = dict(state or {})
    missing: List[str] = []
    assumptions: List[str] = []
    steps: List[str] = []
    hourly_rate: Optional[Decimal] = None
    normal_work_hours = state.get("normal_work_hours")

    if state.get("hourly_rate"):
        hourly_rate = _decimal(state["hourly_rate"])
    elif state.get("daily_wage"):
        if normal_work_hours:
            work_hours = _plain_decimal(normal_work_hours)
            if work_hours <= 0:
                missing.append("normal_work_hours")
            else:
                daily = _decimal(state["daily_wage"])
                hourly_rate = (daily / work_hours).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)
                steps.append(
                    f"Hourly rate = {fmt_money(daily)} / {work_hours.normalize()} = {fmt_money(hourly_rate)}"
                )
        else:
            missing.append("normal_work_hours")
    else:
        missing.append("hourly_rate")

    overtime_hours = None
    if state.get("overtime_hours"):
        overtime_hours = _plain_decimal(state["overtime_hours"])
        if overtime_hours <= 0 or overtime_hours > MAX_REASONABLE_OVERTIME_HOURS:
            missing.append("overtime_hours")
    else:
        missing.append("overtime_hours")

    provided_day_type = str(state.get("day_type") or "").strip()
    day_type = provided_day_type or "ordinary_day"
    is_estimate = not provided_day_type
    if not provided_day_type:
        missing.append("day_type")
        assumptions.append("The overtime was performed on an ordinary working day.")
    if day_type not in OVERTIME_RATES:
        missing.append("day_type")
        day_type = "ordinary_day"
        is_estimate = True

    amount: Optional[Decimal] = None
    inputs: Dict[str, Any] = {
        "hourly_rate": str(hourly_rate) if hourly_rate is not None else None,
        "daily_wage": str(_decimal(state["daily_wage"])) if state.get("daily_wage") else None,
        "normal_work_hours": str(normal_work_hours) if normal_work_hours else None,
        "overtime_hours": str(overtime_hours.normalize()) if overtime_hours is not None else None,
        "day_type": day_type,
        "day_type_label": OVERTIME_RATES[day_type]["label"],
        "multiplier": str(OVERTIME_MULTIPLIERS[day_type]),
    }

    if hourly_rate is not None and overtime_hours is not None and "hourly_rate" not in missing and "overtime_hours" not in missing:
        amount = calculate_overtime_pay(hourly_rate, overtime_hours, day_type)
        overtime_hourly_rate = (hourly_rate * OVERTIME_MULTIPLIERS[day_type]).quantize(
            MONEY_QUANT,
            rounding=ROUND_HALF_UP,
        )
        steps.extend(
            [
                (
                    f"Overtime hourly rate = {fmt_money(hourly_rate)} x "
                    f"{OVERTIME_MULTIPLIERS[day_type]} = {fmt_money(overtime_hourly_rate)}"
                ),
                (
                    f"Overtime pay = {fmt_money(overtime_hourly_rate)} x "
                    f"{overtime_hours.normalize()} hour(s) = {fmt_money(amount)}"
                ),
            ]
        )

    clarification = None
    quick_replies: Optional[List[str]] = None
    unique_missing = list(dict.fromkeys(missing))
    if "hourly_rate" in unique_missing:
        clarification = "What is your hourly wage?"
    elif "normal_work_hours" in unique_missing:
        clarification = "How many normal working hours are used to convert your daily wage to an hourly rate?"
    elif "overtime_hours" in unique_missing:
        clarification = "How many overtime hours did you work?"
    elif "day_type" in unique_missing:
        clarification = (
            "Was the overtime performed on an ordinary working day, rest day, "
            "special non-working day, regular holiday, or a holiday that also fell on your rest day?"
        )
        quick_replies = DAY_TYPE_QUICK_REPLIES

    return CalculationResult(
        topic="overtime_pay",
        inputs=inputs,
        missing_fields=unique_missing,
        assumptions=assumptions,
        formula="Hourly rate x overtime multiplier x overtime hours",
        steps=steps,
        amount=amount,
        currency="PHP",
        is_estimate=is_estimate,
        clarification_question=clarification,
        quick_replies=quick_replies,
    )


def rest_day_or_holiday_overtime_pay(applicable_first_8_hour_rate: Any, overtime_hours: Any) -> Dict[str, Any]:
    rate = money(applicable_first_8_hour_rate)
    hours = Decimal(str(overtime_hours))
    multiplier = Decimal("1.30")
    result = (rate * hours * multiplier).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)
    return {
        "calculator": "rest_day_or_holiday_overtime_pay",
        "formula": "applicable hourly rate for the day x overtime hours x 1.30",
        "inputs": {
            "applicable_hourly_rate": str(rate),
            "overtime_hours": str(hours),
            "multiplier": str(multiplier),
        },
        "steps": [
            f"{fmt_money(rate)} x {hours} hour(s) x 1.30",
            f"Estimated overtime pay = {fmt_money(result)}",
        ],
        "estimated_result": str(result),
        "assumptions": [
            "The overtime was on a rest day, special day, or regular holiday.",
            "The rate supplied is the applicable hourly rate for the first eight hours of that day.",
        ],
    }


def thirteenth_month_pay(total_basic_salary_earned: Any) -> Dict[str, Any]:
    total = money(total_basic_salary_earned)
    divisor = Decimal("12")
    result = (total / divisor).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)
    return {
        "calculator": "thirteenth_month_pay",
        "formula": "total basic salary earned during the year / 12",
        "inputs": {
            "total_basic_salary_earned": str(total),
            "divisor": "12",
        },
        "steps": [
            f"{fmt_money(total)} / 12",
            f"Estimated 13th-month pay = {fmt_money(result)}",
        ],
        "estimated_result": str(result),
        "assumptions": [
            "The total given includes only basic salary earned for the covered year.",
        ],
    }


def overtime_missing_fields(state: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if not state.get("hourly_rate"):
        missing.append("hourly rate")
    if not state.get("overtime_hours"):
        missing.append("number of overtime hours")
    if not state.get("day_type"):
        missing.append("day type: ordinary day, rest day, special day, or regular holiday")
    return missing


def calculate_overtime_from_state(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if overtime_missing_fields(state):
        return None
    day_type = str(state.get("day_type") or "")
    return overtime_pay(state["hourly_rate"], state["overtime_hours"], day_type)
