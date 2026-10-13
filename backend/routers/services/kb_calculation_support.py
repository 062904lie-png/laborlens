"""Arithmetic support only; legal prose is generated from retrieved KB passages."""
import re
from decimal import Decimal
from routers.services.calculators import calculate_worked_holiday


def ordinary_overtime_rule(sources):
    """Extract the normal-hours threshold and premium from operative text.

    No stored legal multiplier is used. Ambiguous or conflicting rates fail closed.
    """
    rules = set()
    for source in sources:
        body = re.sub(r'\s+', ' ', str(source.get('excerpt') or source.get('text') or '')
                      .split('\nText:\n')[-1])
        for match in re.finditer(
            r'(?:beyond|in excess of)\s+(eight|8)\s+hours'
            r'[^.!?]{0,400}?(?:additional compensation|additional|plus|premium)'
            r'[^.!?]{0,100}?(twenty[- ]five|25)\s*(?:\(\s*25\s*%\s*\)|percent|%)',
            body, re.I):
            clause = match.group(0)
            if re.search(r'holiday|rest day|special day', clause, re.I):
                continue
            threshold = Decimal('8') if match[1].lower() == 'eight' else Decimal(match[1])
            premium = Decimal('25') if re.fullmatch(r'twenty[- ]five', match[2], re.I) else Decimal(match[2])
            rules.add((threshold, Decimal('1') + premium / Decimal('100')))
    return next(iter(rules)) if len(rules) == 1 else None


def grounded_ordinary_wage_calculation(query, sources):
    """Compute a single ordinary-day wage from explicit user inputs and KB rule."""
    if not matches(r'\b(?:ordinary|regular|normal)\s+(?:working\s+|work\s*)?day\b', query):
        return None
    if matches(r'holiday|rest day|night|allowance|COLA|government|managerial|'
               r'\b(?:not|isn.t)\s+(?:an?\s+)?(?:ordinary|regular|normal)', query):
        return None
    rates = resolved_numeric_values(r'hourly\s+(?:rate|wage|pay)\s*(?:is|of|[:=])?\s*(?:₱|PHP|P)?\s*'
                       r'(\d[\d,]*(?:\.\d{1,2})?)(?!\d|\.\d)', query)
    daily_rates = resolved_numeric_values(r'daily\s+(?:rate|wage|pay)\s*(?:is|of|[:=])?\s*(?:₱|PHP|P)?\s*'
                       r'(\d[\d,]*(?:\.\d{1,2})?)(?!\d|\.\d)', query)
    hours = resolved_numeric_values(r'work(?:ed)?\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|hors?)\b', query)
    if len(rates) + len(daily_rates) != 1 or len(hours) != 1 or matches(r'hours?\s*(?:of\s+)?(?:overtime|OT)\b', query):
        return None
    rule = ordinary_overtime_rule(sources)
    if rule is None:
        return None
    normal_hours, multiplier = rule
    # Divide the daily rate by the hours established in the retrieved rule,
    # not by a separately stored legal threshold or multiplier.
    stated_rate = Decimal((rates or daily_rates)[0].replace(',', ''))
    hourly = stated_rate if rates else stated_rate / normal_hours
    total_hours = Decimal(hours[0])
    if hourly <= 0 or not 0 < total_hours <= 24:
        return None
    basic_hours = min(total_hours, normal_hours)
    overtime_hours = max(total_hours - normal_hours, Decimal('0'))
    base = hourly * basic_hours
    overtime = hourly * overtime_hours * multiplier
    return {
        'topic': 'overtime_pay', 'kind': 'ordinary_day_total_wage',
        'inputs': {'hourly_rate': str(hourly), 'hours_worked': str(total_hours),
                   'ordinary_hours': str(basic_hours), 'overtime_hours': str(overtime_hours),
                   'day_type': 'ordinary_day', 'multiplier': str(multiplier)},
        'pay_basis': 'hourly' if rates else 'daily',
        'stated_rate': str(stated_rate),
        'base_pay': str(base), 'overtime_pay': str(overtime), 'total': str(base + overtime),
        'assumptions': ['employee covered by the retrieved hours-of-work rule',
                        'stated hours are compensable work hours',
                        'gross pay before deductions; no additional premiums included'],
    }

def resolved_numeric_values(pattern, query):
    """Latest explicit input replaces the old value; unchanged inputs survive.

    Do not choose one of several conflicting values in the latest segment.
    The resolved question contains user inputs, never assistant-made facts.
    """
    parts = re.split(r'\n(?:Current follow-up question|User clarification[^:]*)\s*:', str(query or ''), flags=re.I)
    for part in reversed(parts):
        values = re.findall(pattern, part, re.I)
        if values:
            return values
    return []


def matches(pattern, text):
    return bool(re.search(pattern, text or '', re.I | re.S))


def holiday_inputs(query):
    """Only explicit daily rate + total hours for a single regular holiday.

    Ambiguous, multiple-day, NSD, sector-specific and overtime-only requests
    remain with the existing intake flow. Never infer total hours from OT hours.
    """
    if not matches(r'regular\s+holiday', query) or matches(
        r'double holiday|two.{0,20}holidays|dalawang.{0,20}holiday|night|gabi|COLA|allowance|'
        r'kasambahay|government|managerial|field personnel|not sure|hindi sigurado|multiple days|'
        r'(?:not|hindi)\s+(?:a\s+)?regular holiday', query
    ):
        return None
    rate = re.search(r'(?:daily\s+(?:rate|wage|pay)|arawang\s+sahod)\s*(?:ko\s*)?(?:ay\s*|is\s*|of\s*|[:=]\s*)?(?:₱|PHP|P)?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)(?!\d|\.\d)', query, re.I)
    if not rate:
        rate = re.search(r'(?:₱|PHP)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*(?:daily|per day|a day)', query, re.I)
    hours = re.search(r'(?:nagtrabaho\s+(?:ako\s+)?(?:ng\s+)?|worked\s+(?:for\s+)?)(\d+(?:\.\d+)?|sampung|sampu|walong|walo|siyam|labing[- ]isa|labindalawang)\s*(?:oras|hours?|hrs?)', query, re.I)
    if not rate or not hours or matches(r'\b(?:oras|hours?|hrs?)\s*(?:of\s+)?(?:overtime|OT)\b', query):
        return None
    words = {'sampung': '10', 'sampu': '10', 'walong': '8', 'walo': '8', 'siyam': '9',
             'labing-isa': '11', 'labing isa': '11', 'labindalawang': '12'}
    count = Decimal(words.get(hours[1].lower(), hours[1]))
    daily = Decimal(rate[1].replace(',', ''))
    if daily <= 0 or not 0 < count <= 24:
        return None
    rest = matches(r'rest\s+day', query) and not matches(r'(?:not|hindi)\s+(?:my\s+|ko\s+)?rest\s+day', query)
    return daily, count, rest



def holiday_rate_evidence(sources, also_rest_day=False):
    """Identify uploaded passages proving the required holiday multipliers."""
    bodies = [re.sub(r'\s+', ' ', str(s.get('excerpt') or s.get('text') or '').split('\nText:\n')[-1]) for s in sources]
    required = [
        r'holiday.{0,220}(?:twice|200\s*%|two hundred)',
        r'(?:overtime|eight hours).{0,500}(?:thirty|30)',
    ]
    if also_rest_day:
        required.append(r'260\s*%.{0,100}regular holiday.{0,100}rest day|regular holiday.{0,100}rest day.{0,150}(?:260\s*%|additional thirty|add thirty|additional 30|add 30)')
    used = []
    for pattern in required:
        idx = next((i for i, text in enumerate(bodies) if matches(pattern, text)), None)
        if idx is None:
            return None
        if idx not in used:
            used.append(idx)
    return used


def grounded_holiday_calculation(query, sources):
    """Return numeric working data only when the uploaded text proves each rate."""
    values = holiday_inputs(query)
    if not values:
        return None
    used = holiday_rate_evidence(sources, values[2])
    if used is None:
        return None
    result = calculate_worked_holiday(*values)
    return {
        'kind': 'worked_regular_holiday',
        'daily_rate': str(values[0]), 'total_hours': str(values[1]),
        'also_rest_day': values[2],
        **{key: str(value) for key, value in result.items()},
        'source_labels': [sources[i].get('label') or f'Source {i + 1}' for i in used],
        'assumptions': ['covered private-sector employee', 'eight-hour basic daily rate',
                        'single regular holiday', 'no COLA or night differential included'],
    }


def calculation_answer_errors(answer, calculation):
    """Reject a generated computation that drops or changes the checked total."""
    if not calculation:
        return []
    total = calculation.get('total', calculation.get('amount'))
    if total is None:
        return []
    expected = Decimal(str(total))
    amounts = [Decimal(value.replace(',', '')) for value in
               re.findall(r'(?:₱|PHP\s*|P\s*)(\d[\d,]*(?:\.\d+)?)', answer or '', re.I)]
    if expected not in amounts:
        return ['The generated calculation omitted or changed the verified total.']
    return []


def calculation_evidence_present(calculation, sources):
    """Code computes arithmetic; the uploaded body must establish its multiplier."""
    bodies = [re.sub(r'\s+', ' ', str(s.get('excerpt') or s.get('text') or '').split('\nText:\n')[-1]) for s in sources]
    def has(pattern):
        return any(matches(pattern, body) for body in bodies)
    if calculation.get('topic') == 'thirteenth_month_pay':
        return has(r'(?:basic salary|basic salaries).{0,160}(?:divided by (?:12|twelve)|/\s*12)|(?:one[- ]twelfth|1/12).{0,160}basic salar')
    day = (calculation.get('inputs') or {}).get('day_type')
    if day == 'ordinary_day':
        return ordinary_overtime_rule(sources) is not None
    if day in {'regular_holiday', 'regular_holiday_rest_day'}:
        return holiday_rate_evidence(sources, day == 'regular_holiday_rest_day') is not None
    if day in {'rest_day', 'special_non_working_day', 'special_day_rest_day'}:
        base = r'(?:fifty|50\s*%|150\s*%)' if day == 'special_day_rest_day' else r'(?:thirty|30\s*%|130\s*%)'
        return has(r'(?:rest day|special (?:non-working )?day|special holiday).{0,180}' + base) and has(
            r'(?:overtime|beyond eight hours|beyond 8 hours).{0,250}(?:thirty|30\s*%)')
    return False
