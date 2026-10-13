"""Read explicit numeric quantities, including common written hour counts.

Number words and unit grammar are language data. This module contains no legal
rates, entitlement formulas, or document-specific information.
"""

import re
from decimal import Decimal, InvalidOperation


_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
    "isa": 1, "isang": 1, "dalawa": 2, "dalawang": 2, "duha": 2,
    "tatlo": 3, "tatlong": 3, "tatlu": 3, "apat": 4, "apatna": 4,
    "lima": 5, "limang": 5, "anim": 6, "animna": 6, "pito": 7,
    "pitong": 7, "walo": 8, "walong": 8, "siyam": 9, "siyamna": 9,
    "sampu": 10, "sampung": 10, "labingisa": 11, "labingisang": 11,
    "labindalawa": 12, "labindalawang": 12,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SMALL_PATTERN = "|".join(sorted(_ONES, key=len, reverse=True))
_TENS_PATTERN = "|".join(_TENS)
_BASE_NUMBER = (
    r"(?:\d+(?:\.\d+)?|(?:" + _TENS_PATTERN + r")"
    r"(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|"
    r"labing[- ]isang?|labin[- ]dalawang?|(?:" + _SMALL_PATTERN + r"))"
)
NUMBER_PATTERN = (
    r"(?:" + _BASE_NUMBER + r"(?:[- ]+and[- ]+(?:a[- ]+)?half)?|"
    r"half(?:\s+an?)?|(?:a\s+)?quarter(?:\s+of\s+an?)?)"
)
_HOUR_RE = re.compile(
    r"(?<![\w.\-])(?P<number>" + NUMBER_PATTERN + r")\s*"
    r"(?P<modifier>overtime|extra|OT)?\s*(?:ka\s+|na\s+|nga\s+)?"
    r"(?P<unit>hours?|hrs?|oras)\b"
    r"(?:\s+and\s+(?P<minutes>" + _BASE_NUMBER + r")\s+minutes?\b)?",
    re.I,
)
_RANGE_PREFIX_RE = re.compile(_BASE_NUMBER + r"\s*(?:or|to|[-–])\s*$", re.I)


def parse_number(value):
    """Return a Decimal only for a recognized number, never infer a value."""
    original = str(value or "").strip()
    if re.match(r"^[-−﹣－]\s*", original):
        return None
    text = re.sub(r"\s+", " ", original.lower().replace("-", " "))
    half = bool(re.search(r"\s+and (?:a )?half$", text))
    if half:
        text = re.sub(r"\s+and (?:a )?half$", "", text)
    if text in {"half", "half a", "half an"}:
        return Decimal("0.5")
    if text in {"quarter", "a quarter", "quarter of a", "quarter of an", "a quarter of a", "a quarter of an"}:
        return Decimal("0.25")
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        try:
            number = Decimal(text)
        except InvalidOperation:
            return None
    elif text.replace(" ", "") in _ONES:
        number = Decimal(_ONES[text.replace(" ", "")])
    else:
        words = text.split()
        if words and words[0] in _TENS and len(words) <= 2:
            remainder = _ONES.get(words[1]) if len(words) == 2 else 0
            if remainder is None or not 0 <= remainder < 10:
                return None
            number = Decimal(_TENS[words[0]] + remainder)
        else:
            return None
    return number + Decimal("0.5") if half else number


def hour_quantities(text):
    """Yield explicit hour counts with their original spans and unit labels."""
    text = str(text or "")
    for match in _HOUR_RE.finditer(text):
        if re.search(r"[-−﹣－]\s*$", text[max(0, match.start() - 20):match.start()]):
            continue
        # Do not choose one endpoint of a range or an alternative.
        if _RANGE_PREFIX_RE.search(text[max(0, match.start() - 55):match.start()]):
            continue
        value = parse_number(match["number"])
        if value is None:
            continue
        if match["minutes"]:
            minutes = parse_number(match["minutes"])
            if minutes is None or not 0 <= minutes < 60:
                continue
            value += minutes / 60
        yield {
            "value": format(value.normalize(), "f"),
            "start": match.start(), "end": match.end(),
            "modifier": str(match["modifier"] or "").lower(),
        }


def explicit_overtime_hours(text):
    """Read a unique stated overtime duration, keeping normal shift hours out."""
    text = str(text or "")
    values = []
    previous_end = 0
    for quantity in hour_quantities(text):
        prefix = text[max(previous_end, quantity["start"] - 60):quantity["start"]]
        suffix = text[quantity["end"]:quantity["end"] + 25]
        before = re.search(r"\b(?:overtime|OT|nag[- ]?overtime)\b[^.;!?]{0,55}$", prefix, re.I)
        before_extra = re.search(r"\b(?:extra|additional)\s*$", prefix, re.I)
        after = re.match(r"\s*(?:of\s+)?(?:overtime|OT)\b", suffix, re.I)
        negated = bool(re.search(
            r"\b(?:no|not|without|walang|wala)\s+(?:(?:any|the)\s+)?"
            r"(?:overtime|OT|extra|additional)\b[^.;!?]*$", prefix, re.I,
        ))
        if (quantity["modifier"] or before or before_extra or after) and not negated:
            values.append(quantity["value"])
        previous_end = quantity["end"]
    values = list(dict.fromkeys(values))
    return values[0] if len(values) == 1 else None


def worked_total_hours(text):
    """Read one explicit total-hours figure attached to a past-tense work verb."""
    text = str(text or "")
    totals = []
    for quantity in hour_quantities(text):
        prefix = text[max(0, quantity["start"] - 80):quantity["start"]]
        suffix = text[quantity["end"]:quantity["end"] + 30]
        worked = re.search(r"\bworked\s+(?:for\s+)?$", prefix, re.I)
        overtime_label = bool(quantity["modifier"] or re.search(
            r"\b(?:overtime|OT|extra|additional)\b[^.;!?]*$", prefix, re.I
        ) or re.match(r"\s*(?:overtime|OT|extra|additional)\b", suffix, re.I))
        if worked and not overtime_label:
            totals.append(quantity["value"])
    totals = list(dict.fromkeys(totals))
    return totals[0] if len(totals) == 1 else None
