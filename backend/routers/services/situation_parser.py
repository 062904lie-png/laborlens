"""Extract exact scenario details without deciding what they mean legally."""
import re
import unicodedata


_CONTRAST = re.compile(
    r"(?<=[.!?;])\s+|,\s*(?=(?:is|are|was|were|am|can|could|should|what|how|"
    r"why|when|where|who|does|do|did|will|would|may|must)\b)|"
    r"\s+\b(?:but|however|although|yet|pero|ngunit|subalit)\b\s+",
    re.I,
)
_QUESTION_START = re.compile(
    r"^(?:what|which|who|whose|where|when|why|how|can|could|should|may|must|"
    r"am|is|are|was|were|do|does|did|will|would|paano|ano|sino|saan|kailan|"
    r"bakit|pwede|puwede)\b",
    re.I,
)
_NEGATION = re.compile(
    r"\b(?:not|never|no|without|neither|nor|don't|doesn't|didn't|wasn't|"
    r"weren't|haven't|hasn't|hadn't|hindi|di|indi|wala|walang)\b",
    re.I,
)
_REPORTING = re.compile(
    r"\b(?:said|says|told|stated|claimed|reported|alleged|explained|denied|"
    r"asked|ordered|warned|promised|sinabi|sabi|iniulat|giingnan)\b",
    re.I,
)
_CONDITION_OR_TIME = re.compile(
    r"\b(?:because|since|due\s+to|after|before|during|while|when|if|unless|"
    r"until|from|dahil|bago|pagkatapos|habang|kung|kon|sanglit)\b",
    re.I,
)
_QUANTITY_OR_DATE = re.compile(
    r"(?:₱|\bPHP\s*|\bP\s*\d|\bpesos?\s*\d|\d)|"
    r"\b(?:today|yesterday|tomorrow|last|next|january|february|march|april|"
    r"may|june|july|august|september|october|november|december|ngayon|kahapon|"
    r"bukas|adlaw|bulan|taon)\b",
    re.I,
)


def parse_situation(question, *, limit=8):
    """Return short verbatim fact clauses, never inferred legal classifications.

    The parser labels surface features only. It does not infer employment
    status, legal cause, eligibility, truth of a reported allegation, or what
    facts the governing rule requires.
    """
    text = unicodedata.normalize("NFKC", str(question or "")).strip()
    if not text:
        return []
    facts = []
    seen = set()
    for raw in _CONTRAST.split(text):
        clause = raw.strip(" \t,;:-")
        key = re.sub(r"\s+", " ", clause).casefold()
        if len(clause.split()) < 3 or key in seen or _QUESTION_START.match(clause):
            continue
        if clause.endswith("?"):
            continue
        seen.add(key)
        features = []
        if _QUANTITY_OR_DATE.search(clause):
            features.append("explicit amount, number, or time")
        if _REPORTING.search(clause):
            features.append("reported statement; keep attribution")
        if _NEGATION.search(clause):
            features.append("stated negation or absence")
        if _CONDITION_OR_TIME.search(clause):
            features.append("stated condition or timing")
        facts.append({"features": features or ["user-stated detail"], "text": clause[:320]})
        if len(facts) >= max(0, min(int(limit), 8)):
            break
    return facts


def situation_search_probes(question, *, limit=2):
    """Search explicit scenario clauses as supplements to the complete question."""
    result = []
    for fact in parse_situation(question):
        if len(fact["features"]) == 1 and fact["features"][0] == "user-stated detail":
            continue
        probe = re.sub(r"\s+", " ", fact["text"]).strip()
        if probe and probe.casefold() != str(question or "").strip().casefold():
            result.append(probe)
        if len(result) >= max(0, min(int(limit), 3)):
            break
    return result
