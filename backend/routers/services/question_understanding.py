"""Recognize an answerable request without a catalog of legal subjects.

This gate decides whether searching documents can be useful; it does not decide
the legal answer, coverage, or whether the evidence is sufficient. A described
situation can be explained conditionally before asking for every possible fact.
Personal computations still use the existing checks for missing numeric inputs.
"""

import re
import unicodedata


def employment_context_parts(text):
    """Separate denied descriptions and other actors from worker-scope hints.

    These views are for fact extraction only. Retrieval keeps the full question.
    A denied clause is retained separately so memory can retract an old fact.
    """
    text = str(text or '').replace('’', "'")
    # A supervisor mentioned as an actor does not describe the worker's role.
    text = re.sub(r'\b(?:my|our|his|her|their)\s+(?:manager|supervisor|boss)\b',
                  'another person', text, flags=re.I)
    positive, negative = [], []
    for clause in re.split(r'[.!?;\n]|\b(?:but|however)\b', text, flags=re.I):
        clause = clause.strip()
        negation = re.search(r"\b(?:not|never|no longer|hindi|indi)\b|n't\b", clause, re.I)
        if negation:
            # A later denial must not erase an earlier affirmative description
            # ("I am a regular employee and was not paid").
            positive.append(clause[:negation.start()])
            negative.append(clause[negation.start():])
        else:
            positive.append(clause)
    return '. '.join(positive), '. '.join(negative)


def employment_fact_text(text):
    """Questions about a status must not be stored as assertions of that status."""
    positive, _ = employment_context_parts(text)
    return '. '.join(clause for clause in positive.split('. ') if not re.match(
        r'\s*(?:am|are|is|was|were|can|could|do|does|did|what|which|who|how|if)\b', clause, re.I))


# These are grammatical words, not a list of supported laws, documents, or topics.
_GRAMMAR_WORDS = frozenset("""
    a an the this that it these those they them their he she his her i me my mine
    we us our ours you your yours somebody someone anyone anything something
    is are was were am be been being do does did can could may might will would
    should must have has had what which who whom whose where when why how yes no
    of for to from with without in on at by as and or but if then than only also
    really please pls kindly help question ask tell know want need about regarding
    explain understand explanation information general overview rules rule right
    rights legal correct lawful allowed applies apply entitled covered coverage
    ang ng mga sa ko ako akin aking ka ikaw mo iyo iyong niya siya kami namin
    tayo natin sila nila ito iyan iyon ba po opo na pa naman lang at pero kung
    ano anong alin sino saan kailan paano pwede puwede maaari paki pakisabi
    indi bala gid sang kag nga ini ina sina akon imo amon aton iya nila ano
""".split())

_REQUEST_RE = re.compile(
    r"\b(?:what|which|who|whose|where|when|why|how)\b|"
    r"\b(?:can|could|may|should|must|am|is|are|was|were|does|do)\b.{0,120}[?]|"
    r"\b(?:explain|describe|compare|summari[sz]e|define|clarify)\b|"
    r"\b(?:tell|show|teach|help)\b.{0,40}\b(?:about|understand|difference|how|why)\b|"
    r"\b(?:ano|anong|paano|sino|saan|kailan|alin|ipaliwanag|ipalathag|"
    r"pakiexplain|paki[- ]?explain|pwede|puwede|maaari)\b",
    re.I,
)

_VAGUE_QUERY_RE = re.compile(
    r"^(?:(?:please|pls|hi|hello|hey|thank you)[,!. ]*)*"
    r"(?:can (?:i|you) (?:ask(?: (?:you )?(?:a |some )?(?:question|something))?|help(?: me)?)|"
    r"(?:is|was) (?:it|this|that) (?:legal|correct|allowed|okay|ok)|"
    r"(?:what|how) about (?:it|this|that)|what (?:should|can) i do|"
    r"(?:please |pls )?(?:help(?: me)?|explain(?: (?:it|this|that))?)|"
    r"(?:pwede|puwede) (?:ba )?(?:ako(?:ng)? )?magtanong|"
    r"ano(?:ng)? (?:ang )?(?:gagawin|dapat kong gawin)|legal ba (?:ito|iyan|iyon))"
    r"[?.! ]*$",
    re.I,
)

_GENERAL_METHOD_RE = re.compile(
    r"\b(?:formula|method|general (?:calculation|computation)|"
    r"how (?:is|are)\b.{0,100}\b(?:computed|calculated)|"
    r"how (?:does|do)\b.{0,100}\bwork|"
    r"paano (?:kinakalkula|kinukuwenta|gumagana))\b",
    re.I,
)

_PERSONAL_QUANTITIES_RE = re.compile(
    r"(?:[₱$€£]|\b(?:php|pesos?)\s*\d)|"
    r"\b\d+(?:[.,]\d+)?\s*(?:hours?|hrs?|days?|weeks?|months?|years?|"
    r"oras|araw|buwan|taon|adlaw|bulan|tuig)\b|"
    r"\b(?:my|our|his|her|their)\s+(?:amount|balance|pay|salary|wage|rate)\b",
    re.I,
)


def request_subject_terms(question):
    """Extract the user's words without limiting them to known legal topics."""
    text = unicodedata.normalize("NFKC", str(question or ""))
    words = re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", text.lower(), re.U)
    return tuple(dict.fromkeys(word for word in words if word not in _GRAMMAR_WORDS))


def retrieval_ready_request(question, *, calculation_requested=False):
    """Whether a useful evidence search should precede a generic intake form.

    Call this with a reconstructed user question when resolving a follow-up.
    Unknown terminology is allowed. Personal calculations defer to checks for
    their actual missing inputs; general formulas do not require a pay history.
    Explicit ambiguity checks (such as an unidentified document) remain first.
    """
    text = unicodedata.normalize("NFKC", str(question or "")).strip()
    if not text or _VAGUE_QUERY_RE.fullmatch(text):
        return False
    if not _REQUEST_RE.search(text):
        return False
    if not request_subject_terms(text):
        return False
    if calculation_requested:
        # A formula can be explained without inputs. If actual personal values
        # accompany it, preserve the computation path rather than guessing.
        return bool(_GENERAL_METHOD_RE.search(text)) and not bool(
            _PERSONAL_QUANTITIES_RE.search(text)
        )
    return True


def stated_service_facts(question):
    """Read a stated service duration; calendar years and age are not tenure."""
    from decimal import Decimal
    from routers.services.numeric_quantities import NUMBER_PATTERN, parse_number

    text = str(question or "")
    clarification_markers = list(re.finditer(r"\bUser clarification[^:]*:", text, re.I))
    clarification_start = clarification_markers[-1].end() if clarification_markers else None
    pattern = re.compile(r"(?<![\w.-])(?P<number>" + NUMBER_PATTERN + r")\s*"
                         r"(?P<unit>years?|months?|taon|buwan|tuig|bulan)\b", re.I)
    durations = []
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 100):match.start()]
        suffix = text[match.end():match.end() + 65]
        if re.match(r"\s+old\b", suffix, re.I) or re.search(r"\b(?:age|aged)\s*$", prefix, re.I):
            continue
        before = re.search(r"\b(?:work(?:ed|ing)?|employ(?:ed|ment)?|serv(?:ed|ing)|tenure|"
                           r"service|been\s+with|with\s+(?:the\s+|my\s+)?(?:company|employer)|"
                           r"nagtrabaho|nagtatrabaho|nagaobra|nagobra)\b[^.;!?]{0,85}$", prefix, re.I)
        after = re.match(r"\s+(?:of\s+(?:continuous\s+)?(?:service|employment)|"
                         r"with\s+(?:the\s+|my\s+)?(?:company|employer)|"
                         r"sa\s+(?:company|kumpanya|trabaho))\b", suffix, re.I)
        prior_duration = durations and not re.search(r"[.;!?]", text[durations[-1]["end"]:match.start()])
        clarified_duration = (clarification_start is not None and match.start() >= clarification_start
                              and bool(durations))
        if not (before or after or prior_duration or clarified_duration):
            continue
        if re.search(r"[-−﹣－]\s*$", prefix):
            continue
        value = parse_number(match["number"])
        if value is not None:
            durations.append({"text": match.group(), "value": value, "unit": match["unit"].lower(),
                              "start": match.start(), "end": match.end()})
    if not durations:
        return {}
    if clarification_start is not None and any(part["start"] >= clarification_start for part in durations):
        durations = [part for part in durations if part["start"] >= clarification_start]
    # Repeated descriptions of the same input are not additive. A joined
    # years-and-months duration is one value; distinct claims remain ambiguous.
    groups = []
    for duration in durations:
        if groups and re.fullmatch(r"\s*(?:and|at|kag|,)?\s*", text[groups[-1][-1]["end"]:duration["start"]], re.I):
            groups[-1].append(duration)
        else:
            groups.append([duration])
    values = []
    for group in groups:
        units = ["year" if part["unit"] in {"year", "years", "taon", "tuig"} else "month" for part in group]
        if len(units) != len(set(units)):
            continue
        years = sum((part["value"] if unit == "year" else part["value"] / Decimal(12)
                     for part, unit in zip(group, units)), Decimal(0))
        values.append((years, text[group[0]["start"]:group[-1]["end"]]))
    unique = {value for value, _ in values}
    if len(unique) != 1:
        return {}
    return {"years_of_service": format(values[-1][0].normalize(), "f"),
            "service_duration": values[-1][1]}


def stated_separation_reason(question):
    """Preserve the user's stated reason without assigning a legal category."""
    text = str(question or "")
    if not re.search(r"\b(?:separation|termination|dismiss(?:ed|al)?|terminated|laid\s+off|"
                     r"let\s+go|fired|tinanggal|natanggal|gintanggal|pagtanggal)\b", text, re.I):
        return None
    patterns = (
        r"\b(?:due\s+to|because\s+of|because|on\s+the\s+grounds?\s+of|dahil\s+sa|dahil)\s+",
        r"\b(?:reason\s+(?:for\s+(?:separation|termination|dismissal)\s*)?|"
        r"separation\s+reason|dahilan|rason)\s*(?:given\s+)?(?:is|was|ay|:|=)\s*",
        r"\b(?:dismissed|terminated|laid\s+off|let\s+go)\s+(?:for|on\s+account\s+of)\s+",
        r"\b(?:employer|company|HR|supervisor|manager)\s+(?:says?|said|states?|stated|cites?|cited)\s+",
    )
    reasons = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            reason = re.split(r"[.;!?\n]|\b(?:but|however|pero|User clarification)\b", text[match.end():], maxsplit=1, flags=re.I)[0].strip()
            reason = re.split(r"\b(?:and|at|kag)\s+(?:I|my|I've|I’m|I am|ako|akon)\b", reason, maxsplit=1, flags=re.I)[0].strip()
            if not reason or len(reason) > 220 or re.match(r"^(?:unknown|unspecified|not\s+(?:given|stated)|"
                                                          r"no\s+reason|I\s+(?:do\s+not|don't)\s+know|hindi\s+alam|wala)\b", reason, re.I):
                continue
            # A date or quantity following 'for' is not an employer's reason.
            if re.match(r"^(?:\d|₱|PHP\b)", reason, re.I):
                continue
            reasons.append((match.start(), reason))
    return max(reasons)[1] if reasons else None


def separation_input_facts(question, history=None):
    """Locate only explicit amount, pay period, tenure, and stated cause facts."""
    text = str(question or "")
    facts = stated_service_facts(text)
    reason = stated_separation_reason(text)
    if reason:
        facts["separation_reason"] = reason
    amount = r"(?:₱|PHP\s*|P\s*)?(\d[\d,]*(?:\.\d+)?)"
    patterns = (
        r"\b(?:salary|wage|sahod|sweldo|suweldo|rate)\s*(?:ko\s+)?(?:is|was|of|ay|=|:)?\s*" + amount,
        amount + r"\s*(?:per\s+(?:month|day)|a\s+(?:month|day)|monthly|daily|kada\s+(?:buwan|adlaw|araw))\b",
    )
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)
               if not re.search(r"[-−﹣－]\s*$", text[max(0, match.start() - 10):match.start()])]
    if matches:
        matched = max(matches, key=lambda match: match.start())
        facts["pay_amount"] = matched[1].replace(",", "")
        offset = max(0, matched.start() - 50)
        near = text[offset:matched.end() + 55]
        periods = []
        for period, pattern in (
            ("monthly", r"\b(?:monthly|per\s+month|a\s+month|each\s+month|buwanang|binulan|kada\s+buwan)\b"),
            ("daily", r"\b(?:daily|per\s+day|a\s+day|each\s+day|arawang|kada\s+(?:araw|adlaw))\b"),
        ):
            for unit in re.finditer(pattern, near, re.I):
                start, end = unit.start() + offset, unit.end() + offset
                distance = max(matched.start() - end, start - matched.end(), 0)
                periods.append((distance, period))
        if periods:
            _, period = min(periods)
            facts["salary_period"] = period
            facts["monthly_salary" if period == "monthly" else "daily_wage"] = facts["pay_amount"]
    # A direct amount reply to an explicit pay-period question inherits the
    # question's unit. The amount itself still has to come from the user.
    last_prompt = next((message for message in reversed(list(history or []))
                        if message.get("role") == "assistant"), {})
    prompt = str(last_prompt.get("clarification_question") or last_prompt.get("content") or "")
    if (last_prompt.get("intent") == "needs_clarification" or last_prompt.get("clarification_question")):
        reply = re.split(r"User clarification[^:]*:", text, flags=re.I)[-1].strip()
        if re.fullmatch(amount + r"[.!\s]*", reply, re.I) and re.search(r"\bmonthly\s+(?:basic\s+)?salary\b", prompt, re.I):
            value = re.fullmatch(amount + r"[.!\s]*", reply, re.I)[1].replace(",", "")
            facts.update(pay_amount=value, monthly_salary=value, salary_period="monthly")
    return facts


def missing_separation_inputs(question, history=None):
    facts = separation_input_facts(question, history)
    missing = []
    if not facts.get("separation_reason"):
        missing.append("separation_reason")
    if "years_of_service" not in facts:
        missing.append("service_duration")
    if "pay_amount" not in facts:
        missing.append("pay_amount")
    if "salary_period" not in facts:
        missing.append("salary_period")
    return missing


def separation_intake_question(question, language="en", history=None):
    """Ask only for still-missing inputs, preserving the supplied scenario."""
    missing = missing_separation_inputs(question, history)
    if not missing:
        return None
    labels = {
        "en": {"separation_reason": "the reason your employer gave for the separation",
               "service_duration": "how long you worked for the employer",
               "pay_amount": "your basic salary amount", "salary_period": "whether that salary is daily or monthly"},
        "fil": {"separation_reason": "dahilan ng employer sa iyong pagtanggal",
                "service_duration": "gaano ka katagal nagtrabaho sa employer",
                "pay_amount": "halaga ng iyong basic salary", "salary_period": "kung arawan o buwanan ang sahod"},
        "hil": {"separation_reason": "rason sang employer sa imo pagtanggal",
                "service_duration": "pila ka kadugay nagtrabaho sa employer",
                "pay_amount": "kantidad sang imo basic salary", "salary_period": "kon adlaw-adlaw ukon binulan ang sweldo"},
    }
    if missing == ["pay_amount", "salary_period"]:
        return {"en": "What is your basic salary, and is it paid daily or monthly?",
                "fil": "Magkano ang iyong basic salary, at arawan ba o buwanan ito?",
                "hil": "Pila ang imo basic salary, kag adlaw-adlaw bala ukon binulan ini?"}.get(language, "What is your basic salary, and is it paid daily or monthly?")
    selected = labels.get(language, labels["en"])
    prefix = {"en": "Please provide", "fil": "Pakibigay ang", "hil": "Palihog ihatag ang"}.get(language, "Please provide")
    return prefix + " " + "; ".join(selected[field] for field in missing) + "."
