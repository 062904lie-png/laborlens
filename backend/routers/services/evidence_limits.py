"""Conservative checks for unsupported conclusions observed in chat regressions.

These checks supplement citation validation; they are not a general entailment model.
"""
import re


_LIMITATION = re.compile(r'\b(?:if|may|might|could|possible|potential|subject\s+to|depends?|'
                          r'not\s+(?:automatic|confirmed|established|supported)|cannot\s+(?:confirm|verify)|'
                          r'do(?:es)?\s+not\s+(?:establish|confirm|specify)|'
                          r'no\s+(?:automatic|guaranteed)|requires?\s+(?:proof|assessment))\b', re.I)


def unsupported_safety_claims(answer, evidence, query=''):
    """Separate OSH duties, wage liabilities and conditional EC findings.

    These are conservative safeguards, not a substitute for claim entailment
    or medical/agency assessment. Never derive the legal answer from a scenario.
    """
    from routers.services.issue_evidence import safety_issue_contract
    from routers.services.legal_topic_matrix import supported_passage_concepts, source_passage_text
    if not safety_issue_contract(query):
        return []
    body = source_passage_text(evidence)
    errors = []
    for sentence in re.split(r'(?<=[.!?])\s+|\n+', answer or ''):
        if re.search(r'cannot\s+(?:confirm|verify)|not\s+(?:supported|confirmed|established)|'
                     r'do(?:es)?\s+not\s+(?:establish|confirm|specify)', sentence, re.I):
            continue
        conditional = bool(_LIMITATION.search(sentence))
        # A broad chemical-information rule does not establish every specific
        # document or monitoring obligation an LLM may associate with it.
        if re.search(r'\b(?:safety\s+data\s+sheets?|sds)\b', sentence, re.I) and not re.search(
            r'\b(?:safety\s+data\s+sheets?|sds)\b', body, re.I,
        ):
            errors.append('Chemical hazard-information evidence alone does not establish a specific Safety Data Sheet obligation.')
        if re.search(r'\b(?:must|shall|required)\b.{0,80}\bpost\w*\b.{0,70}\b(?:emergency\s+procedures|evacuation\s+plans)\b', sentence, re.I):
            specific_notice = (r'\b(?:post|display)\w*\b.{0,180}\b(?:emergency\s+procedures|evacuation\s+plans)\b|'
                               r'\b(?:emergency\s+procedures|evacuation\s+plans)\b.{0,180}\b(?:post|display)\w*\b')
            if not re.search(specific_notice, body, re.I):
                errors.append('A general safety-signage rule does not establish a specific emergency-procedure or evacuation-plan posting requirement.')
        ec_claim = re.search(r'\b(?:compensable|ec\s+benefits?|employees.?\s+compensation|'
                             r'(?:qualify|eligible)\s+for\s+(?:ec|compensation))\b', sentence, re.I)
        if ec_claim and not supported_passage_concepts({'text': body}, ['ec_eligibility_conditions']):
            errors.append('An EC benefit or eligibility claim needs a work-connection and eligibility provision, even when phrased conditionally.')
        if re.search(r'\b(?:sss|ecc|ec)\b.{0,60}\b(?:sickness|injury)\s+claims?\b', sentence, re.I) and not re.search(
            r'\b(?:sss|ecc|employees.?\s+compensation)\b.{0,200}\b(?:claims?|benefits?|medical|sickness)\b', body, re.I,
        ):
            errors.append('Claim-filing instructions need the relevant benefit and agency procedure; do not conflate ordinary SSS sickness benefits with EC claims.')
        if re.search(r'\b(?:compensable|entitled\s+to\s+(?:ec|employees.?\s+compensation)|'
                     r'will\s+(?:receive|qualify\s+for)\s+(?:ec|compensation)|'
                     r'is\s+(?:an?\s+)?occupational\s+disease)\b', sentence, re.I) and not conditional:
            errors.append('Symptoms or an incident do not establish a diagnosis or approved EC entitlement; explain eligibility conditions, not guaranteed compensation.')
        if re.search(r'\b(?:joint|solidar|shared\s+liability)\w*\b', sentence, re.I):
            if not supported_passage_concepts({'text': body}, ['osh_compliance_liability']):
                errors.append('Unpaid-wage liability does not establish OSH compliance or incident-damages liability.')
            elif not re.search(r'\b(?:compliance|safety\s+duties|osh\s+(?:duties|obligations))\b', sentence, re.I):
                errors.append('Limit shared OSH liability to the compliance obligation established by the passage; do not imply automatic liability for robbery damages.')
        if re.search(r'\b(?:will|automatically|always|must)\b.{0,65}\b(?:fined|work\s+stoppage|stop\s+work|shut\s+down)\b', sentence, re.I) and not conditional:
            errors.append('Do not present penalties or work stoppage as automatic; retain the enforcement prerequisites in the source.')
        if re.search(r'\b(?:he|she|you|worker|guard)\b.{0,30}\b(?:was|were|is)\s+injured\b', sentence, re.I) and not re.search(r'\binjur\w*\b', query, re.I) and not conditional:
            errors.append('Do not invent an injury that the user did not report; use an explicit conditional explanation.')
    return list(dict.fromkeys(errors))


def unsupported_rule_quantities(answer, evidence):
    """Check quantities against the same rule, even without a wage calculation.

    A percentage elsewhere in a document is not proof of the combined holiday
    and rest-day rate; a date elsewhere is not proof of a drill frequency.
    This deliberately checks only clearly identified rule quantities.
    """
    from routers.services.legal_topic_matrix import source_passage_text
    body = re.sub(r'\s+', ' ', source_passage_text(evidence)).lower()
    # Normalize common written frequencies without embedding a legal rate.
    for pattern, replacement in ((r'\bonce\b', '1 time'), (r'\btwice\b', '2 times'),
                                 (r'\bannually\b', 'per year'), (r'\byearly\b', 'per year'),
                                 (r'\bmonthly\b', 'per month')):
        body = re.sub(pattern, replacement, body)
    errors = []
    for sentence in re.split(r'(?<=[.!?])\s+|\n+', answer or ''):
        if re.search(r'cannot\s+(?:confirm|verify)|not\s+(?:supported|confirmed)|'
                     r'do(?:es)?\s+not\s+(?:specify|establish)', sentence, re.I):
            continue
        alternative_days = re.search(r'\bholidays?\b.{0,30}\bor\b.{0,15}\brest\s+days?\b|'
                                     r'\brest\s+days?\b.{0,30}\bor\b.{0,15}\bholidays?\b', sentence, re.I)
        if (re.search(r'\b(?:regular\s+holiday|holidays?)\b', sentence, re.I)
                and re.search(r'\brest\s+days?\b', sentence, re.I) and not alternative_days):
            rates = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent\b)', sentence, re.I)
            for rate in rates:
                # Check a local rule window containing BOTH day conditions.
                windows = [body[max(0, m.start()-250):m.end()+250] for m in re.finditer(r'\brest\s+day\b', body)]
                if not any(re.search(r'holiday', w) and re.search(r'(?<![\d.])' + re.escape(rate) + r'\s*(?:%|percent\b)', w) for w in windows):
                    errors.append('A holiday/rest-day percentage is not supported by the applicable combined-rate passage.')
                    break
        if re.search(r'\b(?:fire|evacuation|emergency)\s+drills?\b', sentence, re.I):
            normalized = sentence.lower()
            for pattern, replacement in ((r'\bonce\b', '1 time'), (r'\btwice\b', '2 times'),
                                         (r'\bannually\b', 'per year'), (r'\byearly\b', 'per year'),
                                         (r'\bmonthly\b', 'per month')):
                normalized = re.sub(pattern, replacement, normalized)
            frequency = re.search(r'\b(?:\d+\s*(?:times?|drills?)?\s*(?:a|per|every|each)\s+(?:year|month)|per\s+(?:year|month))\b', normalized)
            if frequency:
                windows = [body[max(0, m.start()-150):m.end()+250] for m in re.finditer(r'\b(?:fire|evacuation|emergency)\s+drills?\b', body)]
                number = re.search(r'\d+', frequency[0])
                unit = 'year' if 'year' in frequency[0] else 'month'
                if not any(unit in w and (not number or re.search(r'\b' + number[0] + r'\b', w)) for w in windows):
                    errors.append('The asserted drill frequency is not supported by a drill-specific passage.')
    return list(dict.fromkeys(errors))


def unsupported_scenario_claims(answer, evidence, query=''):
    errors = []
    from routers.services.scenario_retrieval import SSS_REMITTANCE_QUERY, DEDUCTION_QUERY
    if re.search(SSS_REMITTANCE_QUERY, query, re.I):
        if re.search(r'employers?[^.!?\n]{0,100}remit[^.!?\n]{0,60}quarterly|must first[^.!?\n]{0,120}employer[^.!?\n]{0,120}(?:complaint|complain)', answer, re.I):
            errors.append('Do not confuse employer payment deadlines with quarterly reports or invent an employer-first complaint requirement.')
    if re.search(r'cba|collective bargaining', query, re.I):
        if re.search(r'NLRC\s*\([^)]*Board|National Labor Relations Commission\s*\(NLRC\)[^.!?\n]{0,80}conciliation|NLRC[^.!?\n]{0,80}(?:261|250)\s*\(?c', answer, re.I):
            errors.append('The Board in the collective-bargaining procedure is NCMB, not NLRC.')
    if re.search(DEDUCTION_QUERY, query, re.I) and not re.search(r'deduct|kaltas|bawas', answer, re.I):
        errors.append('The answer does not address the wage deduction asked about.')
    if re.search(r'probation', query, re.I) and re.search(r'obligation[^.!?\n]{0,120}standards[^.!?\n]{0,160}only[^.!?\n]{0,160}apprenticeship', answer, re.I):
        errors.append('The apprenticeship duration exception does not establish an exemption from communicating standards.')
    if re.search(r'holiday', query, re.I):
        if re.search(r'same as first.?8.?hour rate[^.!?\n]{0,80}per hour', answer, re.I):
            errors.append('Convert daily pay to hourly pay and apply the overtime premium; do not use daily pay per hour.')
        for sentence in re.split(r'[.!?\n]+', answer):
            if re.search(r'(?:10\s*%|ten\s*percent)', sentence, re.I) and re.search(r'holiday', sentence, re.I) and not re.search(r'night|differential|not\b|hindi', sentence, re.I):
                errors.append('A 10% night differential is not the regular-holiday pay rule.')
    if re.search(r'phone|cellphone|mobile', query, re.I) and re.search(r'penalt|sanction', query, re.I):
        if re.search(r'(?:verify|check|penalty|fine)[^.!?\n]{0,180}(?:fine limits|art\.?\s*39|100,?000|20,?000)|statutory fine amounts', answer, re.I):
            errors.append('Do not apply government enforcement fine ranges to employee discipline for phone use.')
    if re.search(r'final[-\s]+pay', query, re.I):
        if re.search(r'(?:any|all)\s+unused\s+(?:vacation|leave)(?:\s+days)?', answer, re.I) and not re.search(r'convertib|company\s+policy|contract|cba|subject\s+to|if\s+eligible', answer, re.I):
            errors.append('Unused leave is not universally cash-convertible; distinguish statutory SIL from conditional policy benefits.')
    if re.search(r'resign(?:ation)?', answer, re.I) and re.search(
        r'(?:wala nang dapat bayaran|owes? (?:you )?nothing|no further (?:pay|benefits) (?:is|are) owed)[^.\n]{0,45}(?:kumpanya|company|employer)|'
        r'(?:kumpanya|company|employer)[^.\n]{0,45}(?:owes? (?:you )?nothing)', answer, re.I,
    ):
        errors.append('Resignation does not establish that all accrued monetary claims disappear; the answer overstates the effect of resignation.')
    # A number in a source about some other subject is not resignation evidence.
    notice = r"(?:two[- ]weeks?|14\s+days|duha\s+ka\s+semana|dalawang\s+linggo)"
    if re.search(notice, answer, re.I) and re.search(r"resign", answer, re.I):
        if not re.search(r"resign.{0,240}" + notice + "|" + notice + r".{0,240}resign", evidence, re.I | re.S):
            errors.append("The resignation notice duration is not supported by the retrieved text.")
    recall_claim = r"(?:cannot|can't|must not|hindi\s+puwede|hindi\s+maaari).{0,55}(?:force|require|pilit)"
    if re.search(recall_claim, answer, re.I) and re.search(r'leave|\bsil\b', answer + ' ' + query, re.I):
        from routers.services.scenario_retrieval import leave_recall_passage
        if not leave_recall_passage(evidence):
            errors.append("Leave entitlement evidence does not establish an approved-leave recall rule.")
    if re.search(r'DOLE\s+(?:hotline|helpline)\s*[:( -]*1555\b', answer, re.I):
        errors.append('The DOLE hotline is 1349, not 1555. Omit contact numbers unsupported by the sources.')
    if re.search(r'certification\s+election[^.!?\n]{0,130}(?:must|shall)\s+be\s+held', answer, re.I):
        errors.append('Distinguish the period for filing a representation petition from a mandatory date for holding an election; CBA expiration does not require an election.')
    if re.search(r'without\s+(?:any\s+)?risk(?:ing)?\s+(?:dismissal|losing|termination)|guarantee[^.!?\n]{0,50}(?:job|dismissal)', answer, re.I):
        errors.append('Explain the legal protection and its conditions; do not guarantee that retaliation or dismissal cannot happen.')
    if re.search(r'\b(?:freelancer|independent\s+contractor)\b', query, re.I):
        if re.search(r'labor[- ]only\s+contracting', answer, re.I) and not re.search(r'four[- ]fold|right\s+of\s+control|control\s+test|means\s+and\s+methods', answer, re.I):
            errors.append('A direct freelancer employment-status question requires the employer-employee relationship test; intermediary contracting alone does not answer it.')
    if re.search(r'(?:notice|due[- ]process|hearing)', query, re.I) and re.search(r'dismiss|terminat|awol', query, re.I):
        if re.search(r'cannot\s+validly\s+dismiss[^.!?\n]{0,120}(?:notice|heard)|(?:no|without)[^.!?\n]{0,35}notice[^.!?\n]{0,80}(?:automatically|always)[^.!?\n]{0,40}illegal', answer, re.I):
            errors.append('Distinguish substantive cause from procedural due process; deficient notice does not automatically make a justified dismissal invalid.')
    if re.search(r'\b(?:buntis|pregnan\w*)\b', query, re.I) and re.search(r'promot', query, re.I):
        if re.search(r'(?:promot|reinstat|back.?pay|damages)', answer, re.I) and not re.search(r'promotion|discrimination|equal\s+opportunit', evidence, re.I):
            errors.append('Pregnancy dismissal evidence alone does not establish promotion remedies or a general discrimination holding; limit the answer to the supported rule.')
    return errors


def ambiguous_compound_reply(query, history):
    if not re.fullmatch(r"\s*(?:yes|no|hindi|oo|indi|huo)[.!?\s]*", query, re.I):
        return False
    previous = next((m.get('content', '') for m in reversed(history or []) if m.get('role') == 'assistant'), '')
    return bool(re.search(r"(?:reason|dahilan|rason).{0,160}(?:notice|hearing)", previous, re.I | re.S))
