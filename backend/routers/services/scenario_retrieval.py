"""Preserve clarification context and require evidence for the actual issue.

Lexical responsiveness is a necessary filter, not proof of legal entailment.
Only user facts enter the reconstructed query; assistant answers are not evidence.
"""
import re


LEAVE_CHOICE = re.compile(r'\s*(?:sick leave|service incentive leave|vacation leave|maternity(?:\s*/\s*paternity)?(?: leave)?|paternity leave|solo.parent leave|parental leave|other leave)\s*[.!]?\s*', re.I)


WORK_INJURY_QUERY = (
    r'\b(?:(?:work[-\s]?related|workplace|occupational)\s+'
    r'(?:injur(?:y|ies)|accidents?|illness(?:es)?|diseases?)|'
    r'work\s+accidents?|(?:injured|hurt|injur(?:y|ies)|accidents?)\s+'
    r'(?:at|during|while\s+at)\s+work|'
    r'(?:nasaktan|aksidente|disgrasya)\s+sa\s+(?:trabaho|obra))\b'
)
INJURY_MEDICAL_ISSUE = (
    r'\b(?:hospitals?|hospitali[sz](?:ed|ation)|medical|treatment|'
    r'bills?|expenses?|reimburs\w*|gamot|ospital|gastos)\b'
)


def is_work_injury_query(query):
    return bool(re.search(WORK_INJURY_QUERY, query or '', re.I))


def is_answerable_work_injury_query(query):
    """A known workplace incident plus a concrete issue can retrieve rules.

    Dates and exact location are not prerequisites for a conditional explanation
    of medical costs or reporting. A bare incident with no requested issue is
    still an intake question; personal calculations are checked by the caller.
    """
    return is_work_injury_query(query) and bool(re.search(
        INJURY_MEDICAL_ISSUE + r'|\b(?:claims?|report\w*|benefits?|compensation|'
        r'rights?|karapatan|reklamo)\b|\bwhat\s+(?:should|can|do)\s+'
        r'(?:i|we|he|she|they)\s+do\b', query or '', re.I,
    ))


EXIT_QUERY = r'(?:fire|emergency)\s+exit|(?:blocked|locked|padlocked)\s+exit|nakaharang.{0,30}daanan'
DEDUCTION_QUERY = r'(?:binabawas|kinaltas|kinakaltas|deduct\w*).{0,80}(?:sahod|salary|wage)|(?:sahod|salary|wage).{0,80}(?:deduct\w*|binabawas|kinaltas)'
SSS_REMITTANCE_QUERY = r'\bsss\b.{0,180}(?:deduct\w*|contribut\w*|remit\w*|hulog)|(?:contribut\w*|remit\w*|hulog).{0,100}\bsss\b'
SSS_EMPLOYER_REMITTANCE_QUERY = (
    r'\bsss\b.{0,220}\b(?:remit\w*|remittance|posted|proof|payment\s+history|'
    r'contribution\s+history|not\s+paid|unpaid)\b|'
    r'\b(?:remit\w*|remittance|posted|proof|payment\s+history|contribution\s+history|'
    r'not\s+paid|unpaid)\b.{0,220}\bsss\b|'
    r'\bsss\b.{0,220}\b(?:walang\s+(?:bagong\s+)?(?:contributions?|hulog)|'
    r'hindi\s+(?:na)?(?:remit\w*|hulog|bayad)|hindi\s+posted)\b'
)
PROMOTION_HARASSMENT_QUERY = r'(?:supervisor|boss|manager|promotion|promote).{0,150}(?:makikipag[- ]?date|sexual favor|date (?:him|her|them))'
AWOL_QUERY = r'\bawol\b|absen(?:t|ce)\s+without\s+leave|abandon(?:ment|ed|ing)?\s+(?:of\s+)?(?:work|employment|job)|(?:terminat\w*|dismiss\w*|tanggal\w*).{0,100}\babsent\b'
PREGNANCY_DISCRIMINATION_QUERY = r'(?:buntis|pregnan\w*).{0,150}(?:promot\w*|discriminat\w*|pinapahiya|insult\w*)|(?:pinapahiya|discriminat\w*|insult\w*|promot\w*).{0,150}(?:buntis|pregnan\w*)'
EMPLOYMENT_RELATIONSHIP_QUERY = r'freelanc\w*|independent\s+contractor'
UNRETURNED_PROPERTY_QUERY = r'(?:ayaw|hindi|refus\w*|fail\w*|not).{0,65}(?:ibalik|return).{0,90}(?:laptops?|gadgets?|equipment|company\s+property)|(?:keeps?|kept|retained|withhold\w*).{0,60}company.{0,40}(?:laptops?|gadgets?|equipment|property)'
AWOL_EVIDENCE = r'abandonment\s+of\s+(?:work|employment)|termination\s+by\s+employee|hold\s+the\s+employee\s+liable\s+for\s+damages|employee\s+may\s+put\s+an\s+end\s+to\s+the\s+relationship|inhuman\s+and\s+unbearable\s+treatment\s+accorded\s+the\s+employee'


def is_employee_departure_query(query):
    return bool(re.search(AWOL_QUERY, query or '', re.I))


def property_remedy_evidence(passages):
    body = '\n'.join(str(p).split('\nText:\n')[-1] for p in passages)
    return bool(re.search(r'replevin|(?:return|recover|recovery).{0,80}(?:company|employer).{0,30}(?:property|equipment)|(?:estafa|qualified\s+theft).{0,150}(?:entrust|property|possession)', body, re.I))


def everyday_topic(query):
    """Specific user scenarios precede broad words such as salary/emergency."""
    from routers.services.issue_evidence import safety_issue_contract
    if safety_issue_contract(query):
        return 'workplace_safety'
    if is_work_injury_query(query):
        return 'workplace_injury'
    if re.search(PREGNANCY_DISCRIMINATION_QUERY, query or '', re.I):
        return 'discrimination'
    if re.search(EMPLOYMENT_RELATIONSHIP_QUERY, query or '', re.I) and re.search(r'employee|schedule|supervis|time[- ]?in|control', query or '', re.I):
        return 'regular_employment'
    if re.search(r'kasambahay|domestic\s+worker', query or '', re.I):
        return 'domestic_workers'
    if is_employee_departure_query(query):
        return 'termination'
    for pattern, topic in ((EXIT_QUERY, 'workplace_safety'),
                           (SSS_REMITTANCE_QUERY, 'social_security'),
                           (PROMOTION_HARASSMENT_QUERY, 'sexual_harassment'),
                           (DEDUCTION_QUERY, 'wage_deductions')):
        if re.search(pattern, query or '', re.I):
            return topic
    return ''


def is_domestic_rest_query(query):
    return bool(re.search(r'kasambahay|domestic\s+workers?', query or '', re.I) and
                re.search(r'rest|pahinga', query or '', re.I))


# Necessary issue-level evidence, not a substitute for legal entailment.
# Search vocabulary is kept beside its evidence check to avoid routing a
# specific question to a broad but legally unrelated provision.
ISSUE_RULES = (
    (r'\b(?:file|filing|submit|lodge)\s+(?:a\s+)?complain(?:t)?\b|\bcomplain(?:t)?\s+against\s+(?:my\s+|the\s+)?(?:employer|company|boss)\b',
     'labor complaint filing procedure request for assistance Single Entry Approach SEnA conciliation jurisdiction DOLE NLRC',
     r'request\s+for\s+assistance|single[-\s]+entry\s+approach|\bsena\b|(?:fil(?:e|ing)|lodg(?:e|ing)|submit)[^.!?\n]{0,100}\bcomplaints?\b|\bcomplaints?\b[^.!?\n]{0,100}(?:filed|lodged|submitted|jurisdiction)',
     'The retrieved passages do not establish complaint-filing procedures or jurisdiction. General employment terms alone cannot support filing instructions.'),
    (AWOL_QUERY,
     'abandonment of work absence without leave Omnibus Rules notice last known address Labor Code termination by employee written notice employer hold employee liable damages exceptions just causes',
     AWOL_EVIDENCE,
     'No operative abandonment or employee-initiated termination provision was retrieved. Telecommuting, conciliation and apprenticeship rules alone do not answer this scenario.'),
    (EXIT_QUERY,
     'RA 11058 Section 6 right refuse unsafe work Section 7 report hazards DOLE imminent danger emergency exit unobstructed escape',
     r'right\s+to\s+report|report.{0,100}hazards?|right.{0,50}refus.{0,50}(?:work|unsafe)|emergency\s+exits?',
     'Livelihood and emergency-employment programs do not establish workplace hazard-reporting rights.'),
    (SSS_EMPLOYER_REMITTANCE_QUERY,
     'RA 11199 Section 22 remittance contributions employer delinquent failure pay SSS Section 19 monthly contributions',
     r'employer[^.!?\n]{0,400}(?:remit|contribut)|(?:remit|contribut)[^.!?\n]{0,400}employer',
     'Self-employed contribution rules alone do not establish an employer remittance obligation.'),
    (DEDUCTION_QUERY,
     'Labor Code Article 113 Wage Deduction no employer deductions from wages except authorized law regulations',
     r'wage\s+deduction|deductions?\s+from\s+(?:the\s+)?wages',
     'Discipline and work-stoppage pay alone do not establish lawful deductions from wages.'),
    (PROMOTION_HARASSMENT_QUERY,
     'RA 7877 Section 3 promotion sexual favors Section 4 committee decorum investigation RA 11313 workplace harassment',
     r'sexual\s+(?:harassment|favo[u]?rs?)|committee\s+on\s+decorum',
     'No operative workplace sexual-harassment provision was retrieved.'),
    (r'\bfinal[-\s]+pay\b',
     'final pay release unpaid earned salary Labor Advisory 06 2020 SEnA assistance',
     r'final\s+pay|last\s+pay',
     'Final-pay rules were not retrieved; employer dismissal grounds do not establish final-pay rights.'),
    (r'\bquitclaim\b',
     'quitclaim waiver voluntary informed reasonable credible consideration Supreme Court labor claims',
     r'quitclaims?|waivers?',
     'No operative quitclaim or waiver rule was retrieved.'),
    (r'(?:phone|cellphone|mobile).{0,160}(?:penalt|sanction|company rules)|(?:penalt|sanction|pinagmumulta|multa).{0,160}(?:phone|cellphone)',
     'employee discipline reasonable lawful orders made known willful disobedience proportionality wage deductions',
     r'willful\s+disobedience|disciplinary|lawful\s+orders?|wage\s+deductions?|deductions?\s+from\s+(?:the\s+)?wages',
     'Government enforcement fines do not establish lawful employee disciplinary penalties.'),
    (r'(?:aksidente|accident|injur).{0,100}(?:papasok|pauwi|commut|way to work|way home)',
     'Employees Compensation ECC going to coming from work usual route accident disability eligibility SSS',
     r'going\s+to.{0,100}coming\s+from|usual\s+route|commut(?:ing|e)\s+accident',
     'No commuting-accident coverage rule was retrieved; exemption from garnishment is not benefit eligibility.'),
    (r'(?:hindi.{0,15}tinanggap|not hired|rejected).{0,100}(?:babae|lalaki|gender|sex|woman|man)',
     'sex gender discrimination hiring recruitment occupational qualifications equal opportunity',
     r'(?:sex|gender|discriminat)\w*[^.!?\n]{0,240}(?:hiring|recruitment|applicant|employment\s+opportunit)|(?:hiring|recruitment|applicant)[^.!?\n]{0,240}(?:sex|gender|discriminat)',
     'Marriage restrictions alone do not establish rules on sex-based hiring.'),
    (r'(?:meeting|makipag[-\s]?meeting|refus|ayaw|negotia|bargain\s+in\s+good\s+faith).{0,100}\bcba\b|\b(?:cba|collective\s+bargaining)\b.{0,100}(?:refus|ayaw|negotiat|meeting|good\s+faith)',
     'Labor Code Article 261 procedure collective bargaining ten days reply Article 263 duty good faith Board means National Conciliation and Mediation Board NCMB',
     r'duty\s+to\s+bargain|bargain.{0,50}good\s+faith|good\s+faith.{0,50}bargain|procedure\s+in\s+collective\s+bargaining',
     'Union registration or general bureau jurisdiction alone does not establish the duty to bargain.'),
)


def resolve_clarification_query(query, history):
    from routers.services.followup_context import resolve_followup_context
    if not LEAVE_CHOICE.fullmatch(query or ''):
        resolved = resolve_followup_context(query, history)
        if resolved != query:
            return resolved
    query = resolve_wage_followup(query, history)
    messages = list(history or [])
    if messages and messages[-1].get('role') == 'user' and messages[-1].get('content', '').strip() == query.strip():
        messages.pop()
    if re.fullmatch(r'\s*(?:i (?:already )?submitted|already submitted|naisumite ko na)\s*[.!]?\s*', query or '', re.I):
        if messages and messages[-1].get('role') == 'assistant' and re.search(r'submitted.{0,35}resignation|resignation letter|resignation\s+letter', str(messages[-1].get('content') or ''), re.I):
            original = next((str(m.get('content') or '') for m in reversed(messages[:-1]) if m.get('role') == 'user'), '')
            if re.search(r'resign|final[-\s]+pay', original, re.I):
                return f'{original[:700]}\nClarification from user: {query.strip()}'
    if not LEAVE_CHOICE.fullmatch(query or ''):
        return query
    messages = list(history or [])
    # Some callers include the current user turn in history.
    if messages and messages[-1].get('role') == 'user' and messages[-1].get('content', '').strip() == query.strip():
        messages.pop()
    # A second choice corrects the first even after an answer/refusal. Walk only
    # this contiguous clarification chain; never borrow facts from AI answers.
    saw_leave_prompt = False
    for message in reversed(messages):
        content = str(message.get('content') or '')
        if message.get('role') == 'assistant':
            prompt = str(message.get('clarification_question') or content)
            saw_leave_prompt |= bool(message.get('intent') == 'needs_clarification' or re.search(r'which leave|anong leave|ano nga leave', prompt, re.I))
            continue
        if message.get('role') != 'user':
            continue
        if LEAVE_CHOICE.fullmatch(content):
            continue
        if saw_leave_prompt and re.search(r'\b(?:leave|sil)\b', content, re.I):
            return f'{content[:700]}\nClarified leave type: {query.strip()}'
        break
    return query


def resolve_wage_followup(query, history):
    """Carry user-supplied wage context across a sector/location follow-up only."""
    from routers.services.rag_policy import wage_order_query_details
    text = (query or '').strip()
    if not re.match(r'^(?:how about|what about|and for|i mean|paano naman|pila naman)\b', text, re.I):
        return query
    messages = [m for m in history or [] if m.get('role') == 'user']
    if messages and str(messages[-1].get('content', '')).strip() == text:
        messages.pop()
    if not messages:
        return query
    previous = str(messages[-1].get('content') or '')
    if len(messages) > 1:
        previous = resolve_wage_followup(previous, messages[-4:-1])
    # No search through unrelated old turns and no facts copied from answers.
    prior = wage_order_query_details(previous)
    current = wage_order_query_details(text)
    if not prior.get('is_wage_order_query'):
        return query
    if not current.get('region_code') and not re.search(r'wage|salary|rate|domestic workers?|kasambahay', text, re.I):
        return query
    location = '' if current.get('region_code') else str(prior.get('region_name') or prior.get('region_code') or '')
    sector = ''
    if re.search(r'domestic workers?|kasambahay', previous, re.I) and not re.search(r'workers?|kasambahay|agricultur|commercial|factory', text, re.I):
        sector = 'domestic workers'
    return f'minimum wage {location} {sector} {text}'.strip()


def domestic_wage_passage(text):
    body = str(text or '').split('\nText:\n')[-1]
    # A mention in an exclusion clause is not affirmative wage evidence.
    for sentence in re.split(r'[\n.!?]+', body):
        if (re.search(r'domestic\s+workers?|kasambahays?|household\s+(?:helpers?|workers?)', sentence, re.I)
                and not re.search(r'not cover|exclud|except|not apply', sentence, re.I)):
            return True
    return False


def is_leave_recall_query(query):
    return bool(re.search(r'\b(?:leave|sil)\b', query or '', re.I) and
                re.search(r'approved?|inaprubahan|gin.aprubahan', query or '', re.I) and
                re.search(r'work|trabahu|trabaho|pumasok|pinapapasok|papasok|pagtrabahuhin|recall|cancel|balik|ubra', query or '', re.I))


def scenario_search_terms(query):
    from routers.services.issue_evidence import safety_issue_contract
    contract = safety_issue_contract(query)
    if contract:
        return contract['search_terms']
    if is_work_injury_query(query):
        return ('Employees Compensation ECC SSS GSIS work-related injury occupational '
                'disease accident reporting claim benefits medical services hospital '
                'treatment expenses reimbursement employer obligations')
    if re.search(PREGNANCY_DISCRIMINATION_QUERY, query or '', re.I):
        return 'Labor Code discrimination women promotion pregnancy prohibited acts equal opportunities RA 9710 Magna Carta of Women'
    if re.search(EMPLOYMENT_RELATIONSHIP_QUERY, query or '', re.I):
        return 'employer employee relationship four fold test selection engagement payment wages dismissal control means methods independent contractor economic dependence Supreme Court'
    if re.search(r'kasambahay|domestic\s+worker', query or '', re.I) and re.search(r'resign|final[- ]pay|termination', query or '', re.I):
        return 'RA 10361 domestic worker resignation termination notice unpaid wages thirteenth month pay service incentive leave non cumulative non convertible cash'
    if re.search(r'government|civil service|kawani|raccs', query or '', re.I) and re.search(r'administrative|complaint|suspend|penalt', query or '', re.I):
        return '2025 RACCS administrative complaint formal charge answer under oath right counsel formal investigation preventive suspension grounds duration appeal'
    matched = [terms for pattern, terms, _evidence, _reason in ISSUE_RULES if re.search(pattern, query or '', re.I)]
    if re.search(UNRETURNED_PROPERTY_QUERY, query or '', re.I):
        matched.append('return recovery employer company property entrusted equipment civil action damages criminal liability')
    if matched:
        return ' '.join(matched)
    if re.search(r'regular\s+holiday', query or '', re.I) and re.search(r'compute|worked|nagtrabaho|kwenta', query or '', re.I):
        return 'Labor Code Article 94 regular holiday twice regular rate Article 87 overtime hourly thirty percent Omnibus Rules holiday rest day 260 percent 338 percent'
    if re.search(r'probation', query or '', re.I) and re.search(r'standards|performance', query or '', re.I):
        return 'Labor Code Article 296 Omnibus Rules Section 6 probationary standards made known at engagement where no standards deemed regular employee just authorized cause'
    if is_domestic_rest_query(query):
        return 'RA 10361 Section 21 weekly rest period twenty four consecutive hours seven days written agreement waiver accumulation'
    if re.search(r'fire\s+exit|nakaharang.{0,30}daanan|blocked\s+exit', query or '', re.I):
        return 'fire exits unobstructed escape routes safety signage report hazards safety officer DOLE worker right report accidents RA 11058'
    if re.search(r'telecommut|work[-\s]+from[-\s]+home', query or '', re.I):
        return 'RA 11165 DO 237 22 revised telecommuting rules voluntary mutual consent written program agreement non diminution'
    if re.search(r'saan.{0,65}tulong|where.{0,50}(?:help|complain)|humingi ng tulong', query or '', re.I):
        return 'DOLE Single Entry Approach SEnA request for assistance conciliation unpaid wages complaint'
    if re.search(r'hindi na (?:nila )?kailangan|no longer needed|redundan|retrench', query or '', re.I):
        return 'termination authorized causes redundancy retrenchment Article 298 notice separation pay just causes Article 297'
    if re.search(r'\b(?:union|unyon|labor\s+organization|self[-\s]?organization)\b', query or '', re.I) and re.search(r'\b(?:join|form|organize|organise|sumali|bumuo|makasulod|mag[-\s]?organisa)\b', query or '', re.I):
        return 'Labor Code Book V right to self organization employees probationary employment union join form organize coverage'
    if re.search(r'\b(union|unyon)\b', query or '', re.I) and re.search(r'tanggal|fired|dismiss|retaliat', query or '', re.I):
        return 'right self organization unfair labor practice employer discrimination union membership retaliation dismissal'
    if re.search(r'resign(?:ation)?|quitclaim', query or '', re.I):
        return 'voluntary resignation constructive dismissal quitclaim waiver accrued wages final pay rights'
    if re.search(r'malalaswa|malaswa|sexual harassment|lewd', query or '', re.I):
        return 'sexual harassment supervisor sexual favors promotion RA 7877 RA 11313 CODI complaints'
    if is_leave_recall_query(query):
        return 'approved leave recall cancellation employer require return to work company policy'
    return ''


def leave_recall_passage(text):
    # Strip indexed metadata headers: a topic label is not operative evidence.
    body = str(text or '').split('\nText:\n')[-1]
    # Return-to-work accommodation after illness is not an employer recall rule.
    return bool(re.search(
        r'\bleave\b[^.!?\n]{0,120}\b(?:recall|cancel(?:lation|led|ed)?)\b'
        r'|\b(?:recall|cancel(?:lation|led|ed)?)\b[^.!?\n]{0,120}\bleave\b'
        r'|\b(?:require|compel|order)(?:d|s)?\b[^.!?\n]{0,80}(?:work|report)[^.!?\n]{0,50}(?:during|while on)[^.!?\n]{0,30}\bleave\b',
        body, re.I))


def is_final_pay_component_calculation(query):
    """A benefit amount is distinct from final-pay release or collection rights."""
    text = query or ''
    return bool(
        re.search(r'\bfinal[-\s]+pay\b', text, re.I)
        and re.search(r'13th[-\s]*month|thirteenth[-\s]*month|overtime|holiday\s+pay|'
                      r'service\s+incentive\s+leave|separation\s+pay|retirement\s+pay', text, re.I)
        and re.search(r'how\s+much|comput\w*|calculat\w*|\bamount\b|magkano|\bpila\b', text, re.I)
        and not re.search(r'\bwhen\b|deadline|releas\w*|delay\w*|withh\w*|not\s+paid|'
                          r'unpaid|refus\w*|complain\w*|how\s+long|kailan|san-o|reklamo', text, re.I)
    )


def scenario_evidence_gap(query, passages):
    body = '\n'.join(str(p).split('\nText:\n')[-1] for p in passages)
    if (is_work_injury_query(query)
            and re.search(INJURY_MEDICAL_ISSUE, query or '', re.I)
            and not re.search(r'\bmedical\s+(?:services?|benefits?|care|treatment|expenses?)\b|'
                              r'\bhospital(?:ization|isation|\s+(?:bills?|expenses?|treatment))\b|'
                              r'\breimburs\w*\b', body, re.I)):
        return ('No medical-care or expense provision was retrieved for this workplace '
                'injury. General safety or employment-status rules alone cannot establish '
                'who pays hospital bills or how reimbursement works.')
    for pattern, _terms, evidence, reason in ISSUE_RULES:
        if pattern == DEDUCTION_QUERY and re.search(r'\b(?:sss|philhealth|pag[-\s]?ibig|hdmf)\b', query or '', re.I):
            # Statutory contribution deductions are checked against the
            # agency's current contribution schedule, not the generic Article
            # 113 wage-deduction rule used for unrelated payroll deductions.
            continue
        if pattern == r'\bfinal[-\s]+pay\b' and is_final_pay_component_calculation(query):
            # The topic validator still requires the specific benefit's evidence.
            continue
        if re.search(pattern, query or '', re.I) and not re.search(evidence, body, re.I):
            return reason
    if is_domestic_rest_query(query) and not re.search(r'weekly\s+rest|(?:24|twenty[-\s]+four)\s+consecutive\s+hours', body, re.I):
        return 'A general list of domestic-worker rights does not establish the weekly rest rule.'
    if re.search(r'telecommut|work[-\s]+from[-\s]+home', query or '', re.I) and re.search(r'sign|pumirma|agreement', query or '', re.I):
        # Grandfathering of old arrangements is not consent for a new agreement.
        operative = '\n'.join(s for s in re.split(r'[.!?\n]+', body) if not re.search(r'existing|prior\s+to|before\s+the|predat', s, re.I))
        if not re.search(r'mutual\s+consent|voluntar(?:y|ily)[^.!?\n]{0,180}(?:agree|offer|program)|(?:agree|offer|program)[^.!?\n]{0,180}voluntar', operative, re.I):
            return 'No operative consent rule for a new telecommuting agreement was retrieved; an existing-agreements savings clause alone is insufficient.'
    if re.search(r'hindi na (?:nila )?kailangan|no longer needed|redundan|retrench', query or '', re.I):
        if not re.search(r'authorized\s+causes?|redundan|retrench', body, re.I):
            return 'The retrieved text does not establish the relevant authorized-cause rules. Just-cause dismissal grounds alone cannot resolve a redundancy or no-longer-needed scenario.'
    if re.search(r'\b(union|unyon)\b', query or '', re.I) and re.search(r'tanggal|fired|dismiss|retaliat', query or '', re.I):
        if not re.search(r'unfair labor practice|discriminat|interfere|restrain|coerce|retaliat', body, re.I):
            return 'The retrieved text does not establish protection against union retaliation. Union registration or certification procedures alone do not answer the dismissal question.'
    if re.search(r'resign(?:ation)?|quitclaim', query or '', re.I) and re.search(r'rights|habulin|waiv|claim|karapatan', query or '', re.I):
        if not re.search(r'quitclaim|waiver|constructive dismissal|resign(?:ation)?', body, re.I):
            return 'The retrieved passages do not establish resignation or waiver of claims. Employer dismissal grounds alone cannot answer this question.'
    if re.search(r'malalaswa|malaswa|sexual harassment|lewd', query or '', re.I):
        if not re.search(r'sexual\s+(?:harassment|favo[u]?rs?|advances)|committee on decorum', body, re.I):
            return 'No operative sexual-harassment provision was retrieved. Recruitment prohibitions or execution procedures do not establish the answer.'
    if (re.search(r'domestic workers?|kasambahay', query or '', re.I)
            and re.search(r'wage|salary|sahod|sweldo', query or '', re.I)
            and not any(domestic_wage_passage(p) for p in passages)):
        return 'No retrieved operative passage establishes wages for domestic workers. General daily wage orders cannot establish a kasambahay wage.'
    if is_leave_recall_query(query) and not any(leave_recall_passage(p) for p in passages):
        return 'The retrieved passages do not establish rules on recalling or cancelling approved leave; leave entitlement alone does not answer this question.'
    return ''


def incomplete_issuance_query(query):
    return bool(re.fullmatch(r'\s*(?:DOLE\s+)?(?:Department\s+Order|D\.?O\.?)\s*(?:No\.?\s*)?\d{1,3}\s*[?.]?\s*', query or '', re.I))


def unidentified_advisory_query(query):
    return bool(re.search(r'advisory|issuance', query or '', re.I) and re.search(
        r'wala.{0,55}(?:kopya|number)|no.{0,25}(?:copy|number)|(?:don.t|do not).{0,25}(?:have|know).{0,30}(?:copy|number)', query or '', re.I))
