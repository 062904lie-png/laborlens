"""Issue-level retrieval contracts, not prewritten legal answers.

The rules describe evidence to look for. Matching them does not establish a
legal entitlement, medical diagnosis, or the applicability of an exception.
Unknown issues remain unverified rather than inheriting a similarity label.
"""
import re


EMERGENCY = r'\b(?:fire(?:\s+and\s+earthquake)?\s+drills?|evacuation\s+(?:drills?|plans?)|emergency\s+(?:procedures?|plans?|preparedness)|simulacrum|fire\s+safety)\b'
PROTECTIVE = r'\b(?:gloves?|masks?|respirators?|protective\s+(?:equipment|gear|clothing)|ppe)\b'
CHEMICAL = r'\b(?:chemical\s+(?:solvents?|exposure|handling)|solvents?|toxic\s+(?:chemicals?|fumes)|hazardous\s+(?:chemicals?|substances)|safety\s+data\s+sheets?)\b'
SITE_DANGER = r'\b(?:robbed|robber(?:y|ies)|gunpoint|unlit|poor\s+lighting|no\s+lighting|violent\s+incidents?|unsafe\s+(?:post|site|workplace))\b'
WORK_CONTEXT = r'\b(?:employ(?:er|ee|ment)|workers?|workplace|factory|company|agency|guard|on\s+duty|at\s+work|trabaho|kumpanya|manggagawa)\b'


def safety_issue_contract(query):
    """Preserve multiple requested issues; occupation is not the legal issue."""
    text = re.sub(r'\s+', ' ', query or '')
    if not re.search(WORK_CONTEXT, text, re.I):
        return {}
    required, issues, terms = [], [], []
    if re.search(EMERGENCY, text, re.I):
        issues.append('emergency_preparedness')
        required.append('emergency_preparedness')
        terms.append('workplace emergency preparedness response evacuation fire drill safety warning signage RA 11058 Sections 4 9 12 Fire Code implementing rules')
        if re.search(r'\b(?:drills?|simulacrum)\b', text, re.I):
            required.append('fire_drill_requirement')
        if re.search(r'\b(?:post(?:ed|ing|s)?|notices?|signage|display)\b', text, re.I):
            required.append('safety_notice_requirement')
        if re.search(r'\b(?:small|big|large|size|employees|workers|establishments)\b', text, re.I):
            required.append('osh_establishment_coverage')
    if re.search(PROTECTIVE, text, re.I) or re.search(CHEMICAL, text, re.I):
        issues.append('hazard_protection')
        required.extend(['ppe_provision', 'ppe_cost', 'hazard_assessment'])
        terms.append('hazard appropriate personal protective equipment free of charge risk assessment RA 11058 Sections 4 5 8 OSH Standards')
        if re.search(CHEMICAL, text, re.I):
            required.append('chemical_safety_information')
            terms.append('chemical hazard information safety data sheet SDS training exposure controls health surveillance medical examination')
        if re.search(r'\b(?:symptoms?|irritation|headaches?|sick|illness|medical|health\s+surveillance)\b', text, re.I):
            required.append('occupational_health_protection')
    if re.search(SITE_DANGER, text, re.I):
        issues.append('site_safety')
        required.extend(['safe_and_healthful_workplace', 'hazard_assessment'])
        terms.append('employer safe healthful workplace site hazard risk assessment emergency response workplace violence RA 11058 Section 4')
        if re.search(r'\b(?:lighting|unlit)\b', text, re.I):
            required.append('workplace_lighting')
            terms.append('OSH Standards workplace adequate illumination lighting')
        if re.search(r'\b(?:client|contractor|joint|solidary|share[ds]?\s+liability)\b', text, re.I):
            required.append('osh_compliance_liability')
            terms.append('RA 11058 Section 21 joint solidary liability compliance employer contractor persons control supervise work')
    if not issues:
        return {}
    # Compensation is a separate issue, not a conclusion implied by exposure
    # or robbery. Search eligibility only when the user actually asks about it.
    if re.search(r'\b(?:compensat\w*|ecc|claims?|medical\s+bills?|hospital\s+bills?)\b', text, re.I):
        required.append('ec_eligibility_conditions')
        terms.append('Employees Compensation work connection eligibility occupational disease increased risk medical benefits SSS ECC')
    required = list(dict.fromkeys(required))
    return {'topic': 'workplace_safety', 'task': 'issue_specific_safety',
            'issues': issues, 'required': required, 'essential': required[:1],
            'search_terms': ' '.join(terms)}


_STOP_WORDS = set('''a an the is are was were be been being do does did have has had
    i my me we our you your he his she her they their it its this that these those
    for from to of on in at by with without and or but if as than then only all
    what which who how can could should would will must shall may not no yes
    please tell explain generally correct legal law under rights employer
    employee workers worker company employment work workplace labor question
    information specific receive give given get gets know want wants'''.split())


def issue_terms(query):
    """Distinctive terms for conservative unknown-topic responsiveness only."""
    return {word for word in re.findall(r'[a-z]{3,}', (query or '').lower())
            if word not in _STOP_WORDS}


def unclassified_evidence_responsive(query, passages):
    """Lexical responsiveness is necessary, never sufficient for direct match.

    Require distinctive issue terms in the same operative paragraph, rather
    than matching a title, document metadata, or words across unrelated pages.
    """
    from routers.services.legal_topic_matrix import source_passage_text
    terms = issue_terms(query)
    if not terms:
        return False
    threshold = min(2, len(terms))
    for passage in passages:
        body = source_passage_text(passage)
        for paragraph in re.split(r'\n\s*\n', body):
            for start in range(0, len(paragraph), 700):
                window = paragraph[start:start + 900]
                operative = re.search(r'\b(?:shall|must|entitled|prohibited|required|may|means|refers\s+to|defined\s+as)\b', window, re.I)
                if operative and len(terms & issue_terms(window)) >= threshold:
                    return True
    return False


def incomplete_contract(profile):
    return (profile.get('topic') != 'non_legal'
            and not profile.get('required_concepts')
            and profile.get('task') != 'conversation')


GROUNDING_INSTRUCTIONS = (
    'Separate user facts from legal rules and from medical or agency findings. '
    'Answer the actual issue, not incidental occupation or employment-status words. '
    'For every asserted obligation, exception, frequency, deadline, percentage, benefit '
    'or liability, cite the operative passage supporting that proposition and retain '
    'its scope and conditions. Do not substitute wage liability for safety liability. '
    'Exposure or symptoms alone do not establish a diagnosis or compensation approval; '
    'do not invent an injury. Distinguish possible eligibility from confirmed entitlement. '
    'Do not invent drill frequency, automatic fines, work stoppage or shared damages '
    'liability. When only some requested points are supported, explain those points '
    'and identify the specific evidence gaps; do not fill them with unrelated rules. '
    'An empty evidence checklist is unresolved, not proof that the answer is complete.'
)
