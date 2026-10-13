"""Ask about scope before choosing a narrow legal topic for a broad request."""
import re


def broad_intake_question(query, history=None, language='en'):
    from routers.services.followup_context import resolve_followup_context
    from routers.services.scenario_retrieval import is_work_injury_query
    text = resolve_followup_context(query, history).lower()
    from routers.services.issue_evidence import safety_issue_contract
    if safety_issue_contract(text):
        return None
    if is_work_injury_query(text):
        # Injury benefits are a named legal issue, not a request for every
        # benefit. The regular completeness checks still handle bare incidents.
        return None
    # Definitions and explicitly named benefits already have an answerable scope.
    if re.search(r'\b(?:define|definition|what is|what are|ano ang|ano ang buot)\b', text) and not re.search(r'\b(?:my|i|me|ko|akon|claim)\b', text):
        return None
    specific = r'\b(?:leave|maternity|paternity|solo.parent|sss|philhealth|pag.ibig|13th|thirteenth|overtime|holiday|retirement|separation|disability|unemployment)\b'
    if re.search(r'\b(?:benefits?|benepisyo)\b', text) and not re.search(specific, text):
        sector = re.search(r'private[ -]sector|private (?:employee|company)|government|public[ -]sector|civil service|kasambahay|domestic worker|gobyerno|pribado', text)
        if not sector:
            return {
                'en': 'Are you a private-sector employee, government employee, or domestic worker (kasambahay)?',
                'fil': 'Empleyado ka ba sa pribadong sektor, sa gobyerno, o kasambahay?',
                'hil': 'Empleyado ka bala sa pribado nga sektor, sa gobyerno, ukon kasambahay?',
            }.get(language, 'Are you a private-sector employee, government employee, or domestic worker (kasambahay)?')
        # With a sector supplied, retrieve a broad overview instead of imposing leave.
        return None
    if re.fullmatch(r'\s*(?:can i (?:file (?:a )?complain(?:t)?|complain)(?: against (?:my |the )?(?:employer|company|boss))?|how do i complain|pwede ba akong magreklamo|puwede ba akong magreklamo|pwede bala ako magreklamo)[?.! ]*', text):
        return {
            'en': 'What happened at work that you want to report?',
            'fil': 'Ano ang nangyari sa trabaho na gusto mong ireklamo?',
            'hil': 'Ano ang natabo sa obra nga gusto mo ireklamo?',
        }.get(language, 'What happened at work that you want to report?')
    return None


def broad_benefits_query(query):
    from routers.services.issue_evidence import safety_issue_contract
    if safety_issue_contract(query):
        return False
    from routers.services.scenario_retrieval import is_work_injury_query
    if is_work_injury_query(query):
        return False
    return bool(re.search(r'\b(?:benefits?|benepisyo)\b', query, re.I)
                and not re.search(r'\b(?:leave|maternity|paternity|solo.parent|sss|philhealth|pag.ibig|13th|thirteenth|overtime|holiday|retirement|separation|disability|unemployment)\b', query, re.I))
