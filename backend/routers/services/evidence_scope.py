"""Local applicability checks, not a legal answer or support certificate."""
import re
from functools import lru_cache

from .rag_policy import GENERAL_SECTOR, UNKNOWN_SECTOR, canonical_sector, infer_query_sector

CAREGIVERS = 'caregivers'
COS_JO = 'government_cos_jo'
CAREGIVER_RE = re.compile(r'\bcaregivers?\b', re.I)
COS_JO_RE = re.compile(r'\b(?:COS\s*(?:[/&]|and)\s*JO|COS|JO|contract\s+of\s+service|job\s+order)\s*(?:workers?)?\b', re.I)


@lru_cache(maxsize=256)
def requested_scope(question):
    """None means unspecified; it is not a private-employment assertion."""
    from .question_understanding import employment_context_parts
    current, _ = employment_context_parts(question)
    if CAREGIVER_RE.search(current):
        return CAREGIVERS
    if COS_JO_RE.search(current):
        return COS_JO
    government = r"(?:government|public(?:[-\s]+sector)?|civil[-\s]+service)"
    private = r"private(?:[-\s]+sector)?"
    if re.search(rf"\b{government}\b.{{0,60}}\b(?:and|versus|vs\.?|or)\b.{{0,40}}\b{private}\b|"
                 rf"\b{private}\b.{{0,60}}\b(?:and|versus|vs\.?|or)\b.{{0,40}}\b{government}\b",
                 current, re.I):
        return None
    if re.search(
        r'\bprivate[-\s]+(?:sector|employment|employee|company|workplace|employer|business)\b'
        r'|\bprivate\s+(?:\w+\s+){0,2}(?:store|shop|workplace|company|business)\b',
        current, re.I,
    ):
        return GENERAL_SECTOR
    if re.search(r'\b(?:government|public[-\s]+sector|civil[-\s]+service)\s+'
                 r'(?:office|agency|department|employee|worker|employment)\b', current, re.I):
        from .rag_policy import GOVERNMENT_SECTOR
        return GOVERNMENT_SECTOR
    sector = infer_query_sector(current)
    return sector if sector not in {GENERAL_SECTOR, UNKNOWN_SECTOR} else None


def source_scope(item):
    """Use owning-document coverage before legacy passage classification.

    An internal citation does not change the owner or the worker population.
    General documents may contain special sections, so do not classify their
    entire contents from incidental occupation mentions.
    """
    m = item.get('metadata') or item
    owner = ' '.join(str(m.get(k) or '') for k in ('document_title', 'source', 'law_number'))
    if re.search(r'\b(?:GSIS|civil service|RACCS|government service insurance)\b', owner, re.I):
        return 'government_employment'
    if CAREGIVER_RE.search(owner):
        return CAREGIVERS
    for pattern, population in (
        (r'\bhospital(?:\s+and\s+clinic)?\s+personnel\b', 'hospital_clinic_personnel'),
        (r'\b(?:domestic\s+workers?|kasambahay)\b', 'domestic_workers'),
        (r'\bnight\s+workers?\b', 'night_workers'),
    ):
        if re.search(pattern, owner, re.I):
            return population
    # This circular family is government employment; COS/JO coverage is stated
    # in its operative passage even where legacy scope labels say "hospital".
    body = str(item.get('text') or m.get('excerpt') or '').split('\nText:\n')[-1]
    sector = canonical_sector(m.get('sector') or GENERAL_SECTOR)
    if COS_JO_RE.search(owner) or (sector == 'government_employment' and COS_JO_RE.search(body)):
        return COS_JO
    scope = str(m.get('source_scope') or m.get('coverage') or '')
    if CAREGIVER_RE.search(scope):
        return CAREGIVERS
    if sector != GENERAL_SECTOR:
        return sector
    special = any(str(m.get(k) or '').lower() in {'true', '1'}
                  for k in ('is_special_sector', 'is_special_sector_rule', 'is_special_rule'))
    if special or m.get('rule_type') == 'special_sector_rule':
        return 'special_unspecified'
    return GENERAL_SECTOR


def scope_compatible(requested, actual):
    if requested == COS_JO:
        return actual in {COS_JO, 'government_employment'}
    if requested == CAREGIVERS:
        return actual in {CAREGIVERS, GENERAL_SECTOR}
    from .rag_policy import sector_matches
    return sector_matches(requested, actual)


def responsive(question, item):
    from .regional_wage_evidence import wage_scope_rejection, regional_wage_query, regional_rate_present
    if wage_scope_rejection(question, item):
        return False
    # An official rate table need not contain a prose verb such as "shall".
    # This is a discovery signal only; independent checking still verifies it.
    if regional_wage_query(question) and regional_rate_present(question, [item]):
        return True
    from .document_evidence import operative_quality
    from .remedy_discovery import remedy_requested, procedural_passage
    if remedy_requested(question) and procedural_passage(item) and operative_quality(item, question) > 0:
        # General filing rules need not name each benefit that can be disputed.
        # Worker-scope compatibility is still evaluated separately below.
        return True
    from .evidence_selection import discovery_hit, discovery_terms
    from .focused_retrieval_queries import question_focus
    from .legal_topic_matrix import source_passage_text
    body = source_passage_text(item.get('text') or item.get('excerpt') or '')
    subjects = question_focus(question)['subjects'] or [question]
    return operative_quality({'text': body}, question) > 0 and any(
        discovery_hit(discovery_terms(s), discovery_terms(body)) for s in subjects)


def scope_priority(question, item):
    wanted = requested_scope(question) or GENERAL_SECTOR
    applicable = source_scope(item) == wanted and responsive(question, item)
    if not applicable:
        return 0
    from .regional_wage_evidence import regional_rate_present, document_regions, regional_wage_document
    from .rag_policy import wage_order_query_details
    details = wage_order_query_details(question)
    meta = item.get('metadata') or item
    if details['is_regional_wage_query'] and regional_wage_document(meta):
        if details['region_code'] in document_regions(meta, str(item.get('text') or '')):
            return 3 if regional_rate_present(question, [item]) else 2
    return 1


def scope_completeness(question, items):
    wanted = requested_scope(question)
    scopes = sorted({source_scope(c) for c in items})
    from .regional_wage_evidence import regional_rate_present
    rate_present = regional_rate_present(question, items)
    from .remedy_discovery import remedy_evidence_present
    remedy_present = remedy_evidence_present(question, items)
    complete = any(scope_priority(question, c) for c in items) and rate_present and remedy_present
    special = [s for s in scopes if s != GENERAL_SECTOR]
    # Do not turn this guard into a second topic checklist. It intervenes when
    # a worker population is requested or sector evidence creates a scope gap.
    gap = bool(not rate_present or not remedy_present or (items and not complete and (special or wanted is not None)))
    return {'requested_scope': wanted or 'unspecified', 'source_scopes': scopes,
            'general_present': any(source_scope(c) == GENERAL_SECTOR and responsive(question, c) for c in items),
            'scope_complete': complete, 'scope_gap': gap, 'regional_rate_present': rate_present,
            'remedy_evidence_present': remedy_present,
            'discovery_gap': ('matching regional wage rate' if not rate_present else
                             'requested complaint or remedy procedure' if not remedy_present else
                             'general applicable worker coverage' if wanted in {None, GENERAL_SECTOR}
                              else wanted.replace('_', ' ') + ' applicable worker coverage') if gap else None}


def scope_notice(state, language='en'):
    if state.get('remedy_evidence_present') is False:
        return {
            'en': 'The retrieved passages do not yet establish the complaint or remedy procedure you asked about.',
            'fil': 'Hindi pa naitatatag ng mga nakuha na source ang proseso ng reklamo o tulong na tinatanong mo.',
            'hil': 'Wala pa napamatud-an sang nakuha nga mga source ang proseso sang pagreklamo ukon bulig nga imo ginapangayo.',
        }.get(language, 'The requested remedy procedure still needs supporting evidence.')
    if state.get('regional_rate_present') is False:
        return {
            'en': 'The retrieved passages do not establish the minimum wage for the requested region and worker category. A matching wage-rate provision is still needed.',
            'fil': 'Hindi pa naitatatag ng mga nakuha na bahagi ng source ang minimum wage para sa rehiyon at uri ng manggagawang tinukoy mo. Kailangan pa ang naaangkop na wage-rate provision.',
            'hil': 'Wala pa ginapamatud-an sang nakuha nga mga bahin sang source ang minimum wage para sa rehiyon kag klase sang manugpangabudlay nga imo gintumod. Kinahanglan pa ang nagakaigo nga wage-rate provision.',
        }.get(language, 'A matching regional wage-rate provision is still needed.')
    if state['requested_scope'] == 'unspecified':
        notices = {
            'en': 'The retrieved passages cover special worker categories and do not establish a general rule for this question. Which worker category applies—for example, a private-sector employee, caregiver, or government COS/JO worker?',
            'fil': 'Ang mga nakuha na bahagi ng source ay para sa espesyal na uri ng manggagawa at hindi sapat para sa pangkalahatang tuntunin. Anong uri ng manggagawa ang tinutukoy—empleyado sa pribadong sektor, caregiver, o COS/JO sa gobyerno?',
            'hil': 'Ang nakuha nga mga bahin sang source para sa espesyal nga mga klase sang manugpangabudlay kag indi igo para sa pangkabilugan nga pagsulundan. Ano nga klase sang manugpangabudlay ang ginatumod—empleyado sa pribado, caregiver, ukon COS/JO sa gobyerno?',
        }
        return notices.get(language, notices['en'])
    return {
        'en': 'The retrieved passages do not establish the rule for the worker category you specified. Applicable source evidence is still needed.',
        'fil': 'Hindi naitatatag ng mga nakuha na bahagi ng source ang tuntunin para sa uri ng manggagawang tinukoy mo. Kailangan pa ng naaangkop na source.',
        'hil': 'Wala ginapamatud-an sang nakuha nga mga bahin sang source ang pagsulundan para sa klase sang manugpangabudlay nga imo gintumod. Kinahanglan pa sang nagakaigo nga source.',
    }.get(language, 'The retrieved passages do not establish the rule for the worker category you specified. Applicable source evidence is still needed.')
