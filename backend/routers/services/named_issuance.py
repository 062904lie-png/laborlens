"""Exact DOLE section lookup, independent of parent-law citations in titles."""
import asyncio
import re


def requested_issuance(query):
    text = re.sub(r"_+", " ", str(query or ''))
    match = re.search(
        r'\b(?:DOLE[\s._-]+)?(?:Department|Dept\.?)'
        r'[\s._-]*Order[\s._-]*(?:No\.?[\s._-]*)?[-\s]*'
        r'(\d{2,3})[-\s]+(?:series\s+of\s+)?(\d{2,4})\b',
        text,
        re.I,
    )
    if not match:
        match = re.search(
            r'\b(?:DOLE\s+)?D\.?O\.?\s*(?:No\.?\s*)?'
            r'(\d{2,3})[-\s]+(?:series\s+of\s+)?(\d{2,4})\b',
            text,
            re.I,
        )
    if match:
        number, year = match.groups()
        return number, year[-2:]
    if re.search(r'rules on the administration and enforcement of labor standards', text, re.I):
        return '238', '23'
    return None


def requested_section(query):
    match = re.search(r'\bsec(?:tion)?\.?\s*(\d+)\b', query, re.I)
    return match.group(1) if match else None


def issuance_matches(meta, identity):
    # Never infer identity from body citations or old canonical IDs.
    for key in ('law_number', 'document_title', 'original_filename'):
        if requested_issuance(str(meta.get(key) or '').replace('_', ' ')) == identity:
            return True
    return False


def section_passage(text, section, query):
    body = str(text or '').split('\nText:\n')[-1]
    blocks = re.split(r'(?=\bSection\s+\d+\.)', body, flags=re.I)
    candidates = [b.strip() for b in blocks if re.match(rf'Section\s+{re.escape(section)}\.', b, re.I)]
    if re.search(r'coverage and applicability', query, re.I):
        candidates = [b for b in candidates if re.match(r'Section\s+\d+\.\s*Coverage and Applicability', b, re.I)]
    # A heading-only chunk is not evidence of the operative rule.
    return [b for b in candidates if len(b) > 90]


async def answer_named_issuance(query, language='en'):
    identity, section = requested_issuance(query), requested_section(query)
    if not identity or not section:
        return None
    from routers.services.chroma_service import get_collection, _records_from_get_results
    from routers.chat import filter_searchable_chunks
    from routers.services.openai_service import build_sources_payload
    collection = get_collection()
    matches = []
    # Scan section candidates, not only nearest vectors. Approval/currentness
    # is still checked against the document registry before exposing any text.
    for offset in range(0, 2000, 200):
        result = await asyncio.to_thread(collection.get, where_document={'$contains': f'Section {section}.'},
                                         limit=200, offset=offset, include=['documents', 'metadatas'])
        records = _records_from_get_results(result, retrieval_method='named_issuance_section')
        records = [r for r in records if issuance_matches(r.get('metadata') or {}, identity)]
        records = await filter_searchable_chunks(records)
        for record in records:
            for passage in section_passage(record.get('text'), section, query):
                matches.append((passage, record))
        if len(result.get('ids') or []) < 200:
            break
    unavailable = {'en': 'I could not uniquely verify the requested DOLE section in the searchable knowledge base. Please provide its Rule number and exact issuance number. I will not substitute a Labor Code article.',
                   'fil': 'Hindi ko matiyak ang hiniling na seksyon ng DOLE sa knowledge base. Pakibigay ang Rule number at eksaktong issuance number. Hindi ko ito papalitan ng artikulo ng Labor Code.',
                   'hil': 'Indi ko mapamatud-an ang ginpangayo nga seksyon sang DOLE sa knowledge base. Palihog ihatag ang Rule number kag eksakto nga issuance number. Indi ko ini pag-ilisan sang artikulo sang Labor Code.'}
    missing = {'answer': unavailable.get(language, unavailable['en']), 'sources': [], 'kb_found': False, 'intent': 'document_section_lookup'}
    if not matches:
        return missing
    # Section numbers repeat across Rules; don't choose an arbitrary one.
    headings = {p.split('—')[0].split(' - ')[0].split('. ', 2)[1].strip() for p, _ in matches}
    if len(headings) > 1:
        return missing
    passage, record = max(matches, key=lambda item: len(item[0]))
    title = f'DOLE Department Order No. {identity[0]}-{identity[1]}'
    meta = dict(record.get('metadata') or {})
    for key in ('article_number', 'article_title', 'article_section', 'normalized_authority_id', 'citation', 'rule', 'rule_title'):
        meta.pop(key, None)
    meta.update(document_title=title, canonical_document_id=f'dole_do_{identity[0]}_{identity[1]}', section=f'Section {section}')
    sources = build_sources_payload([{**record, 'text': passage, 'metadata': meta}])
    intro = {'en': 'Here is the requested section from the indexed DOLE document. The source text is in English.',
             'fil': 'Ito ang hiniling na seksyon mula sa naka-index na dokumento ng DOLE. Nasa Ingles ang orihinal na teksto.',
             'hil': 'Ari ang ginpangayo nga seksyon halin sa naka-index nga dokumento sang DOLE. Ingles ang orihinal nga teksto.'}.get(language)
    caveat = {'en': 'This is the indexed version, not confirmation that no later amendment exists.',
              'fil': 'Ito ang naka-index na bersyon; hindi nito kinukumpirma na walang mas bagong pagbabago.',
              'hil': 'Ini ang naka-index nga bersyon; wala sini ginakumpirma nga wala sang mas bag-o nga pagbag-o.'}.get(language)
    return {'answer': f'{intro}\n\n**{title} — Section {section}**\n\n{passage}\n\n(Source 1)\n\n{caveat}',
            'sources': sources, 'kb_found': True, 'intent': 'document_section_lookup'}
