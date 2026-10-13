"""Conservative version precedence using existing curated legal metadata.

Never infer repeal from publication/upload dates or a document's mere mention
of another law. These constraints reorder evidence; they do not prove its law.
"""
import re
from datetime import date, datetime


VERSION_FIELDS = (
    'canonical_document_id', 'authority_id', 'law_number', 'citation',
    'article_number', 'section', 'article_section', 'document_title', 'source',
    'legal_status', 'version_status', 'effective_date', 'date_issued',
    'superseded_by', 'superseding_document', 'amending_act_section',
    'amended_law_section', 'sector', 'source_scope',
    'amendment_metadata_conflict',
)


def version_metadata(chunk):
    metadata = chunk.get('metadata') or {}
    return {key: str(metadata[key])[:255] for key in VERSION_FIELDS
            if metadata.get(key) not in (None, '')}


def infer_amendment_metadata(text, metadata):
    """Extract an explicit section-to-section amendment, not mere citations.

    This is read-time evidence enrichment. Never infer status, effectivity,
    wholesale repeal, or relationships from document years.
    """
    from routers.services.legal_topic_matrix import source_passage_text
    result = dict(metadata)
    title = str(result.get('document_title') or result.get('source') or '')
    # An IRR quoting an Act's amendment is not itself the amending Act.
    if re.search(r'\birr\b|implementing\s+rules', title, re.I):
        return result
    owner = re.fullmatch(r'(ra|pd|eo)_(\d+)', str(result.get('canonical_document_id') or ''), re.I)
    if not owner:
        return result
    body = re.sub(r'\s+', ' ', source_passage_text(text))
    clause = re.search(
        r'\b(?:SEC(?:TION)?\.?|ART(?:ICLE)?\.?)\s*(\d+[A-Za-z]?)\s*[.:–-]\s*'
        r'(Section|Article)\s*(\d+[A-Za-z]?)\s+of[ -]+'
        r'(Republic\s+Act|RA|Presidential\s+Decree|PD|Executive\s+Order|EO)'
        r'\s*(?:No\.?\s*)?(\d+)\b[^.;]{0,180}\b(?:is\s+hereby|is|shall\s+be)\s+amended\b',
        body, re.I,
    )
    if not clause:
        return result
    prefix = {'republicact': 'RA', 'ra': 'RA', 'presidentialdecree': 'PD',
              'pd': 'PD', 'executiveorder': 'EO', 'eo': 'EO'}[_key(clause[4])]
    owning_label = 'Article' if clause[0].lower().startswith('art') else 'Section'
    amending = f'{owner[1].upper()} {owner[2]} {owning_label} {clause[1]}'
    affected = f'{prefix} {clause[5]} {clause[2].title()} {clause[3]}'
    # A curated contradictory relationship must be reviewed, not overwritten.
    if any(result.get(k) and _key(result[k]) != _key(v) for k, v in (
        ('amending_act_section', amending), ('amended_law_section', affected),
    )):
        result['amendment_metadata_conflict'] = True
        return result
    result.update(amending_act_section=amending, amended_law_section=affected)
    result['section' if owning_label == 'Section' else 'article_number'] = (
        f'Section {clause[1]}' if owning_label == 'Section' else clause[1])
    result['article_section'] = f'{owning_label} {clause[1]}'
    result['citation'] = f'{amending} (amending {affected})'
    return result


def _key(value):
    text = str(value or '').lower().replace('_', ' ')
    for pattern, replacement in (
        (r'\brepublic\s+act\b', 'ra'),
        (r'\bpresidential\s+decree\b', 'pd'),
        (r'\bexecutive\s+order\b', 'eo'),
        (r'\bdepartment\s+order\b', 'do'),
        (r'\bdole\s+(?=do\b)', ''),
        (r'\b(?:no|number)\b\.?\s*', ''),
        (r'\bsec\.', 'section'),
        (r'\bart\.', 'article'),
    ):
        text = re.sub(pattern, replacement, text)
    return re.sub(r'[^a-z0-9]', '', re.sub(r'\.pdf$', '', text))


def _documents(meta):
    return {_key(meta.get(field)) for field in
            ('canonical_document_id', 'law_number', 'document_title', 'source')
            if meta.get(field)}


def _provisions(meta):
    # Only owning-provision fields, never an amendment's affected provision.
    keys = {_key(meta.get(field)) for field in ('citation', 'article_section', 'authority_id')
            if meta.get(field)}
    for document in _documents(meta):
        for field, label in (('section', 'section'), ('article_number', 'article')):
            number = str(meta.get(field) or '')
            if re.fullmatch(r'\d+[A-Za-z]?', number):
                keys.add(document + label + number.lower())
        for field in ('section', 'article_section'):
            local = str(meta.get(field) or '').strip()
            match = re.fullmatch(r'(section|sec\.?|article|art\.?)\s*(\d+[A-Za-z]?)', local, re.I)
            if match:
                label = 'section' if match[1].lower().startswith('sec') else 'article'
                keys.add(document + label + match[2].lower())
    return keys


def _effective(meta):
    status = str(meta.get('legal_status') or meta.get('version_status') or '').strip().lower()
    if meta.get('is_active') is False or meta.get('is_active') == 0:
        return False
    if status not in {'active', 'current', 'approved', 'indexed', 'effective', 'valid'}:
        return False
    value = str(meta.get('effective_date') or '').strip()
    if not value:
        return True
    for fmt in ('%Y-%m-%d', '%B %d, %Y', '%B %d %Y', '%d %B %Y', '%b %d, %Y', '%d %b %Y'):
        try:
            return datetime.strptime(value, fmt).date() <= date.today()
        except ValueError:
            pass
    # Ambiguous dates must not silently become effective today.
    return False


def historical_query(query):
    """Whether discovery needs earlier legal versions (not deleted uploads)."""
    # Historical/comparison questions and dated scenarios need temporal legal
    # interpretation. Do not force today's version ahead of the requested rule.
    return bool(re.search(r'\b(?:historical|original|previous|old|compare|comparison|difference|'
                 r'as\s+of)\b|\b(?:in|during|before|after|on|for)\s+(?:19|20)\d{2}\b|'
                 r'\bbefore\s+(?:the\s+)?amendment\b|'
                 r'\b(?:january|february|march|april|may|june|july|august|september|'
                 r'october|november|december)\s+\d{1,2}\b|'
                 r'\b\d{1,2}[/-]\d{1,2}\b|\b(?:19|20)\d{2}-\d{2}-\d{2}\b', query or '', re.I))


def version_precedence(query, chunks):
    """Return (successor index, predecessor index) constraints for this turn."""
    if historical_query(query):
        return []
    metadata = [chunk.get('metadata') or {} for chunk in chunks]
    replacements = {_key(m.get('superseded_by') or m.get('superseding_document')) for m in metadata}
    replacements.discard('')
    successors = [(i, m) for i, m in enumerate(metadata)
                  if (m.get('amended_law_section') or replacements.intersection(_documents(m)))
                  and _effective(m) and not m.get('amendment_metadata_conflict')]
    if not successors:
        return []
    edges = []
    for old_index, old in enumerate(metadata):
        replacement = _key(old.get('superseded_by') or old.get('superseding_document'))
        old_status = str(old.get('legal_status') or old.get('version_status') or '').strip().lower()
        for new_index, new in successors:
            if old_index == new_index:
                continue
            if old.get('amendment_metadata_conflict') or new.get('amendment_metadata_conflict'):
                continue
            if old.get('sector') and new.get('sector') and old['sector'] != new['sector']:
                continue
            if old.get('source_scope') and new.get('source_scope') and old['source_scope'] != new['source_scope']:
                continue
            document_replacement = (old_status in {'superseded', 'replaced', 'obsolete', 'outdated'}
                                    and replacement and replacement in _documents(new))
            affected = _key(new.get('amended_law_section'))
            provision_amendment = bool(
                affected and affected in _provisions(old)
                and new.get('amending_act_section')
                and _key(new['amending_act_section']) in _provisions(new)
                # Partial amendments need current/effectivity evidence as well
                # as an explicit provision relationship. Active alone is not it.
                and (new.get('effective_date') or str(new.get('legal_status')
                     or new.get('version_status') or '').strip().lower() == 'current')
            )
            if document_replacement or provision_amendment:
                edges.append((new_index, old_index))
    return edges


def apply_version_precedence(query, chunks):
    """Preserve objects/membership and relevance except for explicit precedence."""
    edges = version_precedence(query, chunks)
    if not edges or all(new < old for new, old in edges):
        return chunks
    remaining = set(range(len(chunks)))
    order = []
    while remaining:
        available = [index for index in sorted(remaining)
                     if not any(old == index and new in remaining for new, old in edges)]
        if not available:
            # Conflicting/cyclic metadata is not a license to guess a version.
            return chunks
        index = available[0]
        order.append(index)
        remaining.remove(index)
    return [chunks[index] for index in order]
