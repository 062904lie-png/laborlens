"""Overview discovery from request grammar and document wording, not legal lists."""
import re


def definition_subject(question):
    """Preserve the complete term being defined, including connective words."""
    text = str(question or '').split('Current follow-up question:')[-1].strip()
    match = re.match(r'^(?:what\s+is|define|(?:what\s+is\s+)?the\s+meaning\s+of)\s+'
                     r'(.{2,180}?)[?.!]*$', text, re.I)
    if not match:
        return ''
    return re.sub(r'^(?:a|an|the)\s+', '', match[1], flags=re.I).strip(' ?.!"“”')


def calculation_request(question):
    """Task wording, not a benefit/topic catalog or a legal formula."""
    return bool(re.search(r'\b(?:calculat\w*|comput\w*|formula|how\s+much)\b',
                          str(question or ''), re.I))


def calculation_search_query(question):
    """A lexical task probe; full question retains scope in semantic search."""
    if not calculation_request(question):
        return ''
    from .issue_evidence import issue_terms
    words = issue_terms(question)
    scaffolding = {'government', 'public', 'private', 'sector', 'company', 'office',
                   'employee', 'employees', 'worker', 'workers', 'benefit', 'benefits'}
    subject = [word.casefold() for word in re.findall(r'[a-z]{3,}', str(question), re.I)
               if word.casefold() in words - scaffolding
               and not re.match(r'calculat|comput|formula', word, re.I)]
    return ' '.join([*dict.fromkeys(subject), 'computation', 'formula']) if subject else ''


def broad_explanation_request(question):
    """Recognize requests likely to need several supporting passages.

    This checks question wording only; it contains no legal topics, authorities,
    or rules about which conclusions a complete answer must reach.
    """
    text = str(question or '').split('Current follow-up question:')[-1].strip()
    if overview_subject(text) or calculation_request(text):
        return True
    return bool(re.search(
        r'\b(?:rules?|overview|types?|kinds?|forms?|options?|rights?|benefits?|'
        r'compare|comparison|differences?|calculate|computation|compute|formula|'
        r'worked\s+example|show\s+(?:me\s+)?(?:an?\s+)?example|how\s+much)\b',
        text, re.I,
    ))


def overview_subject(question):
    text = str(question or '').split('Current follow-up question:')[-1].strip()
    match = re.search(r'\b(?:what|which)\s+(.{2,80}?)\s+(?:may|can|do|does)\b', text, re.I)
    if match is None:
        match = re.search(r'\b(?:types?|kinds?|forms?|list|overview)\s+(?:of\s+)?(.{2,80}?)(?:[?.!]|$)', text, re.I)
    if match is None:
        match = re.search(r'\bwhat\s+are\s+(?:the\s+)?(.{2,80}?(?:benefits|options|rights|rules))\s*[?.!]*$', text, re.I)
    if match is None:
        return ''
    subject = re.sub(r'^(?:the|different|available|employee|employees)\s+', '', match[1].strip(), flags=re.I)
    subject = re.sub(r'\s+(?:benefits|options|types|kinds|rules|rights)$', '', subject, flags=re.I)
    from .issue_evidence import issue_terms
    return subject if issue_terms(subject) and len(subject.split()) <= 6 else ''


def overview_facets(question, chunk):
    """Distinct descriptive phrases found in bodies; never evidence certificates."""
    subject = overview_subject(question)
    if not subject:
        return set()
    from .legal_topic_matrix import source_passage_text
    from .question_understanding import request_subject_terms
    body = source_passage_text(chunk.get('text') or chunk.get('excerpt') or '')
    pattern = r'\b(?:[a-z-]+\s+){0,3}' + r'\s+'.join(re.escape(w) for w in subject.split()) + r'\b'
    ignored = {'shall', 'must', 'granted', 'grant', 'receive', 'receives', 'entitled', 'paid',
               'unpaid', 'taking', 'take', 'additional', 'benefits', 'benefit'}
    facets = set()
    for match in re.finditer(pattern, body, re.I):
        words = [w for w in request_subject_terms(match[0]) if w not in ignored]
        if words:
            facets.add(' '.join(sorted(words)))
    return facets
