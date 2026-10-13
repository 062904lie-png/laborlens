"""Keep a retrieved circular's identity separate from its internal references."""
import re

CIRCULAR_RE = re.compile(
    r'\b(?P<agency>(?:CSC|COA|DBM|DOLE)(?:\s*[-–]\s*(?:CSC|COA|DBM|DOLE))*)'
    r'[-\s]+(?:Joint[-\s]+Circular|JC)[-\s]+(?:No\.?[-\s]*)?'
    r'(?P<number>\d+)(?:\s*,?[-\s]*(?:s\.?|series\s+of)[-\s]*(?P<year>\d{4}))?', re.I)


def circulars(text):
    normalized = str(text or '').replace('_', ' ').translate(str.maketrans('‐‑‒–—', '-----'))
    return [{'agency': re.sub(r'\s+', '', m['agency']).upper().replace('–', '-'),
             'number': m['number'], 'year': m['year'] or '', 'label': m[0]}
            for m in CIRCULAR_RE.finditer(normalized)]


def same_identity(first, second):
    return (first['agency'] == second['agency'] and first['number'] == second['number']
            and (not first['year'] or not second['year'] or first['year'] == second['year']))


def owning_circulars(source):
    # A body-derived citation label must never override a curated owner title.
    for key in ('document_title', 'source', 'law_number', 'citation'):
        found = circulars(source.get(key))
        if found:
            return found
    return []


def referenced_circulars(source, body):
    owners = owning_circulars(source)
    result = []
    for identity in circulars(body):
        if not any(same_identity(identity, own) for own in owners + result):
            result.append(identity)
    return result


def validate_circular_identity(answer, sources):
    from .legal_answer_validator import cited_source_numbers
    errors = []
    for sentence in re.split(r'\n+|(?<=[.!?])\s+(?=[A-Z])', answer or ''):
        cited = cited_source_numbers(sentence) | {int(n) for n in re.findall(r'\[(\d+)\]', sentence)}
        relevant = [s for i, s in enumerate(sources, 1) if not cited or i in cited]
        for mention in circulars(sentence):
            if any(any(same_identity(mention, own) for own in owning_circulars(s)) for s in relevant):
                continue
            referenced = any(any(same_identity(mention, ref) for ref in
                                 referenced_circulars(s, s.get('excerpt') or '')) for s in relevant)
            attributed = bool(re.search(r'\b(?:cites?|referenc(?:e[sd]?|ing)|refers?\s+to|'
                                        r'cross[-\s]references?|incorporates?)\b', sentence, re.I))
            if not (referenced and attributed):
                errors.append(f"{mention['label']} is an internal referenced or unprovided authority, not the identity of the cited retrieved document. Attribute the reference explicitly to its owning document.")
    return list(dict.fromkeys(errors))
