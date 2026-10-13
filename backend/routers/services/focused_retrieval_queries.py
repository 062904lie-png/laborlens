"""Small discovery probes derived from the user's wording, never a law catalog.

These probes supplement the unchanged full question. They do not select a
legal authority, decide applicability, or supply an answer. Use an already
loaded local parser when available; never download or call a model here.
"""
import re
import sys
import unicodedata

from .issue_evidence import issue_terms
from .question_analysis_cache import reuse_analysis


_LEADING = {'a', 'an', 'the', 'my', 'our', 'your', 'his', 'her', 'their', 'its',
            'this', 'that', 'these', 'those', 'which', 'some'}
_BOUNDARIES = {'in', 'at', 'for', 'from', 'on', 'with', 'without', 'is', 'are',
               'was', 'were', 'has', 'have', 'had', 'will', 'can', 'should',
               'would', 'must', 'shall', 'because', 'but', 'that',
               'who', 'which', 'to', 'if', 'work', 'apply', 'mean', 'means'}
_PERSONAL = {'my', 'our', 'your', 'his', 'her', 'their', 'its'}
_WORD = re.compile(r"[^\W_]+(?:[-’'][^\W_]+)*", re.UNICODE)


def _normalized(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value or ''))).strip()


def _clean_phrase(value):
    words = list(_WORD.finditer(value))[:8]
    while words and words[0].group().casefold() in _LEADING:
        words.pop(0)
    if not words:
        return ''
    words = words[:6]
    while words and words[-1].group().casefold() in {'here', 'there', 'anymore'}:
        words.pop()
    if not words:
        return ''
    if len(words) >= 3 and words[-1].group().casefold() in {'rules', 'details', 'information'}:
        words.pop()
    meaningful = [index for index, word in enumerate(words) if issue_terms(word.group())]
    if not meaningful:
        return ''
    # Generic heads such as "employment" remain part of a meaningful compound
    # even though the responsiveness vocabulary excludes them as stop words.
    return _normalized(value[words[meaningful[0]].start():words[-1].end()])


def _ready_parser():
    # The chat preprocessor already loads this parser. Standalone extraction
    # must remain useful without loading a large model or invoking its downloader.
    module = sys.modules.get('routers.services.nlp_service')
    return getattr(module, '_nlp', None) if module is not None else None


@reuse_analysis
def question_focus(question, nlp_profile=None):
    """Return explicit noun subjects, named anchors, and dates from this text."""
    text = _normalized(question)
    subjects, anchors, dates = [], [], []

    def anchor(value, target=anchors):
        value = _normalized(value)
        if (value and len(value) <= 80 and len(value.split()) <= 6
                and value.casefold() in text.casefold()
                and value.casefold() not in {v.casefold() for v in target}):
            target.append(value)

    def subject(value, grammar=0, requested=False):
        phrase = _clean_phrase(value)
        if not phrase or phrase.casefold() not in text.casefold():
            return
        words = list(_WORD.finditer(value))
        leading = words[0].group().casefold() if words else ''
        score = min(3, len(issue_terms(phrase))) + grammar + (3 if requested else 0)
        score += .7 if leading == 'the' else 0
        score -= 3 if leading in _PERSONAL else 0
        score -= 2 if any(word.group().isdigit() for word in words) else 0
        subjects.append((score, text.casefold().rfind(phrase.casefold()), phrase))

    profile = nlp_profile or {}
    for key in ('locations', 'organizations', 'law_references'):
        for value in (profile.get('entities') or {}).get(key) or []:
            anchor(value)
    for value in (profile.get('entities') or {}).get('dates') or []:
        anchor(value, dates)
    parser = _ready_parser()
    if parser is not None:
        try:
            doc = parser(text)
            for entity in doc.ents:
                if entity.label_ in {'GPE', 'LOC', 'FAC', 'ORG', 'LAW'}:
                    anchor(entity.text)
                elif entity.label_ == 'DATE':
                    anchor(entity.text, dates)
            for chunk in doc.noun_chunks:
                dependency = chunk.root.dep_
                grammar = 2 if dependency in {'nsubj', 'nsubjpass', 'dobj', 'obj'} else 1 if dependency == 'attr' else 0
                subject(chunk.text, grammar, chunk.sent.text.rstrip().endswith('?'))
        except Exception:
            # The deterministic wording fallback remains available even if the
            # cached parser has no dependency component for this language.
            pass
    for phrase in profile.get('noun_phrases') or []:
        subject(phrase)
    # Grammar fallback: articles and question cues identify phrases supplied by
    # the user. There are no recognized benefits, regions, laws, or document IDs.
    patterns = (
        (r'\b(?:what\s+(?:is|are)|explain|describe|tell\s+me\s+about|how\s+(?:does|do)|rules\s+for)\s+([^.!?;:]+)', True),
        (r'\b(?:what|which)\s+([^.!?;:]+?)\s+(?:may|can|do|does|are)\b', True),
        # A reported refusal identifies the disputed subject even without an
        # article ("we don't provide beaconcredit allowance here"). This is
        # grammar, not a catalog of benefits or an endorsement of the refusal.
        (r"\b(?:(?:do(?:es)?|did|has|have)\s+not|don['’]t|doesn['’]t|didn['’]t)\s+(?:do|provide|pay|paid|allow|offer|give|permit|recognize)\s+([^.!?;:]+)", True),
        (r'\b(?:not\s+(?:allowed|permitted)\s+to|cannot|can\s+not|can[\x27’]t)\s+(?:get|receive|claim|collect|take|use)\s+([^.!?;:]+)', True),
        (r'(?=(\b(?:the|a|an|my|our|your|his|her|their)\s+[^.!?;,:]+))', False),
        (r'\b(?:about|concerning|regarding)\s+([^.!?;:]+)', True),
        # Entitlement requests often lack an article before their object.
        # Extract the user's named object, not the earlier narrated tenure.
        (r'\b(?:eligible\s+for|entitled\s+to(?:\s+receive)?|receive|claim|collect)\s+([^.!?;:]+)', True),
    )
    for pattern, requested in patterns:
        for match in re.finditer(pattern, text, re.I):
            phrase = match.group(1) if match.lastindex else match.group()
            words = list(_WORD.finditer(phrase))
            boundary = next((word.start() for word in words[1:] if word.group().casefold() in _BOUNDARIES), len(phrase))
            subject(phrase[:boundary], 0, requested)
    for match in re.finditer(r'\b(?:in|at|within|from|under)\s+((?:[A-Z][\w-]*)(?:\s+[A-Z][\w-]*){0,3})', text):
        anchor(match.group(1))
    for match in re.finditer(r'\b(?:[A-Z]{2,}(?:[-/][A-Z0-9]+)*|[A-Z][a-z]+[A-Z]\w*)\b', text):
        anchor(match.group())

    anchor_keys = {value.casefold() for value in anchors}
    ranked, seen = [], set()
    for score, position, phrase in sorted(subjects, reverse=True):
        key = phrase.casefold()
        if key in seen or key in anchor_keys:
            continue
        seen.add(key)
        ranked.append((score, phrase))
    if ranked:
        threshold = max(1, ranked[0][0] * .65)
        ranked = [(score, phrase) for score, phrase in ranked if score >= threshold]
    return {'subjects': [phrase for _score, phrase in ranked[:3]],
            'anchors': anchors[:3], 'dates': dates[:2]}


@reuse_analysis
def scenario_actions(question):
    """Actions stated by the user, not inferred legal classifications."""
    from .situation_parser import parse_situation
    result = []
    parser = _ready_parser()
    control = {'plan', 'planning', 'say', 'ask', 'want', 'need', 'think', 'tell', 'hope', 'decide'}
    for fact in parse_situation(question):
        text = fact['text']
        values = [m.group(1) for m in re.finditer(r'\bto\s+([a-z]+)\b', text, re.I)]
        if parser is not None:
            try:
                values.extend(token.lemma_ for token in parser(text) if token.pos_ == 'VERB')
            except Exception:
                pass
        for value in values:
            value = value.casefold()
            if value not in control and issue_terms(value) and value not in result:
                result.append(value)
    return result[:3]


def focused_search_queries(question, limit=3, nlp_profile=None):
    """Add exact scenario details and subject/entity probes to full-question search."""
    focus = question_focus(question, nlp_profile)
    result = []
    for subject in focus['subjects']:
        if len(result) >= limit:
            break
        probe = _normalized(' '.join([subject, *focus['anchors']]))
        if probe.casefold() != _normalized(question).casefold() and probe not in result:
            result.append(probe)
    actions = scenario_actions(question)
    if focus['subjects'] and actions and len(result) < limit:
        probe = _normalized(' '.join([focus['subjects'][0], *actions, *focus['anchors']]))
        if probe not in result:
            result.append(probe)
    # The disputed object must get a discovery slot before long reported
    # clauses/tenure details; these still supplement the unchanged full query.
    from .situation_parser import situation_search_probes
    for probe in situation_search_probes(question, limit=2):
        if len(result) >= limit:
            break
        if probe.casefold() not in {_normalized(question).casefold(), *(v.casefold() for v in result)}:
            result.append(probe)
    dated_facts = [value for value in focus['dates'] if re.search(r'\d', value)]
    if result and dated_facts and len(result) < limit:
        result.append(_normalized(' '.join([result[0], *dated_facts])))
    return result[:max(0, min(int(limit), 4))]


def literal_search_probes(question, limit=5):
    """Prefer actual subjects and named anchors over incidental first phrases."""
    focus = question_focus(question)
    # Keep adjacent user-supplied content words together as bounded probes.
    # This preserves phrase meaning without adding topic-specific synonyms.
    query_terms = issue_terms(question)
    text = _normalized(question)
    words = list(re.finditer(r'[a-z]{3,}', text, re.I))
    phrases = [f'{left.group().casefold()} {right.group().casefold()}'
               for left, right in zip(words, words[1:])
               if left.group().casefold() in query_terms and right.group().casefold() in query_terms
               and re.fullmatch(r'[\s\-–—]+', text[left.end():right.start()])]
    values = [*focus['subjects'][:2], *focus['anchors'][:1]]
    # Named locations often appear in a table as one city among several, not as
    # the exact phrase used by the user. Keep bounded constituent probes too.
    if focus['anchors']:
        values.extend(word.group() for word in _WORD.finditer(focus['anchors'][0]) if issue_terms(word.group()))
    values.extend(reversed(phrases))
    values.extend(sorted(issue_terms(question), key=lambda word: (-len(word), word)))
    result, seen = [], set()
    for value in values:
        value = _normalized(value).casefold()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
        if len(result) >= max(1, min(int(limit), 5)):
            break
    return result
