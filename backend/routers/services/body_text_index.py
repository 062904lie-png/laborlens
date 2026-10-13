"""Local, document-derived lexical discovery; no laws, topics or answer catalog.

The cache contains only searchable text features and is discarded on index
changes. Approximate word-family matching is a candidate hint, not evidence.
No stored vectors, documents or metadata are changed.
"""
import math
import re
import threading
import logging
from collections import defaultdict

from .issue_evidence import issue_terms
from .evidence_selection import discovery_terms
from .legal_topic_matrix import source_passage_text

_lock = threading.Lock()
_cache = None
_epoch = 0
_scheduler_lock = threading.Lock()
_requested_build = None
_worker_running = False
logger = logging.getLogger(__name__)
_CONTENT_WORD = re.compile(r'[a-z]{3,}', re.I)
_ALIAS = re.compile(r'([A-Za-z][A-Za-z -]{2,120})\s*\(([A-Z][A-Z0-9]{1,9})\)')


def document_aliases(body):
    """Learn explicit full-name (ACRONYM) pairs, never infer legal synonyms."""
    pairs = set()
    for match in _ALIAS.finditer(body):
        short = match[2]
        words = re.findall(r'[A-Za-z]+', match[1])[-10:]
        # Find the shortest matching suffix. Compound words can contribute more
        # than one letter (e.g. a repeated initial inside a compound word).
        def aligned(parts, letters):
            if not parts:
                return not letters
            word = parts[0].casefold()
            if not letters or letters[0] != word[0]:
                return False
            for size in range(1, min(3, len(letters)) + 1):
                iterator = iter(word)
                if all(any(c == letter for c in iterator) for letter in letters[:size]):
                    if aligned(parts[1:], letters[size:]):
                        return True
            return False
        for size in range(2, min(len(words), len(short)) + 1):
            parts = words[-size:]
            if aligned(parts, short.casefold()):
                pairs.add((' '.join(parts).casefold(), short.casefold()))
                break
    return pairs


def expand_aliases(question, aliases):
    """One-hop query hints only; preserve the original question and source text."""
    original = str(question or '')
    additions = []
    for full, short in sorted(aliases):
        full_pattern = r'\b' + r'[\s-]+'.join(map(re.escape, full.split())) + r'\b'
        has_full = bool(re.search(full_pattern, original, re.I))
        has_short = bool(re.search(r'\b' + re.escape(short) + r'\b', original, re.I))
        if has_full and not has_short:
            additions.append(short.upper())
        elif has_short and not has_full:
            additions.append(full)
        if len(additions) == 6:
            break
    return ' '.join([original, *dict.fromkeys(additions)])


def expanded_query(collection, generation, question):
    cached = _cache
    if collection is None or cached is None or cached[0] != (id(collection), generation, collection.count()):
        return question
    return expand_aliases(question, cached[3])


def invalidate():
    global _cache, _epoch
    # An upload must not block the event loop behind a corpus scan. A build
    # finishing concurrently retains its old generation key and cannot be used
    # by a query for the newly published generation.
    _cache = None
    _epoch += 1


def schedule_build(collection, generation, is_current):
    """Warm the complete body index off the request path; coalesce updates."""
    global _requested_build, _worker_running
    with _scheduler_lock:
        _requested_build = (collection, generation, is_current)
        if _worker_running:
            return
        _worker_running = True
    threading.Thread(target=_build_pending, name='laborlens-body-index', daemon=True).start()


def _build_pending():
    global _requested_build, _worker_running
    while True:
        with _scheduler_lock:
            request, _requested_build = _requested_build, None
            if request is None:
                _worker_running = False
                return
        collection, generation, is_current = request
        try:
            if is_current():
                candidates(collection, generation, '', limit=0, is_current=is_current)
        except Exception as exc:
            logger.warning('Body index warm-up unavailable (%s)', type(exc).__name__)


def terms(text):
    # Prefix families supplement exact matches, e.g. agreement/agree and
    # house/household. Ordered pairs retain short phrases even when the source
    # inserts a qualifier ("double regular holiday" vs. "double holiday").
    # Semantic retrieval and verification remain required.
    exact = discovery_terms(text)
    phrase_features = set()
    for sentence in re.split(r'[.!?;\n]+', str(text or '')):
        words = [match.group().casefold() for match in _CONTENT_WORD.finditer(sentence)
                 if match.group().casefold() in exact]
        phrase_features.update(
            f'^p:{left}:{right}'
            for index, left in enumerate(words)
            for right in words[index + 1:index + 3]
            if left != right
        )
    task_features = {'^task:calculation'} if re.search(
        r'\b(?:calculat\w*|comput\w*|formula)\b', str(text or ''), re.I) else set()
    return (exact | {'~' + term[:5] for term in exact if len(term) >= 5}
            | phrase_features | task_features)


def candidates(collection, generation, question, limit=100, *, is_current=None):
    global _cache
    count = collection.count()
    if type(count) is not int or count < 1:
        return []
    key = (id(collection), generation, count)
    epoch = _epoch
    with _lock:
        cached = _cache
        if cached is None or cached[0] != key:
            postings, sizes, aliases = defaultdict(set), {}, set()
            for offset in range(0, count, 500):
                if epoch != _epoch or (is_current is not None and not is_current()):
                    return []
                rows = collection.get(offset=offset, limit=500, include=['documents'])
                for identity, text in zip(rows.get('ids') or [], rows.get('documents') or []):
                    body = source_passage_text(text or '')
                    aliases.update(document_aliases(body))
                    features = terms(body)
                    sizes[identity] = max(1, len(features))
                    for feature in features:
                        postings[feature].add(identity)
            cached = key, postings, sizes, frozenset(aliases)
            if epoch == _epoch and (is_current is None or is_current()):
                _cache = cached
    return _rank(cached, question, limit)


def cached_candidates(collection, generation, question, limit=100):
    """Read a ready index without waiting for its lock or building the corpus.

    A cold index returns None (not a negative match). The caller can perform
    a bounded direct text lookup instead. The complete immutable cache tuple
    is published atomically only after a build finishes.
    """
    cached = _cache
    if cached is None or cached[0] != (id(collection), generation, collection.count()):
        return None
    return _rank(cached, question, limit)


def _rank(cached, question, limit):
    _, postings, sizes, aliases = cached
    scores, matches = defaultdict(float), defaultdict(set)
    wanted = terms(expand_aliases(question, aliases))
    for term in sorted(wanted):
        ids = postings.get(term, ())
        phrase_feature = term.startswith('^p:')
        multiplier = 1.35 if phrase_feature else .55 if term.startswith('~') else 1
        weight = math.log(1 + (len(sizes) + 1) / (len(ids) + 1)) * multiplier
        for identity in ids:
            scores[identity] += weight
            if not phrase_feature:
                matches[identity].add(term.lstrip('~')[:5])
    ranked = sorted(scores, key=lambda identity: (
        scores[identity] / (1 + .10 * math.log(sizes[identity])), len(matches[identity]), identity), reverse=True)
    minimum = min(2, len({term[:5] for term in discovery_terms(question)}))
    ranked = [identity for identity in ranked if len(matches[identity]) >= minimum][:limit]
    maximum = max((scores[identity] for identity in ranked), default=1)
    return [(identity, min(1., scores[identity] / maximum)) for identity in ranked]
