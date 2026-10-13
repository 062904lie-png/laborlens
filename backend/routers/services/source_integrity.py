"""Validate citation locators against the actual passage, not indexed headers."""
import re
import hashlib
import json


def source_passage_identity(text, metadata, origin=''):
    """Identify the exact displayed evidence, never just its parent chunk hash.

    A chunk can supply several different windows or table rows. Equal titles,
    pages or stored content_hash values do not make those passages equivalent.
    Preserve document/version/scope identity even when their wording is equal.
    """
    meta = metadata or {}
    fields = ('source_passage_id', 'doc_db_id', 'index_version', 'canonical_document_id',
              'document_title', 'source_name', 'source', 'source_url', 'official_url',
              'article_number', 'article_section', 'citation', 'book', 'title',
              'chapter', 'rule', 'section', 'source_scope', 'coverage', 'sector',
              'law_version', 'effective_date', 'legal_status')
    values = [str(meta.get(key) or '').strip() for key in fields]
    values.extend([str(meta.get('page_number') or meta.get('page') or '').strip(),
                   str(origin or '').strip(),
                   re.sub(r'\s+', ' ', str(text or '')).strip()])
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode('utf-8')).hexdigest()


def verified_article_metadata(text, metadata):
    result = dict(metadata or {})
    body = str(text or '').split('\nText:\n')[-1]
    number = str(result.get('article_number') or '')
    headings = re.findall(r'\b(?:ART\.|ARTICLE)\s*(\d{1,3})\b', body, re.I)
    # An overlapping chunk can include more than one article. Do not replace
    # one locator with an arbitrary nearby number.
    if number and headings and number not in headings:
        for key in ('article_number', 'article_title', 'article', 'article_section', 'citation'):
            result[key] = ''
    title = str(result.get('article_title') or '')
    normalize = lambda value: re.sub(r'[^a-z0-9]+', ' ', value.lower()).strip()
    if title and normalize(title) not in normalize(body):
        result['article_title'] = ''
        if title in str(result.get('citation') or ''):
            result['citation'] = f"Article {result['article_number']}" if result.get('article_number') else ''
    return result


def verified_structural_locators(text, metadata):
    """Suppress contradictory numeric locators without guessing a replacement."""
    result = dict(metadata or {})
    body = str(text or '').split('\nText:\n')[-1]

    def locator_number(token):
        token = token.lower().rstrip('.')
        if token.isdigit():
            return int(token)
        words = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
        if token in words:
            return words[token]
        if not token or not re.fullmatch(r'm{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})', token):
            return None
        values = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100, 'd': 500, 'm': 1000}
        total, previous = 0, 0
        for char in reversed(token):
            value = values[char]
            total += -value if value < previous else value
            previous = max(previous, value)
        return total

    for key in ('book', 'title', 'chapter', 'rule', 'section'):
        pattern = re.compile(r'\b' + key + r'\s+(\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b', re.I)
        locator = pattern.search(str(result.get(key) or ''))
        if not locator and result.get(key):
            locator = pattern.search(key + ' ' + str(result[key]))
        headings = {locator_number(m.group(1)) for m in pattern.finditer(body)} - {None}
        if not locator or not headings or locator_number(locator.group(1)) is None:
            continue
        # A window containing several numbered provisions cannot be labeled
        # as one specific provision merely because that number occurs in it.
        # Retain the document/page and exact excerpt, without guessing which
        # provision supplies a particular claim.
        if len(headings) == 1 and locator_number(locator.group(1)) in headings:
            continue
        rejected_number = locator_number(locator.group(1))
        result[key] = ''
        result[key + '_title'] = ''
        for field in ('citation', 'article_section'):
            if any(locator_number(m.group(1)) == rejected_number
                   for m in pattern.finditer(str(result.get(field) or ''))):
                result[field] = ''
    return result
