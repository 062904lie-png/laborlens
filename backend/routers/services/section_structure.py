"""Document-derived section boundaries; no topic, law or answer catalog."""
import hashlib
import re


_HEADINGS = re.compile(
    r'(?<!\w)(?P<kind>BOOK|TITLE|CHAPTER|RULE|ARTICLE|ART\.|SECTION|SEC\.)'
    r'\s+(?P<number>\d+[A-Z]?|[IVXLC]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)'
    r'(?P<separator>\s*[.:\u2013\u2014-]\s*|(?=\s|$))', re.I)
_LEVELS = {'book': 0, 'title': 1, 'chapter': 2, 'rule': 3}


def section_ranges(text):
    """Return explicit section ranges, with hierarchy and occurrence identity.

    Bare inline references ("under Section 4") are not headings. When extraction
    has flattened lines, uppercase numbered headings need a separator. A range
    stops at the next structural heading, never at a fixed character count.
    """
    headings = []
    for match in _HEADINGS.finditer(text):
        raw = match.group('kind')
        kind = {'sec.': 'section', 'art.': 'article'}.get(raw.lower(), raw.lower())
        line_start = text.rfind('\n', 0, match.start()) + 1
        at_line_start = not text[line_start:match.start()].strip()
        separator = match.group('separator').strip()
        if not at_line_start and not (raw.isupper() and separator):
            continue
        if kind in {'section', 'article'} and not separator:
            continue
        if re.search(r'\.{3,}', text[match.end():match.end() + 100].split('\n')[0]):
            continue  # A contents entry is not an operative section.
        headings.append((match, kind))
    hierarchy = {}
    result = []
    for i, (match, kind) in enumerate(headings):
        number = match.group('number').upper()
        if kind in _LEVELS:
            hierarchy = {k: v for k, v in hierarchy.items() if _LEVELS[k] < _LEVELS[kind]}
            hierarchy[kind] = number
            continue
        if kind != 'section':
            continue  # Articles already have document-spanning parents.
        end = headings[i + 1][0].start() if i + 1 < len(headings) else len(text)
        if end - match.start() < 80:
            continue
        path = '/'.join(f'{k}:{hierarchy[k]}' for k in _LEVELS if k in hierarchy)
        locator = (path + '/' if path else '') + 'section:' + number
        identity = hashlib.sha256(f'{match.start()}:{locator}'.encode()).hexdigest()[:24]
        result.append((match.start(), end, {
            'section_key': identity, 'section_path': path, 'section_number': number,
            'section': 'Section ' + number, **hierarchy,
        }))
    return result
