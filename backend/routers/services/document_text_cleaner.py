"""Shared cleanup for legal source text before indexing or prompting."""

import re


_FOOTNOTE_MARKER = r"(?:\[?\d{1,3}\]?|[¹²³⁴⁵⁶⁷⁸⁹⁰]+)"
_FOOTNOTE_LEAD = r"(?:as\s+amended\s+by|refer\s+to|see\s+also|please\s+see)"
FOOTNOTE_PREFIX_RE = re.compile(
    rf"^\s*{_FOOTNOTE_MARKER}\s*(?:[.)\]-]\s*)?{_FOOTNOTE_LEAD}\b",
    re.IGNORECASE,
)
FOOTNOTE_CITATION_ONLY_RE = re.compile(
    rf"^\s*{_FOOTNOTE_MARKER}\s*(?:[.)\]-]\s*)?"
    r"(?:r\.?\s*a\.?|republic\s+act|p\.?\s*d\.?|presidential\s+decree|"
    r"d\.?\s*o\.?|department\s+order|article|art\.?|section|sec\.?)"
    r"\s*(?:no\.?\s*)?[a-z0-9.-]+(?:\s*\([^)]*\))?[.;]?\s*$",
    re.IGNORECASE,
)
INLINE_FOOTNOTE_START_RE = re.compile(
    rf"(?<![\w(]){_FOOTNOTE_MARKER}\s+(?={_FOOTNOTE_LEAD}\b)",
    re.IGNORECASE,
)
INLINE_FOOTNOTE_END_RE = re.compile(
    rf"(?=\s+{_FOOTNOTE_MARKER}\s+(?={_FOOTNOTE_LEAD}\b)|"
    r"\s+(?=(?:ART\.|Article|BOOK|Book|TITLE|Title|CHAPTER|Chapter|RULE|Rule|"
    r"SECTION|Section)\s+(?:[IVXLCDM]+|\d+))|\Z)",
    re.IGNORECASE,
)
STRUCTURAL_HEADING_RE = re.compile(
    r"^\s*(?:ART\.|Article|BOOK|Book|TITLE|Title|CHAPTER|Chapter|RULE|Rule|"
    r"SECTION|Section)\s+(?:[IVXLCDM]+|\d+)",
    re.IGNORECASE,
)


def _remove_flattened_footnotes(text: str) -> str:
    """Remove amendment/cross-reference footnotes after line breaks were flattened."""
    output = text
    search_from = 0
    while True:
        start = INLINE_FOOTNOTE_START_RE.search(output, search_from)
        if not start:
            break
        end = INLINE_FOOTNOTE_END_RE.search(output, start.end())
        if not end:
            output = output[: start.start()]
            break
        output = output[: start.start()] + " " + output[end.start() :]
        search_from = max(0, start.start() - 1)
    return output


def strip_document_footnotes(text: str) -> str:
    """Remove legal-editor footnotes without removing numbered substantive clauses."""
    kept = []
    dropping_footnote = False
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if FOOTNOTE_PREFIX_RE.match(line) or FOOTNOTE_CITATION_ONLY_RE.match(line):
            dropping_footnote = True
            continue
        if dropping_footnote:
            if not line:
                dropping_footnote = False
                continue
            if not STRUCTURAL_HEADING_RE.match(line):
                continue
            dropping_footnote = False
        kept.append(raw_line.rstrip())
    cleaned = _remove_flattened_footnotes("\n".join(kept))
    if re.search(r"\bRegion\s+VI\b|\bWestern\s+Visayas\b", cleaned, re.IGNORECASE):
        # Correct stable OCR substitutions in the official RTWPB-VI wage page.
        cleaned = re.sub(r"\bRBVEDW-07\b", "RBVI-DW-07", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bRBVE29\b", "RBVI-29", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
