"""Distinguish requests to explain a rule from requests to list its records."""
import re


def legal_basis_explanation_requested(query):
    text = str(query or '')
    basis = re.search(r'\blegal\s+(?:basis|na\s+batayan|nga\s+basihan)\b', text, re.I)
    explanation = re.search(r'\b(?:explain|explanation|meaning|why|how|ipaliwanag|paliwanag|'
                            r'ipaathag|paathag|bakit|ngaa|paano)\b', text, re.I)
    return bool(basis and explanation)


def source_records_requested(query):
    """A bare 'legal basis' is not a source-list command; default to synthesis.

    This only selects the metadata-list shortcut. It does not change search
    words, legal applicability, answer content or source attribution.
    """
    text = str(query or '')
    if legal_basis_explanation_requested(text):
        return False
    records = re.search(r'\b(?:sources?|citations?|references?|source\s+links?|'
                        r'sanggunian|pinagkunan)\b', text, re.I)
    list_action = re.search(r'\b(?:show|list|give|provide|display|pakita|ipakita|'
                            r'ilista|ihatag|ipakita)\b', text, re.I)
    if records and list_action:
        return True
    if records and re.search(r'\b(?:what|which|where|ano|alin|diin)\b', text, re.I):
        return bool(re.search(r'\b(?:supports?|supporting|used|cited|retrieved|'
                              r'previous|earlier|answer|response|formula|calculation)\b', text, re.I))
    # Preserve explicit requests for the basis of an earlier answer. The
    # source-history path normally handles these before global discovery.
    return bool(list_action and re.search(r'\blegal\s+basis\b', text, re.I)
                and re.search(r'\b(?:your|previous|last|earlier|that|this)\s+'
                              r'(?:answer|response)\b', text, re.I))
