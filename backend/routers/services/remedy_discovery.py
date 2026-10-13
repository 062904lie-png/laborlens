"""Procedural discovery hints, not filing instructions or legal conclusions."""
import re


def remedy_requested(question):
    return bool(re.search(
        r'\b(?:what\s+(?:can|should|do)\s+(?:I|we)\s+do|'
        r'how\s+(?:can|do|should)\s+(?:I|we)\s+(?:claim|recover|report|complain)|'
        r'(?:file|filing|submit|lodge)\s+(?:a\s+)?complain(?:t)?|'
        r'where\s+.{0,35}(?:complain|report|help)|'
        r'ano\s+(?:ang\s+)?(?:pwede|puwede|dapat|maaari).{0,30}(?:gawin|himuon)|'
        r'paano.{0,35}(?:magreklamo|reklamo)|diin.{0,35}(?:reklamo|bulig))\b',
        str(question or ''), re.I))


def procedural_passage(item):
    body = str(item.get('text') or item.get('excerpt') or '').split('\nText:\n')[-1]
    return bool(re.search(
        r'request\s+for\s+assistance|single[-\s]+entry\s+approach|\bsena\b|'
        r'(?:fil(?:e|ing)|lodg(?:e|ing)|submit)[^.!?\n]{0,100}\bcomplaints?\b|'
        r'\bcomplaints?\b[^.!?\n]{0,100}(?:filed|lodged|submitted|jurisdiction)', body, re.I))


def remedy_evidence_present(question, items):
    return not remedy_requested(question) or any(procedural_passage(c) for c in items)


def remedy_search_probe(question):
    # Generic procedure vocabulary supplements the unchanged factual query.
    # No document/category is an allowlist; no filing rule is asserted here.
    return ('labor complaint filing procedure request for assistance conciliation') \
        if remedy_requested(question) else None
