"""Regional applicability and discovery hints; never a catalog of wage amounts."""
import re

from .rag_policy import REGION_ALIASES, WAGE_ORDER_QUERY_RE, wage_order_query_details


def regional_wage_query(question):
    # A location in an overtime/final-pay question does not request a regional
    # minimum wage. Do not impose the wage-rate completeness gate on that task.
    question = re.sub(r'\b(minimum\s+)wages\b', r'\1wage', question, flags=re.I)
    return bool(wage_order_query_details(question)['is_regional_wage_query'] and (
        WAGE_ORDER_QUERY_RE.search(question) or re.search(r'\bwage\s+rates?\b', question, re.I)))


def document_regions(metadata, text=''):
    """Owning-document identifiers outrank incidental citations in a passage."""
    meta = metadata or {}
    explicit = str(meta.get('region_code') or meta.get('region') or '')
    owner = ' '.join(str(meta.get(k) or '') for k in (
        'document_title', 'original_filename', 'source', 'law_number',
        'source_url', 'official_url'))
    codes = re.findall(
        r'\b(?:RTWPB|RO|RB|R)[- _–—]*(NCR|CAR|BARMM|MIMAROPA|IVA|IVB|XIII|XII|XI|IX|VIII|VII|VI|V|III|II|I|X|\d{1,2})'
        r'(?=[- _–—./]|$)', owner, re.I)
    if not codes:
        codes = re.findall(r'wage[- _]+order[- _]*(?:no\.?[- _]*)?(\d{1,2})(?=[- _])', owner, re.I)
    codes = [str(int(c)) if c.isdigit() else c for c in codes]
    descriptors = str(meta.get('owning_region') or '') or ' '.join('Region ' + c if c.upper() not in {
        'NCR', 'CAR', 'BARMM', 'MIMAROPA'} else c for c in codes) or owner
    if not codes and explicit:
        descriptors = explicit
    descriptors = re.sub(r'[-_]', ' ', descriptors)
    regions = {code for pattern, code, _ in REGION_ALIASES
               if code not in {'Mindanao', 'Visayas', 'Luzon'} and pattern.search(descriptors)}
    if regions:
        return regions
    return {code for pattern, code, _ in REGION_ALIASES
            if code not in {'Mindanao', 'Visayas', 'Luzon'} and pattern.search(text)}


def regional_wage_document(metadata):
    owner = ' '.join(str((metadata or {}).get(k) or '') for k in (
        'document_title', 'original_filename', 'source', 'source_url', 'doc_type'))
    return bool(re.search(r'wage[- _]+orders?|rtwpb|regional[- _]+wage|region-[ivx\d]+/', owner, re.I))


def domestic_wage_document(metadata):
    owner = ' '.join(str((metadata or {}).get(k) or '') for k in (
        'document_title', 'original_filename', 'source', 'source_url'))
    return bool(re.search(r'[- _]DW[- _]|domestic[- _]+workers?|kasambahay', owner, re.I))


def wage_scope_rejection(question, item):
    """Reject a known inapplicable wage order, not national supporting law."""
    details = wage_order_query_details(question)
    meta = item.get('metadata') or item
    if not regional_wage_query(question) or not regional_wage_document(meta):
        return None
    # Comparisons may deliberately request more than one regional order.
    parts = re.split(r'\n(?:Current follow-up question|User clarification[^:]*)\s*:', question, flags=re.I)
    if len(parts) > 1 and not re.search(r'\b(?:compare|comparison|versus|vs\.?|difference|both)\b', parts[-1], re.I):
        # A new region in the latest reply replaces the earlier location, while
        # an elliptical worker-type reply still inherits the previous location.
        requested_regions = next((regions for part in reversed(parts)
                                  if (regions := document_regions({}, part))), set())
    else:
        requested_regions = document_regions({}, question)
    regions = document_regions(meta, str(item.get('text') or item.get('excerpt') or ''))
    if regions and not regions.intersection(requested_regions or {details['region_code']}):
        return 'wrong_region_wage_order'
    domestic = bool(re.search(r'\b(?:domestic\s+workers?|kasambahay|household\s+helpers?)\b', question, re.I))
    if domestic_wage_document(meta) and not domestic:
        return 'wrong_worker_type_wage_order'
    if domestic and not domestic_wage_document(meta):
        body = str(item.get('text') or item.get('excerpt') or '').split('\nText:\n')[-1]
        if re.search(r'\b(?:daily|per\s+day)\b', body, re.I) and not re.search(
                r'\b(?:domestic\s+workers?|kasambahay)\b', body, re.I):
            return 'wrong_worker_type_wage_order'
    return None


def regional_rate_present(question, items):
    """Discovery cannot stop on a matching title or coverage paragraph alone."""
    details = wage_order_query_details(question)
    if not regional_wage_query(question):
        return True
    domestic = bool(re.search(r'\b(?:domestic\s+workers?|kasambahay)\b', question, re.I))
    for item in items:
        meta = item.get('metadata') or item
        body = str(item.get('text') or item.get('excerpt') or '').split('\nText:\n')[-1]
        if (wage_scope_rejection(question, item) or
                details['region_code'] not in document_regions(meta, body)):
            continue
        amount = re.search(r'(?:₱|\bPHP|\bP\s*|pesos?)\s*\d[\d,]*(?:\.\d+)?', body, re.I)
        table_amount = (re.search(r'\bnew\s+minimum\s+wage\s+rates?\b', body, re.I)
                        and re.search(r'\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b|\b\d+\.\d{2}\b', body))
        if (re.search(r'\b(?:minimum\s+wage|wage\s+rates?)\b', body, re.I)
                and (amount or table_amount)
                and (not domestic or re.search(r'\bmonthly|per\s+month\b', body, re.I))):
            return True
    return False
