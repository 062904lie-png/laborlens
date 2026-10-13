"""Conservative local task routing; no legal rules or support certificates."""
import math
import re

from config import settings
from .document_evidence import (
    duplicate_key, eligible_candidates, operative_quality, question_candidate_scorer,
    rank_question_candidates,
)
from .evidence_selection import discovery_hit, discovery_terms, discovery_qualifiers, requested_points, evidence_tokens, context_budget
from .legal_topic_matrix import source_passage_text
from .legal_version_ranking import apply_version_precedence, historical_query, version_precedence


def conflict_reason(question, chunks, *, unresolved_only=False):
    metadata = [c.get('metadata') or {} for c in chunks]
    if any(m.get('amendment_metadata_conflict') for m in metadata):
        return 'amendment_conflict'
    if not unresolved_only and version_precedence(question, chunks):
        return 'legal_versions'
    sectors = {str(m.get('sector')) for m in metadata if m.get('sector')}
    scopes = {str(m.get('source_scope') or m.get('coverage')) for m in metadata
              if m.get('source_scope') or m.get('coverage')}
    if len(sectors) > 1 or len(scopes) > 1:
        return 'source_scope_conflict'
    # Multiple versions of the same provision without a curated precedence
    # relationship deserve a model hint, even if vector scores are far apart.
    groups = {}
    for i, m in enumerate(metadata):
        owner = m.get('canonical_document_id') or m.get('doc_db_id')
        provision = m.get('section_key') or m.get('article_number') or m.get('section')
        version = m.get('index_version') or m.get('effective_date')
        if owner and provision and version:
            groups.setdefault((str(owner), str(provision)), []).append((i, str(version)))
    edges = {tuple(pair) for pair in version_precedence(question, chunks)}
    for members in groups.values():
        for i, first in members:
            for j, second in members:
                if first != second and (not unresolved_only or (i, j) not in edges and (j, i) not in edges):
                    return 'competing_versions'
    return None


def straightforward_question(question, profile=None):
    profile = profile or {}
    return bool(
        len(question or '') <= int(getattr(settings, 'LABORLENS_LIGHT_QUESTION_CHAR_LIMIT', 160))
        and len(requested_points(question)) == 1
        and re.match(r'^(?:what|how|when|who|ano|paano|kailan|sino)\b', question or '', re.I)
        and not historical_query(question)
        and profile.get('task') not in {'comparison_question', 'jurisprudence_request'}
        and not profile.get('validation_feedback')
        and not profile.get('sector_restricted')
        and not re.search(r'\b(?:terminat\w*|dismiss\w*|separation|resign\w*|fired|redundancy|dispute|case|court|'
                         r'jurisprudence|illegal|lawful|conflict|amendment|except|unless|if|i|my|our|'
                         r'government|domestic|kasambahay|managerial|hospital)\b|'
                         r'(?:₱|php|peso)|\d+[.,]\d+|\b\d+\b', question or '', re.I))


def rerank_decision(question, chunks, *, profile=None, confidence_only=False):
    ranked = rank_question_candidates(chunks, question)
    if not ranked:
        return False, 'no_candidates', ranked
    if not confidence_only and not getattr(settings, 'LABORLENS_LLM_RERANK_ENABLED', True):
        return False, 'disabled', ranked
    if not confidence_only and not getattr(settings, 'LABORLENS_CONDITIONAL_RERANK_ENABLED', True):
        return True, 'always_enabled', ranked
    if len(requested_points(question)) > 1:
        return True, 'multiple_requested_points', ranked
    conflict = conflict_reason(question, ranked)
    if conflict:
        return True, conflict, ranked
    from .evidence_scope import scope_completeness, requested_scope, source_scope
    scope = scope_completeness(question, ranked)
    if scope['scope_gap']:
        return True, 'scope_incomplete', ranked
    wanted = requested_scope(question)
    if wanted is None and any(source_scope(c) != 'general_private_employment' for c in ranked):
        return True, 'general_special_scope_conflict', ranked
    simple = bool((profile or {}).get('simple_legal_rag'))
    if wanted is None and not simple and re.search(r'\b(?:rules|regulations|rights)\b', question, re.I):
        return True, 'broad_scope_question', ranked
    if not simple and not straightforward_question(question):
        return True, 'application_or_complex_question', ranked
    top = ranked[0]
    terms = discovery_terms(question)
    body = discovery_terms(source_passage_text(top.get('text') or ''))
    # A single clear requested point can be checked with the same discovery
    # terms/qualifiers without running the full context packer a second time.
    # Actual selection and final claim verification still run normally.
    if not discovery_hit(terms, body, discovery_qualifiers(question)):
        return True, 'discovery_gap', ranked
    if evidence_tokens([top]) > context_budget(question):
        return True, 'evidence_budget_gap', ranked
    try:
        vector_order = sorted(ranked, key=lambda c: float(c.get('score') or 0), reverse=True)
        vector_top = float(vector_order[0].get('score') or 0)
        vector_second = float(vector_order[1].get('score') or 0) if len(vector_order) > 1 else 0.0
        lexical = float(top.get('lexical_score') or 0)
        score = question_candidate_scorer(question)
        scores = [score(c) for c in ranked]
        margin = scores[0] - max(scores[1:], default=0)
    except (ValueError, TypeError):
        return True, 'invalid_local_score', ranked
    if not all(math.isfinite(v) for v in [vector_top, vector_second, lexical, margin, *scores]):
        return True, 'invalid_local_score', ranked
    if vector_order[0] is not top:
        return True, 'vector_local_order_disagreement', ranked
    content_overlap = len(terms & body)
    if (vector_top < float(getattr(settings, 'LABORLENS_RERANK_SKIP_MIN_VECTOR_SCORE', .72))
            or vector_top - vector_second < float(getattr(settings, 'LABORLENS_RERANK_SKIP_MIN_VECTOR_MARGIN', .10))
            or content_overlap < int(getattr(settings, 'LABORLENS_RERANK_SKIP_MIN_CONTENT_OVERLAP', 3))
            or lexical < float(getattr(settings, 'LABORLENS_RERANK_LEXICAL_AGREEMENT', .60))
            or not discovery_hit(terms, body) or operative_quality(top, question) <= 0):
        return True, 'insufficient_local_agreement', ranked
    if margin < float(getattr(settings, 'LABORLENS_RERANK_LOCAL_MARGIN', .10)):
        return True, 'close_local_scores', ranked
    return False, 'confident_local_vector_match', ranked


def integrate_recovery(question, prior, additions, profile, *, prior_scores=None):
    """Score only new views; never sort an existing model order by local score.

    Insert new evidence ahead of its local-score peer. Prior relative ordering
    stays intact except for the existing explicit legal-version constraints.
    Do not cap the union before coverage packing: a recovered point must have
    a chance to compete with all previously ranked evidence.
    """
    old = eligible_candidates(prior, profile)
    known = {duplicate_key(c) for c in old}
    new = rank_question_candidates([c for c in eligible_candidates(additions, profile)
                                    if duplicate_key(c) not in known], question)
    score = question_candidate_scorer(question)
    scores = dict(prior_scores or {})
    for item in [*old, *new]:
        key = duplicate_key(item)
        if key not in scores:
            scores[key] = score(item)
    merged = list(old)
    for item in new:
        value = scores[duplicate_key(item)]
        index = next((i for i, c in enumerate(merged)
                      if value > scores[duplicate_key(c)]
                      or value == scores[duplicate_key(c)] and duplicate_key(c) in known), len(merged))
        merged.insert(index, item)
    # Preserve prior model order within each scope class; applicability outranks
    # a topical score, while explicit legal-version relationships still win.
    from .evidence_scope import scope_priority, source_scope
    if len({source_scope(c) for c in merged}) > 1:
        merged = sorted(merged, key=lambda c: scope_priority(question, c), reverse=True)
    return apply_version_precedence(question, merged)


def recovery_introduces_conflict(question, prior, additions):
    """Do not pay twice to rerank an unchanged set of sector alternatives."""
    combined = conflict_reason(question, [*prior, *additions], unresolved_only=True)
    if not combined:
        return False
    if (combined != 'source_scope_conflict'
            or conflict_reason(question, prior, unresolved_only=True) != combined):
        return True
    from .evidence_scope import source_scope
    if not {source_scope(c) for c in additions} <= {source_scope(c) for c in prior}:
        return True
    for field in ('sector', 'source_scope', 'coverage'):
        known = {str((c.get('metadata') or {}).get(field)) for c in prior
                 if (c.get('metadata') or {}).get(field)}
        new = {str((c.get('metadata') or {}).get(field)) for c in additions
               if (c.get('metadata') or {}).get(field)}
        if not new <= known:
            return True
    # Explicit amendment/competing-version conflicts returned above still rank.
    return False


def provision_overlap_diagnostics(chunks):
    """Measure parent/child provision overlap without reading/logging bodies."""
    groups = {}
    parents = 0
    for chunk in chunks:
        m = chunk.get('metadata') or {}
        role = str(m.get('chunk_role') or '')
        parent = role in {'article_parent', 'document_article_parent', 'document_section_parent'}
        parents += int(parent)
        provision = m.get('section_key') or m.get('article_number') or m.get('section')
        if m.get('doc_db_id') and m.get('index_version') and provision:
            key = (str(m['doc_db_id']), str(m['index_version']), str(provision))
            groups.setdefault(key, []).append(parent)
    return {'retrieval_parent_candidates': parents,
            'retrieval_provision_duplicate_candidates': sum(max(0, len(g) - 1) for g in groups.values()),
            'retrieval_parent_child_overlap_groups': sum(any(g) and not all(g) for g in groups.values())}
