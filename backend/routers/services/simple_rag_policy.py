"""Local request routing only; confidence never certifies a legal claim."""
import re

from .evidence_selection import requested_points
from .legal_version_ranking import historical_query
from .question_understanding import retrieval_ready_request


def simple_legal_rag(question, profile=None):
    profile = profile or {}
    text = str(question or '')
    from .overview_discovery import overview_subject
    if overview_subject(text):
        # A short menu/overview still needs complementary provisions.
        return False
    if not re.search(r'\?|^(?:what|how|when|who|can|is|are|do|does|ano|paano)\b', text, re.I):
        return False
    if (len(text) > 500 or len(requested_points(text)) != 1
            or not retrieval_ready_request(text) or historical_query(text)
            or profile.get('task') in {'calculation_request', 'comparison_question', 'jurisprudence_request'}
            or profile.get('calculation_requested') or profile.get('validation_feedback')
            or len(profile.get('requested_issues') or []) > 1
            or profile.get('actor_ambiguous') or profile.get('sector_conflict')):
        return False
    # Supplied hours can be scenario facts. Asking for a quantity or formula is
    # computation; a condition/exception request needs the stronger pipeline.
    if re.search(r'\b(?:calculat\w*|comput\w*|formula|how much|compare|versus|difference|'
                 r'historical|previous|amend\w*|version|except\w*|unless|if|conditions?|circumstances|'
                 r'someone|somebody|a person)\b|\b(?:he|she|they|their|his|her)\b', text, re.I):
        return False
    populations = [r'caregivers?', r'\b(?:COS|JO|government)\b',
                   r'\b(?:domestic|kasambahay)\b', r'\bhospital\b', r'\bprivate[- ]sector\b']
    if sum(bool(re.search(p, text, re.I)) for p in populations) > 1:
        return False
    # Open-ended scenario checklists are not one clearly bounded legal issue.
    if re.search(r'\bwhat should I check\b', text, re.I):
        return False
    return True


def strong_simple_retrieval(question, chunks, profile):
    from .ranking_policy import rerank_decision
    from .evidence_selection import select_evidence
    from .evidence_scope import scope_completeness, source_scope
    if not profile.get('simple_legal_rag'):
        return False
    if len({source_scope(c) for c in chunks}) > 1:
        return False
    # Discovery confidence must not depend on whether model ranking is mandatory.
    # This evaluates the existing thresholds; it never skips the ranking stage.
    call, reason, ranked = rerank_decision(question, chunks, profile=profile, confidence_only=True)
    if call or reason != 'confident_local_vector_match':
        return False
    selected, diagnostics = select_evidence(ranked, question, preserve_order=True)
    return bool(selected and not diagnostics['discovery_gaps']
                and scope_completeness(question, selected)['scope_complete'])


def ordinary_overtime_calculation_evidence(question, chunks, profile):
    """Allow a fully specified ordinary-day calculation to stop broad discovery.

    This only ends retrieval expansion after the general rule, ordinary-day
    multiplier, calculation inputs, and general worker scope are all present.
    It does not skip conditional ranking, evidence recovery, generation, or
    independent verification.
    """
    if (profile.get('topic') != 'overtime_pay'
            or profile.get('task') not in {'calculation_request', 'personalized_computation'}
            or not profile.get('calculation_requested')):
        return False
    scenario = profile.get('scenario') or {}
    if (scenario.get('day_type') != 'ordinary_day'
            or not (scenario.get('hourly_rate') or scenario.get('daily_wage'))
            or not scenario.get('overtime_hours')
            or (not scenario.get('hourly_rate') and not scenario.get('normal_work_hours'))):
        return False
    if not chunks:
        return False

    from config import settings
    from .evidence_scope import source_scope
    from .kb_calculation_support import calculation_evidence_present
    from .legal_topic_matrix import supported_passage_concepts
    from .rag_policy import GENERAL_SECTOR, missing_required_authorities
    from .ranking_policy import conflict_reason
    from .calculators import build_overtime_calculation

    if any(source_scope(chunk) != GENERAL_SECTOR for chunk in chunks):
        return False
    if conflict_reason(question, chunks):
        return False
    if max(float(chunk.get('rerank_score', chunk.get('score', 0)) or 0) for chunk in chunks) \
            < float(settings.RAG_RELIABLE_SOURCE_THRESHOLD):
        return False
    required = list(profile.get('required_concepts') or [])
    if not required or not set(required).issubset(
            set().union(*(supported_passage_concepts(chunk, required) for chunk in chunks))):
        return False
    if missing_required_authorities(chunks, profile):
        return False
    calculation = build_overtime_calculation(scenario).to_payload()
    return calculation.get('amount') is not None and calculation_evidence_present(calculation, chunks)
