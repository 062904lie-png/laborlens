"""Question-derived context packing. Selection is relevance, never legal proof."""
import json
import math
import re

from config import settings
from .request_budget import estimate_request_tokens


_REQUEST_NOISE = {'legally', 'entitled', 'allowed', 'whether', 'shouldn', 'doesn',
                  'don', 'wouldn', 'couldn', 'isn', 'aren', 'wasn', 'weren'}
# Generic quantity wording, never legal topics, categories or scope qualifiers.
_WORDING_MODIFIERS = {'extra', 'additional', 'more', 'further', 'supplemental',
                     'supplementary', 'added'}


def discovery_terms(text):
    """User-wording signals only; not a catalog of laws or an entailment test."""
    from .issue_evidence import issue_terms
    terms = issue_terms(text) - _REQUEST_NOISE
    # Linguistic inflections only, never topic/law synonyms or legal answers.
    # A noun request must not exclude its operative verb (pay versus paid).
    irregular = {'paid': 'pay', 'took': 'take', 'taken': 'take', 'made': 'make',
                 'gave': 'give', 'given': 'give', 'written': 'write', 'went': 'go', 'done': 'do'}
    return terms | {irregular[word] for word in terms if word in irregular}


def discovery_qualifiers(point):
    """Keep the distinctive ends of user-supplied qualification phrases.

    Grammar, not legal knowledge: a topic match cannot stand in for 'for X',
    'without X', or 'before X'. Literal absence is a recovery hint, not a veto
    on a semantically supported rule or a claim of legal incompleteness.
    """
    anchors = set()
    for match in re.finditer(r'\b(?:for|without|with|before|after|at|by|unless|except)\s+'
                             r'([^?.,;]+)', str(point or ''), re.I):
        terms = discovery_terms(match[1])
        words = [w for w in re.findall(r'[a-z]{3,}', match[1].lower()) if w in terms]
        if words:
            anchors.add(words[-1])
    return anchors


def requested_points(question):
    """Do not split a narrated list of objects into separate legal requests.

    Full scenario facts remain in retrieval/generation. These labels identify
    what was asked, so 'a hat and boots ... should these be free?' is not marked
    complete merely because each item was mentioned somewhere in the context.
    """
    from .document_evidence import question_search_queries
    focus = str(question or '').split('Current follow-up question:')[-1].strip()
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', focus) if s.strip()]
    requests = [s for s in sentences if re.match(
        r"^(?:what|how|who|when|where|why|which|can|could|may|should|must|am|"
        r"is|are|was|were|do|does|did|explain|describe|compare|ano|paano|sino|"
        r"kailan|saan|pwede|puwede)\b", re.sub(r"n['’]t\b", ' not', s), re.I)]
    if not requests:
        return question_search_queries(focus) or [focus]
    points = []
    for request in requests:
        # A referential 'Is that allowed?' needs its scenario to identify the
        # subject; it must not become an empty standalone search request.
        if not discovery_terms(request):
            request = focus
        points.extend(question_search_queries(request) or [request])
    return list(dict.fromkeys(points))


def discovery_hit(terms, words, qualifiers=()):
    """Require distinctive words together, not one shared generic topic word.

    This remains a discovery hint: synonyms/translations can trigger a bounded
    extra search; only the independent checker can certify a legal conclusion.
    """
    return (bool(terms) and set(qualifiers) <= words
            and len(terms & words) >= max(min(2, len(terms)), math.ceil(.6 * len(terms))))


def context_budget(question=''):
    configured = max(256, int(getattr(settings, 'RAG_CONTEXT_TOKEN_BUDGET', 2200)))
    # Independent checking is disabled on the normal chat path, so its quota
    # must not reserve or shrink answer-generation evidence. Keep only the
    # configured retrieval ceiling and the real generation request budget.
    question_cost = estimate_request_tokens([{'content': question}], 0)
    generation = int(getattr(settings, 'GROQ_REQUEST_TOKEN_BUDGET', 6000)) - 3000 - question_cost
    return max(0, min(configured, generation))


def evidence_tokens(chunks):
    # Include citation metadata and per-source transport overhead. The complete
    # provider request is checked again; no selected evidence is silently cut.
    return sum(estimate_request_tokens([{'content': json.dumps({
        'text': c.get('text') or c.get('excerpt') or '',
        'metadata': {k: (c.get('metadata') or {}).get(k, c.get(k, '')) for k in (
            'document_title', 'source_scope', 'sector', 'section', 'article_number',
            'page', 'legal_status', 'effective_date')},
    }, ensure_ascii=False)}], 0) for c in chunks)


def refresh_selection_diagnostics(selection, chunks):
    """Report the actual retained context after parent expansion/filtering.

    Discovery coverage is deliberately not an entailment certificate. A point
    without a literal hit remains a hint even when semantic retrieval found it.
    """
    from .legal_topic_matrix import source_passage_text
    result = dict(selection)
    bodies = [discovery_terms(source_passage_text(c.get('text') or '')) for c in chunks]
    points = result.get('requested_points') or []
    result.update(selected_passages=len(chunks), estimated_evidence_tokens=evidence_tokens(chunks),
                  discovery_gaps=[p for p in points if not any(discovery_hit(
                      discovery_terms(p), b, discovery_qualifiers(p)) for b in bodies)],
                  coverage_verified=False)
    return result


def select_evidence(chunks, question, assessment=None, *, token_budget=None, min_score=.25,
                    preserve_order=False, additional_points=(), semantic_ranked_keys=()):
    """Keep complementary responsive bodies and atomic reference bundles.

    Exact bodies/IDs survive packing. A malformed or incomplete assessment never
    vetoes locally responsive evidence. Unrepresented points are search hints;
    only final verification can certify claims or legal completeness.
    """
    from .document_evidence import (
        assessment_coverage, duplicate_key, question_candidate_scorer, rank_question_candidates,
    )
    from .legal_version_ranking import version_precedence
    from .evidence_scope import scope_priority, source_scope
    scope_competition = len({source_scope(c) for c in chunks}) > 1
    from .focused_retrieval_queries import question_focus, scenario_actions
    from .legal_topic_matrix import source_passage_text
    budget = context_budget(question) if token_budget is None else max(0, int(token_budget))
    mapping = assessment_coverage(assessment, chunks, question) or {}
    # Caller supplies exact body/version identities only after a successful
    # semantic ranking. Local fallback order and newly recovered bodies do not
    # inherit this signal. It is relevance evidence, never claim verification.
    semantic_ranked_keys = frozenset(semantic_ranked_keys)
    focus = question_focus(question)
    # An assessment cannot silently replace the user's actual request with a
    # narrower topic. Its bound support map still contributes complementary rules.
    canonical = {p.casefold().rstrip('?. '): p for p in mapping.get('evidence_by_point', {})}
    actual = [canonical.get(p.casefold().rstrip('?. '), p) for p in requested_points(question)]
    points = list(dict.fromkeys([*additional_points, *actual, *mapping.get('evidence_by_point', {})]))
    direct_subject_terms = [discovery_terms(value) for point in [*actual, *additional_points]
                            for value in (question_focus(point)['subjects'] or [point])]
    priority_points = set(range(len(list(dict.fromkeys(additional_points)))))
    terms = [discovery_terms(p) for p in points]
    qualifiers = [discovery_qualifiers(p) for p in points]
    request_words = set().union(*terms) if terms else set()
    scenario_words = discovery_terms(question) - request_words
    focused_subjects = [discovery_terms(p) for p in focus['subjects']]
    subject = set().union(*focused_subjects) if focused_subjects else discovery_terms(question)
    # Focused search can omit an explicitly requested qualifier (for example,
    # "amount" in a question about how a voucher works). Keep the direct
    # request and any checker-discovered point in the subject vocabulary too.
    if not focused_subjects:
        subject.update(*(discovery_terms(p) for p in actual))
    subject.update(*(discovery_terms(p) for p in additional_points))
    # These describe the kind of question, not the named subject. Keeping them
    # out of subject matching lets an unfamiliar named topic match its uploaded
    # source wording ("frostcredit entitlement" -> "frostcredit is available")
    # without letting one incidental event word match a different benefit.
    subject.difference_update({'entitlement', 'entitlements'})
    # The noun subject and the scenario are different signals. A manager/job
    # title or a shared tenure word cannot substitute for the disputed benefit.
    # Keep explicit multi-part subjects, not an arbitrary union of all facts.
    subject_groups = [group - {'entitlement', 'entitlements'} for group in focused_subjects] or [subject]
    subject_heads = set().union(*(discovery_terms(value.split()[-1])
                                  for value in focus['subjects'] if value.split()))
    actions = scenario_actions(question)
    local_relevance = (question_candidate_scorer(question)
                       if not (preserve_order or assessment is not None) else None)
    mapped = {id(c): {p for p, ids in mapping.get('evidence_by_point', {}).items() if i in ids}
              for i, c in enumerate(chunks, 1)}
    groups, seen = {}, set()
    candidate_trace = []

    def trace_candidate(chunk, decision, score=None):
        metadata = chunk.get('metadata') or {}
        from .regional_wage_evidence import document_regions
        candidate_trace.append({
            'chunk_id': metadata.get('chunk_id') or chunk.get('id'),
            'doc_db_id': metadata.get('doc_db_id'),
            'region': ','.join(sorted(document_regions(metadata))),
            'score': score,
            'lexical_score': chunk.get('lexical_score'),
            'rrf_score': chunk.get('rrf_score'),
            'retrieval_method': metadata.get('retrieval_methods') or metadata.get('retrieval_method'),
            'decision': decision,
        })
    # A valid model order must survive packing. Local scores are fallback
    # relevance signals, not a reason to undo a successful semantic ranking.
    ordered = chunks if preserve_order or assessment is not None else rank_question_candidates(chunks, question)
    from .remedy_discovery import remedy_requested, procedural_passage
    needs_remedy = remedy_requested(question)
    for c in ordered:
        from .regional_wage_evidence import wage_scope_rejection
        rejection = wage_scope_rejection(question, c)
        if rejection:
            trace_candidate(c, rejection, float(c.get('score') or 0))
            continue
        key = duplicate_key(c)
        if key in seen:
            trace_candidate(c, 'duplicate')
            continue
        if not str(c.get('text') or '').strip():
            trace_candidate(c, 'empty_body')
            continue
        seen.add(key)
        try:
            score = float(c.get('score') or 0)
        except (TypeError, ValueError):
            trace_candidate(c, 'invalid_score')
            continue
        methods = (c.get('metadata') or {}).get('retrieval_methods') or [(c.get('metadata') or {}).get('retrieval_method')]
        procedural_probe_hit = needs_remedy and 'requested_remedy' in methods and procedural_passage(c)
        # A broad procedural rule may not repeat the disputed benefit. Preserve
        # actual procedural-probe hits for independent scope/claim checking;
        # the full-question cosine score is still reported unchanged.
        if score < min_score and not (c.get('metadata') or {}).get('evidence_bundle') and not procedural_probe_hit:
            trace_candidate(c, 'score_floor', score)
            continue
        trace_candidate(c, 'eligible', score)
        bundle = (c.get('metadata') or {}).get('evidence_bundle') or str(key)
        groups.setdefault(bundle, []).append(c)
    # Literal overlap is a ranking bonus, never a keep/drop requirement. This
    # lets translated or differently worded legal passages survive discovery;
    # independent reranking, applicability checks and verification still gate
    # what may be published.
    def subject_match(words):
        if not subject:
            return False
        return any(bool(group) and len(group & words) >= min(2, len(group)) for group in subject_groups)

    def explicit_subject_match(words):
        return any(bool(group) and len(group & words) >= min(2, len(group))
                   for group in direct_subject_terms)

    candidates = []
    from .remedy_discovery import remedy_requested, procedural_passage
    needs_remedy = remedy_requested(question)
    for group in groups.values():
        body = '\n'.join(source_passage_text(c['text']) for c in group)
        words = discovery_terms(body)
        support = set().union(*(mapped.get(id(c), set()) for c in group))
        # A known one-word collision between distinct named subjects remains
        # excluded. Wholly non-overlapping passages are retained above the
        # score floor, which is essential for translated evidence.
        partial_subject_collision = bool(subject & words) and not (
            subject_match(words) or explicit_subject_match(words)
        )
        semantic_head_match = bool(support) and bool(subject_heads & words)
        semantically_ranked = all(duplicate_key(c) in semantic_ranked_keys for c in group)
        # A shared noun head plus the narrated event can express the same
        # request without its adjective ("extra pay" / "additional compensation").
        # Do not use this shortcut for multiple requests or explicit qualifiers.
        scenario_match = (len(actual) == 1 and not additional_points
                          and not discovery_qualifiers(actual[0])
                          and bool(subject - words)
                          and (subject - words) <= _WORDING_MODIFIERS
                          and bool(subject_heads & words)
                          and discovery_hit(scenario_words, words))
        procedural_match = needs_remedy and any(procedural_passage(c) for c in group)
        if partial_subject_collision and not (semantic_head_match or semantically_ranked or scenario_match or procedural_match):
            trace_candidate(group[0], 'subject_collision', float(group[0].get('score') or 0))
            continue
        if partial_subject_collision:
            trace_candidate(group[0], 'semantic_wording_match' if semantically_ranked
                            else 'scenario_wording_match' if scenario_match
                            else 'assessment_wording_match', float(group[0].get('score') or 0))
        hits = {i for i, t in enumerate(terms)
                if discovery_hit(t, words, qualifiers[i]) or points[i] in support}
        if scenario_match:
            hits.update(i for i, point in enumerate(points) if point == actual[0])
        if procedural_match:
            hits.update(i for i, point in enumerate(points) if remedy_requested(point))
        if assessment is not None and mapping.get('evidence_by_point') and not hits and not support:
            # A completed evidence map identifies passages responsive to the
            # requested points. Keep multilingual matches when the map cites
            # them, but avoid filling the generation context with unrelated
            # high-vector-score candidates from a broad discovery pool.
            trace_candidate(group[0], 'no_requested_point_support', float(group[0].get('score') or 0))
            continue
        candidates.append((group, hits, support, len(candidates), words))
    predecessors = {}
    for new, old in version_precedence(question, chunks):
        predecessors.setdefault(id(chunks[old]), set()).add(id(chunks[new]))

    # Most priority inputs are properties of the group and query, so they do
    # not change as coverage grows. Computing scope and local relevance inside
    # the greedy loop recalculated expensive text/quality features once for
    # every remaining candidate on every selection pass.
    priority_cache = []
    from .overview_discovery import overview_facets
    for group, hits, support, ordinal, words in candidates:
        group_ids = {id(c) for c in group}
        required_predecessors = set()
        for chunk in group:
            required_predecessors.update(predecessors.get(id(chunk), ()))
        required_predecessors.difference_update(group_ids)
        priority_cache.append((
            frozenset(required_predecessors),
            max((scope_priority(question, c) for c in group), default=0)
            if scope_competition else 0,
            bool(actions) and any(
                any(word.startswith(action[:5]) for word in words)
                for action in actions if len(action) >= 5
            ) and subject_match(words),
            max((local_relevance(c) for c in group), default=0)
            if not (preserve_order or assessment is not None) else 0,
            frozenset(words & request_words),
            frozenset(words & scenario_words),
            (all(duplicate_key(c) in semantic_ranked_keys for c in group), -ordinal),
            needs_remedy and any(procedural_passage(c) for c in group),
            frozenset().union(*(overview_facets(question, c) for c in group)),
        ))

    selected, covered, budget_omitted = [], set(), 0
    covered_words, supported_points = set(), set()
    selected_ids = set()
    used = 0
    procedure_selected = False
    covered_facets = set()
    while candidates:
        # Cover distinct requested points first, then responsive bodies in
        # relevance order. Exact duplicates were removed above. Do not discard
        # a near-duplicate merely for low word novelty: a single changed number
        # or qualification can materially change a legal rule.
        available = [i for i, cached in enumerate(priority_cache)
                     if cached[0] <= selected_ids]
        if not available:
            break  # Never keep an old rule alone after its amendment could not fit.
        def priority(i):
            group, hits, support, ordinal, words = candidates[i]
            (_, scope_score, action_priority, relevance, request_terms,
             scenario_terms, semantic_priority, procedure_match, facets) = priority_cache[i]
            # Prefer bound responsive rules, then unresolved requested parts.
            # With no successful assessment, reward complementary user-wording
            # evidence instead of packing several repeated topical mentions.
            novelty = (len(request_terms - covered_words)
                       + .2 * len(scenario_terms - covered_words))
            return (scope_score,
                    procedure_match and not procedure_selected,
                    len((hits - covered) & priority_points),
                    # Keep a successful semantic order ahead of lexical bonuses.
                    # Scope/version gates and checker-identified gaps still lead.
                    semantic_priority if semantic_priority[0] else (False, 0),
                    bool(facets - covered_facets),
                    action_priority,
                    len(support - supported_points), len(hits - covered),
                    relevance,
                    0 if preserve_order or assessment is not None else novelty, -ordinal)
        index = max(available, key=priority)
        group, hits, support, _, words = candidates.pop(index)
        chosen_priority = priority_cache.pop(index)
        cost = evidence_tokens(group)
        if used + cost > budget:
            budget_omitted += len(group)
            for chunk in group:
                trace_candidate(chunk, 'token_budget', float(chunk.get('score') or 0))
            continue
        selected.extend(group)
        covered_facets.update(chosen_priority[-1])
        procedure_selected |= needs_remedy and any(procedural_passage(c) for c in group)
        selected_ids.update(id(c) for c in group)
        covered.update(hits)
        covered_words.update(words)
        supported_points.update(support)
        used += cost
    # Retain version precedence, which is independent of discovery score.
    from .legal_version_ranking import apply_version_precedence
    before_precedence = list(selected)
    selected = apply_version_precedence(question, selected)
    selected_keys = {duplicate_key(c) for c in selected}
    for chunk in before_precedence:
        if duplicate_key(chunk) not in selected_keys:
            trace_candidate(chunk, 'version_precedence', float(chunk.get('score') or 0))
    for chunk in selected:
        trace_candidate(chunk, 'selected', float(chunk.get('score') or 0))
    return selected, {'requested_points': points,
        'discovery_gaps': [p for i, p in enumerate(points) if i not in covered],
        'estimated_evidence_tokens': used, 'token_budget': budget,
        'budget_omitted_passages': budget_omitted, 'selected_passages': len(selected),
        'candidate_trace': candidate_trace[:80],
        'coverage_verified': False}
