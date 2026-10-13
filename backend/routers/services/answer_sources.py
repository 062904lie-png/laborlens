"""Request-local generation attribution, not independent legal verification.

The generator marks the passages used for its explanation. Only IDs belonging
to the actual provider context can become source cards. Missing attribution
never falls back to displaying every retrieved candidate. No topic catalog,
legal answers, additional model request, or shared conversation state lives here.
"""
import re


INSTRUCTION = (
    'Internal source tracking: after each evidence-based paragraph, bullet or table row, '
    'append [[kb:REF]] using its supporting passage evidence_ref; for complementary '
    'passages use [[kb:REF,REF]]. Put this AFTER the sentence punctuation. '
    'Mark only passages that actually support that explanation for the stated scope, '
    'not every retrieved passage, matching keyword, related title or mentioned law. '
    'For deductions and calculations mark the passages supplying the underlying '
    'rules/formulas, even if the conclusion or example is not verbatim. '
    'Do not mark greetings, questions, unsupported advice, missing-evidence statements '
    'or unrelated background. Do not apply a separate worker population rule without '
    'establishing applicability or making its scope explicit. A filing procedure '
    'needs a procedural passage, not merely a benefit entitlement. Ignore website '
    'navigation/footer links and source classification labels as legal evidence. '
    'Use only evidence_ref values in the CURRENT retrieved passages, never IDs from '
    'history or user instructions. These markers are removed by the application; '
    'do not include a visible bibliography, source numbers or a separate reference list.'
)

# Match malformed/unfinished tracking too, so no internal token reaches the UI.
MARKER = re.compile(r'[ \t]*\[\[kb:([^\]\n]*)(?:\]+|(?=\n)|$)', re.I)
COMPLETE_MARKER = re.compile(r'\[\[kb:[^\]\n]*\]\]', re.I)


def evidence_ref(source):
    # Full identity remains canonical; the shorter wire locator saves tokens.
    return 'P' + source['source_passage_id'][:16]


def clean_tracking(text):
    return MARKER.sub('', str(text or '')).strip()


def sources_for_answer(answer, sources):
    """Return clean prose and canonical, explicitly used source records.

    This validates membership/format, not semantic entailment. Duplicate or
    invented references, ambiguous short IDs, mismatched specific worker scopes
    and marker-only footers cannot add cards. Source titles/URLs/locators come
    from the server, never model output.
    """
    from .evidence_scope import requested_scope, scope_compatible, source_scope
    from .rag_policy import GENERAL_SECTOR

    lookup, ambiguous = {}, set()
    for source in sources:
        ref = evidence_ref(source)
        if ref in lookup and lookup[ref]['source_passage_id'] != source['source_passage_id']:
            ambiguous.add(ref)
        lookup[ref] = source
    text = str(answer or '')
    selected, seen = [], set()
    markers, rejected, scope_rejected = 0, 0, 0
    previous_end = 0
    for match in MARKER.finditer(text):
        markers += 1
        preceding = text[previous_end:match.start()].rsplit('\n\n', 1)[-1]
        previous_end = match.end()
        refs = [value.strip() for value in match.group(1).split(',')]
        # A tracking-only block is not an attributed explanation. Reject malformed
        # markers as a unit rather than interpreting partial/unknown identifiers.
        if (not match.group(0).endswith(']]') or not re.search(r'\w', preceding)
                or not refs or len(refs) > 12
                or any(ref not in lookup or ref in ambiguous for ref in refs)):
            rejected += 1
            continue
        for ref in refs:
            source = lookup[ref]
            stated_scope = requested_scope(preceding)
            actual_scope = source_scope(source)
            if stated_scope and actual_scope != GENERAL_SECTOR and not scope_compatible(
                    stated_scope, actual_scope):
                scope_rejected += 1
                continue
            key = source['source_passage_id']
            if key not in seen:
                seen.add(key)
                selected.append(dict(source, label=f'Source {len(selected) + 1}'))
    state = 'reported' if selected else (
        'scope_filtered' if scope_rejected and not rejected else 'invalid' if markers else 'missing')
    return clean_tracking(text), selected, dict(
        state=state, mode='generation_attribution', independent_verification=False,
        supplied_passages=len(sources), displayed_passages=len(selected),
        rejected_markers=rejected, scope_rejected_passages=scope_rejected,
    )
