"""Privacy-safe claim audit summaries. Never log drafts, questions or excerpts."""

VERDICTS = {'supported', 'unsupported', 'uncertain', 'user_fact',
            'evidence_gap', 'practical_suggestion'}
ISSUES = {'citation_mismatch', 'evidence_conflict', 'invalid_evidence_gap',
          'invalid_practical_suggestion'}


def audit_diagnostics(verification):
    value = verification if isinstance(verification, dict) else {}
    result = []
    for check in (value.get('claim_checks') or [])[:40]:
        if not isinstance(check, dict):
            continue
        verdict = check.get('verdict')
        issue = check.get('issue')
        if verdict not in {'unsupported', 'uncertain'} and not issue:
            continue
        unit = check.get('unit_id')
        result.append({
            'unit_id': unit if type(unit) is int and 0 <= unit < 40 else None,
            'verdict': verdict if verdict in VERDICTS else 'invalid_verdict',
            'reason': issue if issue in ISSUES else
                      'unsupported_claim' if verdict == 'unsupported' and not issue else
                      'uncertain_claim' if verdict == 'uncertain' and not issue else 'other_validation_failure',
            'source_ids': [i for i in (check.get('source_ids') or [])
                           if type(i) is int and 0 <= i < 10000][:100],
            'cited_source_ids': [i for i in (check.get('cited_source_ids') or [])
                                 if type(i) is int and 0 <= i < 10000][:100],
        })
    return {'rejected_units': result,
            'unchecked_units': len(value.get('unchecked_claims') or []),
            'validation_errors': len(value.get('errors') or [])}
