"""Bounded local operational reports, independent of the application database.

Only allowlisted metrics/codes are persisted. Never save prompts, chats, source
text, provider response bodies, credentials, or arbitrary logging arguments.
"""
import json
import logging
import math
import queue
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from config import BACKEND_DIR, settings

_CODES = {
    'request_complete': 'Chat request completed',
    'model_task_failed': 'No usable model completed this pipeline step',
    'model_task_skipped': 'Model call skipped by a local guard',
    'model_attempt_failed': 'Model attempt failed; fallback may have recovered',
    'answer_check_failed': 'Independent answer check could not complete',
    'answer_rejected': 'Explanation rejected by independent evidence checking',
    'verification_failed_due_to_missing_evidence': 'Selected evidence did not cover a verifier-identified point',
    'generation_failed': 'Answer generation failed; fallback may have recovered',
    'backend_warning': 'Backend warning',
    'backend_error': 'Backend error',
}
_METRICS = {
    'retrieval_budget_ms', 'answer_time_reserved_ms', 'retrieval_optional_limited',
    'retrieval_optional_limit_count',
    'simple_fast_path', 'retrieval_fast_exit', 'retrieval_discovery_complete', 'rerank_called', 'generation_calls',
    'verification_calls', 'recovery_called', 'total_llm_calls', 'generation_ms',
    'verification_ms', 'recovery_ms', 'total_answer_ms',
    'request_total_ms', 'routing_ms', 'retrieval_ms', 'embedding_ms', 'rerank_ms',
    'primary_model_ms', 'fallback_model_ms', 'validation_ms', 'citation_validation_ms',
    'database_read_ms', 'database_write_ms', 'session_update_ms', 'answer_check_ms',
    'time_to_first_token_ms', 'retrieved_chunk_count', 'retrieval_stage_count',
    'retrieval_candidate_count', 'retrieval_gap_count', 'retrieval_expanded',
    'retrieval_active_filter_dropped', 'retrieval_applicability_filter_dropped',
    'retrieval_explicit_scope_filter_dropped', 'retrieval_duplicate_count',
    'retrieval_scope_candidate_count', 'retrieval_scope_balanced_selected',
    'retrieval_selection_limit_dropped',
    'estimated_evidence_tokens', 'token_budget', 'budget_omitted_passages',
    'supported_claim_count', 'checked_claim_count', 'missing_point_count',
    'validation_error_count',
    'estimated_tokens', 'request_budget', 'latency_ms', 'http_status', 'attempt_count',
    'output_token_count', 'fallback_triggered', 'provider_timeout', 'answer_compressed',
    'query_reformulation_ms',
    'local_rank_count', 'llm_rerank_called', 'recovery_triggered', 'recovery_new_candidates',
    'prior_llm_ranking_preserved', 'selected_passages', 'evidence_tokens',
    'generation_provider_calls', 'verification_called', 'verification_output_budget',
    'repair_called', 'total_model_calls', 'estimated_model_tokens',
    'rerank_provider_calls', 'verify_provider_calls', 'query_provider_calls',
    'retrieval_parent_candidates', 'retrieval_provision_duplicate_candidates',
    'retrieval_parent_child_overlap_groups',
    'retrieval_elapsed_ms', 'retrieval_remaining_deadline_ms',
    'retrieval_total_ms', 'retrieval_active_stage_count', 'retrieval_cancelled_stage_count',
    'retrieval_candidates_produced', 'simple_legal_rag', 'retrieval_cancelled',
    'retrieval_selected_chunk_count', 'retrieval_parents_reconstructed',
    'retrieval_sections_expanded', 'retrieval_section_expansion_calls',
    'calculation_requested', 'scenario_field_count', 'category_hint_count',
    'reconstructed_query_chars', 'retrieval_variant_count', 'pending_scenario',
    'retrieval_completed', 'source_ranking_completed', 'evidence_selected',
    'evidence_sufficient', 'evidence_chunk_count', 'generation_eligibility_checked',
    'generation_eligible', 'generation_api_call_started', 'generation_api_call_returned',
    'generation_skip_count', 'generation_provider_attempts', 'verification_eligibility_checked',
    'verification_eligible', 'verification_api_call_started',
    'generation_draft_created',
    'generation_queue_ms', 'generation_active_requests', 'generation_concurrency_limit',
    'generation_concurrency_uncapped',
    'attempt_timeout_ms', 'generation_remaining_ms',
    'rerank_skipped', 'verification_failed_due_to_missing_evidence',
}
_LABELS = {'role', 'model', 'provider', 'reason', 'source_match', 'verification_state',
           'final_status', 'exception_type', 'intent', 'provider_used', 'model_used',
           'fallback_reason', 'provider_error_type'}
_LABELS.update({'llm_rerank_reason', 'llm_rerank_model', 'generation_model', 'generation_prompt_profile'})
_LABELS.add('timeout_scope')
_LABELS.update({'routing_topic', 'routing_category', 'routing_task', 'routing_sector'})
_LABELS.update({'generation_gate_reason', 'generation_pool_admission',
                'generation_call_state', 'generation_outcome', 'generation_skip_reason',
                'generation_selected_model', 'generation_exception', 'verification_gate_reason'})
_LABELS.update({'retrieval_active_stage', 'retrieval_cancelled_stage', 'retrieval_stage',
                'retrieval_category', 'retrieval_primary_category', 'failure_type'})
_LABELS.update({'rerank_skip_reason', 'verification_failure_cause', 'retrieval_last_limited_stage'})


def _label(value):
    # Values here are identifiers, never arbitrary explanations or messages.
    value = str(value or '')
    return value if re.fullmatch(r'[A-Za-z0-9_./:-]{1,100}', value) else 'unknown'


def safe_details(details):
    result = {}
    for key, value in (details or {}).items():
        if key in _METRICS and (value is None or type(value) in (int, float, bool)):
            result[key] = value
        elif key in _LABELS:
            result[key] = _label(value)
        elif key == 'stages' and isinstance(value, list):
            result[key] = [{'stage': _label(item.get('stage')), 'retrieved': int(item.get('retrieved') or 0)}
                           for item in value[:20] if isinstance(item, dict)]
        elif key == 'retrieval_active_stages' and isinstance(value, list):
            result[key] = [_label(item) for item in value[:64]]
        elif key == 'retrieval_fallback_categories_attempted' and isinstance(value, list):
            result[key] = [_label(item) for item in value[:13]]
        elif key == 'retrieval_cancelled_stages' and isinstance(value, list):
            result[key] = [_label(item) for item in value[:64]]
        elif key == 'retrieval_operation_counts' and isinstance(value, list):
            result[key] = [{'stage': _label(item.get('stage')),
                            'operation_count': int(item.get('operation_count') or 0)}
                           for item in value[:64] if isinstance(item, dict)]
        elif key == 'retrieval_stage_details' and isinstance(value, list):
            fields = ('started_at_ms', 'start_elapsed_ms', 'ended_at_ms', 'elapsed_ms',
                      'remaining_deadline_ms', 'remaining_at_end_ms', 'operation_count',
                      'candidate_count', 'input_candidate_count')
            result[key] = []
            for item in value[-200:]:
                if not isinstance(item, dict):
                    continue
                stage = {'stage': _label(item.get('stage')),
                         'category': _label(item.get('category')),
                         'categories': [_label(value) for value in (item.get('categories') or [])[:13]]}
                for field in fields:
                    number = item.get(field)
                    if number is None or type(number) in (int, float):
                        stage[field] = number
                for field in ('completed', 'failed', 'cancelled', 'cancelled_by_deadline'):
                    stage[field] = bool(item.get(field))
                if item.get('failure_type'):
                    stage['failure_type'] = _label(item['failure_type'])
                result[key].append(stage)
        elif key == 'selected_sources' and isinstance(value, list):
            result[key] = [{k: _label(item[k]) for k in ('chunk_id', 'doc_db_id', 'page') if item.get(k) is not None}
                           for item in value[:20] if isinstance(item, dict)]
        elif key in {'retrieval_candidate_trace', 'evidence_candidate_trace',
                     'reranked_candidate_order', 'selected_passage_trace'} and isinstance(value, list):
            rows = []
            for item in value[:80]:
                if not isinstance(item, dict):
                    continue
                row = {}
                for field in ('chunk_id', 'doc_db_id', 'decision', 'scope', 'region'):
                    if item.get(field) is not None:
                        row[field] = _label(str(item[field]).lower().replace(' ', '_')) if field == 'region' else _label(item[field])
                for field in ('rank', 'score', 'local_score', 'lexical_score', 'rrf_score', 'rerank_score'):
                    number = item.get(field)
                    if type(number) in (int, float) and math.isfinite(number):
                        row[field] = number
                methods = item.get('retrieval_method')
                if isinstance(methods, str):
                    row['retrieval_method'] = [_label(methods)]
                elif isinstance(methods, list):
                    row['retrieval_method'] = [_label(v) for v in methods[:4]]
                rows.append(row)
            result[key] = rows
        elif key == 'stage4_checks' and isinstance(value, list):
            result[key] = [
                {'check': _label(item.get('check')), 'passed': bool(item.get('passed')),
                 'reason': _label(item.get('reason'))}
                for item in value[:5] if isinstance(item, dict)
            ]
    return result


class DiagnosticStore:
    def __init__(self, path, max_events=10000, retention_days=30):
        self.path = Path(path)
        self.max_events = max(1, int(max_events))
        self.retention_days = max(1, int(retention_days))
        self.lock = threading.Lock()

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=2)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, created REAL NOT NULL, '
                   'level TEXT NOT NULL, component TEXT NOT NULL, code TEXT NOT NULL, request_id TEXT, details TEXT NOT NULL)')
        db.execute('CREATE INDEX IF NOT EXISTS events_created ON events(created)')
        try:
            with db:
                yield db
        finally:
            db.close()

    def record(self, component, code, *, level='info', request_id='', details=None):
        code = code if code in _CODES else 'backend_error'
        level = level if level in {'info', 'warning', 'error'} else 'info'
        with self.lock, self._connect() as db:
            db.execute('INSERT INTO events(created,level,component,code,request_id,details) VALUES(?,?,?,?,?,?)',
                       (time.time(), level, _label(component), code, _label(request_id),
                        json.dumps(safe_details(details), allow_nan=False)))
            db.execute('DELETE FROM events WHERE created < ?', (time.time() - self.retention_days * 86400,))
            db.execute('DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY id DESC LIMIT -1 OFFSET ?)',
                       (self.max_events,))

    def report(self, days=7, limit=50, offset=0, level=None, component=None, request_id=None):
        cutoff = time.time() - max(1, min(int(days), self.retention_days)) * 86400
        where, args = 'created >= ?', [cutoff]
        for key, value in (('level', level), ('component', component), ('request_id', request_id)):
            if value:
                where += ' AND ' + key + ' = ?'
                args.append(value)
        with self.lock, self._connect() as db:
            count = db.execute('SELECT COUNT(*) FROM events WHERE ' + where, args).fetchone()[0]
            rows = db.execute('SELECT * FROM events WHERE ' + where + ' ORDER BY id DESC LIMIT ? OFFSET ?',
                              [*args, max(1, min(int(limit), 200)), max(0, int(offset))]).fetchall()
            # Summary always covers the chosen time window, independently of
            # table filters/pagination. Partial is not automatically a failure.
            outcomes = db.execute("SELECT details FROM events WHERE created >= ? AND code='request_complete'", (cutoff,)).fetchall()
            issues = db.execute("SELECT code,COUNT(*) count FROM events WHERE created >= ? AND level!='info' GROUP BY code ORDER BY count DESC", (cutoff,)).fetchall()
        requests = [json.loads(row['details']) for row in outcomes]
        timings = [r['request_total_ms'] for r in requests if type(r.get('request_total_ms')) in (int, float)]
        states = {}
        for item in requests:
            state = item.get('source_match') or item.get('final_status') or 'other'
            states[state] = states.get(state, 0) + 1
        return {'items': [{**dict(row), 'created_at': datetime.fromtimestamp(row['created'], timezone.utc).isoformat(),
                           'summary': _CODES[row['code']], 'details': json.loads(row['details'])} for row in rows],
                'total': count, 'offset': max(0, int(offset)), 'retention_days': self.retention_days,
                'summary': {'requests': len(requests), 'average_ms': round(sum(timings) / len(timings), 1) if timings else None,
                            'outcomes': states, 'issues': [dict(row) for row in issues]},
                'note': 'Reports start when diagnostics are enabled. Warning events may be recovered by a fallback. '
                        'Partial/insufficient evidence is not necessarily a backend outage. No chat or document text is recorded.'}


_store = None
_events = queue.Queue(maxsize=1000)
_worker = None
_worker_lock = threading.Lock()
_dropped = 0


def get_store():
    global _store
    if _store is None:
        path = getattr(settings, 'LABORLENS_DIAGNOSTICS_PATH', '') or BACKEND_DIR / 'data' / 'backend_diagnostics.sqlite3'
        _store = DiagnosticStore(path, getattr(settings, 'LABORLENS_DIAGNOSTICS_MAX_EVENTS', 10000),
                                 getattr(settings, 'LABORLENS_DIAGNOSTICS_RETENTION_DAYS', 30))
    return _store


def _write_events():
    global _dropped
    while True:
        event = _events.get()
        try:
            get_store().record(**event)
        except Exception:
            _dropped += 1  # Never disrupt chat or recursively log a storage error.
        finally:
            _events.task_done()


def record_event(component, code, *, level='info', details=None, request_id=None):
    global _worker, _dropped
    if not getattr(settings, 'LABORLENS_DIAGNOSTICS_ENABLED', True):
        return
    from .performance import current_timing
    timing = current_timing()
    identity = request_id or (timing.request_id if timing else '')
    try:
        details = safe_details(details)
        with _worker_lock:
            if _worker is None:
                worker = threading.Thread(target=_write_events, name='backend-diagnostics', daemon=True)
                worker.start()
                _worker = worker
        _events.put_nowait(dict(component=component, code=code, level=level,
                               request_id=identity, details=details))
    except Exception:
        _dropped += 1


def storage_status():
    return {'enabled': bool(getattr(settings, 'LABORLENS_DIAGNOSTICS_ENABLED', True)),
            'pending_events': _events.qsize(), 'dropped_events': _dropped}


class DiagnosticLogHandler(logging.Handler):
    def emit(self, record):
        if not record.name.startswith(('routers.', 'main')) or record.name == __name__:
            return
        # Inspect the static format, never interpolate private logging args.
        template = str(record.msg).lower()
        if record.name.endswith('.model_pools') and template.startswith(('model task failed:', 'model task unavailable:', 'model task skipped:')):
            return  # Already recorded with safe structured reason/model fields.
        code = ('answer_check_failed' if 'answer check' in template else
                'generation_failed' if 'generation' in template or 'providers failed' in template else
                'backend_error' if record.levelno >= logging.ERROR else 'backend_warning')
        details = {'exception_type': record.exc_info[0].__name__} if record.exc_info else {}
        record_event(record.name, code, level='error' if record.levelno >= logging.ERROR else 'warning', details=details)


def install_log_handler():
    root = logging.getLogger()
    if not any(isinstance(handler, DiagnosticLogHandler) for handler in root.handlers):
        root.addHandler(DiagnosticLogHandler(logging.WARNING))
