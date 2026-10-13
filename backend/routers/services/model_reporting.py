"""Read-only model inventory for admin reporting; never rotates or probes models."""
import math
import time

from config import settings
from . import model_pools
from .provider_runtime import provider_health_snapshot
from .lda_topic_reranker import lda_topic_model_status, lda_topic_configuration


def lda_status_report():
    """Observe this worker's existing LDA model; never load text or train it."""
    try:
        snapshot = lda_topic_model_status()
    except Exception:
        snapshot = {'enabled': False, 'ready': False, 'building': False,
                    'topic_count': 0, 'trained_chunk_count': 0,
                    'last_error': 'status_unavailable'}
    configured = bool(getattr(settings, 'LDA_RERANK_ENABLED', False))
    if snapshot.get('last_error') == 'status_unavailable':
        status, detail = 'Unavailable', 'Topic-learning status could not be read. Try Refresh.'
    elif not configured:
        status, detail = 'Disabled', 'Topic learning is turned off. Normal document search still works.'
    elif not snapshot.get('enabled'):
        status, detail = 'Unavailable', 'The topic-learning library is unavailable. Normal document search still works.'
    elif snapshot.get('building'):
        if snapshot.get('build_phase') == 'assigning':
            status, detail = 'Matching chunks', 'Checking every stored chunk for related topics in background batches. Normal document search still works.'
        else:
            status, detail = 'Training', 'Learning topics from stored document text in the background. Use Refresh to check again.'
    elif snapshot.get('assignment_error'):
        status, detail = 'Matching incomplete', 'The topic model is ready, but some chunk matches could not be checked. Normal document search still works.'
    elif snapshot.get('ready'):
        status, detail = 'Ready', 'Topic learning has finished for the current document sample.'
    elif snapshot.get('last_error') == 'not_enough_usable_document_text':
        status, detail = 'Needs documents', 'There was not enough usable document text to build topics. Normal document search still works.'
    elif snapshot.get('last_error'):
        status, detail = 'Failed', 'Topic training could not finish. Normal document search still works.'
    else:
        status, detail = 'Not trained', 'No trained topic model is available in this backend worker yet.'
    return {**snapshot, **lda_topic_configuration(settings), 'status': status, 'detail': detail,
            'search_hints_enabled': bool(configured and snapshot.get('enabled') and
                                         getattr(settings, 'LDA_SEARCH_HINTS_ENABLED', False))}


def generation_scheduler_report(providers=None):
    """A process-local capacity snapshot, not a provider availability probe."""
    rows = configured_model_report() if providers is None else providers
    generators = [row for row in rows if row.get('role') == 'generation']
    pooled = bool(settings.LABORLENS_MODEL_POOLS_ENABLED)
    limit = model_pools.concurrency_limit('generation') if pooled else None
    primary = [row for row in generators if row.get('generation_tier', 'primary') == 'primary']
    backups = [row for row in generators if row.get('generation_tier') == 'backup']
    return {
        'enabled': pooled,
        'policy': 'primary_idle_first_then_least_busy' if pooled else 'legacy_provider_order',
        'model_count': len(generators),
        'primary_model_count': len(primary),
        'backup_model_count': len(backups),
        'active_requests': sum(row['active_requests'] for row in generators) if pooled else 0,
        'configured_slots': sum(row['concurrency_limit'] for row in generators) if pooled and limit is not None else None,
        'concurrency_uncapped': pooled and limit is None,
        'waiting_requests': len(model_pools._generation_waiters) if pooled else 0,
        'concurrency_per_model': limit,
        'queue_wait_seconds': max(0, float(getattr(settings, 'LABORLENS_GENERATION_QUEUE_WAIT_SECONDS', 2))) if pooled and limit is not None else None,
        'queue_max_waiters': max(0, int(getattr(settings, 'LABORLENS_GENERATION_QUEUE_MAX_WAITERS', 64))) if pooled and limit is not None else None,
        'scope': 'backend_worker',
    }


def configured_model_report():
    snapshot = provider_health_snapshot()
    if settings.LABORLENS_MODEL_POOLS_ENABLED:
        generators = model_pools.models(settings.LABORLENS_GENERATOR_POOL)
    else:
        from .gemini_service import configured_gemini_models
        generators = ['groq/' + settings.GROQ_MODEL] + ['gemini/' + m for m in configured_gemini_models()]
    groups = [('generation', generators, True),
              ('reranking', model_pools.models(settings.LABORLENS_RERANKER_POOL), settings.LABORLENS_LLM_RERANK_ENABLED)]
    rows = []
    primary_generators = set(model_pools.generation_tiers(generators)[0])
    now = time.monotonic()
    for role, pool, role_enabled in groups:
        for priority, name in enumerate(pool, 1):
            provider, model = name.split('/', 1)
            enabled = bool(role_enabled and model_pools.provider_enabled(provider))
            configured = bool(model_pools.provider_api_key(provider))
            key = ('rerank:' + name if role == 'reranking' else
                   'gemini:' + model if provider == 'gemini' else
                   name if settings.LABORLENS_MODEL_POOLS_ENABLED else 'groq')
            health = dict(snapshot.get(key, {}))
            completed = health.get('successes', 0) + health.get('failures', 0)
            if not completed:
                health['average_latency_ms'] = None
            elif health.get('average_latency_ms') is not None:
                # Runtime averages divide by started requests; omit unfinished
                # requests from the displayed completed-attempt average.
                health['average_latency_ms'] = round(health['average_latency_ms'] * health.get('requests', completed) / completed, 2)
            local_wait = max(0, model_pools._cooldowns.get(name, 0) - now)
            retry = math.ceil(max(local_wait, health.get('retry_in_seconds') or 0))
            busy = name in model_pools._busy
            request_cap = model_pools.request_limit(name)
            if not enabled:
                status = 'Disabled'
            elif not configured:
                status = 'API key missing'
            elif retry or health.get('circuit_state') == 'OPEN':
                status = 'Waiting to retry'
            elif busy:
                status = 'In use'
            elif (role == 'reranking' or settings.LABORLENS_MODEL_POOLS_ENABLED) and request_cap is not None and sum(
                    stamp > now - 60 for stamp in model_pools._calls.get(name, ())) >= request_cap:
                status = 'Local request limit reached'
            elif model_pools.token_limit(name) and model_pools.tokens_last_minute(name) >= model_pools.token_limit(name):
                status = 'Local token limit reached'
            elif (health.get('last_failure_at') or 0) > (health.get('last_success_at') or 0):
                status = {'timeout': 'Last attempt timed out', 'rate_limit': 'Request limit reached',
                          'cancelled': 'Last call interrupted',
                          'local_budget': 'Stopped at local budget',
                          'invalid_response_format': 'Response format failed checking',
                          'incomplete_response': 'Last answer was incomplete'}.get(health.get('last_error_type'), 'Last attempt failed')
            elif health.get('successes'):
                status = 'Observed successes'
            else:
                status = 'Not used since restart'
            row = {**health, 'provider': provider, 'model': model, 'role': role,
                   'model_key': name, 'enabled': enabled, 'configured': configured,
                   'status': status, 'priority': priority, 'retry_in_seconds': retry,
                   'metrics_available': True}
            row['active_requests'] = model_pools.active_requests(name)
            row['concurrency_limit'] = model_pools.concurrency_limit('generation' if role == 'generation' else None)
            row['concurrency_uncapped'] = row['concurrency_limit'] is None
            row['available_slots'] = (max(0, row['concurrency_limit'] - row['active_requests'])
                                      if row['concurrency_limit'] is not None else None)
            if role == 'generation':
                row['generation_tier'] = 'primary' if name in primary_generators else 'backup'
            if role == 'reranking' or settings.LABORLENS_MODEL_POOLS_ENABLED:
                row['calls_last_minute'] = sum(stamp > now - 60 for stamp in model_pools._calls.get(name, ()))
                row['local_rpm_limit'] = request_cap
                row['estimated_tokens_last_minute'] = model_pools.tokens_last_minute(name)
                row['local_tpm_limit'] = model_pools.token_limit(name) or None
            rows.append(row)
    return rows
