"""Deadline/attempt ownership shared across child tasks and provider fallbacks."""
import asyncio
from contextvars import ContextVar
from functools import wraps
import time

from config import settings
from .answer_pipeline_budget import AnswerPipelineBudget
from .performance import current_timing, add_timing, set_timing_metadata

_attempts = ContextVar('simple_pipeline_attempts', default=None)
_budget = ContextVar('answer_request_budget', default=None)


def request_budget(query, profile=None, rag=None):
    timing = current_timing()
    budget = (rag or {}).get('_answer_budget') or _budget.get() or (timing.pipeline_budget if timing else None)
    if budget is None:
        budget = AnswerPipelineBudget.from_settings(settings, query, profile)
        if timing:
            # Routing time also consumes the one deadline; never reset on retry.
            budget._deadline -= max(0, time.perf_counter() - timing.started)
            timing.pipeline_budget = budget
    return budget


def clamp_deadline(deadline):
    timing = current_timing()
    budget = _budget.get() or (timing.pipeline_budget if timing else None)
    return min(deadline, time.monotonic() + budget.remaining()) if budget else deadline


def timed_stage(name, **metadata):
    def decorate(fn):
        @wraps(fn)
        async def wrapped(*args, **kwargs):
            started = time.perf_counter()
            set_timing_metadata(**metadata)
            try:
                return await fn(*args, **kwargs)
            finally:
                add_timing(name, (time.perf_counter() - started) * 1000)
        return wrapped
    return decorate


def start_generation_attempt():
    state = _attempts.get()
    if state:
        state['attempt_generation'] = 0


def call_allowed(role):
    state = _attempts.get()
    if not state:
        return True
    if role == 'generation':
        return state['generation'] < state.get('generation_limit', 2) and state['attempt_generation'] == 0
    return role != 'verify' or state['verify'] < state.get('verify_limit', 1)


def record_llm_call(role):
    # Called immediately before the wire request, after local quota checks.
    if not call_allowed(role):
        from .model_pools import LocalModelLimitError
        raise LocalModelLimitError('simple_request_call_budget')
    state = _attempts.get()
    if state and role in {'generation', 'verify'}:
        state[role] += 1
        if role == 'generation':
            state['attempt_generation'] += 1
    timing = current_timing()
    if timing:
        timing.metadata['total_llm_calls'] = timing.metadata.get('total_llm_calls', 0) + 1
        if role in {'generation', 'verify'}:
            name = 'generation_calls' if role == 'generation' else 'verification_calls'
            timing.metadata[name] = timing.metadata.get(name, 0) + 1
            if role == 'generation':
                timing.set_meta(generation_api_call_started=True, generation_call_state='dispatched')
            else:
                timing.set_meta(verification_eligible=True, verification_api_call_started=True)
        if role == 'rerank':
            timing.set_meta(rerank_called=True)


def bounded_answer(fn):
    @wraps(fn)
    async def wrapped(rag, query, *args, **kwargs):
        profile = rag.get('query_profile') or {}
        budget = request_budget(query, profile, rag)
        rag['_answer_budget'] = budget
        fast = bool(profile.get('simple_fast_path'))
        if fast:
            budget.restrict_to_simple(minimum_remaining=budget.verify_seconds + min(15, budget.generation_seconds))
        token = _attempts.set({
            'generation': 0,
            'verify': 0,
            'attempt_generation': 0,
            'generation_limit': max(1, int(getattr(settings, 'LABORLENS_SIMPLE_GENERATION_MAX_CALLS', 5))),
            'verify_limit': max(1, int(getattr(settings, 'LABORLENS_SIMPLE_VERIFY_MAX_CALLS', 5))),
        } if fast else None)
        budget_token = _budget.set(budget)
        set_timing_metadata(simple_fast_path=fast)
        try:
            return await budget.run(fn, rag, query, *args, **kwargs)
        except asyncio.TimeoutError:
            from .rag_policy import SOURCE_STATUS_INSUFFICIENT
            notice = 'The request deadline was reached before an independently checked answer was available.'
            return ({'answer': notice, 'sources': []}, SOURCE_STATUS_INSUFFICIENT,
                    notice, False, {'errors': ['request_deadline'], 'source_match': SOURCE_STATUS_INSUFFICIENT})
        finally:
            _attempts.reset(token)
            _budget.reset(budget_token)
            timing = current_timing()
            if timing:
                timing.add_ms('total_answer_ms', 0)
                timing.values['total_answer_ms'] = (time.perf_counter() - timing.started) * 1000
    return wrapped
