"""Request-local reuse of deterministic linguistic analysis, never answers."""
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import json

_cache = ContextVar('question_analysis_cache', default=None)


def with_question_analysis(fn):
    @wraps(fn)
    async def wrapped(*args, **kwargs):
        if _cache.get() is not None:
            return await fn(*args, **kwargs)
        token = _cache.set({})
        try:
            return await fn(*args, **kwargs)
        finally:
            _cache.reset(token)
    return wrapped


def reuse_analysis(fn):
    @wraps(fn)
    def wrapped(question, *args, **kwargs):
        cache = _cache.get()
        if cache is None:
            return fn(question, *args, **kwargs)
        from .focused_retrieval_queries import _ready_parser
        # Parser identity and every supplied profile field are part of the key.
        key = (fn.__name__, id(_ready_parser()), json.dumps(
            [question, args, kwargs], sort_keys=True, ensure_ascii=False, default=str))
        if key not in cache:
            result = fn(question, *args, **kwargs)
            if len(cache) < 64:
                cache[key] = deepcopy(result)
            return result
        # Callers must never mutate another selection pass's analysis.
        return deepcopy(cache[key])
    return wrapped
