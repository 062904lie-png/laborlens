"""Persistent, allowlisted admin controls for non-secret runtime settings."""
import json
import logging
import math
import re
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path

from config import BACKEND_DIR, settings


DATABASE_PATH = BACKEND_DIR / 'data' / 'admin_runtime_settings.sqlite3'
_POOL_FIELDS = {
    'LABORLENS_GENERATOR_POOL', 'LABORLENS_RERANKER_POOL',
}
_INTEGER_FIELDS = {
    'LABORLENS_MAX_OUTPUT_TOKENS': (400, 2048),
    'LABORLENS_MAX_OUTPUT_TOKENS_COMPLEX': (400, 2048),
    'LABORLENS_SIMPLE_OUTPUT_TOKENS': (400, 2048),
    'RAG_INITIAL_TOP_K': (4, 30),
    'RAG_EXPANDED_TOP_K': (4, 50),
    'RAG_CONTEXT_TOKEN_BUDGET': (256, 30000),
    'RAG_CANDIDATE_POOL_SIZE': (10, 50),
    'LABORLENS_RETRIEVAL_CACHE_SIZE': (16, 2048),
}
_NUMBER_FIELDS = {
    'LABORLENS_NORMAL_REQUEST_BUDGET_SECONDS': (10.0, 120.0),
    'LABORLENS_COMPLEX_REQUEST_BUDGET_SECONDS': (10.0, 180.0),
    'LABORLENS_OPENAI_TIMEOUT_SECONDS': (5.0, 30.0),
    'LABORLENS_GEMINI_TIMEOUT_SECONDS': (5.0, 30.0),
    'LABORLENS_GROQ_TIMEOUT_SECONDS': (5.0, 30.0),
    'LABORLENS_RETRIEVAL_CACHE_TTL_SECONDS': (0.0, 86400.0),
}
_BOOLEAN_FIELDS = {
    'LABORLENS_OPENAI_ENABLED', 'LABORLENS_GEMINI_ENABLED',
    'LABORLENS_MODEL_POOLS_ENABLED', 'LABORLENS_LLM_RERANK_ENABLED',
    'LDA_RERANK_ENABLED',
}
_STRING_FIELDS = {'APP_NAME'}
EDITABLE_FIELDS = frozenset(_POOL_FIELDS | set(_INTEGER_FIELDS) | set(_NUMBER_FIELDS)
                            | _BOOLEAN_FIELDS | _STRING_FIELDS)
_BASELINE = {key: getattr(settings, key) for key in EDITABLE_FIELDS}
_MODEL_ID = re.compile(r'^(?:openai|gemini|groq)/[A-Za-z0-9][A-Za-z0-9._:/-]{0,119}$')
logger = logging.getLogger(__name__)


class SettingsValidationError(ValueError):
    pass


@contextmanager
def _connect():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DATABASE_PATH), timeout=3)
    try:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS runtime_settings '
                   '(setting_key TEXT PRIMARY KEY, value_json TEXT NOT NULL)')
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _read_overrides():
    if not DATABASE_PATH.exists():
        return {}
    with _connect() as db:
        rows = db.execute('SELECT setting_key,value_json FROM runtime_settings').fetchall()
    result = {}
    for key, raw in rows:
        if key in EDITABLE_FIELDS:
            try:
                result[key] = json.loads(raw)
            except (TypeError, ValueError):
                continue
    return result


def _pool_value(value, key):
    if isinstance(value, list):
        if any(not isinstance(item, str) for item in value):
            raise SettingsValidationError(f'{key} must contain model names only.')
        value = ','.join(value)
    if not isinstance(value, str):
        raise SettingsValidationError(f'{key} must be a comma- or line-separated model list.')
    models = list(dict.fromkeys(item.strip() for item in re.split(r'[,\n\r]+', value) if item.strip()))
    if not models or len(models) > 32:
        raise SettingsValidationError(f'{key} must contain between 1 and 32 models; there is no five-model cap.')
    if any(not _MODEL_ID.fullmatch(model) for model in models):
        raise SettingsValidationError(f'{key} entries must use provider/model format (openai/, gemini/, or groq/).')
    return ','.join(models)


def validate_changes(changes, *, current=None):
    if not isinstance(changes, dict) or not changes:
        raise SettingsValidationError('Provide at least one setting to save.')
    if len(changes) > 24:
        raise SettingsValidationError('Save at most 24 settings at a time.')
    unknown = sorted(set(changes) - EDITABLE_FIELDS)
    if unknown:
        raise SettingsValidationError('One or more settings are not editable.')
    clean = {}
    for key, value in changes.items():
        if key in _POOL_FIELDS:
            clean[key] = _pool_value(value, key)
        elif key in _INTEGER_FIELDS:
            low, high = _INTEGER_FIELDS[key]
            if type(value) is not int or not low <= value <= high:
                raise SettingsValidationError(f'{key} must be a whole number from {low} to {high}.')
            clean[key] = value
        elif key in _NUMBER_FIELDS:
            low, high = _NUMBER_FIELDS[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise SettingsValidationError(f'{key} must be a number from {low:g} to {high:g}.')
            clean[key] = float(value)
        elif key in _BOOLEAN_FIELDS:
            if type(value) is not bool:
                raise SettingsValidationError(f'{key} must be true or false.')
            clean[key] = value
        elif key in _STRING_FIELDS:
            if not isinstance(value, str) or not 2 <= len(value.strip()) <= 80:
                raise SettingsValidationError(f'{key} must contain 2 to 80 characters.')
            clean[key] = value.strip()

    effective = dict(current or {key: getattr(settings, key) for key in EDITABLE_FIELDS})
    effective.update(clean)
    if int(effective['RAG_EXPANDED_TOP_K']) < int(effective['RAG_INITIAL_TOP_K']):
        raise SettingsValidationError('Wider retrieval count must be at least the initial retrieval count.')
    if float(effective['LABORLENS_COMPLEX_REQUEST_BUDGET_SECONDS']) < float(effective['LABORLENS_NORMAL_REQUEST_BUDGET_SECONDS']):
        raise SettingsValidationError('The complex request budget must be at least the normal request budget.')
    return clean


def _apply(values):
    for key, value in values.items():
        if key in EDITABLE_FIELDS:
            setattr(settings, key, value)
    # Retrieval limits are cached as module constants; refresh them whenever
    # persistent settings are loaded or changed so they match the displayed UI.
    module = sys.modules.get('routers.services.retrieval_router')
    if module is not None:
        module.INITIAL_TOP_K = max(4, min(int(settings.RAG_INITIAL_TOP_K), 30))
        module.EXPANDED_TOP_K = max(module.INITIAL_TOP_K, min(int(settings.RAG_EXPANDED_TOP_K), 50))


def effective_configuration():
    return {key: getattr(settings, key) for key in EDITABLE_FIELDS}


def saved_keys():
    return sorted(_read_overrides())


def load_saved_settings():
    """Apply persisted, validated values during application startup."""
    stored = _read_overrides()
    if not stored:
        return []
    try:
        validated = validate_changes(stored, current=_BASELINE)
    except SettingsValidationError:
        logger.warning('Ignoring invalid persisted admin runtime settings')
        return []
    _apply({**_BASELINE, **validated})
    return sorted(validated)


def save_settings(changes):
    current = effective_configuration()
    clean = validate_changes(changes, current=current)
    merged = {**_read_overrides(), **clean}
    # Validate cross-field constraints against all active overrides before writing.
    validate_changes(merged, current=_BASELINE)
    with _connect() as db:
        db.executemany('INSERT INTO runtime_settings(setting_key,value_json) VALUES(?,?) '
                       'ON CONFLICT(setting_key) DO UPDATE SET value_json=excluded.value_json',
                       [(key, json.dumps(value, allow_nan=False)) for key, value in merged.items()])
    _apply({**_BASELINE, **merged})
    return effective_configuration()


def reset_settings():
    if DATABASE_PATH.exists():
        with _connect() as db:
            db.execute('DELETE FROM runtime_settings')
    _apply(_BASELINE)
    return effective_configuration()
