"""Reviewed-label NLP benchmarks. No generation, retrieval, or training calls."""
import hashlib
import inspect
import json
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from config import BACKEND_DIR

LANGUAGES = {'en', 'fil', 'hil'}
INTENTS = {'wage_inquiry', 'leave_benefits', 'termination_rights', '13th_month',
           'employment_status', 'occupational_safety', 'holiday_pay', 'general_inquiry',
           'greeting', 'small_talk', 'identity', 'capability_question', 'gratitude', 'farewell'}
ENTITY_TYPES = {'money', 'dates', 'organizations', 'law_references', 'numbers'}
RUN_LOCK = threading.Lock()


def validate_dataset(data):
    if not isinstance(data, dict) or set(data) - {'name', 'version', 'cases'}:
        raise ValueError('Use an object containing name, version and cases.')
    for key in ('name', 'version'):
        if not isinstance(data.get(key), str) or not 1 <= len(data[key].strip()) <= 120:
            raise ValueError(f'{key} must contain 1–120 characters.')
    cases = data.get('cases')
    if not isinstance(cases, list) or not 1 <= len(cases) <= 200:
        raise ValueError('Provide 1–200 cases per dataset.')
    seen = set()
    for index, case in enumerate(cases, 1):
        prefix = f'Case {index}: '
        if not isinstance(case, dict) or set(case) - {'id', 'question', 'expected_language', 'expected_intent', 'expected_entities'}:
            raise ValueError(prefix + 'unrecognized fields.')
        identifier = case.get('id')
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', identifier) or identifier in seen:
            raise ValueError(prefix + 'id must be unique, using letters, digits, underscores or hyphens.')
        seen.add(identifier)
        if not isinstance(case.get('question'), str) or not 1 <= len(case['question'].strip()) <= 4000:
            raise ValueError(prefix + 'question must contain 1–4000 characters.')
        if not any(key in case for key in ('expected_language', 'expected_intent', 'expected_entities')):
            raise ValueError(prefix + 'provide at least one expected output.')
        if 'expected_language' in case and (not isinstance(case['expected_language'], str) or case['expected_language'] not in LANGUAGES):
            raise ValueError(prefix + 'expected_language must be en, fil or hil.')
        if 'expected_intent' in case and (not isinstance(case['expected_intent'], str) or case['expected_intent'] not in INTENTS):
            raise ValueError(prefix + 'expected_intent is not a supported intent.')
        if 'expected_entities' in case:
            entities = case['expected_entities']
            if not isinstance(entities, dict) or not entities or set(entities) - ENTITY_TYPES:
                raise ValueError(prefix + 'expected_entities must specify supported entity types.')
            for values in entities.values():
                if not isinstance(values, list) or len(values) > 50 or any(not isinstance(v, str) or not v.strip() or len(v) > 500 for v in values):
                    raise ValueError(prefix + 'entity values must be lists of up to 50 nonempty strings (500 characters each).')
    return {**data, 'name': data['name'].strip(), 'version': data['version'].strip()}


def normalized(value):
    return ' '.join(value.split()).casefold()


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'precision': precision, 'recall': recall,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def classification(pairs):
    if not pairs:
        return {'samples': 0, 'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'classes': [], 'confusion': []}
    labels = sorted({value for pair in pairs for value in pair})
    classes = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in pairs)
        fp = sum(a != label and b == label for a, b in pairs)
        fn = sum(a == label and b != label for a, b in pairs)
        classes.append({'label': label, 'support': sum(a == label for a, _ in pairs), **prf(tp, fp, fn)})
    return {'samples': len(pairs), 'accuracy': sum(a == b for a, b in pairs) / len(pairs),
            **{key: sum(row[key] for row in classes) / len(classes) for key in ('precision', 'recall', 'f1')},
            'classes': classes, 'labels': labels,
            'confusion': [[sum(a == expected and b == predicted for a, b in pairs) for predicted in labels] for expected in labels]}


def summarize(rows):
    output = {}
    for task in ('language', 'intent'):
        output[task] = classification([(row['expected'][f'expected_{task}'], row['predicted'][task]) for row in rows
                                       if row['status'] == 'completed' and f'expected_{task}' in row['expected']])
    entities = []
    for kind in sorted(ENTITY_TYPES):
        counts = [0, 0, 0]
        samples = 0
        for row in rows:
            if row['status'] != 'completed' or kind not in row['expected'].get('expected_entities', {}):
                continue
            samples += 1
            expected = {normalized(v) for v in row['expected']['expected_entities'][kind]}
            predicted = {normalized(v) for v in row['predicted']['entities'].get(kind, [])}
            for index, value in enumerate((len(expected & predicted), len(predicted - expected), len(expected - predicted))):
                counts[index] += value
        if samples:
            entities.append({'type': kind, 'samples': samples, 'tp': counts[0], 'fp': counts[1], 'fn': counts[2], **prf(*counts)})
    totals = [sum(row[key] for row in entities) for key in ('tp', 'fp', 'fn')]
    output['entities'] = {'samples': sum(row['status'] == 'completed' and 'expected_entities' in row['expected'] for row in rows),
                          'accuracy': None, **(prf(*totals) if entities else dict.fromkeys(('precision', 'recall', 'f1'))),
                          'types': entities}
    return output


def evaluate(dataset, predict=None):
    from routers.services import nlp_service
    if predict is None:
        nlp_service.learned_nlp.prime([case['question'] for case in dataset['cases']])
    predict = predict or nlp_service.preprocess_query
    rows = []
    for case in dataset['cases']:
        row = {'id': case['id'], 'question': case['question'], 'expected': {k: v for k, v in case.items() if k.startswith('expected_')}}
        try:
            value = predict(case['question'])
            row.update(status='completed', predicted={k: value[k] for k in ('language', 'intent', 'entities', 'normalized', 'search_text')})
            if 'classification' in value:
                row['predicted']['classification'] = value['classification']
            # Validate prediction structure before scoring, without leaking exception text.
            if not isinstance(row['predicted']['language'], str) or not isinstance(row['predicted']['intent'], str):
                raise ValueError('Invalid labels')
            if not isinstance(value['entities'], dict) or any(not isinstance(v, list) or any(not isinstance(x, str) for x in v) for v in value['entities'].values()):
                raise ValueError('Invalid entities')
            mismatches = []
            for task in ('language', 'intent'):
                key = 'expected_' + task
                if key in case and case[key] != value[task]:
                    mismatches.append(task)
            for kind, expected in case.get('expected_entities', {}).items():
                if {normalized(v) for v in expected} != {normalized(v) for v in value['entities'].get(kind, [])}:
                    mismatches.append('entities.' + kind)
            row['mismatches'] = mismatches
        except Exception:
            row.update(status='failed', predicted=None, mismatches=[], error='NLP processing failed for this case.')
        rows.append(row)
    fingerprint = hashlib.sha256((inspect.getsource(nlp_service) + inspect.getsource(nlp_service.learned_nlp)).encode()).hexdigest()
    return {'completed': sum(r['status'] == 'completed' for r in rows), 'failed': sum(r['status'] == 'failed' for r in rows),
            'mismatch_cases': sum(bool(r['mismatches']) for r in rows), 'metrics': summarize(rows),
            'by_language': {language: summarize([r for r in rows if r['expected'].get('expected_language') == language]) for language in sorted(LANGUAGES)},
            'configuration': {'nlp_source_sha256': fingerprint, **nlp_service.model_configuration(), 'scoring_version': '1'},
            'method': 'Classification: macro average over expected and predicted labels; zero denominators score 0. Entities: micro counts of unique text/type pairs after casefolding and whitespace normalization. Only explicitly annotated tasks/types are scored; [] means no entity expected. Failed cases are excluded from scores and counted separately. Zero-entity denominators score 0, not perfect performance.',
            'cases': rows}


class EvaluationStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else BACKEND_DIR / 'data' / 'nlp_evaluations.sqlite3'

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                db.execute('CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, created_at TEXT, payload TEXT)')
                db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, dataset_id TEXT, created_at TEXT, payload TEXT)')
                yield db
        finally:
            db.close()

    def put_dataset(self, data):
        data = validate_dataset(data)
        identifier = uuid.uuid4().hex
        data = {**data, 'id': identifier, 'created_at': datetime.now(timezone.utc).isoformat(),
                'sha256': hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        with self.connect() as db:
            db.execute('INSERT INTO datasets VALUES (?, ?, ?)', (identifier, data['created_at'], json.dumps(data, ensure_ascii=False)))
        return data

    def get(self, table, identifier):
        if table not in ('datasets', 'runs'):
            raise ValueError('Invalid table')
        with self.connect() as db:
            row = db.execute(f'SELECT payload FROM {table} WHERE id=?', (identifier,)).fetchone()
        return json.loads(row['payload']) if row else None

    def listing(self, table, limit=50):
        if table not in ('datasets', 'runs'):
            raise ValueError('Invalid table')
        with self.connect() as db:
            rows = db.execute(f'SELECT payload FROM {table} ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()
        items = []
        for row in rows:
            value = json.loads(row['payload'])
            items.append({k: v for k, v in value.items() if k not in ('cases', 'by_language')})
            items[-1]['case_count'] = len(value.get('cases', []))
        return items

    def run(self, dataset, predict=None):
        if not RUN_LOCK.acquire(blocking=False):
            raise RuntimeError('An NLP evaluation is already running on this worker. Try again when it finishes.')
        try:
            result = evaluate(dataset, predict)
            result.update(id=uuid.uuid4().hex, dataset_id=dataset['id'], dataset_name=dataset['name'],
                          dataset_version=dataset['version'], dataset_sha256=dataset['sha256'],
                          created_at=datetime.now(timezone.utc).isoformat())
            with self.connect() as db:
                db.execute('INSERT INTO runs VALUES (?, ?, ?, ?)', (result['id'], dataset['id'], result['created_at'], json.dumps(result, ensure_ascii=False)))
            return result
        finally:
            RUN_LOCK.release()
