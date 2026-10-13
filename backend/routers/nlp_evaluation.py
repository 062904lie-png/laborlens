"""Admin-only NLP benchmark upload, execution and reports."""
import asyncio
import csv
import io
import json
import re
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from routers.auth import require_admin
from routers.services.nlp_evaluation import EvaluationStore, LANGUAGES, INTENTS, ENTITY_TYPES

router = APIRouter(prefix='/nlp-evaluation')
store = EvaluationStore()


async def stored(fn, *args):
    try:
        return await asyncio.to_thread(fn, *args)
    except (sqlite3.Error, OSError):
        raise HTTPException(503, 'NLP evaluation storage is unavailable.') from None


@router.get('')
async def overview(user=Depends(require_admin)):
    return {'datasets': await stored(store.listing, 'datasets'), 'runs': await stored(store.listing, 'runs'),
            'schema': {'languages': sorted(LANGUAGES), 'intents': sorted(INTENTS), 'entity_types': sorted(ENTITY_TYPES), 'max_cases': 200}}


@router.post('/datasets', status_code=201)
async def upload(request: Request, user=Depends(require_admin)):
    body = bytearray()
    async for part in request.stream():
        body.extend(part)
        if len(body) > 1_000_000:
            raise HTTPException(413, 'Dataset must be at most 1 MB.')
    try:
        data = json.loads(body.decode('utf-8-sig'))
        result = await stored(store.put_dataset, data)
    except (ValueError, TypeError, RecursionError):
        raise HTTPException(422, 'Invalid dataset. Use name, version and 1–200 cases with unique ids, questions and supported expected outputs. See the dataset format in the panel.') from None
    return {k: v for k, v in result.items() if k != 'cases'} | {'case_count': len(result['cases'])}


@router.post('/datasets/{identifier}/runs', status_code=201)
async def run(identifier: str, user=Depends(require_admin)):
    dataset = await stored(store.get, 'datasets', identifier)
    if not dataset:
        raise HTTPException(404, 'Dataset not found.')
    try:
        return await stored(store.run, dataset)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None


@router.get('/runs/{identifier}')
async def detail(identifier: str, user=Depends(require_admin)):
    result = await stored(store.get, 'runs', identifier)
    if not result:
        raise HTTPException(404, 'Evaluation run not found.')
    return result


def csv_cell(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    if value is None:
        return ''
    text = str(value)
    if isinstance(value, str) and re.match(r'^[\x00-\x20\uFEFF\u200B]*[=+\-@]', text):
        text = "'" + text
    return text


@router.get('/runs/{identifier}/export/{kind}')
async def export(identifier: str, kind: str, user=Depends(require_admin)):
    if kind not in ('summary', 'cases'):
        raise HTTPException(404, 'Export type not found.')
    result = await detail(identifier, user)
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['run_id', result['id'], 'dataset', csv_cell(result['dataset_name']), 'version', csv_cell(result['dataset_version'])])
    writer.writerow(['created_at', result['created_at'], 'completed', result['completed'], 'failed', result['failed']])
    writer.writerow(['dataset_sha256', result['dataset_sha256'], 'configuration', json.dumps(result['configuration'])])
    if kind == 'summary':
        writer.writerow(['task', 'reviewed_language', 'samples', 'accuracy', 'precision', 'recall', 'f1'])
        for language, metrics in [('all', result['metrics']), *result['by_language'].items()]:
            for task, values in metrics.items():
                writer.writerow([task, language, *[csv_cell(values.get(key)) for key in ('samples', 'accuracy', 'precision', 'recall', 'f1')]])
        writer.writerow([])
        writer.writerow(['task', 'class_or_entity_type', 'support_or_annotated_cases', 'precision', 'recall', 'f1'])
        for task in ('language', 'intent'):
            for row in result['metrics'][task]['classes']:
                writer.writerow([task, csv_cell(row['label']), row['support'], row['precision'], row['recall'], row['f1']])
        for row in result['metrics']['entities']['types']:
            writer.writerow(['entities', row['type'], row['samples'], row['precision'], row['recall'], row['f1']])
        writer.writerow(['method', result['method']])
    else:
        writer.writerow(['case_id', 'question', 'status', 'expected', 'predicted', 'mismatches', 'error'])
        for row in result['cases']:
            writer.writerow([csv_cell(row.get(key)) for key in ('id', 'question', 'status', 'expected', 'predicted', 'mismatches', 'error')])
    return Response('\ufeff' + stream.getvalue(), media_type='text/csv', headers={'Content-Disposition': f'attachment; filename="nlp-{identifier}-{kind}.csv"'})
