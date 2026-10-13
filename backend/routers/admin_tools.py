"""Administrator maintenance tools. Playground calls never persist chat records."""
import asyncio
import json
import time
from contextvars import ContextVar
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from routers.auth import require_admin
from routers.services.database import execute_affected, execute_insert, execute_one, execute_query

router = APIRouter()
playground_trace = ContextVar('admin_playground_trace', default=None)
TOPICS = {
    'thirteenth_month': ('13th-month pay', ['13th', 'thirteenth']),
    'holiday_pay': ('Holiday pay', ['holiday']),
    'leave': ('Leave benefits', ['leave', 'service incentive']),
    'employment_types': ('Employment types', ['probationary', 'regular', 'project-based', 'employment status']),
    'termination': ('Termination', ['termination', 'dismissal', 'just cause', 'authorized cause']),
    'wages': ('Wages and working hours', ['wage', 'overtime', 'hours of work']),
    'safety': ('Occupational safety', ['occupational', 'safety']),
}


async def audit(user, action, target, identifier, details):
    from routers.admin import log_admin_action
    await log_admin_action(user['id'], action, target, str(identifier), details)


class CorrectionUpdate(BaseModel):
    revision: int = Field(default=0, ge=0)
    stage: Literal['needs_kb_update', 'update_linked', 'ready_for_retest', 'closed']
    document_id: int | None = Field(default=None, gt=0)
    note: str = Field(min_length=1, max_length=2000)
    retest_note: str = Field(default='', max_length=12000)


@router.get('/corrections')
async def corrections(current_user=Depends(require_admin)):
    return await execute_query("""SELECT ac.*,LEFT(cm.content,200) AS answer_preview,
        ud.document_title FROM admin_corrections ac
        JOIN chat_messages cm ON cm.id=ac.message_id
        LEFT JOIN uploaded_documents ud ON ud.id=ac.document_id
        ORDER BY ac.updated_at DESC LIMIT 500""")


@router.put('/corrections/{message_id}')
async def save_correction(message_id: int, body: CorrectionUpdate, current_user=Depends(require_admin)):
    if not body.note.strip():
        raise HTTPException(422, 'Describe the correction required')
    if not await execute_one("SELECT id FROM chat_messages WHERE id=%s AND role='assistant'", (message_id,)):
        raise HTTPException(404, 'Answer not found')
    try:
        retest = json.loads(body.retest_note) if body.retest_note else {}
    except (ValueError, TypeError):
        retest = {}
    automatic = isinstance(retest, dict) and isinstance(retest.get('documents'), list)
    linked = []
    if automatic:
        for item in retest['documents']:
            if not isinstance(item, dict) or not isinstance(item.get('id'), int) or item['id'] <= 0:
                raise HTTPException(422, 'Invalid retest document link')
            if any(d['id'] == item['id'] for d in linked):
                continue
            document = await execute_one('SELECT id,document_title,original_filename FROM uploaded_documents WHERE id=%s', (item['id'],))
            if not document:
                raise HTTPException(404, 'A retest source document no longer exists')
            linked.append({'id':document['id'], 'title':document.get('document_title') or document.get('original_filename') or 'Document'})
        retest['documents'] = linked
        body.document_id = linked[0]['id'] if len(linked) == 1 else None
        body.retest_note = json.dumps(retest, ensure_ascii=False)
        if len(body.retest_note) > 12000:
            raise HTTPException(422, 'Retest record exceeds 12,000 characters')
    if body.stage == 'closed' and not body.document_id and not linked:
        raise HTTPException(422, 'Run Test Chatbot and review its returned source documents before closing')
    if body.document_id and not await execute_one('SELECT id FROM uploaded_documents WHERE id=%s', (body.document_id,)):
        raise HTTPException(404, 'Document not found')
    if body.stage == 'closed':
        if not body.retest_note.strip():
            raise HTTPException(422, 'Record the retest question, result and evidence before closing')
        if not isinstance(retest, dict):
            raise HTTPException(422, 'Retest record must contain a question, result and evidence')
        missing = [key for key in ('question', 'result', 'evidence')
                   if not isinstance(retest.get(key), str) or not retest[key].strip()]
        if missing:
            raise HTTPException(422, 'Record a non-empty retest question, result and evidence before closing')
    existing = await execute_one('SELECT stage,revision FROM admin_corrections WHERE message_id=%s', (message_id,))
    if body.revision != (existing['revision'] if existing else 0):
        raise HTTPException(409, 'Correction changed. Reopen it before saving.')
    allowed = {
        None: {'needs_kb_update'},
        'needs_kb_update': {'needs_kb_update', 'update_linked'},
        'update_linked': {'needs_kb_update', 'update_linked', 'ready_for_retest'},
        'ready_for_retest': {'needs_kb_update', 'update_linked', 'ready_for_retest', 'closed'},
        'closed': {'closed', 'needs_kb_update'},
    }
    if body.stage not in allowed.get(existing['stage'] if existing else None, set()):
        raise HTTPException(409, 'Complete the preceding correction step first')
    # Reopening invalidates the previous passing retest, rather than carrying it
    # forward to a later closure after another KB change.
    reopening = body.stage == 'needs_kb_update' and existing and existing['stage'] != 'needs_kb_update'
    retest_note = '' if reopening else body.retest_note.strip()
    values = (body.stage, body.document_id, body.note.strip(), retest_note, current_user['id'])
    if existing:
        changed = await execute_affected('''UPDATE admin_corrections SET stage=%s,document_id=%s,note=%s,
            retest_note=%s,updated_by=%s,revision=revision+1 WHERE message_id=%s AND revision=%s''',
            values + (message_id, body.revision))
    else:
        changed = await execute_affected('''INSERT IGNORE INTO admin_corrections
            (stage,document_id,note,retest_note,updated_by,message_id) VALUES (%s,%s,%s,%s,%s,%s)''', values + (message_id,))
    if not changed:
        raise HTTPException(409, 'Correction changed. Reopen it before saving.')
    await audit(current_user, 'correction_update', 'chat_message', message_id,
                {'stage': body.stage, 'document_id': body.document_id, 'retest_note': retest_note})
    return {'message': 'Correction saved. Original answer and query status are unchanged.', 'revision': body.revision + 1}


@router.get('/corrections/{message_id}')
async def correction_detail(message_id: int, current_user=Depends(require_admin)):
    return await execute_one('SELECT * FROM admin_corrections WHERE message_id=%s', (message_id,))


class CoverageUpdate(BaseModel):
    document_id: int = Field(gt=0)
    note: str = Field(min_length=1, max_length=2000)


@router.get('/coverage')
async def coverage(current_user=Depends(require_admin)):
    docs = await execute_query('''SELECT id,document_title,original_filename,topics_json,sub_intents_json,
        article_section,status,is_active,version_status,chunk_count,updated_at FROM uploaded_documents''')
    reviews = await execute_query('SELECT * FROM admin_coverage_reviews')
    items = []
    for key, (label, terms) in TOPICS.items():
        candidates = []
        for doc in docs:
            review = next((r for r in reviews if r['topic'] == key and r['document_id'] == doc['id']), None)
            text = ' '.join(str(doc.get(k) or '') for k in ('document_title','original_filename','topics_json','sub_intents_json','article_section')).lower()
            if not review and not any(term in text for term in terms):
                continue
            usable = bool(doc['is_active'] and doc['status'] == 'indexed' and doc['chunk_count']
                          and doc['version_status'] in ('current', 'active'))
            fresh = bool(review and (not doc.get('updated_at') or review['reviewed_at'] >= doc['updated_at']))
            candidates.append({**doc, 'review': review, 'review_current': usable and fresh})
        items.append({'topic': key, 'label': label, 'documents': candidates,
                      'status': 'Reviewed coverage' if any(d['review_current'] for d in candidates)
                      else 'Needs review' if candidates else 'No mapped source'})
    return {'items': items, 'note': 'Candidates use metadata keywords only. Reviewed coverage is a human assessment, not proof of complete legal coverage. Document edits require renewed review.'}


@router.put('/coverage/{topic}')
async def save_coverage(topic: str, body: CoverageUpdate, current_user=Depends(require_admin)):
    if topic not in TOPICS or not body.note.strip():
        raise HTTPException(422, 'Select a supported topic and record the supporting provisions')
    doc = await execute_one('SELECT status,is_active,version_status,chunk_count FROM uploaded_documents WHERE id=%s', (body.document_id,))
    if not doc:
        raise HTTPException(404, 'Document not found')
    if not (doc['is_active'] and doc['status'] == 'indexed' and doc['chunk_count'] and doc['version_status'] in ('current','active')):
        raise HTTPException(409, 'Only current, active, indexed documents with chunks can be reviewed')
    await execute_insert('''INSERT INTO admin_coverage_reviews (topic,document_id,note,reviewed_by)
        VALUES (%s,%s,%s,%s) ON DUPLICATE KEY UPDATE note=VALUES(note),reviewed_by=VALUES(reviewed_by),reviewed_at=CURRENT_TIMESTAMP(6)''',
        (topic, body.document_id, body.note.strip(), current_user['id']))
    await audit(current_user, 'coverage_review', 'document', body.document_id, {'topic': topic, 'note': body.note})
    return {'message': 'Coverage review saved'}


@router.delete('/coverage/{topic}/{document_id}')
async def remove_coverage(topic: str, document_id: int, current_user=Depends(require_admin)):
    await execute_insert('DELETE FROM admin_coverage_reviews WHERE topic=%s AND document_id=%s', (topic, document_id))
    await audit(current_user, 'coverage_review_removed', 'document', document_id, {'topic': topic})
    return {'message': 'Review removed; document retained'}


class PlaygroundMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1, max_length=20000)
    sources: list[dict] = Field(default_factory=list, max_length=30)
    intent: str | None = Field(default=None, max_length=100)


class PlaygroundRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    language: Literal['en', 'fil', 'hil'] = 'en'
    history: list[PlaygroundMessage] = Field(default_factory=list, max_length=10)


_playground_active = set()


@router.post('/playground')
async def playground(body: PlaygroundRequest, request: Request, current_user=Depends(require_admin)):
    if not body.message.strip():
        raise HTTPException(422, 'Enter a test question')
    uid = current_user['id']
    if uid in _playground_active or len(_playground_active) >= 3:
        raise HTTPException(429, 'A playground request is already running. Please wait.')
    from routers import chat
    trace = {}
    trace_token = playground_trace.set(trace)
    _playground_active.add(uid)
    started = time.perf_counter()
    try:
        response = await asyncio.wait_for(chat.run_guest_message(chat.GuestChatRequest(
            message=body.message, language=body.language,
            previous_intent=next((m.intent for m in reversed(body.history) if m.role == 'assistant'), None),
            history=[m.model_dump() for m in body.history]), request, sandbox=True), timeout=90)
        return {'response': response.model_dump(), 'elapsed_ms': round((time.perf_counter()-started)*1000),
                'diagnostics': trace or {'note': 'No scoped retrieval trace for this response. It may be a clarification, static response or a specialized route; see final sources.'},
                'note': 'Test only: not saved to chat history or query analytics. Uses the guest chat pipeline and live providers; provider health counters and retrieval caches are shared.'}
    except asyncio.TimeoutError:
        raise HTTPException(504, 'Playground timed out; retry after checking system performance')
    finally:
        _playground_active.discard(uid)
        playground_trace.reset(trace_token)
