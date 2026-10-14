"""LaborLens - Admin Router."""

import json
from datetime import date, datetime, timedelta
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from routers.auth import get_current_user, require_admin
from routers.services.database import execute_admin_role_change, AdminRoleChangeConflict, execute_insert, execute_one, execute_query
from routers.services.ingestion_service import get_ingestion_status
from routers.services.report_topics import rank_question_topics, conversation_topics, question_topic, LABELS

router = APIRouter()
from routers.admin_tools import router as maintenance_router
router.include_router(maintenance_router)
from routers.nlp_evaluation import router as nlp_evaluation_router
router.include_router(nlp_evaluation_router)


@router.get('/backend-diagnostics')
async def backend_diagnostics_report(
    days: int = Query(7, ge=1, le=30), limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0), level: Optional[Literal['info', 'warning', 'error']] = None,
    component: Optional[str] = Query(None, max_length=100),
    request_id: Optional[str] = Query(None, max_length=100), current_user=Depends(require_admin),
):
    import asyncio
    from routers.services.backend_diagnostics import get_store, storage_status
    try:
        report = await asyncio.to_thread(get_store().report, days, limit, offset, level, component, request_id)
    except Exception:
        raise HTTPException(503, 'Diagnostic storage is unavailable. Check its configured path and permissions.') from None
    return {**report, 'storage': storage_status()}

FAQ_LANGUAGES = {"en", "fil", "hil"}
REVIEW_STATUSES = {"pending", "reviewed", "dismissed", "under_review", "resolved"}
_PRIVATE_USER_KEYS = {
    "username",
    "email",
    "full_name",
    "display_name",
    "user_name",
    "user_email",
}


def redact_user_identity(value):
    """Remove end-user identity fields from admin-facing audit payloads."""

    if isinstance(value, dict):
        return {
            key: "[redacted]" if str(key).lower() in _PRIVATE_USER_KEYS else redact_user_identity(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_user_identity(item) for item in value]
    return value


def redact_audit_details(details):
    if not isinstance(details, str):
        return redact_user_identity(details)
    try:
        parsed = json.loads(details)
    except (TypeError, ValueError):
        return details
    return json.dumps(redact_user_identity(parsed), ensure_ascii=False)


class FAQCreate(BaseModel):
    question: str
    answer: str
    category: Optional[str] = None
    language: str = "en"
    publication_status: Literal["draft", "published", "needs_review"] = "draft"
    source_document_ids: List[int] = Field(default_factory=list, max_length=20)
    source_citation: str = Field(default="", max_length=2000)


class FAQImportRequest(BaseModel):
    items: List[FAQCreate]


class FAQUpdate(FAQCreate):
    pass


class ReviewQueueUpdate(BaseModel):
    review_status: Literal["pending", "reviewed", "dismissed", "under_review", "resolved"] = "reviewed"
    admin_notes: str = Field(default="", max_length=2000)


class AnswerReviewUpdate(BaseModel):
    status: Literal['unreviewed', 'correct', 'needs_review', 'incorrect', 'kb_update', 'resolved', 'partial_correct']
    note: str = Field(default='', max_length=2000)
    query_status: Optional[Literal['resolved', 'unresolved']] = None


class AnswerQueryStatusUpdate(BaseModel):
    query_status: Optional[Literal['resolved', 'unresolved']] = None


@router.get('/configuration')
async def admin_configuration(current_user=Depends(require_admin)):
    from config import settings
    from routers.services import admin_runtime_settings
    # Explicit allowlist: never serialize settings or environment wholesale.
    keys=['APP_NAME','GROQ_MODEL','LABORLENS_PRIMARY_PROVIDER','LABORLENS_FALLBACK_PROVIDER',
          'LABORLENS_MAX_OUTPUT_TOKENS','LABORLENS_MAX_OUTPUT_TOKENS_COMPLEX',
          'LABORLENS_CONTINUATION_TOKENS','LABORLENS_NORMAL_REQUEST_BUDGET_SECONDS',
          'LABORLENS_COMPLEX_REQUEST_BUDGET_SECONDS','RAG_INITIAL_TOP_K','RAG_EXPANDED_TOP_K',
          'LABORLENS_SIMPLE_OUTPUT_TOKENS','LABORLENS_PROVIDER_TIMEOUT_SECONDS',
          'LABORLENS_RETRIEVAL_CACHE_SIZE','LABORLENS_RETRIEVAL_CACHE_TTL_SECONDS','JWT_EXPIRE_MINUTES']
    values = {key:getattr(settings,key) for key in keys}
    values.update(admin_runtime_settings.effective_configuration())
    return {'values':values,'editable_keys':sorted(admin_runtime_settings.EDITABLE_FIELDS),
            'saved_overrides':admin_runtime_settings.saved_keys(),
            'note':'Secret keys and live status are never editable here. Saved values apply to this worker and are loaded after restart.'}


class RuntimeSettingsUpdate(BaseModel):
    values: dict = Field(min_length=1, max_length=24)


@router.patch('/configuration')
async def update_admin_configuration(payload: RuntimeSettingsUpdate, request: Request,
                                     current_user=Depends(require_admin)):
    from routers.services import admin_runtime_settings
    try:
        values = admin_runtime_settings.save_settings(payload.values)
    except admin_runtime_settings.SettingsValidationError as exc:
        raise HTTPException(422, str(exc)) from None
    request.app.title = str(values.get('APP_NAME') or request.app.title)
    changed_fields = sorted(payload.values)
    await log_admin_action(current_user['id'], 'update_runtime_settings', 'system',
                           details={'fields':changed_fields})
    return {'values':values,'editable_keys':sorted(admin_runtime_settings.EDITABLE_FIELDS),
            'saved_overrides':admin_runtime_settings.saved_keys(),
            'message':'Settings saved. They apply to this worker now and load automatically after restart.'}


@router.delete('/configuration')
async def reset_admin_configuration(current_user=Depends(require_admin)):
    from routers.services import admin_runtime_settings
    values = admin_runtime_settings.reset_settings()
    await log_admin_action(current_user['id'], 'reset_runtime_settings', 'system')
    return {'values':values,'saved_overrides':[],
            'message':'Saved overrides cleared. Deployment defaults are active in this worker.'}


@router.get('/conversations')
async def admin_conversations(q: str = Query('',max_length=200), language: str = '', offset: int = Query(0,ge=0), limit: int = Query(25,ge=1,le=100), current_user=Depends(require_admin)):
    where='cs.title LIKE %s'
    params=[f'%{q}%']
    if language:
        where+=" AND EXISTS (SELECT 1 FROM chat_messages m WHERE m.session_id=cs.id AND m.language=%s)"
        params.append(normalize_language(language))
    total=await execute_one(f'SELECT COUNT(*) AS total FROM chat_sessions cs WHERE {where}',tuple(params))
    rows=await execute_query(f"""SELECT cs.id,cs.title,cs.category,cs.created_at,cs.updated_at,
        CONCAT('Anonymous User #',cs.user_id) AS anonymous_user,
        (SELECT COUNT(*) FROM chat_messages m WHERE m.session_id=cs.id AND m.role IN ('user','assistant')) AS message_count
        FROM chat_sessions cs WHERE {where} ORDER BY cs.updated_at DESC LIMIT %s OFFSET %s""",tuple(params+[limit,offset]))
    return {'items':rows,'total':total['total']}


class UserAccessUpdate(BaseModel):
    is_active: bool


@router.put('/users/{user_id}/access')
async def admin_user_access(user_id: int, payload: UserAccessUpdate, current_user=Depends(require_admin)):
    user=await execute_one('SELECT id,role FROM users WHERE id=%s',(user_id,))
    if not user:
        raise HTTPException(404,'User not found')
    if user['role']=='admin' or user_id==current_user['id']:
        raise HTTPException(409,'Administrator accounts cannot be disabled through this action')
    await execute_insert('UPDATE users SET is_active=%s WHERE id=%s',(payload.is_active,user_id))
    await log_admin_action(current_user['id'],'user_reactivated' if payload.is_active else 'user_disabled','user',str(user_id))
    return {'message':'Account access updated'}


# User reports are explicit reports, not every negative helpfulness rating.
REPORT_SQL = "(LOWER(TRIM(COALESCE(cf.comment,''))) = 'report incorrect answer' OR LOWER(TRIM(COALESCE(cf.comment,''))) LIKE 'report incorrect answer:%%')"

# Keep the admin labels aligned with the confidence labels shown to chat users.
# Some older rows only have `confidence_level`, so the CASE intentionally supports
# both fields without modifying historical chat records.
SOURCE_MATCH_LABEL_SQL = """CASE
    WHEN cm.intent='needs_clarification' THEN 'Awaiting user response'
    WHEN LOWER(REPLACE(TRIM(COALESCE(cm.confidence_level,'')), '_', ' ')) = 'not required' THEN 'Not Required'
    WHEN LOWER(TRIM(COALESCE(cm.intent,''))) IN
    ('greeting','small_talk','identity','identity_question','capability_question',
     'gratitude','thanks','farewell','goodbye','acknowledgement') THEN 'Not Required'
    WHEN LOWER(REPLACE(TRIM(COALESCE(cm.confidence_level,'')), '_', ' ')) IN ('provider unavailable', 'temporary response issue') THEN 'Temporary response issue'
    WHEN LOWER(REPLACE(TRIM(COALESCE(cm.confidence_level,'')), '_', ' ')) IN ('conflicting', 'conflicting sources') THEN 'Conflicting sources'
    WHEN LOWER(TRIM(COALESCE(cm.confidence_level,''))) = 'unverified' THEN 'Unverified test answer'
    WHEN LOWER(TRIM(COALESCE(cm.confidence_label,''))) = 'high confidence match' THEN 'High confidence match'
    WHEN LOWER(TRIM(COALESCE(cm.confidence_label,''))) = 'source match' THEN 'Source match'
    WHEN LOWER(TRIM(COALESCE(cm.confidence_label,''))) = 'partial source match' THEN 'Partial source match'
    WHEN LOWER(TRIM(COALESCE(cm.confidence_label,''))) = 'conflicting sources' THEN 'Conflicting sources'
    WHEN LOWER(TRIM(COALESCE(cm.confidence_label,''))) = 'temporary response issue' THEN 'Temporary response issue'
    WHEN LOWER(TRIM(COALESCE(cm.confidence_label,''))) = 'insufficient legal source' THEN 'Insufficient legal source'
    WHEN LOWER(REPLACE(TRIM(COALESCE(cm.confidence_level,'')), '_', ' ')) IN ('high', 'high confidence', 'high confidence match') THEN 'High confidence match'
    WHEN LOWER(REPLACE(TRIM(COALESCE(cm.confidence_level,'')), '_', ' ')) IN ('direct', 'source', 'source match') THEN 'Source match'
    WHEN LOWER(REPLACE(TRIM(COALESCE(cm.confidence_level,'')), '_', ' ')) IN ('partial', 'partial source match') THEN 'Partial source match'
    ELSE 'Insufficient legal source'
END"""
SOURCE_QUERY_STATUS_SQL = f"""CASE
    WHEN cm.intent='needs_clarification' THEN 'awaiting_user'
    WHEN ({SOURCE_MATCH_LABEL_SQL})='Not Required' THEN 'resolved'
    WHEN ({SOURCE_MATCH_LABEL_SQL}) IN
    ('Insufficient legal source', 'Conflicting sources', 'Temporary response issue', 'Unverified test answer')
    THEN 'unresolved' ELSE 'resolved' END"""
# A source match supplies the initial recommendation. An explicit query workflow
# decision is stored separately from the answer assessment to preserve both.
QUERY_STATUS_SQL = f"""CASE
    WHEN cm.admin_query_status='automatic' THEN ({SOURCE_QUERY_STATUS_SQL})
    WHEN cm.admin_query_status IN ('resolved','unresolved') THEN cm.admin_query_status
    WHEN cm.admin_review_status='resolved' THEN 'resolved'
    WHEN cm.admin_review_status IN ('needs_review','incorrect','kb_update') THEN 'unresolved'
    ELSE ({SOURCE_QUERY_STATUS_SQL})
END"""


# Reuse the existing taxonomy. Historical messages currently inherit their
# session's stored category; no frontend topic inference or new storage.
from routers.services.kb_categories import KB_CATEGORIES, LEGACY_KB_CATEGORY_MAP
def _category_sql():
    quote = lambda value: "'" + value.replace("'", "''") + "'"
    cases = ' '.join(f"WHEN cs.category={quote(old)} THEN {quote(new)}"
                     for old, new in LEGACY_KB_CATEGORY_MAP.items())
    known = ','.join(quote(value) for value in KB_CATEGORIES)
    return f"CASE WHEN cs.category IN ({known}) THEN cs.category {cases} ELSE 'Others' END"

CATEGORY_SQL = _category_sql()
CLOSED_CORRECTION_SQL = "EXISTS(SELECT 1 FROM admin_corrections ac WHERE ac.message_id=cm.id AND ac.stage='closed')"
REVIEW_STATUS_SQL = f"(CASE WHEN {CLOSED_CORRECTION_SQL} THEN 'resolved' ELSE cm.admin_review_status END)"
COMPLETED_REVIEW_SQL = f"{REVIEW_STATUS_SQL} IN ('correct','partial_correct','resolved')"
NEEDS_REVIEW_SQL = f"""({REVIEW_STATUS_SQL} IN ('needs_review','incorrect','kb_update') OR
    (NOT ({COMPLETED_REVIEW_SQL}) AND (({QUERY_STATUS_SQL})='unresolved' OR
    EXISTS(SELECT 1 FROM chat_feedback nf WHERE nf.message_id=cm.id AND nf.rating='no'
        AND (COALESCE(cm.intent,'')<>'needs_clarification' OR LEFT(LOWER(TRIM(COALESCE(nf.comment,''))),14)<>'system review:')) OR
    EXISTS(SELECT 1 FROM chat_feedback cf WHERE cf.message_id=cm.id AND {REPORT_SQL}))))"""


async def attach_review_topics(rows):
    """Use the same chronological question classifier as Reports, not categories."""
    sessions = sorted({row["session_id"] for row in rows if row.get("session_id")})
    history = []
    if sessions:
        placeholders = ",".join(["%s"] * len(sessions))
        history = await execute_query(
            f"""SELECT id,session_id,role,content,intent FROM chat_messages
                WHERE session_id IN ({placeholders}) AND id<=%s
                AND role IN ('user','assistant') ORDER BY session_id,id""",
            tuple(sessions + [max(row.get("question_id") or 0 for row in rows)]),
        )
    contextual = conversation_topics(history)
    for row in rows:
        key = contextual.get(row.get("question_id"))
        if key is None:
            key = question_topic(row.get("topic_question") or row.get("question") or "",
                                 row.get("question_intent") or "")
        row["topic"] = key
        row["topic_label"] = LABELS.get(key, key.replace("_", " ").title())
        row.pop("topic_question", None)
    return rows


@router.get('/answer-reviews')
async def answer_reviews(
    q: str = Query('', max_length=200), language: str = '', category: str = Query('', max_length=100), review_status: str = '', report_status: str = '',
    reports_only: bool = False, needs_attention: bool = False, completed: bool = False, other_issue: bool = False, offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100),
    feedback: Literal['', 'yes', 'no'] = '', query_status: Literal['', 'resolved', 'unresolved'] = '', date_from: Optional[date] = None, date_to: Optional[date] = None,
    current_user=Depends(require_admin),
):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail='date_from must be on or before date_to')
    conditions = ["cm.role='assistant'"]
    params = []
    if q:
        conditions.append("(cm.content LIKE %s OR um.content LIKE %s OR cm.session_id LIKE %s OR CONCAT('Anonymous User #',cs.user_id) LIKE %s OR cm.admin_review_note LIKE %s)")
        params.extend([f'%{q}%']*5)
    if feedback:
        conditions.append("EXISTS (SELECT 1 FROM chat_feedback f WHERE f.message_id=cm.id AND f.rating=%s AND LEFT(LOWER(TRIM(COALESCE(f.comment,''))),14) <> 'system review:')")
        params.append(feedback)
    if date_from:
        conditions.append('cm.created_at>=%s'); params.append(date_from)
    if date_to:
        conditions.append('cm.created_at<%s'); params.append(date_to+timedelta(days=1))
    if language:
        conditions.append('cm.language=%s')
        params.append(normalize_language(language))
    if category:
        conditions.append(f'({CATEGORY_SQL})=%s')
        params.append(category)
    if review_status:
        conditions.append(f'{REVIEW_STATUS_SQL}=%s')
        params.append(review_status)
    if query_status:
        conditions.append(f'({QUERY_STATUS_SQL})=%s')
        params.append(query_status)
    if needs_attention:
        conditions.append(NEEDS_REVIEW_SQL)
    if completed:
        conditions.append(COMPLETED_REVIEW_SQL)
    if other_issue:
        conditions.append(NEEDS_REVIEW_SQL)
        conditions.append(f"NOT EXISTS(SELECT 1 FROM chat_feedback cf WHERE cf.message_id=cm.id AND ({REPORT_SQL} OR cf.rating='no' AND LEFT(LOWER(TRIM(COALESCE(cf.comment,''))),14) <> 'system review:'))")
    if reports_only:
        status_filter = ' AND cf.review_status=%s' if report_status else ''
        conditions.append(f'EXISTS (SELECT 1 FROM chat_feedback cf WHERE cf.message_id=cm.id AND {REPORT_SQL}{status_filter})')
        if report_status:
            params.append(report_status)
    joins = """FROM chat_messages cm
        LEFT JOIN chat_messages um ON um.id=(SELECT MAX(prev.id) FROM chat_messages prev
            WHERE prev.session_id=cm.session_id AND prev.role='user' AND prev.id<cm.id)
        JOIN chat_sessions cs ON cs.id=cm.session_id AND cs.user_id=cm.user_id AND cs.is_active=1"""
    where = ' AND '.join(conditions)
    total = await execute_one(f'SELECT COUNT(*) AS total {joins} WHERE {where}', tuple(params))
    rows = await execute_query(f"""SELECT cm.id,cm.session_id,cm.language,cm.intent,
        cm.confidence_level,cm.confidence_label,cm.confidence_reason,
        {SOURCE_MATCH_LABEL_SQL} AS source_match_label,
        {QUERY_STATUS_SQL} AS query_status,
        {REVIEW_STATUS_SQL} AS admin_review_status,{CLOSED_CORRECTION_SQL} AS correction_closed,cm.created_at,{CATEGORY_SQL} AS category,
        CONCAT('Anonymous User #',cs.user_id) AS anonymous_user,
        (SELECT COUNT(*) FROM chat_feedback f WHERE f.message_id=cm.id AND f.rating='no' AND LEFT(LOWER(TRIM(COALESCE(f.comment,''))),14) <> 'system review:') AS not_helpful_count,
        (SELECT COUNT(*) FROM chat_feedback f WHERE f.message_id=cm.id AND f.rating='yes') AS helpful_count,
        LEFT(cm.content,240) AS answer_preview,LEFT(um.content,240) AS question,
        um.id AS question_id,um.content AS topic_question,um.intent AS question_intent,
        (SELECT cf.comment FROM chat_feedback cf WHERE cf.message_id=cm.id AND {REPORT_SQL} ORDER BY cf.id DESC LIMIT 1) AS report_reason,
        (SELECT cf.review_status FROM chat_feedback cf WHERE cf.message_id=cm.id AND {REPORT_SQL} ORDER BY cf.id DESC LIMIT 1) AS report_status,
        (SELECT cf.created_at FROM chat_feedback cf WHERE cf.message_id=cm.id AND {REPORT_SQL} ORDER BY cf.id DESC LIMIT 1) AS reported_at,
        (SELECT COUNT(*) FROM chat_feedback cf WHERE cf.message_id=cm.id AND {REPORT_SQL}) AS report_count
        {joins} WHERE {where} ORDER BY CASE
            WHEN EXISTS(SELECT 1 FROM chat_feedback cf WHERE cf.message_id=cm.id AND {REPORT_SQL}) THEN 0
            WHEN {REVIEW_STATUS_SQL} IN ('needs_review','incorrect','kb_update') THEN 1
            WHEN EXISTS(SELECT 1 FROM chat_feedback cf WHERE cf.message_id=cm.id AND cf.rating='no'
                AND SUBSTR(LOWER(TRIM(COALESCE(cf.comment,''))),1,14) <> 'system review:') THEN 2
            WHEN ({QUERY_STATUS_SQL})='unresolved' THEN 3
            ELSE 4 END,cm.created_at DESC,cm.id DESC LIMIT %s OFFSET %s""", tuple(params + [limit, offset]))
    rows = await attach_review_topics(rows)
    return {'items': rows, 'total': total['total'], 'offset': offset, 'limit': limit}


@router.get('/review-summary')
async def review_summary(current_user=Depends(require_admin)):
    return await execute_one(f"""SELECT COUNT(*) AS total,
        COALESCE(SUM({NEEDS_REVIEW_SQL}),0) AS needs_review,
        COALESCE(SUM({COMPLETED_REVIEW_SQL}),0) AS completed,
        COALESCE(SUM(({QUERY_STATUS_SQL})='resolved'),0) AS resolved_queries,
        COALESCE(SUM(({QUERY_STATUS_SQL})='unresolved'),0) AS unresolved_queries,
        COALESCE(SUM(EXISTS(SELECT 1 FROM chat_feedback cf WHERE cf.message_id=cm.id AND {REPORT_SQL})),0) AS reported,
        COALESCE(SUM(EXISTS(SELECT 1 FROM chat_feedback f WHERE f.message_id=cm.id AND f.rating='no' AND LEFT(LOWER(TRIM(COALESCE(f.comment,''))),14) <> 'system review:')),0) AS not_helpful,
        COALESCE(SUM(EXISTS(SELECT 1 FROM chat_feedback f WHERE f.message_id=cm.id AND f.rating='yes')),0) AS helpful
        FROM chat_messages cm JOIN chat_sessions cs
          ON cs.id=cm.session_id AND cs.user_id=cm.user_id AND cs.is_active=1
        WHERE cm.role='assistant'""")


@router.get('/answer-reviews/{message_id}/context')
async def answer_review_context(message_id: int, current_user=Depends(require_admin)):
    anchor=await execute_one("""SELECT cm.id,cm.session_id FROM chat_messages cm
        JOIN chat_sessions cs ON cs.id=cm.session_id AND cs.user_id=cm.user_id AND cs.is_active=1
        WHERE cm.id=%s AND cm.role='assistant'""",(message_id,))
    if not anchor:
        raise HTTPException(404,'Answer not found')
    before=await execute_query("""SELECT id,role,content,language,created_at FROM chat_messages
        WHERE session_id=%s AND role IN ('user','assistant') AND id<=%s ORDER BY id DESC LIMIT 5""",(anchor['session_id'],message_id))
    after=await execute_query("""SELECT id,role,content,language,created_at FROM chat_messages
        WHERE session_id=%s AND role IN ('user','assistant') AND id>%s ORDER BY id LIMIT 2""",(anchor['session_id'],message_id))
    return {'session_id':anchor['session_id'],'current_id':message_id,'items':list(reversed(before))+after}


@router.get('/answer-reviews/{message_id}')
async def answer_review_detail(message_id: int, current_user=Depends(require_admin)):
    row = await execute_one(f"""SELECT cm.id,cm.session_id,cm.content AS answer,cm.language,
        cm.intent,cm.sources,cm.confidence_level,cm.confidence_label,cm.confidence_reason,{QUERY_STATUS_SQL} AS query_status,
        cm.admin_query_status,cm.created_at,
        (SELECT category FROM chat_sessions WHERE id=cm.session_id) AS category,
        (SELECT CONCAT('Anonymous User #',user_id) FROM chat_sessions WHERE id=cm.session_id) AS anonymous_user,
        cm.admin_review_status AS answer_review_status,
        {REVIEW_STATUS_SQL} AS admin_review_status,{CLOSED_CORRECTION_SQL} AS correction_closed,cm.admin_review_note,cm.admin_reviewed_by,cm.admin_reviewed_at,
        (SELECT prev.content FROM chat_messages prev WHERE prev.session_id=cm.session_id
          AND prev.role='user' AND prev.id<cm.id ORDER BY prev.id DESC LIMIT 1) AS question
        FROM chat_messages cm JOIN chat_sessions cs
          ON cs.id=cm.session_id AND cs.user_id=cm.user_id AND cs.is_active=1
        WHERE cm.id=%s AND cm.role='assistant'""", (message_id,))
    if not row:
        raise HTTPException(404, 'Answer not found')
    if isinstance(row.get('sources'), str):
        try:
            row['sources'] = json.loads(row['sources'])
        except ValueError:
            row['sources'] = []
    row['feedback'] = await execute_query(f"""SELECT id,rating,comment,review_status,admin_notes,
        reviewed_at,created_at,({REPORT_SQL}) AS reported FROM chat_feedback
        WHERE LEFT(LOWER(TRIM(COALESCE(comment,''))),14) <> 'system review:'
          AND message_id=%s ORDER BY id DESC""", (message_id,))
    # Historical rows have no reliable message-to-provider/timing/trace link.
    row['diagnostic_note'] = 'Provider, model, per-answer timing and retrieval trace were not recorded on these messages. No values are inferred from nearby logs.'
    return row


@router.get('/conversation-review/{session_id}')
async def conversation_review(session_id: str, offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100), current_user=Depends(require_admin)):
    session = await execute_one("SELECT id,title,category,created_at,updated_at FROM chat_sessions WHERE id=%s", (session_id,))
    if not session:
        raise HTTPException(404, 'Conversation not found')
    total = await execute_one("SELECT COUNT(*) AS total FROM chat_messages WHERE session_id=%s AND role IN ('user','assistant')", (session_id,))
    rows = await execute_query("""SELECT id,role,content,language,confidence_level,created_at,sources,
        (SELECT rating FROM chat_feedback WHERE LEFT(LOWER(TRIM(COALESCE(comment,''))),14) <> 'system review:' AND message_id=chat_messages.id ORDER BY id DESC LIMIT 1) AS feedback
        FROM chat_messages WHERE session_id=%s AND role IN ('user','assistant') ORDER BY id LIMIT %s OFFSET %s""", (session_id,limit,offset))
    return {'session': session, 'items': rows, 'total': total['total'], 'offset': offset, 'limit': limit}


@router.put('/answer-reviews/{message_id}')
async def save_answer_review(message_id: int, payload: AnswerReviewUpdate, current_user=Depends(require_admin)):
    if not await execute_one("""SELECT cm.id FROM chat_messages cm JOIN chat_sessions cs
        ON cs.id=cm.session_id AND cs.user_id=cm.user_id AND cs.is_active=1
        WHERE cm.id=%s AND cm.role='assistant'""", (message_id,)):
        raise HTTPException(404, 'Answer not found')
    fields_set = getattr(payload, 'model_fields_set', getattr(payload, '__fields_set__', set()))
    if 'query_status' in fields_set:
        await execute_insert("""UPDATE chat_messages SET admin_review_status=%s,admin_query_status=%s,
            admin_review_note=%s,admin_reviewed_by=%s,admin_reviewed_at=UTC_TIMESTAMP() WHERE id=%s""",
            (payload.status, payload.query_status or 'automatic', payload.note.strip(), current_user['id'], message_id))
    else:
        await execute_insert("""UPDATE chat_messages SET admin_review_status=%s,admin_review_note=%s,
            admin_reviewed_by=%s,admin_reviewed_at=UTC_TIMESTAMP() WHERE id=%s""",
            (payload.status, payload.note.strip(), current_user['id'], message_id))
    await log_admin_action(current_user['id'], 'review_answer', 'chat_message', str(message_id), {'status': payload.status})
    return {'message': 'Answer review saved'}


@router.put('/answer-reviews/{message_id}/query-status')
async def save_answer_query_status(message_id: int, payload: AnswerQueryStatusUpdate,
                                   current_user=Depends(require_admin)):
    if not await execute_one("""SELECT cm.id FROM chat_messages cm JOIN chat_sessions cs
        ON cs.id=cm.session_id AND cs.user_id=cm.user_id AND cs.is_active=1
        WHERE cm.id=%s AND cm.role='assistant'""", (message_id,)):
        raise HTTPException(404, 'Answer not found')
    await execute_insert("""UPDATE chat_messages SET admin_query_status=%s,
        admin_reviewed_by=%s,admin_reviewed_at=UTC_TIMESTAMP() WHERE id=%s""",
        (payload.query_status or 'automatic', current_user['id'], message_id))
    await log_admin_action(current_user['id'], 'update_query_status', 'chat_message', str(message_id),
                           {'query_status': payload.query_status})
    return {'message': 'Query status saved'}


@router.get('/evaluation-summary')
async def evaluation_summary(days: int = Query(30, ge=7, le=30), current_user=Depends(require_admin)):
    import math
    timing_params = (days,)
    totals = await execute_one("""SELECT COUNT(*) AS samples,AVG(response_time_ms) AS average,
        MIN(response_time_ms) AS fastest,MAX(response_time_ms) AS slowest
        FROM query_logs WHERE response_time_ms IS NOT NULL AND response_time_ms>=0
        AND created_at>=DATE_SUB(UTC_TIMESTAMP(),INTERVAL %s DAY)""", timing_params)
    for label, percentile in [('median', .5), ('p90', .9), ('p95', .95)]:
        value = None
        if totals['samples']:
            item = await execute_one('SELECT response_time_ms AS value FROM query_logs WHERE response_time_ms IS NOT NULL AND response_time_ms>=0 AND created_at>=DATE_SUB(UTC_TIMESTAMP(),INTERVAL %s DAY) ORDER BY response_time_ms LIMIT 1 OFFSET %s', (days, max(0, math.ceil(totals['samples']*percentile)-1)))
            value = item['value']
        totals[label] = value
    return {
        'performance': totals,
        'usage': await execute_one("""SELECT
            COALESCE(SUM(created_at>=UTC_DATE()),0) AS today,
            COALESCE(SUM(created_at>=DATE_SUB(UTC_TIMESTAMP(),INTERVAL 7 DAY)),0) AS week,
            COALESCE(SUM(created_at>=DATE_SUB(UTC_TIMESTAMP(),INTERVAL 30 DAY)),0) AS month
            FROM query_logs"""),
        'documents': await execute_one('SELECT COUNT(*) AS total,COALESCE(SUM(chunk_count),0) AS stored_chunks FROM uploaded_documents'),
        'reviews': await execute_query("SELECT admin_review_status AS status,COUNT(*) AS count FROM chat_messages WHERE role='assistant' GROUP BY admin_review_status"),
        'reports': await execute_one(f"""SELECT COUNT(*) AS total,
            COALESCE(SUM(COALESCE(cf.review_status,'pending') IN ('pending','under_review')),0) AS pending
            FROM chat_feedback cf WHERE {REPORT_SQL}"""),
        'note': 'Usage windows use UTC. Response percentiles use nearest rank over recorded query logs. Helpful rate and source match are not measured accuracy; admin assessments are not an unbiased evaluation sample.',
    }


@router.get('/system-health')
async def system_health(current_user=Depends(require_admin)):
    import asyncio
    from config import settings
    from routers.services import chroma_service
    components = []
    try:
        await asyncio.wait_for(execute_one('SELECT 1 AS connected'), timeout=3)
        components.append({'name': 'Database', 'status': 'Connected'})
    except Exception:
        components.append({'name': 'Database', 'status': 'Unavailable'})
    chunks = None
    try:
        if chroma_service._collection is not None:
            chunks = await asyncio.wait_for(asyncio.to_thread(chroma_service._collection.count), timeout=3)
    except Exception:
        pass
    components.extend([
        {'name': 'Chroma', 'status': 'Ready' if chunks is not None else 'Unavailable'},
        {'name': 'Knowledge Base', 'status': 'Ready' if chunks else ('Empty' if chunks == 0 else 'Unavailable'), 'chunks': chunks},
        {'name': 'Embedding service', 'status': 'Loaded (not probed)' if chroma_service._embedder is not None else 'Not loaded'},
    ])
    from routers.services.model_reporting import configured_model_report, lda_status_report, generation_scheduler_report
    providers = configured_model_report()
    return {'checked_at': datetime.utcnow().isoformat()+'Z', 'components': components, 'providers': providers,
            'lda': lda_status_report(),
            'generation_scheduler': generation_scheduler_report(providers),
            'model_pools_enabled': settings.LABORLENS_MODEL_POOLS_ENABLED,
            'note': 'Observations since backend restart, including admin tests; not unique questions or live availability checks. Generation success records a completed draft, not measured legal accuracy. Recovery ranking runs only when conflicting evidence is found. Model calls share any configured local minute limits; these are not verified provider quotas. Estimated tokens include input and the maximum completion allowance, not actual billed usage. Capacity and limits are process-local; other workers and API clients are not counted.'}


@router.get('/report-date-bounds')
async def report_date_bounds(current_user=Depends(require_admin)):
    """Earliest recorded report event, not an arbitrary all-time cutoff."""
    row = await execute_one("""
        SELECT MIN(first_date) AS first_date FROM (
            SELECT MIN(created_at) AS first_date FROM chat_messages
            UNION ALL SELECT MIN(created_at) FROM chat_feedback
            UNION ALL SELECT MIN(created_at) FROM query_logs
            UNION ALL SELECT MIN(created_at) FROM users
        ) report_dates
    """)
    first = (row or {}).get("first_date")
    return {"from": first.date().isoformat() if first else datetime.utcnow().date().isoformat()}


@router.get('/insights')
async def admin_insights(
    days: int = Query(30, ge=7, le=30), category: str = Query('',max_length=100), language: str = '',
    date_from: Optional[date] = None, date_to: Optional[date] = None, current_user=Depends(require_admin),
):
    """Aggregate real stored events; do not join timings to an arbitrary answer."""
    import math
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail='date_from must be on or before date_to')
    end_day = date_to or datetime.utcnow().date()
    start_day = date_from or end_day-timedelta(days=days-1)
    range_start = datetime.combine(start_day, datetime.min.time())
    range_end = datetime.combine(end_day+timedelta(days=1), datetime.min.time())
    where='cm.created_at>=%s AND cm.created_at<%s'
    params=[range_start,range_end]
    if category:
        where+=f' AND ({CATEGORY_SQL})=%s';params.append(category)
    if language:
        where+=' AND cm.language=%s';params.append(normalize_language(language))
    join='FROM chat_messages cm LEFT JOIN chat_sessions cs ON cs.id=cm.session_id'
    usage=await execute_one(f"SELECT COUNT(*) AS questions,COUNT(DISTINCT cm.user_id) AS active_users {join} WHERE cm.role='user' AND {where}",tuple(params))
    volume=await execute_query(f"SELECT DATE(cm.created_at) AS date,COUNT(*) AS count {join} WHERE cm.role='user' AND {where} GROUP BY DATE(cm.created_at) ORDER BY date",tuple(params))
    categories=await execute_query(f"SELECT ({CATEGORY_SQL}) AS label,COUNT(*) AS count {join} WHERE cm.role='user' AND {where} GROUP BY label ORDER BY count DESC,label ASC",tuple(params))
    languages=await execute_query(f"SELECT COALESCE(cm.language,'unknown') AS label,COUNT(*) AS count {join} WHERE cm.role='user' AND {where} GROUP BY cm.language ORDER BY count DESC",tuple(params))
    top_users=await execute_query(f"SELECT COALESCE(CONCAT('Anonymous User #',cm.user_id),'Guest') AS label,COUNT(*) AS count {join} WHERE cm.role='user' AND {where} GROUP BY cm.user_id ORDER BY count DESC LIMIT 10",tuple(params))
    active=await execute_query(f"SELECT DATE(cm.created_at) AS date,COUNT(DISTINCT cm.user_id) AS count {join} WHERE cm.role='user' AND {where} GROUP BY DATE(cm.created_at) ORDER BY date",tuple(params))
    new_user_sql='SELECT DATE(created_at) AS date,COUNT(*) AS count FROM users WHERE created_at>=%s AND created_at<%s'
    new_user_params=[range_start,range_end]
    new_users=await execute_query(new_user_sql+' GROUP BY DATE(created_at) ORDER BY date',tuple(new_user_params))
    fwhere=where.replace('cm.created_at','cf.created_at') + " AND LEFT(LOWER(TRIM(COALESCE(cf.comment,''))),14) <> 'system review:'"
    fjoin='FROM chat_feedback cf JOIN chat_messages cm ON cf.message_id=cm.id LEFT JOIN chat_sessions cs ON cs.id=cm.session_id'
    # Report is an exclusive distribution bucket; it may also have a negative rating.
    kind=f"CASE WHEN {REPORT_SQL} THEN 'reported' WHEN cf.rating='yes' THEN 'helpful' ELSE 'not_helpful' END"
    feedback=await execute_query(f'SELECT {kind} AS label,COUNT(*) AS count {fjoin} WHERE {fwhere} GROUP BY label',tuple(params))
    feedback_trend=await execute_query(f'SELECT DATE(cf.created_at) AS date,{kind} AS label,COUNT(*) AS count {fjoin} WHERE {fwhere} GROUP BY date,label ORDER BY date',tuple(params))
    feedback_categories=await execute_query(f"SELECT ({CATEGORY_SQL}) AS label,COUNT(*) AS count {fjoin} WHERE {fwhere} GROUP BY label ORDER BY count DESC,label ASC",tuple(params))
    feedback_languages=await execute_query(f"SELECT COALESCE(cm.language,'unknown') AS label,COUNT(*) AS count {fjoin} WHERE {fwhere} GROUP BY cm.language ORDER BY count DESC",tuple(params))
    feedback_question_rows=await execute_query(f"""SELECT um.id,um.session_id,um.content,um.intent,COUNT(*) AS count {fjoin}
        LEFT JOIN chat_messages um ON um.id=(SELECT MAX(prev.id) FROM chat_messages prev
            WHERE prev.session_id=cm.session_id AND prev.role='user' AND prev.id<cm.id)
        WHERE {fwhere} GROUP BY um.id,um.session_id,um.content,um.intent""",tuple(params))
    feedback_issues=await execute_query(f"SELECT LEFT(TRIM(cf.comment),180) AS label,COUNT(*) AS count {fjoin} WHERE {fwhere} AND (cf.rating='no' OR {REPORT_SQL}) AND NULLIF(TRIM(cf.comment),'') IS NOT NULL GROUP BY label ORDER BY count DESC LIMIT 5",tuple(params))
    qwhere='ql.created_at>=%s AND ql.created_at<%s';qparams=[range_start,range_end]
    if category:qwhere+=f' AND ({CATEGORY_SQL})=%s';qparams.append(category)
    # Query logs do not store language: omit rather than incorrectly filter by session language.
    timing_rows=[] if language else await execute_query(f'SELECT DATE(ql.created_at) AS date,ql.response_time_ms FROM query_logs ql LEFT JOIN chat_sessions cs ON cs.id=ql.session_id WHERE {qwhere} AND ql.response_time_ms>=0 ORDER BY ql.created_at',tuple(qparams))
    def timing(values):
        values=sorted(float(v) for v in values)
        return {'samples':len(values),'average':sum(values)/len(values) if values else None,**{key:values[max(0,math.ceil(len(values)*p)-1)] if values else None for key,p in [('median',.5),('p90',.9),('p95',.95)]}}
    groups={}
    for row in timing_rows:groups.setdefault(str(row['date']),[]).append(row['response_time_ms'])
    question_rows=await execute_query(f"SELECT cm.id,cm.session_id,cm.content,cm.intent,1 AS count {join} WHERE cm.role='user' AND {where} ORDER BY cm.id",tuple(params))
    # Context may precede the selected period. Only selected rows contribute
    # counts, and every message keeps its own session and chronological topic.
    selected_topic_rows=[*question_rows, *feedback_question_rows]
    topic_sessions=sorted({r['session_id'] for r in selected_topic_rows if r.get('session_id')})
    topic_history=[]
    for start in range(0,len(topic_sessions),100):
        batch=topic_sessions[start:start+100]
        placeholders=','.join(['%s']*len(batch))
        topic_history.extend(await execute_query(f"""SELECT id,session_id,role,content,intent
            FROM chat_messages WHERE session_id IN ({placeholders}) AND role IN ('user','assistant')
            AND id<=%s ORDER BY session_id,id""",tuple(batch+[max(r.get('id') or 0 for r in selected_topic_rows)])))
    topics=rank_question_topics(question_rows,topic_history)
    feedback_topics=rank_question_topics(feedback_question_rows,topic_history)
    return {'usage':usage,'volume':volume,'categories':categories,'topics':topics,'languages':languages,'top_users':top_users,
            'active_users':active,'new_users':new_users,'feedback':feedback,'feedback_trend':feedback_trend,
            'feedback_categories':feedback_categories,'feedback_topics':feedback_topics,'feedback_languages':feedback_languages,'feedback_issues':feedback_issues,
            'timing':timing([r['response_time_ms'] for r in timing_rows]),'timing_trend':[{'date':key,**timing(values)} for key,values in groups.items()],
            'date_range':{'from':start_day.isoformat(),'to':end_day.isoformat(),'is_custom':bool(date_from or date_to)}}


@router.get('/feedback-page')
async def feedback_page(q: str = Query('',max_length=200), category: str = Query('',max_length=100), language: str = '', kind: Literal['','helpful','not_helpful','reported'] = '', days: int = Query(30, ge=7, le=30), date_from: Optional[date] = None, date_to: Optional[date] = None, offset: int = Query(0,ge=0), limit: int = Query(25,ge=1,le=100), current_user=Depends(require_admin)):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail='date_from must be on or before date_to')
    conditions=["LEFT(LOWER(TRIM(COALESCE(cf.comment,''))),14) <> 'system review:'"]
    if date_from or date_to:
        start_day=date_from or (date_to-timedelta(days=days-1))
        end_day=date_to or date.today()
        conditions.extend(['cf.created_at>=%s','cf.created_at<%s'])
        params=[datetime.combine(start_day, datetime.min.time()),datetime.combine(end_day+timedelta(days=1),datetime.min.time())]
    else:
        conditions.append('cf.created_at>=%s')
        params=[datetime.utcnow()-timedelta(days=days)]
    if q:conditions.append('(cm.content LIKE %s OR cf.comment LIKE %s OR um.content LIKE %s)');params.extend([f'%{q}%']*3)
    if category:conditions.append(f'({CATEGORY_SQL})=%s');params.append(category)
    if language:conditions.append('cm.language=%s');params.append(normalize_language(language))
    if kind=='reported':conditions.append(REPORT_SQL)
    elif kind:conditions.append(f'NOT ({REPORT_SQL}) AND cf.rating=%s');params.append('yes' if kind=='helpful' else 'no')
    join="""FROM chat_feedback cf JOIN chat_messages cm ON cm.id=cf.message_id
        LEFT JOIN chat_sessions cs ON cs.id=cm.session_id LEFT JOIN chat_messages um ON um.id=(SELECT MAX(p.id) FROM chat_messages p WHERE p.session_id=cm.session_id AND p.id<cm.id AND p.role='user')"""
    where=' AND '.join(conditions)
    total=await execute_one(f'SELECT COUNT(*) AS total {join} WHERE {where}',tuple(params))
    rows=await execute_query(f"""SELECT cf.id,cf.message_id,cf.rating,cf.comment,cf.created_at,cf.review_status,cm.language,cs.category,
        CONCAT('Anonymous User #',cf.user_id) AS anonymous_user,um.content AS question,
        ({REPORT_SQL}) AS reported {join} WHERE {where} ORDER BY cf.id DESC LIMIT %s OFFSET %s""",tuple(params+[limit,offset]))
    return {'items':rows,'total':total['total']}


class GlossaryCreate(BaseModel):
    term_en: str
    term_hil: str
    notes: str = ""
    sample_usage: str = ""
    is_active: bool = True


class GlossaryUpdate(BaseModel):
    term_en: str
    term_hil: str
    notes: str = ""
    sample_usage: str = ""
    is_active: bool = True


def normalize_language(language: str) -> str:
    value = (language or "en").strip().lower()
    if value not in FAQ_LANGUAGES:
        raise HTTPException(status_code=400, detail="Language must be en, fil, or hil")
    return value


@router.get("/ingestion-status")
async def ingestion_status(current_user=Depends(require_admin)):
    """Let the admin UI poll ingestion progress without holding an upload open."""

    return get_ingestion_status()


async def log_admin_action(
    admin_user_id: int,
    action: str,
    target_type: str,
    target_id: Optional[str] = None,
    details: Optional[dict] = None,
) -> None:
    await execute_insert(
        """
        INSERT INTO admin_audit_logs (admin_user_id, action, target_type, target_id, details)
        VALUES (%s,%s,%s,%s,%s)
        """,
        (
            admin_user_id,
            action[:120],
            (target_type or "")[:80],
            str(target_id or "")[:120] or None,
            json.dumps(details or {}),
        ),
    )


@router.get("/dashboard")
async def dashboard(current_user=Depends(require_admin)):
    feedback_counts = await execute_query(
        "SELECT rating, COUNT(*) as count FROM chat_feedback WHERE LEFT(LOWER(TRIM(COALESCE(comment,''))),14) <> 'system review:' GROUP BY rating"
    )
    resolution_summary = await execute_one(
        """
        SELECT
            COUNT(*) AS total_queries,
            COALESCE(SUM(CASE WHEN resolved=1 THEN 1 ELSE 0 END), 0) AS resolved_queries,
            COALESCE(SUM(CASE WHEN resolved=0 THEN 1 ELSE 0 END), 0) AS unresolved_queries,
            COALESCE(ROUND(AVG(response_time_ms)), 0) AS avg_response_ms,
            COALESCE(SUM(CASE WHEN response_time_ms >= 2000 THEN 1 ELSE 0 END), 0) AS slow_queries
        FROM query_logs
        """
    )
    feedback_summary = await execute_one(
        """
        SELECT
            COUNT(*) AS total,
            COALESCE(SUM(CASE WHEN rating='yes' THEN 1 ELSE 0 END), 0) AS helpful,
            COALESCE(SUM(CASE WHEN rating='no' THEN 1 ELSE 0 END), 0) AS not_helpful
        FROM chat_feedback
        WHERE LEFT(LOWER(TRIM(COALESCE(comment,''))),14) <> 'system review:'
        """
    )
    return {
        "totals": {
            "users": (await execute_one("SELECT COUNT(*) as c FROM users"))["c"],
            "sessions": (await execute_one("SELECT COUNT(*) as c FROM chat_sessions"))["c"],
            "messages": (await execute_one("SELECT COUNT(*) as c FROM chat_messages WHERE role='user'"))["c"],
            "docs": (await execute_one("SELECT COUNT(*) as c FROM uploaded_documents WHERE status='indexed'"))["c"],
            "feedback": (await execute_one("SELECT COUNT(*) as c FROM chat_feedback WHERE LEFT(LOWER(TRIM(COALESCE(comment,''))),14) <> 'system review:'"))["c"],
            "glossary_terms": (await execute_one("SELECT COUNT(*) as c FROM hiligaynon_glossary WHERE is_active=1"))["c"],
            "review_queue_pending": (
                await execute_one(
                    "SELECT COUNT(*) as c FROM chat_feedback WHERE rating='no' AND COALESCE(review_status,'pending')='pending'"
                )
            )["c"],
        },
        "feedback_counts": feedback_counts,
        "feedback_summary": feedback_summary,
        "resolution_summary": resolution_summary,
        "intent_distribution": await execute_query(
            "SELECT intent, COUNT(*) as count FROM query_logs WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY) GROUP BY intent ORDER BY count DESC LIMIT 10"
        ),
        "daily_volume": await execute_query(
            "SELECT DATE(created_at) as date, COUNT(*) as count FROM query_logs WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY) GROUP BY DATE(created_at) ORDER BY date"
        ),
        "unresolved": (await execute_one("SELECT COUNT(*) as c FROM query_logs WHERE resolved=0"))["c"],
        "language_distribution": await execute_query(
            "SELECT language, COUNT(*) as count FROM chat_messages WHERE role='user' GROUP BY language"
        ),
        "source_match_distribution": await execute_query(
            """
            SELECT COALESCE(confidence_level, 'Unknown') AS source_match, COUNT(*) AS count
            FROM chat_messages
            WHERE role='assistant'
            GROUP BY COALESCE(confidence_level, 'Unknown')
            ORDER BY count DESC
            """
        ),
        "top_questions": await execute_query(
            """
            SELECT
                MIN(query_text) AS question,
                COALESCE(intent, 'general') AS intent,
                COUNT(*) AS count,
                MAX(created_at) AS last_asked
            FROM query_logs
            WHERE query_text IS NOT NULL AND TRIM(query_text) <> ''
            GROUP BY LOWER(TRIM(query_text)), COALESCE(intent, 'general')
            ORDER BY count DESC, last_asked DESC
            LIMIT 10
            """
        ),
        "disliked_topics": await execute_query(
            """
            SELECT
                COALESCE(cm.intent, 'general') AS intent,
                COUNT(*) AS count
            FROM chat_feedback cf
            JOIN chat_messages cm ON cf.message_id=cm.id
            WHERE cf.rating='no' AND LEFT(LOWER(TRIM(COALESCE(cf.comment,''))),14) <> 'system review:'
            GROUP BY COALESCE(cm.intent, 'general')
            ORDER BY count DESC
            LIMIT 10
            """
        ),
        "version_distribution": await execute_query(
            """
            SELECT COALESCE(version_status, 'active') AS version_status, COUNT(*) AS count
            FROM uploaded_documents
            GROUP BY COALESCE(version_status, 'active')
            ORDER BY version_status
            """
        ),
    }


@router.get("/users")
async def list_users(
    q: str = Query('', max_length=200),
    role: str = Query('', max_length=40),
    status: Literal['', 'active', 'inactive'] = '',
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    current_user=Depends(require_admin),
):
    # Keep administrator accounts out of the ordinary-user management view.
    where = "LOWER(TRIM(COALESCE(role,''))) NOT IN ('admin','administrator','super admin','super_admin')"
    params = []
    if q:
        where += " AND (CAST(id AS CHAR) LIKE %s OR CONCAT('Anonymous User #',id) LIKE %s)"
        params.extend([f'%{q}%', f'%{q}%'])
    if role:
        where += " AND LOWER(TRIM(COALESCE(role,'user')))=%s"
        params.append(role.strip().lower())
    if status:
        where += " AND is_active=%s"
        params.append(status == 'active')

    total = await execute_one(f"SELECT COUNT(*) AS total FROM users WHERE {where}", tuple(params))
    rows = await execute_query(
        f"""
        SELECT id, CONCAT('Anonymous User #', id) AS anonymous_user,
            role, language_preference, created_at, is_active
        FROM users WHERE {where}
        ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s
        """,
        tuple(params + [limit, offset]),
    )
    stats = await execute_one("""SELECT COUNT(*) AS total,
        COALESCE(SUM(is_active=1),0) AS active,
        COALESCE(SUM(is_active=0),0) AS inactive
        FROM users
        WHERE LOWER(TRIM(COALESCE(role,''))) NOT IN ('admin','administrator','super admin','super_admin')""")
    roles = await execute_query("""SELECT DISTINCT LOWER(TRIM(COALESCE(role,'user'))) AS role
        FROM users WHERE LOWER(TRIM(COALESCE(role,''))) NOT IN ('admin','administrator','super admin','super_admin')
        ORDER BY role""")
    return {'items': rows, 'total': total['total'], 'stats': stats,
            'roles': [item['role'] for item in roles if item.get('role')]}


@router.put("/users/{user_id}/role")
async def update_user_role(user_id: int, role: str, current_user=Depends(require_admin)):
    if role not in {"user", "admin"}:
        raise HTTPException(status_code=400, detail="Role must be user or admin")
    try:
        user = await execute_admin_role_change(user_id, role, current_user['id'])
    except AdminRoleChangeConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    await log_admin_action(
        current_user["id"],
        "update_user_role",
        "user",
        str(user_id),
        {"from": user["role"], "to": role},
    )
    return {"message": "Role updated", "user_id": user_id, "role": role}


@router.get("/faqs")
async def list_faqs(current_user=Depends(require_admin)):
    from routers.services.faq_service import list_faq_records
    return await list_faq_records()


@router.get("/faqs/export")
async def export_faqs(current_user=Depends(require_admin)):
    return await execute_query(
        "SELECT id,question,answer,category,language,view_count,created_at,updated_at,publication_status,source_document_ids,source_versions,source_citation FROM faqs ORDER BY id DESC"
    )


@router.post("/faqs")
async def create_faq(faq: FAQCreate, current_user=Depends(require_admin)):
    if not faq.question.strip() or not faq.answer.strip():
        raise HTTPException(422, 'Question and answer are required')
    language = normalize_language(faq.language)
    from routers.services.faq_service import prepare_faq_sources
    source_ids, versions = await prepare_faq_sources(faq)
    fid = await execute_insert(
        "INSERT INTO faqs (question,answer,category,language,publication_status,source_document_ids,source_versions,source_citation) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        (faq.question.strip(), faq.answer.strip(), (faq.category or "").strip() or None, language, faq.publication_status, source_ids, versions, faq.source_citation.strip()),
    )
    await log_admin_action(
        current_user["id"],
        "create_faq",
        "faq",
        str(fid),
        {"question": faq.question[:180], "category": faq.category, "language": language, "publication_status": faq.publication_status, "source_document_ids": faq.source_document_ids},
    )
    return {"message": "Created", "id": fid}


@router.put("/faqs/{faq_id}")
async def update_faq(faq_id: int, faq: FAQUpdate, current_user=Depends(require_admin)):
    if not faq.question.strip() or not faq.answer.strip():
        raise HTTPException(422, 'Question and answer are required')
    language = normalize_language(faq.language)
    existing = await execute_one("SELECT id FROM faqs WHERE id=%s", (faq_id,))
    if not existing:
        raise HTTPException(status_code=404, detail="FAQ not found")
    from routers.services.faq_service import prepare_faq_sources
    source_ids, versions = await prepare_faq_sources(faq)
    await execute_insert(
        """
        UPDATE faqs
        SET question=%s, answer=%s, category=%s, language=%s,
            publication_status=%s,source_document_ids=%s,source_versions=%s,source_citation=%s
        WHERE id=%s
        """,
        (faq.question.strip(), faq.answer.strip(), (faq.category or "").strip() or None, language, faq.publication_status, source_ids, versions, faq.source_citation.strip(), faq_id),
    )
    await log_admin_action(
        current_user["id"],
        "update_faq",
        "faq",
        str(faq_id),
        {"question": faq.question[:180], "category": faq.category, "language": language, "publication_status": faq.publication_status, "source_document_ids": faq.source_document_ids},
    )
    return {"message": "Updated", "id": faq_id}


@router.post("/faqs/import")
async def import_faqs(payload: FAQImportRequest, current_user=Depends(require_admin)):
    created = 0
    for item in payload.items:
        question = item.question.strip()
        answer = item.answer.strip()
        if not question or not answer:
            continue
        language = normalize_language(item.language)
        await execute_insert(
            "INSERT INTO faqs (question,answer,category,language) VALUES (%s,%s,%s,%s)",
            (question, answer, (item.category or "").strip() or None, language),
        )
        created += 1
    await log_admin_action(
        current_user["id"],
        "import_faqs",
        "faq",
        "batch",
        {"created": created},
    )
    return {"message": "Imported", "created": created}


@router.delete("/faqs/{faq_id}")
async def delete_faq(faq_id: int, current_user=Depends(require_admin)):
    faq = await execute_one("SELECT id,question FROM faqs WHERE id=%s", (faq_id,))
    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")
    await execute_insert("DELETE FROM faqs WHERE id=%s", (faq_id,))
    await log_admin_action(
        current_user["id"],
        "delete_faq",
        "faq",
        str(faq_id),
        {"question": faq["question"][:180]},
    )
    return {"message": "Deleted"}


@router.get("/unresolved-queries")
async def unresolved(current_user=Depends(require_admin)):
    return await execute_query(
        """
        SELECT
            ql.*,
            CASE
                WHEN ql.user_id IS NULL THEN 'Guest'
                ELSE CONCAT('Anonymous User #', ql.user_id)
            END AS anonymous_user
        FROM query_logs ql
        WHERE ql.resolved=0
        ORDER BY ql.created_at DESC
        LIMIT 50
        """
    )


@router.get("/query-performance")
async def query_performance(current_user=Depends(require_admin)):
    return await execute_query(
        """
        SELECT
            ql.id,
            ql.user_id,
            ql.session_id,
            ql.query_text,
            ql.intent,
            ql.response_time_ms,
            ql.resolved,
            ql.created_at,
            CASE
                WHEN ql.user_id IS NULL THEN 'Guest'
                ELSE CONCAT('Anonymous User #', ql.user_id)
            END AS anonymous_user,
            CASE
                WHEN ql.resolved = 0 THEN 'not_found'
                WHEN ql.intent = 'needs_clarification' THEN 'clarified'
                WHEN ql.intent IN ('greeting','identity','gratitude','farewell') THEN 'handled'
                ELSE 'kb_found'
            END AS result_status
        FROM query_logs ql
        ORDER BY ql.created_at DESC
        LIMIT 100
        """
    )


@router.get("/chat-logs")
async def chat_logs(current_user=Depends(require_admin)):
    return await execute_query(
        """
        SELECT
            cm.id,
            cm.session_id,
            CASE
                WHEN cm.user_id IS NULL THEN 'Guest'
                ELSE CONCAT('Anonymous User #', cm.user_id)
            END AS anonymous_user,
            cm.role,
            cm.content,
            cm.language,
            cm.intent,
            cm.confidence_level,
            cm.confidence_reason,
            cf.rating as feedback_rating,
            cf.comment as feedback_comment,
            cm.created_at
        FROM chat_messages cm
        LEFT JOIN chat_feedback cf ON cf.message_id=cm.id AND LEFT(LOWER(TRIM(COALESCE(cf.comment,''))),14) <> 'system review:'
        ORDER BY cm.created_at DESC
        LIMIT 1000
        """
    )


@router.get("/feedback")
async def feedback_logs(current_user=Depends(require_admin)):
    return await execute_query(
        """
        SELECT
            cf.id,
            cf.rating,
            cf.comment,
            cf.review_status,
            cf.admin_notes,
            cf.reviewed_at,
            cf.created_at,
            cm.content as assistant_answer,
            cm.intent,
            cm.confidence_level,
            cm.confidence_reason,
            CASE
                WHEN cf.user_id IS NULL THEN 'Guest'
                ELSE CONCAT('Anonymous User #', cf.user_id)
            END AS anonymous_user,
            reviewer.username as reviewer_name
        FROM chat_feedback cf
        JOIN chat_messages cm ON cf.message_id=cm.id
        LEFT JOIN users reviewer ON cf.reviewed_by=reviewer.id
        WHERE LEFT(LOWER(TRIM(COALESCE(cf.comment,''))),14) <> 'system review:'
        ORDER BY cf.created_at DESC
        LIMIT 200
        """
    )


@router.get("/review-queue")
async def review_queue(current_user=Depends(require_admin)):
    return await execute_query(
        """
        SELECT
            cf.id,
            cf.rating,
            cf.comment,
            cf.review_status,
            cf.admin_notes,
            cf.reviewed_at,
            cf.created_at,
            cm.id as assistant_message_id,
            cm.content as assistant_answer,
            cm.intent,
            cm.confidence_level,
            cm.confidence_reason,
            um.content as user_question,
            CASE
                WHEN cf.user_id IS NULL THEN 'Guest'
                ELSE CONCAT('Anonymous User #', cf.user_id)
            END AS anonymous_user,
            reviewer.username as reviewer_name
        FROM chat_feedback cf
        JOIN chat_messages cm ON cf.message_id=cm.id
        LEFT JOIN chat_messages um
            ON um.id = (
                SELECT MAX(prev.id)
                FROM chat_messages prev
                WHERE prev.session_id=cm.session_id
                  AND prev.role='user'
                  AND prev.id < cm.id
            )
        LEFT JOIN users reviewer ON cf.reviewed_by=reviewer.id
        WHERE cf.rating='no' OR COALESCE(cf.review_status,'pending')='pending'
        ORDER BY
            CASE COALESCE(cf.review_status,'pending')
                WHEN 'pending' THEN 0
                WHEN 'reviewed' THEN 1
                ELSE 2
            END,
            cf.created_at DESC
        LIMIT 200
        """
    )


@router.put("/review-queue/{feedback_id}")
async def update_review_queue_item(
    feedback_id: int,
    payload: ReviewQueueUpdate,
    current_user=Depends(require_admin),
):
    if payload.review_status not in REVIEW_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid review status")
    existing = await execute_one("SELECT id FROM chat_feedback WHERE id=%s", (feedback_id,))
    if not existing:
        raise HTTPException(status_code=404, detail="Feedback item not found")
    reviewed_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    await execute_insert(
        """
        UPDATE chat_feedback
        SET review_status=%s, admin_notes=%s, reviewed_by=%s, reviewed_at=%s
        WHERE id=%s
        """,
        (payload.review_status, payload.admin_notes.strip()[:2000], current_user["id"], reviewed_at, feedback_id),
    )
    await log_admin_action(
        current_user["id"],
        "review_feedback",
        "feedback",
        str(feedback_id),
        {"review_status": payload.review_status, "notes": payload.admin_notes[:200]},
    )
    return {"message": "Review updated", "id": feedback_id}


@router.get("/glossary")
async def list_glossary(current_user=Depends(require_admin)):
    return await execute_query(
        """
        SELECT id,term_en,term_hil,notes,sample_usage,is_active,created_at,updated_at
        FROM hiligaynon_glossary
        ORDER BY updated_at DESC, id DESC
        """
    )


@router.post("/glossary")
async def create_glossary(item: GlossaryCreate, current_user=Depends(require_admin)):
    term_en = item.term_en.strip()
    term_hil = item.term_hil.strip()
    if not term_en or not term_hil:
        raise HTTPException(status_code=400, detail="English and Hiligaynon terms are required")
    gid = await execute_insert(
        """
        INSERT INTO hiligaynon_glossary (term_en,term_hil,notes,sample_usage,is_active)
        VALUES (%s,%s,%s,%s,%s)
        """,
        (term_en, term_hil, item.notes.strip(), item.sample_usage.strip(), bool(item.is_active)),
    )
    await log_admin_action(
        current_user["id"],
        "create_glossary_term",
        "glossary",
        str(gid),
        {"term_en": term_en, "term_hil": term_hil, "is_active": item.is_active},
    )
    return {"message": "Created", "id": gid}


@router.put("/glossary/{glossary_id}")
async def update_glossary(glossary_id: int, item: GlossaryUpdate, current_user=Depends(require_admin)):
    existing = await execute_one("SELECT id FROM hiligaynon_glossary WHERE id=%s", (glossary_id,))
    if not existing:
        raise HTTPException(status_code=404, detail="Glossary term not found")
    term_en = item.term_en.strip()
    term_hil = item.term_hil.strip()
    if not term_en or not term_hil:
        raise HTTPException(status_code=400, detail="English and Hiligaynon terms are required")
    await execute_insert(
        """
        UPDATE hiligaynon_glossary
        SET term_en=%s, term_hil=%s, notes=%s, sample_usage=%s, is_active=%s
        WHERE id=%s
        """,
        (term_en, term_hil, item.notes.strip(), item.sample_usage.strip(), bool(item.is_active), glossary_id),
    )
    await log_admin_action(
        current_user["id"],
        "update_glossary_term",
        "glossary",
        str(glossary_id),
        {"term_en": term_en, "term_hil": term_hil, "is_active": item.is_active},
    )
    return {"message": "Updated", "id": glossary_id}


@router.delete("/glossary/{glossary_id}")
async def delete_glossary(glossary_id: int, current_user=Depends(require_admin)):
    existing = await execute_one(
        "SELECT id,term_en,term_hil FROM hiligaynon_glossary WHERE id=%s",
        (glossary_id,),
    )
    if not existing:
        raise HTTPException(status_code=404, detail="Glossary term not found")
    await execute_insert("DELETE FROM hiligaynon_glossary WHERE id=%s", (glossary_id,))
    await log_admin_action(
        current_user["id"],
        "delete_glossary_term",
        "glossary",
        str(glossary_id),
        {"term_en": existing["term_en"], "term_hil": existing["term_hil"]},
    )
    return {"message": "Deleted"}


@router.get("/audit-logs")
async def audit_logs(current_user=Depends(require_admin)):
    rows = await execute_query(
        """
        SELECT
            al.id,
            al.action,
            al.target_type,
            al.target_id,
            al.details,
            al.created_at,
            u.username as admin_username
        FROM admin_audit_logs al
        LEFT JOIN users u ON al.admin_user_id=u.id
        ORDER BY al.created_at DESC
        LIMIT 200
        """
    )
    for row in rows:
        row["details"] = redact_audit_details(row.get("details"))
    return rows


@router.get('/audit-page')
async def audit_page(q: str = Query('',max_length=200), action: str = Query('',max_length=100), module: str = Query('',max_length=100), user: str = Query('',max_length=100), days: int = Query(30, ge=7, le=30), offset: int = Query(0,ge=0), limit: int = Query(25,ge=1,le=100), current_user=Depends(require_admin)):
    conditions=['al.created_at>=%s'];params=[datetime.utcnow()-timedelta(days=days)]
    if q:conditions.append('(al.action LIKE %s OR al.details LIKE %s)');params.extend([f'%{q}%']*2)
    if action:conditions.append('al.action=%s');params.append(action)
    if module:conditions.append('al.target_type=%s');params.append(module)
    if user:conditions.append('u.username LIKE %s');params.append(f'%{user}%')
    join='FROM admin_audit_logs al LEFT JOIN users u ON al.admin_user_id=u.id';where=' AND '.join(conditions)
    total=await execute_one(f'SELECT COUNT(*) AS total {join} WHERE {where}',tuple(params))
    groups=await execute_query(f'SELECT al.target_type AS label,COUNT(*) AS count {join} WHERE {where} GROUP BY al.target_type ORDER BY count DESC,label ASC',tuple(params))
    actions=await execute_query(f'SELECT al.action AS label,COUNT(*) AS count {join} WHERE {where} GROUP BY al.action ORDER BY count DESC,label ASC',tuple(params))
    rows=await execute_query(f'SELECT al.id,al.action,al.target_type,al.target_id,al.details,al.created_at,u.username AS admin_username {join} WHERE {where} ORDER BY al.id DESC LIMIT %s OFFSET %s',tuple(params+[limit,offset]))
    for row in rows:row['details']=redact_audit_details(row.get('details'))
    return {'items':rows,'total':total['total'],'groups':groups,'actions':actions}


@router.get("/system-backup")
async def system_backup(current_user=Depends(require_admin)):
    data = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "documents": await execute_query(
            """
            SELECT id,original_filename,file_type,file_size,category,secondary_categories,document_title,
                   document_type,legal_role,implements_json,topics_json,sub_intents_json,
                   source_name,source_url,law_number,article_section,issuing_agency,
                   date_issued,effective_date,date_downloaded,document_year,superseding_document,
                   law_version,version_status,is_active,status,
                   processing_stage,chunk_count,error_message,last_indexed_at,created_at
            FROM uploaded_documents
            ORDER BY created_at DESC
            """
        ),
        "faqs": await execute_query(
            "SELECT id,question,answer,category,language,view_count,created_at,updated_at,publication_status,source_document_ids,source_versions,source_citation FROM faqs ORDER BY id DESC"
        ),
        "feedback": await execute_query(
            """
            SELECT
                cf.id,cf.rating,cf.comment,cf.review_status,cf.admin_notes,cf.reviewed_at,cf.created_at,
                cm.intent,cm.confidence_level,cm.confidence_reason,cm.content as assistant_answer,
                CASE
                    WHEN cf.user_id IS NULL THEN 'Guest'
                    ELSE CONCAT('Anonymous User #', cf.user_id)
                END AS anonymous_user
            FROM chat_feedback cf
            JOIN chat_messages cm ON cf.message_id=cm.id
            ORDER BY cf.created_at DESC
            LIMIT 1000
            """
        ),
        "query_logs": await execute_query(
            """
            SELECT
                ql.id,ql.session_id,ql.query_text,ql.intent,ql.response_time_ms,ql.resolved,ql.created_at,
                CASE
                    WHEN ql.user_id IS NULL THEN 'Guest'
                    ELSE CONCAT('Anonymous User #', ql.user_id)
                END AS anonymous_user
            FROM query_logs ql
            ORDER BY ql.created_at DESC
            LIMIT 2000
            """
        ),
        "audit_logs": await execute_query(
            """
            SELECT al.id,al.action,al.target_type,al.target_id,al.details,al.created_at,u.username as admin_username
            FROM admin_audit_logs al
            LEFT JOIN users u ON al.admin_user_id=u.id
            ORDER BY al.created_at DESC
            LIMIT 1000
            """
        ),
    }
    for row in data["audit_logs"]:
        row["details"] = redact_audit_details(row.get("details"))
    await log_admin_action(
        current_user["id"],
        "export_system_backup",
        "backup",
        "system",
        {"generated_at": data["generated_at"]},
    )
    return data


@router.put("/unresolved-queries/{log_id}/resolve")
async def resolve(log_id: int, current_user=Depends(require_admin)):
    query = await execute_one("SELECT id,query_text,intent FROM query_logs WHERE id=%s", (log_id,))
    if not query:
        raise HTTPException(status_code=404, detail="Query log not found")
    await execute_insert("UPDATE query_logs SET resolved=1 WHERE id=%s", (log_id,))
    await log_admin_action(
        current_user["id"],
        "resolve_unresolved_query",
        "query_log",
        str(log_id),
        {"query_text": (query["query_text"] or "")[:180], "intent": query["intent"]},
    )
    return {"message": "Resolved"}
