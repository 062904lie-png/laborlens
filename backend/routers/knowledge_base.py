"""LaborLens – Knowledge Base Router"""
import asyncio
import contextlib
import hashlib
import json
import logging
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import (
    APIRouter,
    Depends,
    UploadFile,
    File,
    Form,
    HTTPException,
    BackgroundTasks,
    status,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from routers.auth import get_current_user, require_admin
from routers.services.database import execute_insert, execute_query, execute_one, get_pool
from routers.services.ingestion_service import (
    extract_original_document_text,
    save_upload,
    ingest_document,
)
from routers.services.kb_categories import (
    KB_CATEGORIES,
    DEFAULT_KB_CATEGORY,
    canonical_kb_category,
)
from routers.services.chroma_service import (
    count_document_chunks_by_db_id,
    delete_document_chunks_by_ids,
    delete_document_chunks_by_db_id,
    get_all_document_chunks_by_db_id,
    get_collection_stats,
    get_document_chunk_ids_by_db_id,
    get_document_chunks_by_db_id,
    get_document_chunk_counts_by_db_id,
    reset_collection,
    invalidate_retrieval_caches,
    update_document_chunk_metadata_by_db_id,
)
from routers.services.rag_policy import infer_document_year
from routers.services.document_categories import (
    category_metadata,
    json_list,
    normalize_document_categories,
    parse_string_list,
    suggest_document_categories,
)
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)
STALE_PROCESSING_MINUTES = 60
_rebuild_index_lock = asyncio.Lock()
VERSION_STATUSES = {
    "active",
    "amended",
    "repealed",
    "archived",
    "inactive",
    "replaced",
    "outdated",
    "current",
    "superseded",
    "historical",
    "under_review",
}


async def read_upload_with_limit(file: UploadFile, detail: str) -> bytes:
    """Read an upload in bounded chunks and stop once it exceeds the configured cap."""
    maximum_bytes = max(0, int(settings.MAX_FILE_SIZE_MB)) * 1048576
    content = bytearray()
    while True:
        remaining = maximum_bytes - len(content)
        chunk = await file.read(min(1024 * 1024, remaining + 1))
        if not chunk:
            return bytes(content)
        content.extend(chunk)
        if len(content) > maximum_bytes:
            raise HTTPException(status_code=400, detail=detail)

ALLOWED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpeg",
    ".tif": "tif",
    ".tiff": "tiff",
    ".bmp": "bmp",
    ".webp": "webp",
}

MIME_TO_EXTENSION = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tif",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
}

FILE_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
    "webp": "image/webp",
}

REPLACEMENT_ACTIVE_STATUSES = {
    "queued",
    "validating",
    "extracting",
    "chunking",
    "embedding",
    "activating",
    "cleaning",
}


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def safe_upload_basename(filename: str) -> str:
    safe_name = Path(filename or "document").name
    safe_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", safe_name).strip(" ._")
    return (safe_name or "document")[:180]


def save_staged_replacement_file(content: bytes, original_filename: str) -> Path:
    staging_dir = Path(settings.UPLOAD_DIR) / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_name = f"replace_{uuid.uuid4().hex}_{safe_upload_basename(original_filename)}"
    staged_path = staging_dir / staged_name
    staged_path.write_bytes(content)
    return staged_path


def move_staged_replacement_file(staged_path: Path, original_filename: str) -> Path:
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    final_name = f"{uuid.uuid4().hex}_{safe_upload_basename(original_filename)}"
    final_path = upload_dir / final_name
    shutil.move(str(staged_path), str(final_path))
    return final_path


def temporary_replacement_doc_db_id(doc_id: int) -> int:
    # Positive, very large temporary id so staged chunks are invisible to normal
    # document queries until they are explicitly activated.
    return 800_000_000 + (int(time.time() * 1000) % 100_000_000) + (doc_id % 1000)


class _IngestionWarningCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        if message:
            self.messages.append(message)


def extract_text_with_parser_warnings(filepath: str) -> tuple[dict, list[str]]:
    capture = _IngestionWarningCapture()
    watched_loggers = [
        logging.getLogger("pypdf"),
        logging.getLogger("PyPDF2"),
        logging.getLogger("routers.services.ingestion_service"),
    ]
    for watched in watched_loggers:
        watched.addHandler(capture)
    try:
        result = extract_original_document_text(filepath)
    finally:
        for watched in watched_loggers:
            with contextlib.suppress(Exception):
                watched.removeHandler(capture)
    seen = set()
    warnings = []
    for message in capture.messages:
        compact = str(message).strip()
        if compact and compact not in seen:
            seen.add(compact)
            warnings.append(compact[:500])
    return result, warnings


async def update_replacement_job(
    job_id: str,
    *,
    status: str | None = None,
    progress: int | None = None,
    message: str | None = None,
    warning_message: str | None = None,
    error_message: str | None = None,
    new_filename: str | None = None,
    old_chunk_count: int | None = None,
    new_chunk_count: int | None = None,
    completed: bool = False,
) -> None:
    updates = []
    params = []
    if status is not None:
        updates.append("status=%s")
        params.append(status[:40])
    if progress is not None:
        updates.append("progress=%s")
        params.append(max(0, min(int(progress), 100)))
    if message is not None:
        updates.append("message=%s")
        params.append(message[:500])
    if warning_message is not None:
        updates.append("warning_message=%s")
        params.append(warning_message[:2000])
    if error_message is not None:
        updates.append("error_message=%s")
        params.append(error_message[:2000])
    if new_filename is not None:
        updates.append("new_filename=%s")
        params.append(Path(new_filename).name[:255])
    if old_chunk_count is not None:
        updates.append("old_chunk_count=%s")
        params.append(int(old_chunk_count))
    if new_chunk_count is not None:
        updates.append("new_chunk_count=%s")
        params.append(int(new_chunk_count))
    if completed:
        updates.append("completed_at=NOW()")
    if not updates:
        return
    params.append(job_id)
    await execute_insert(
        f"UPDATE kb_replacement_jobs SET {', '.join(updates)} WHERE job_id=%s",
        tuple(params),
    )


async def recover_interrupted_replacements() -> None:
    """Recover jobs at startup, before this single-worker server accepts requests.

    The durable document filename is the activation commit marker. Never delete
    uploaded files here: interrupted uploads remain available for investigation.
    Recovery errors abort startup rather than expose mixed-version evidence.
    """
    from routers.services.chroma_service import get_collection

    placeholders = ",".join(["%s"] * len(REPLACEMENT_ACTIVE_STATUSES))
    jobs = await execute_query(
        f"SELECT * FROM kb_replacement_jobs WHERE status IN ({placeholders}) ORDER BY started_at",
        tuple(sorted(REPLACEMENT_ACTIVE_STATUSES)),
    )
    for job in jobs:
        doc = await get_document_or_404(job["document_id"])
        job_id = job["job_id"]
        collection = get_collection()
        staged = await asyncio.to_thread(
            collection.get, where={"replacement_job_id": job_id}, include=["metadatas"],
        )
        replacement_ids = list(staged.get("ids") or [])
        committed = bool(job.get("new_filename")) and doc["filename"] == job["new_filename"]
        if committed:
            # Activation metadata must have reached the real document ID.
            valid_ids = [
                chunk_id for chunk_id, meta in zip(replacement_ids, staged.get("metadatas") or [])
                if str((meta or {}).get("doc_db_id")) == str(doc["id"])
            ]
            expected = int(job.get("new_chunk_count") or 0)
            healthy = (
                get_upload_path(doc["filename"]).is_file()
                and expected > 0 and len(valid_ids) == expected
            )
            if healthy:
                current_ids = await get_document_chunk_ids_by_db_id(doc["id"])
                obsolete = [chunk_id for chunk_id in current_ids if chunk_id not in set(valid_ids)]
                await delete_document_chunks_by_ids(obsolete)
            else:
                # Do not certify a partially activated replacement as indexed.
                valid_ids = []
            message = (
                "Replacement recovered after backend restart."
                if healthy else
                "Replacement activation was interrupted. Saved files were retained; use Process Again."
            )
            status = "completed" if healthy else "failed"
            count = len(valid_ids)
        else:
            # Includes temporary-ID chunks and chunks promoted before the SQL commit.
            await delete_document_chunks_by_ids(replacement_ids)
            count = await count_document_chunks_by_db_id(doc["id"])
            healthy = get_upload_path(doc["filename"]).is_file() and count > 0
            status = "failed"
            message = (
                "Replacement interrupted by backend restart. The previous file remains selected; "
                "you can retry replacement."
            )

        await execute_insert(
            """UPDATE uploaded_documents
               SET status=%s, processing_stage=%s, chunk_count=%s, error_message=%s
               WHERE id=%s""",
            ("indexed" if healthy else "error", "ready" if healthy else "error",
             count, None if healthy else message, doc["id"]),
        )
        await update_replacement_job(
            job_id, status=status, progress=100, message=message,
            warning_message="Recovery retained uploaded files for safety.",
            completed=True,
        )


async def get_active_replacement_job(document_id: int) -> dict | None:
    placeholders = ",".join(["%s"] * len(REPLACEMENT_ACTIVE_STATUSES))
    return await execute_one(
        f"""
        SELECT *
        FROM kb_replacement_jobs
        WHERE document_id=%s
          AND status IN ({placeholders})
        ORDER BY started_at DESC
        LIMIT 1
        """,
        (document_id, *sorted(REPLACEMENT_ACTIVE_STATUSES)),
    )


class DocumentMetadataUpdate(BaseModel):
    document_title: str = ""
    source_name: str = ""
    source_url: str = ""
    law_number: str = ""
    article_section: str = ""
    issuing_agency: str = ""
    date_issued: str = ""
    effective_date: str = ""
    date_downloaded: str = ""
    superseding_document: str = ""
    law_version: str = ""
    category: Optional[str] = None
    primary_category: Optional[str] = None
    secondary_categories: list[str] = Field(default_factory=list)
    document_type: str = ""
    legal_role: str = ""
    implements: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    sub_intents: list[str] = Field(default_factory=list)
    is_active: bool = True
    version_status: str = "active"


class DocumentCategorySuggestionRequest(BaseModel):
    title: str = ""
    law_number: str = ""
    document_type: str = ""
    issuing_agency: str = ""
    text: str = ""


class BulkDocumentMetadataUpdate(BaseModel):
    document_ids: list[int] = Field(default_factory=list)
    primary_category: Optional[str] = None
    secondary_categories: Optional[list[str]] = None
    version_status: Optional[str] = None
    document_type: Optional[str] = None


def resolve_upload_type(file: UploadFile) -> tuple[str, str, str]:
    original_filename = Path(file.filename or "document").name
    original_path = Path(original_filename)
    original_extension = original_path.suffix.lower()
    extension = original_extension
    if extension == ".doc":
        raise HTTPException(
            status_code=400,
            detail="Legacy .doc files are not supported. Convert the document to DOCX or PDF before uploading.",
        )
    if extension not in ALLOWED_EXTENSIONS:
        extension = MIME_TO_EXTENSION.get((file.content_type or "").lower(), "")
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX, PNG, JPG, TIFF, BMP, and WEBP files are allowed",
        )
    if original_extension != extension:
        original_filename = f"{original_path.stem or 'document'}{extension}"
    return original_filename, extension, ALLOWED_EXTENSIONS[extension]


def get_upload_path(stored_filename: str) -> Path:
    stored_path = Path(stored_filename or "")
    if stored_path.is_absolute():
        return stored_path
    return Path(settings.UPLOAD_DIR) / stored_path


def normalize_version_status(value: str) -> str:
    normalized = (value or "active").strip().lower()
    if normalized not in VERSION_STATUSES:
        raise HTTPException(
            status_code=400,
            detail="Invalid legal/version status",
        )
    return normalized


def normalize_document_row(row: dict) -> dict:
    normalized = dict(row or {})
    primary, secondary = normalize_document_categories(
        normalized.get("category") or DEFAULT_KB_CATEGORY,
        normalized.get("secondary_categories"),
    )
    normalized["category"] = primary
    normalized["primary_category"] = primary
    normalized["secondary_categories"] = secondary
    normalized["implements"] = parse_string_list(normalized.get("implements_json"))
    normalized["topics"] = parse_string_list(normalized.get("topics_json"))
    normalized["sub_intents"] = parse_string_list(normalized.get("sub_intents_json"))
    normalized["legal_status"] = normalized.get("version_status") or "active"
    normalized["superseded_by"] = normalized.get("superseding_document") or ""
    return normalized


def document_metadata_from_row(row: dict) -> dict:
    row = normalize_document_row(row)
    metadata = {
        "document_title": (row.get("document_title") or Path(row.get("original_filename") or "document").stem).strip()[:255],
        "source_name": (row.get("source_name") or row.get("category") or DEFAULT_KB_CATEGORY).strip()[:255],
        "source_url": (row.get("source_url") or "").strip()[:500],
        "law_number": (row.get("law_number") or "").strip()[:120],
        "article_section": (row.get("article_section") or "").strip()[:255],
        "issuing_agency": (row.get("issuing_agency") or "").strip()[:255],
        "date_issued": (row.get("date_issued") or "").strip()[:120],
        "effective_date": (row.get("effective_date") or "").strip()[:120],
        "date_downloaded": (row.get("date_downloaded") or "").strip()[:120],
        "document_year": row.get("document_year"),
        "superseding_document": (row.get("superseding_document") or "").strip()[:255],
        "law_version": (row.get("law_version") or "").strip()[:120],
        "index_version": str(row.get('index_version') or 1),
        "is_active": bool(row.get("is_active", True)),
        "version_status": (row.get("version_status") or "active").strip()[:120],
        "legal_status": (row.get("version_status") or "active").strip()[:120],
        "document_type": (row.get("document_type") or "").strip()[:120],
        "legal_role": (row.get("legal_role") or "").strip()[:120],
        "implements": json_list(row.get("implements")),
        "topics": json_list(row.get("topics")),
        "sub_intents": json_list(row.get("sub_intents")),
        "superseded_by": (row.get("superseding_document") or "").strip()[:255],
    }
    metadata.update(category_metadata(row.get("category"), row.get("secondary_categories")))
    metadata["document_year"] = infer_document_year(
        row.get("original_filename") or "",
        metadata,
    )
    return metadata


def document_metadata_patch_from_row(row: dict) -> dict:
    """Build the complete search-index metadata patch for a stored document."""
    metadata = document_metadata_from_row(row)
    metadata["doc_type"] = metadata.get("document_type", "")
    return metadata


async def update_document_record_with_index(
    doc_id: int,
    query: str,
    params: tuple,
    index_patch: dict,
    rollback_index_patch: dict,
) -> None:
    """Keep a document row and its search metadata in sync on update failures."""
    pool = await get_pool()
    async with pool.acquire() as connection:
        await connection.begin()
        index_update_started = False
        database_commit_started = False
        try:
            async with connection.cursor() as cursor:
                await cursor.execute(query, params)

            index_update_started = True
            await update_document_chunk_metadata_by_db_id(doc_id, index_patch)
            database_commit_started = True
            await connection.commit()
        except Exception as exc:
            database_rollback_failed = False
            try:
                await connection.rollback()
            except Exception:
                database_rollback_failed = True
                logger.exception(
                    "Could not roll back document metadata transaction for doc_id=%s",
                    doc_id,
                )

            index_rollback_failed = False
            if index_update_started:
                try:
                    await update_document_chunk_metadata_by_db_id(
                        doc_id, rollback_index_patch
                    )
                except Exception:
                    index_rollback_failed = True
                    logger.critical(
                        "Could not restore search metadata after a failed document update; "
                        "doc_id=%s requires reindexing",
                        doc_id,
                        exc_info=True,
                    )

            if index_update_started:
                if database_commit_started:
                    detail = (
                        "The metadata save could not be confirmed. Reload the document and "
                        "reprocess it if its search status or results are inconsistent."
                    )
                elif index_rollback_failed or database_rollback_failed:
                    detail = (
                        "Metadata update failed and automatic rollback was incomplete. "
                        "Reprocess this document to reconcile the search index."
                    )
                else:
                    detail = "The search index update failed; the database change was rolled back."
                raise HTTPException(status_code=503, detail=detail) from exc
            raise


# Category and active/archive state describe the KB record; these fields instead
# describe the uploaded authority and must not leak into a different document.
REPLACEMENT_CITATION_FIELDS = (
    "document_title", "source_name", "source_url", "law_number",
    "article_section", "issuing_agency", "date_issued", "effective_date",
    "date_downloaded", "document_year", "superseding_document", "law_version",
    "document_type", "legal_role", "implements_json", "topics_json",
    "sub_intents_json",
)


def replacement_metadata_row(doc: dict, original_filename: str, preserve: bool) -> dict:
    row = dict(doc)
    row["original_filename"] = original_filename
    if not preserve:
        for field in REPLACEMENT_CITATION_FIELDS:
            row[field] = None if field == "document_year" else "[]" if field.endswith("_json") else ""
        # normalize_document_row rebuilds derived list/alias values from these fields.
        row["document_title"] = Path(original_filename).stem[:255]
        row["source_name"] = doc.get("category") or DEFAULT_KB_CATEGORY
    return row


async def log_kb_admin_action(
    admin_user_id: int,
    action: str,
    target_id: int,
    details: dict | None = None,
) -> None:
    # Source changes invalidate approved FAQs; ingestion must still finish if this
    # notification fails. The FAQ reader also verifies each source snapshot.
    if action in {"replace_document_started", "replace_document", "retry_document_index", "update_document_metadata", "delete_document", "bulk_update_document_metadata", "rebuild_vector_index_started"}:
        try:
            from routers.services.faq_service import flag_source_faqs
            ids = (details or {}).get("updated_document_ids", []) if action == "bulk_update_document_metadata" else [target_id]
            await flag_source_faqs(None if action == "rebuild_vector_index_started" else ids)
        except Exception:
            logger.exception("Unable to flag FAQs for source review after %s", action)
    await execute_insert(
        """
        INSERT INTO admin_audit_logs (admin_user_id, action, target_type, target_id, details)
        VALUES (%s,%s,%s,%s,%s)
        """,
        (
            admin_user_id,
            action[:120],
            "document",
            str(target_id),
            json.dumps(details or {}),
        ),
    )


async def get_document_or_404(doc_id: int) -> dict:
    doc = await execute_one(
        """
        SELECT id,filename,original_filename,file_type,file_size,category,secondary_categories,status,
               document_title,document_type,legal_role,implements_json,topics_json,sub_intents_json,
               source_name,source_url,law_number,article_section,
               issuing_agency,date_issued,effective_date,date_downloaded,document_year,
               superseding_document,law_version,version_status,is_active,
               processing_stage,index_version,replacement_notes,replaced_at,file_hash,
               mime_type,chunk_count,error_message,last_indexed_at,created_at,uploaded_by
        FROM uploaded_documents
        WHERE id=%s
        """,
        (doc_id,),
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return normalize_document_row(doc)


async def build_document_health(doc_id: int, doc: dict | None = None) -> dict:
    document = doc or await get_document_or_404(doc_id)
    upload_path = get_upload_path(document["filename"])
    chunk_items = await get_all_document_chunks_by_db_id(doc_id)
    extraction_methods = {
        str((item.get("metadata") or {}).get("extraction_method") or "").lower()
        for item in chunk_items
    }
    extraction_methods.discard("")
    page_numbers = []
    for item in chunk_items:
        page = (item.get("metadata") or {}).get("page")
        try:
            page_numbers.append(int(page))
        except (TypeError, ValueError):
            continue
    total_chars = sum(len(item.get("text") or "") for item in chunk_items)
    has_chunks = len(chunk_items) > 0
    has_ocr = "ocr" in extraction_methods
    retrieval_eligible = (
        bool(document.get("is_active"))
        and str(document.get("version_status") or "active").strip().lower()
        in {"active", "current"}
    )
    health = {
        "document_id": doc_id,
        "file_exists": upload_path.exists() and upload_path.is_file(),
        "version_status": document.get("version_status") or "active",
        "is_active": bool(document.get("is_active")),
        "processing_stage": document.get("processing_stage") or "uploaded",
        "chunk_count": len(chunk_items),
        "text_char_count": total_chars,
        "page_count": max(page_numbers) if page_numbers else 0,
        "extraction_method": (
            "mixed"
            if len(extraction_methods) > 1
            else next(iter(extraction_methods), "none")
        ),
        "stages": {
            "uploaded": upload_path.exists() and upload_path.is_file(),
            "text_extracted": total_chars > 0,
            "ocr_used": has_ocr,
            "chunked": has_chunks,
            "embedded": has_chunks,
            "ready": (
                document.get("status") == "indexed"
                and has_chunks
                and retrieval_eligible
            ),
        },
    }
    return health


async def delete_uploaded_file(upload_path: Path) -> None:
    if not upload_path.exists() or not upload_path.is_file():
        return

    last_error = None
    for _attempt in range(6):
        try:
            upload_path.unlink()
            return
        except OSError as exc:
            last_error = exc
            await asyncio.sleep(0.35)

    raise HTTPException(
        status_code=409,
        detail=(
            "The uploaded file is still open or being used by another process. "
            "Close the document preview, wait a few seconds, then try Delete again. "
            f"Windows message: {last_error}"
        ),
    )


async def repair_stale_processing_documents() -> None:
    stale_docs = await execute_query(
        """
        SELECT id
        FROM uploaded_documents
        WHERE status IN ('pending','processing')
          AND (processing_stage IS NULL OR processing_stage NOT LIKE %s)
          AND TIMESTAMPDIFF(MINUTE, COALESCE(updated_at, created_at), NOW()) >= %s
        """,
        ("replacing%", STALE_PROCESSING_MINUTES),
    )
    for doc in stale_docs:
        doc_id = doc["id"]
        try:
            chunk_count = await count_document_chunks_by_db_id(doc_id)
        except Exception:
            chunk_count = 0
        if chunk_count > 0:
            await execute_insert(
                """
                UPDATE uploaded_documents
                SET status='indexed', chunk_count=%s, error_message=NULL,
                    processing_stage='ready', last_indexed_at=NOW()
                WHERE id=%s AND status IN ('pending','processing')
                """,
                (chunk_count, doc_id),
            )
            continue
        await execute_insert(
            """
            UPDATE uploaded_documents
            SET status='error',
                processing_stage='error',
                chunk_count=0,
                error_message=%s
            WHERE id=%s AND status IN ('pending','processing')
            """,
            (
                "Indexing did not finish within 60 minutes. The upload may have been interrupted. "
                "Please delete this document and upload it again.",
                doc_id,
            ),
        )


async def _bg_ingest(
    filepath: str,
    doc_id: int,
    original_filename: str,
    category: str,
    document_metadata: dict | None = None,
):
    try:
        await execute_insert(
            """
            UPDATE uploaded_documents
            SET status='processing', processing_stage='extracting_text', error_message=NULL
            WHERE id=%s
            """,
            (doc_id,),
        )
        await ingest_document(filepath, doc_id, original_filename, category, document_metadata)
        await execute_insert(
            "UPDATE uploaded_documents SET processing_stage='embedding' WHERE id=%s",
            (doc_id,),
        )
        count = await count_document_chunks_by_db_id(doc_id)
        if count <= 0:
            raise ValueError(
                "No readable text was extracted. If this is a scanned PDF or image, "
                "make sure OCR dependencies are installed on the backend server."
            )
        await execute_insert(
            """
            UPDATE uploaded_documents
            SET status='indexed', chunk_count=%s, processing_stage='ready',
                error_message=NULL, last_indexed_at=NOW()
            WHERE id=%s
            """,
            (count, doc_id),
        )
        invalidate_retrieval_caches()
    except Exception as e:
        await execute_insert(
            "UPDATE uploaded_documents SET status='error', processing_stage='error', error_message=%s WHERE id=%s",
            (str(e)[:500], doc_id),
        )


async def _bg_rebuild_index(admin_user_id: int, lock_acquired: bool = False):
    if not lock_acquired:
        await _rebuild_index_lock.acquire()
    try:
        await _run_bg_rebuild_index(admin_user_id)
    finally:
        _rebuild_index_lock.release()


async def _run_bg_rebuild_index(admin_user_id: int):
    rows = await execute_query(
        """
        SELECT id,filename,original_filename,file_type,file_size,category,secondary_categories,status,
               document_title,document_type,legal_role,implements_json,topics_json,sub_intents_json,
               source_name,source_url,law_number,article_section,
               issuing_agency,date_issued,effective_date,date_downloaded,document_year,
               superseding_document,law_version,version_status,is_active,index_version
        FROM uploaded_documents
        WHERE is_active=1
          AND COALESCE(version_status, 'active') IN ('active','current')
          AND status <> 'processing'
        ORDER BY id ASC
        """
    )
    reset_collection("admin rebuild after legal-scope metadata upgrade")
    rebuilt = 0
    failed = 0
    for row in rows:
        doc_id = row["id"]
        upload_path = get_upload_path(row["filename"])
        if not upload_path.exists() or not upload_path.is_file():
            failed += 1
            await execute_insert(
                """
                UPDATE uploaded_documents
                SET status='error',
                    processing_stage='error',
                    chunk_count=0,
                    error_message='Uploaded file was not found on disk during index rebuild.'
                WHERE id=%s
                """,
                (doc_id,),
            )
            continue
        try:
            await execute_insert(
                """
                UPDATE uploaded_documents
                SET status='processing',
                    processing_stage='rebuilding_index',
                    chunk_count=0,
                    error_message=NULL
                WHERE id=%s
                """,
                (doc_id,),
            )
            await ingest_document(
                str(upload_path),
                doc_id,
                row.get("original_filename"),
                row.get("category") or DEFAULT_KB_CATEGORY,
                document_metadata_from_row(row),
            )
            count = await count_document_chunks_by_db_id(doc_id)
            if count <= 0:
                raise ValueError("No chunks were created during index rebuild.")
            await execute_insert(
                """
                UPDATE uploaded_documents
                SET status='indexed',
                    processing_stage='ready',
                    chunk_count=%s,
                    last_indexed_at=NOW(),
                    error_message=NULL
                WHERE id=%s
                """,
                (count, doc_id),
            )
            rebuilt += 1
        except Exception as exc:
            failed += 1
            await execute_insert(
                """
                UPDATE uploaded_documents
                SET status='error',
                    processing_stage='error',
                    chunk_count=0,
                    error_message=%s
                WHERE id=%s
                """,
                (str(exc)[:500], doc_id),
            )
    await log_kb_admin_action(
        admin_user_id,
        "rebuild_vector_index_finished",
        0,
        {"documents_seen": len(rows), "rebuilt": rebuilt, "failed": failed},
    )


async def _finish_published_replacement(
    job_id: str, doc_id: int, admin_user_id: int, doc: dict,
    old_upload_path: Path, final_path: Path, original_filename: str,
    old_chunk_ids: list[str], activated: int, warning: str = "",
) -> None:
    """Clean up after publication without ever rolling back published evidence."""
    warnings = [warning] if warning else []
    try:
        await update_replacement_job(
            job_id, status="cleaning", progress=90,
            message="Replacement published. Removing old chunks and old uploaded file.",
        )
    except Exception:
        logger.exception("Replacement cleanup progress could not be saved for job_id=%s", job_id)
        warnings.append("Cleanup progress could not be saved; the replacement remains published.")

    deleted_old_chunks = 0
    try:
        deleted_old_chunks = await delete_document_chunks_by_ids(old_chunk_ids)
    except Exception:
        logger.exception("Old replacement chunks could not be cleaned for job_id=%s", job_id)
        warnings.append("Old chunks could not be fully removed. The new version remains selected; old files were retained.")
    else:
        if old_upload_path != final_path:
            try:
                await delete_uploaded_file(old_upload_path)
            except Exception:
                logger.exception("Old replacement file could not be cleaned for job_id=%s", job_id)
                warnings.append("Old file cleanup failed; the replacement remains published.")

    try:
        await log_kb_admin_action(
            admin_user_id, "replace_document", doc_id,
            {"job_id": job_id, "old_filename": doc.get("filename"),
             "new_filename": final_path.name, "original_filename": original_filename,
             "old_chunks_deleted": deleted_old_chunks, "new_chunks": activated,
             "warning": " | ".join(warnings)},
        )
    except Exception:
        logger.exception("Replacement audit log could not be saved for job_id=%s", job_id)
        warnings.append("The replacement succeeded, but its admin activity log could not be saved.")
    try:
        await update_replacement_job(
            job_id, status="completed", progress=100,
            message=f"Replacement completed with {activated} indexed chunks.",
            warning_message=" | ".join(warnings) or None, completed=True,
        )
    except Exception:
        # The active job retains its durable new_filename marker. Startup
        # recovery can finish it; a reporting outage must not erase the data.
        logger.exception("Published replacement completion could not be saved for job_id=%s", job_id)


async def _bg_replace_document(
    job_id: str,
    doc_id: int,
    staged_filepath: str,
    original_filename: str,
    stored_file_type: str,
    replacement_size: int,
    replacement_hash: str,
    replacement_mime_type: str,
    replacement_notes: str,
    preserve_metadata: bool,
    admin_user_id: int,
    old_document_snapshot: dict,
):
    staged_path = Path(staged_filepath)
    final_path: Path | None = None
    temp_doc_db_id = temporary_replacement_doc_db_id(doc_id)
    old_chunk_ids: list[str] = []
    new_chunk_ids: list[str] = []
    parser_warning = ""
    doc = dict(old_document_snapshot or {})
    publication_attempted = False
    published = False

    try:
        doc = old_document_snapshot or await get_document_or_404(doc_id)
        old_upload_path = get_upload_path(doc["filename"])
        old_chunk_ids = await get_document_chunk_ids_by_db_id(doc_id)
        await update_replacement_job(
            job_id,
            status="validating",
            progress=10,
            message="Validating replacement file.",
            old_chunk_count=len(old_chunk_ids),
        )

        if not staged_path.exists() or not staged_path.is_file():
            raise ValueError("Replacement upload was not found in the staging folder.")

        extracted, warnings = await asyncio.to_thread(
            extract_text_with_parser_warnings,
            str(staged_path),
        )
        replacement_text = (extracted.get("text") or "").strip()
        if not replacement_text:
            raise ValueError(
                "The replacement file did not contain readable text. "
                "If it is scanned, check OCR installation and image quality."
            )
        if warnings:
            parser_warning = " | ".join(warnings[:3])[:2000]
            logger.warning(
                "Replacement parser warning for document %s / job %s: %s",
                doc_id,
                job_id,
                parser_warning,
            )
            await update_replacement_job(
                job_id,
                warning_message=parser_warning,
            )

        await execute_insert(
            """
            UPDATE uploaded_documents
            SET status='processing',
                processing_stage='replacing_chunking',
                error_message=NULL
            WHERE id=%s
            """,
            (doc_id,),
        )
        await update_replacement_job(
            job_id,
            status="chunking",
            progress=35,
            message="Extracting, chunking, and embedding the replacement file.",
        )

        replacement_row = replacement_metadata_row(doc, original_filename, preserve_metadata)
        metadata = document_metadata_from_row(replacement_row)
        # Persist the same citation inputs passed to ingestion, including inferred year.
        replacement_row["document_year"] = metadata.get("document_year")
        citation_updates = ", ".join(f"{field}=%s" for field in REPLACEMENT_CITATION_FIELDS)
        next_index_version = int(doc.get("index_version") or 1) + 1
        metadata.update(
            {
                "replacement_job_id": job_id,
                "index_version": str(next_index_version),
                "source_filename": original_filename,
            }
        )

        await ingest_document(
            str(staged_path),
            temp_doc_db_id,
            original_filename,
            doc.get("category") or DEFAULT_KB_CATEGORY,
            metadata,
        )
        new_chunk_ids = await get_document_chunk_ids_by_db_id(temp_doc_db_id)
        if not new_chunk_ids:
            raise ValueError("No searchable chunks were created from the replacement file.")

        await update_replacement_job(
            job_id,
            status="embedding",
            progress=65,
            message=f"Created {len(new_chunk_ids)} replacement chunks.",
            new_chunk_count=len(new_chunk_ids),
        )

        final_path = move_staged_replacement_file(staged_path, original_filename)
        await update_replacement_job(
            job_id,
            status="activating",
            progress=78,
            message="Activating replacement chunks.",
            new_filename=final_path.name,
        )

        activated = await update_document_chunk_metadata_by_db_id(
            temp_doc_db_id,
            {
                "doc_db_id": str(doc_id),
                "doc_id": f"doc_{doc_id}_v{next_index_version}",
                "document_id": f"doc_{doc_id}_v{next_index_version}",
                "source_filename": original_filename,
                "stored_filename": final_path.name,
                "index_version": str(next_index_version),
                "replacement_job_id": job_id,
            },
        )
        if activated <= 0:
            raise ValueError("Replacement chunks could not be activated.")

        publication_attempted = True
        await execute_insert(
            f"""
            UPDATE uploaded_documents
            SET filename=%s,
                original_filename=%s,
                file_type=%s,
                file_size=%s,
                status='indexed',
                processing_stage='ready',
                chunk_count=%s,
                error_message=NULL,
                last_indexed_at=NOW(),
                replacement_notes=%s,
                replaced_at=NOW(),
                index_version=%s,
                file_hash=%s,
                mime_type=%s,
                {citation_updates}
            WHERE id=%s
            """,
            (
                final_path.name,
                original_filename,
                stored_file_type,
                replacement_size,
                activated,
                replacement_notes[:2000],
                next_index_version,
                replacement_hash,
                replacement_mime_type[:120],
                *(replacement_row.get(field) for field in REPLACEMENT_CITATION_FIELDS),
                doc_id,
            ),
        )
        published = True

    except Exception as exc:
        logger.exception("Document replacement failed for doc_id=%s job_id=%s", doc_id, job_id)
        if publication_attempted:
            # A disconnected client can see an error after MySQL committed.
            # Resolve the durable filename/version before destructive rollback.
            current = None
            try:
                current = await get_document_or_404(doc_id)
            except Exception:
                logger.exception("Replacement publication could not be confirmed for job_id=%s", job_id)
            published = bool(current and current.get("filename") == final_path.name
                             and int(current.get("index_version") or 1) == next_index_version)
            previous_selected = bool(current and current.get("filename") == doc.get("filename")
                                     and int(current.get("index_version") or 1) == int(doc.get("index_version") or 1))
            if not published and not previous_selected:
                # Retain both versions and the active job for startup recovery.
                # Neither an unreadable registry nor an unexpected version is
                # evidence that deleting the replacement is safe.
                with contextlib.suppress(Exception):
                    await update_replacement_job(
                        job_id, warning_message="Publication could not be confirmed. Both versions were retained for recovery.",
                        message="Replacement needs recovery; no files or chunks were removed.",
                    )
                return
            if published:
                parser_warning = " | ".join(filter(None, [parser_warning,
                    "Publication succeeded, but the database acknowledgment failed; the saved version was confirmed."]))

        if not published:
            with contextlib.suppress(Exception):
                if new_chunk_ids:
                    await delete_document_chunks_by_ids(new_chunk_ids)
            with contextlib.suppress(Exception):
                await delete_document_chunks_by_db_id(temp_doc_db_id)
            for cleanup_path in (final_path, staged_path):
                with contextlib.suppress(Exception):
                    if cleanup_path and cleanup_path.exists():
                        cleanup_path.unlink()

            # File/citation/version fields were never published. Restore only
            # processing state, and only while the original version is selected.
            await execute_insert(
                """UPDATE uploaded_documents
                   SET status=%s, processing_stage=%s, chunk_count=%s, error_message=%s
                   WHERE id=%s AND filename=%s AND COALESCE(index_version, 1)=%s""",
                (doc.get("status") or "indexed", doc.get("processing_stage") or "ready",
                 int(doc.get("chunk_count") or len(old_chunk_ids) or 0), doc.get("error_message"),
                 doc_id, doc.get("filename"), int(doc.get("index_version") or 1)),
            )
            try:
                await log_kb_admin_action(
                    admin_user_id, "replace_document_failed", doc_id,
                    {"job_id": job_id, "error": str(exc)[:500]},
                )
            except Exception:
                logger.exception("Replacement failure audit could not be saved for job_id=%s", job_id)
            await update_replacement_job(
                job_id, status="failed", progress=100,
                message="Replacement failed before publication. The original document and chunks were kept.",
                error_message=str(exc), completed=True,
            )
            return

    # This phase is intentionally outside the rollback handler. Publication
    # selected the new filename, citations and index version atomically.
    await _finish_published_replacement(
        job_id, doc_id, admin_user_id, doc, old_upload_path, final_path,
        original_filename, old_chunk_ids, activated, parser_warning,
    )


@router.get("/categories")
async def categories(current_user=Depends(get_current_user)):
    return {
        "categories": KB_CATEGORIES,
        "default": DEFAULT_KB_CATEGORY,
        "max_secondary_categories": 4,
    }


@router.post("/categories/suggest")
async def suggest_categories(
    payload: DocumentCategorySuggestionRequest,
    current_user=Depends(require_admin),
):
    return suggest_document_categories(
        title=payload.title,
        law_number=payload.law_number,
        document_type=payload.document_type,
        issuing_agency=payload.issuing_agency,
        text=payload.text,
    )


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(DEFAULT_KB_CATEGORY),
    primary_category: str = Form(""),
    secondary_categories: str = Form("[]"),
    document_title: str = Form(""),
    document_type: str = Form(""),
    legal_role: str = Form(""),
    implements: str = Form("[]"),
    topics: str = Form("[]"),
    sub_intents: str = Form("[]"),
    source_name: str = Form(""),
    source_url: str = Form(""),
    law_number: str = Form(""),
    article_section: str = Form(""),
    issuing_agency: str = Form(""),
    date_issued: str = Form(""),
    effective_date: str = Form(""),
    date_downloaded: str = Form(""),
    superseding_document: str = Form(""),
    law_version: str = Form(""),
    version_status: str = Form("active"),
    current_user=Depends(require_admin),
):
    original_filename, _extension, stored_file_type = resolve_upload_type(file)
    try:
        category, secondary = normalize_document_categories(
            primary_category or category,
            secondary_categories,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if category not in KB_CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid knowledge base category")
    version_status = normalize_version_status(version_status)
    content = await read_upload_with_limit(
        file, f"Maximum file size is {settings.MAX_FILE_SIZE_MB} MB."
    )
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    file_hash = sha256_hex(content)
    duplicate = await execute_one(
        "SELECT id,document_title,original_filename FROM uploaded_documents WHERE file_hash=%s LIMIT 1",
        (file_hash,),
    )
    if duplicate:
        raise HTTPException(
            status_code=409,
            detail=(
                "This exact file is already stored as one knowledge-base document "
                f"(ID {duplicate['id']}). Edit its categories instead of uploading a duplicate."
            ),
        )
    filepath = None
    document_year = infer_document_year(
        original_filename,
        {
            "document_title": document_title,
            "date_issued": date_issued,
            "effective_date": effective_date,
            "date_downloaded": date_downloaded,
            "law_version": law_version,
        },
    )
    try:
        filepath = await asyncio.to_thread(save_upload, content, original_filename)
        doc_id = await execute_insert(
            """
            INSERT INTO uploaded_documents
                (filename,original_filename,file_type,file_size,category,secondary_categories,
                 document_title,document_type,legal_role,implements_json,topics_json,sub_intents_json,
                 source_name,source_url,law_number,article_section,
                 issuing_agency,date_issued,effective_date,date_downloaded,
                 document_year,superseding_document,law_version,version_status,is_active,
                 file_hash,mime_type,uploaded_by,status,processing_stage)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE,%s,%s,%s,'processing','uploaded')
            """,
            (
                Path(filepath).name,
                original_filename,
                stored_file_type,
                len(content),
                category,
                json.dumps(secondary, ensure_ascii=False),
                (document_title or Path(original_filename).stem).strip()[:255],
                document_type.strip()[:120],
                legal_role.strip()[:120],
                json_list(implements),
                json_list(topics),
                json_list(sub_intents),
                (source_name or category).strip()[:255],
                (source_url or "").strip()[:500],
                (law_number or "").strip()[:120],
                (article_section or "").strip()[:255],
                (issuing_agency or "").strip()[:255],
                (date_issued or "").strip()[:120],
                (effective_date or "").strip()[:120],
                (date_downloaded or "").strip()[:120],
                document_year,
                (superseding_document or "").strip()[:255],
                (law_version or "").strip()[:120],
                version_status,
                file_hash,
                (file.content_type or FILE_MEDIA_TYPES.get(stored_file_type, ""))[:120],
                current_user["id"],
            ),
        )
    except Exception as exc:
        if filepath:
            try:
                Path(filepath).unlink(missing_ok=True)
            except OSError:
                pass
        raise HTTPException(
            status_code=500,
            detail="Upload failed before indexing. Please check the database connection and try again.",
        ) from exc
    document_metadata = {
        "document_title": (document_title or Path(original_filename).stem).strip()[:255],
        "source_name": (source_name or category).strip()[:255],
        "source_url": (source_url or "").strip()[:500],
        "law_number": (law_number or "").strip()[:120],
        "article_section": (article_section or "").strip()[:255],
        "issuing_agency": (issuing_agency or "").strip()[:255],
        "date_issued": (date_issued or "").strip()[:120],
        "effective_date": (effective_date or "").strip()[:120],
        "date_downloaded": (date_downloaded or "").strip()[:120],
        "document_year": document_year,
        "superseding_document": (superseding_document or "").strip()[:255],
        "law_version": (law_version or "").strip()[:120],
        "is_active": True,
        "version_status": version_status,
        "legal_status": version_status,
        "document_type": document_type.strip()[:120],
        "legal_role": legal_role.strip()[:120],
        "implements": json_list(implements),
        "topics": json_list(topics),
        "sub_intents": json_list(sub_intents),
    }
    document_metadata.update(category_metadata(category, secondary))
    await log_kb_admin_action(
        current_user["id"],
        "upload_document",
        doc_id,
        {
            "original_filename": original_filename,
            "primary_category": category,
            "secondary_categories": secondary,
            "version_status": version_status,
        },
    )
    background_tasks.add_task(
        _bg_ingest,
        filepath,
        doc_id,
        original_filename,
        category,
        document_metadata,
    )
    return {
        "status": "processing",
        "message": "Documents are being ingested in the background. The chatbot remains online.",
        "doc_id": doc_id,
        "category": category,
        "primary_category": category,
        "secondary_categories": secondary,
    }


@router.post("/rebuild-index")
async def rebuild_index(
    background_tasks: BackgroundTasks,
    current_user=Depends(require_admin),
):
    if _rebuild_index_lock.locked():
        raise HTTPException(409, "A knowledge-base rebuild is already running.")
    await _rebuild_index_lock.acquire()
    try:
        await repair_stale_processing_documents()
        active_jobs = await execute_query(
            """
            SELECT id, original_filename, processing_stage
            FROM uploaded_documents
            WHERE is_active=1
              AND COALESCE(version_status, 'active') IN ('active','current')
                AND (status IN ('pending','processing') OR processing_stage LIKE 'replacing%%')
            ORDER BY id ASC
            LIMIT 5
            """
        )
        if active_jobs:
            raise HTTPException(
                409,
                "Wait for current document uploads or replacements to finish before rebuilding the entire knowledge base.",
            )
        await log_kb_admin_action(
            current_user["id"],
            "rebuild_vector_index_started",
            0,
            {"reason": "Rebuild vectors with legal-scope metadata and chunk headers."},
        )
        background_tasks.add_task(_bg_rebuild_index, current_user["id"], True)
    except Exception:
        _rebuild_index_lock.release()
        raise
    return {
        "message": "Index rebuild started",
        "status": "processing",
        "note": "Active approved documents will be re-indexed with the new legal-scope metadata.",
    }


@router.get("/rebuild-index/status")
async def rebuild_index_status(current_user=Depends(require_admin)):
    return {"running": _rebuild_index_lock.locked()}


@router.post(
    "/documents/{doc_id}/retry-index",
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_document_index(
    doc_id: int,
    background_tasks: BackgroundTasks,
    current_user=Depends(require_admin),
):
    """Retry an interrupted or failed ingestion using the durable saved file."""

    if _rebuild_index_lock.locked():
        raise HTTPException(409, "Wait until the knowledge-base rebuild finishes before processing one document.")
    await repair_stale_processing_documents()
    document = await get_document_or_404(doc_id)
    if document.get("status") in {"pending", "processing"}:
        raise HTTPException(
            status_code=409,
            detail="This document is already being indexed.",
        )

    upload_path = get_upload_path(document["filename"])
    if not upload_path.exists() or not upload_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="The saved upload is missing. Upload the document again.",
        )

    await delete_document_chunks_by_db_id(doc_id)
    await execute_insert(
        """
        UPDATE uploaded_documents
        SET status='processing', processing_stage='retry_queued',
            chunk_count=0, error_message=NULL
        WHERE id=%s
        """,
        (doc_id,),
    )
    await log_kb_admin_action(
        current_user["id"],
        "retry_document_index",
        doc_id,
        {"original_filename": document.get("original_filename")},
    )
    background_tasks.add_task(
        _bg_ingest,
        str(upload_path),
        doc_id,
        document.get("original_filename") or document.get("filename"),
        document.get("category") or DEFAULT_KB_CATEGORY,
        document_metadata_from_row(document),
    )
    return {
        "status": "processing",
        "message": "Document re-indexing started in the background.",
        "doc_id": doc_id,
    }


@router.post("/documents/{doc_id}/replace")
async def replace_document(
    doc_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    replacement_notes: str = Form(""),
    preserve_metadata: bool = Form(False),
    current_user=Depends(require_admin),
):
    await repair_stale_processing_documents()
    existing = await get_document_or_404(doc_id)
    if existing.get("status") in {"pending", "processing"}:
        raise HTTPException(
            status_code=409,
            detail="This document is currently being processed. Please wait until it is ready before replacing it.",
        )
    if str(existing.get("processing_stage") or "").startswith("replacing"):
        raise HTTPException(
            status_code=409,
            detail="A replacement is already running for this document.",
        )
    active_job = await get_active_replacement_job(doc_id)
    if active_job:
        raise HTTPException(
            status_code=409,
            detail=f"Replacement already in progress for this document. Job: {active_job['job_id']}",
        )

    original_filename, _extension, stored_file_type = resolve_upload_type(file)
    content = await read_upload_with_limit(
        file, f"Maximum replacement file size is {settings.MAX_FILE_SIZE_MB} MB."
    )
    if not content:
        raise HTTPException(status_code=400, detail="Replacement file is empty")

    job_id = f"replace_{uuid.uuid4().hex[:20]}"
    staged_path = None
    try:
        staged_path = await asyncio.to_thread(
            save_staged_replacement_file,
            content,
            original_filename,
        )
        await execute_insert(
            """
            INSERT INTO kb_replacement_jobs
                (job_id,document_id,admin_user_id,status,progress,message,
                 replacement_notes,old_filename,old_file_size,new_filename,new_file_size)
            VALUES (%s,%s,%s,'queued',0,%s,%s,%s,%s,%s,%s)
            """,
            (
                job_id,
                doc_id,
                current_user["id"],
                "Replacement upload received.",
                replacement_notes[:2000],
                existing.get("filename"),
                existing.get("file_size") or 0,
                original_filename,
                len(content),
            ),
        )
        await execute_insert(
            """
            UPDATE uploaded_documents
            SET status='processing',
                processing_stage='replacing_queued',
                error_message=NULL
            WHERE id=%s
            """,
            (doc_id,),
        )
    except Exception as exc:
        if staged_path:
            with contextlib.suppress(Exception):
                staged_path.unlink()
        raise HTTPException(
            status_code=500,
            detail="Replacement could not be queued. Please try again.",
        ) from exc

    await log_kb_admin_action(
        current_user["id"],
        "replace_document_started",
        doc_id,
        {
            "job_id": job_id,
            "old_filename": existing.get("filename"),
            "replacement_filename": original_filename,
        },
    )
    background_tasks.add_task(
        _bg_replace_document,
        job_id,
        doc_id,
        str(staged_path),
        original_filename,
        stored_file_type,
        len(content),
        sha256_hex(content),
        (file.content_type or FILE_MEDIA_TYPES.get(stored_file_type, "")),
        replacement_notes,
        preserve_metadata,
        current_user["id"],
        existing,
    )
    return {
        "message": "Replacement started",
        "status": "processing",
        "job_id": job_id,
        "document_id": doc_id,
        "note": "The old document remains available until the replacement is fully indexed.",
    }


@router.get("/replacement-jobs/{job_id}")
async def replacement_job_status(job_id: str, current_user=Depends(require_admin)):
    job = await execute_one(
        """
        SELECT job_id,document_id,status,progress,message,replacement_notes,
               old_filename,new_filename,old_file_size,new_file_size,
               old_chunk_count,new_chunk_count,warning_message,error_message,
               started_at,completed_at
        FROM kb_replacement_jobs
        WHERE job_id=%s
        """,
        (job_id,),
    )
    if not job:
        raise HTTPException(status_code=404, detail="Replacement job not found")
    return job


@router.get("/documents")
async def list_docs(
    primary_category: Optional[str] = None,
    secondary_category: Optional[str] = None,
    issuing_agency: Optional[str] = None,
    document_type: Optional[str] = None,
    version_status: Optional[str] = None,
    current_user=Depends(get_current_user),
):
    await repair_stale_processing_documents()
    rows = await execute_query(
        """
        SELECT id,filename,original_filename,file_type,file_size,category,secondary_categories,status,
               document_title,document_type,legal_role,implements_json,topics_json,sub_intents_json,
               source_name,source_url,law_number,article_section,
               issuing_agency,date_issued,effective_date,date_downloaded,document_year,
               superseding_document,law_version,version_status,is_active,
               processing_stage,index_version,replacement_notes,replaced_at,file_hash,
               mime_type,chunk_count,error_message,last_indexed_at,created_at,uploaded_by
        FROM uploaded_documents
        ORDER BY created_at DESC
        """
    )
    documents = [normalize_document_row(row) for row in rows]
    if primary_category:
        try:
            requested_primary, _ = normalize_document_categories(primary_category, [])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        documents = [doc for doc in documents if doc["primary_category"] == requested_primary]
    if secondary_category:
        try:
            requested_secondary, _ = normalize_document_categories(secondary_category, [])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        documents = [
            doc for doc in documents
            if requested_secondary in doc.get("secondary_categories", [])
        ]
    if issuing_agency:
        needle = issuing_agency.strip().casefold()
        documents = [doc for doc in documents if needle in str(doc.get("issuing_agency") or "").casefold()]
    if document_type:
        needle = document_type.strip().casefold()
        documents = [doc for doc in documents if needle == str(doc.get("document_type") or "").strip().casefold()]
    if version_status:
        needle = version_status.strip().casefold()
        documents = [doc for doc in documents if needle == str(doc.get("version_status") or "active").casefold()]
    return documents


@router.get("/documents/index-status")
async def document_index_status(current_user=Depends(require_admin)):
    """Report actual Chroma chunk counts per document for admin status views."""
    if _rebuild_index_lock.locked():
        return {"available": False, "document_chunk_counts": {}}
    try:
        counts = await get_document_chunk_counts_by_db_id()
    except Exception:
        logger.exception("Unable to read per-document knowledge-base index status")
        return {"available": False, "document_chunk_counts": {}}
    return {
        "available": True,
        "document_chunk_counts": {str(doc_id): count for doc_id, count in counts.items()},
    }


@router.get("/documents/{doc_id}")
async def get_doc(doc_id: int, current_user=Depends(get_current_user)):
    await repair_stale_processing_documents()
    return await get_document_or_404(doc_id)


@router.get("/documents/{doc_id}/original-content")
async def get_doc_original_content(
    doc_id: int,
    current_user=Depends(require_admin),
):
    await repair_stale_processing_documents()
    doc = await get_document_or_404(doc_id)
    upload_path = get_upload_path(doc["filename"])
    if not upload_path.exists() or not upload_path.is_file():
        raise HTTPException(status_code=404, detail="Uploaded file was not found on disk")
    try:
        content = extract_original_document_text(str(upload_path))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read the original document text: {exc}",
        ) from exc
    if not content.get("text"):
        raise HTTPException(
            status_code=422,
            detail=(
                "No readable text was extracted from the original file. "
                "If this is scanned, OCR may be missing on the server or the image quality may be too low."
            ),
        )
    return {"document": doc, **content}


@router.get("/documents/{doc_id}/file")
async def get_doc_file(
    doc_id: int,
    current_user=Depends(require_admin),
):
    await repair_stale_processing_documents()
    doc = await get_document_or_404(doc_id)
    upload_path = get_upload_path(doc["filename"])
    if not upload_path.exists() or not upload_path.is_file():
        raise HTTPException(status_code=404, detail="Uploaded file was not found on disk")
    media_type = FILE_MEDIA_TYPES.get(
        str(doc.get("file_type") or "").lower(),
        "application/octet-stream",
    )
    original_name = doc.get("original_filename") or upload_path.name
    headers = {
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(original_name)}"
    }
    headers.update({"Cache-Control": "private, no-store, max-age=0", "Pragma": "no-cache", "Expires": "0"})
    return FileResponse(str(upload_path), media_type=media_type, headers=headers)


@router.get("/documents/{doc_id}/content")
async def get_doc_content(
    doc_id: int,
    limit: int = 50,
    offset: int = 0,
    current_user=Depends(require_admin),
):
    await repair_stale_processing_documents()
    doc = await get_document_or_404(doc_id)
    chunks = await get_document_chunks_by_db_id(doc_id, limit=limit, offset=offset)
    return {"document": doc, **chunks}


@router.get("/documents/{doc_id}/health")
async def get_doc_health(
    doc_id: int,
    current_user=Depends(require_admin),
):
    await repair_stale_processing_documents()
    doc = await get_document_or_404(doc_id)
    return {"document": doc, "health": await build_document_health(doc_id, doc)}


@router.put("/documents/{doc_id}/metadata")
async def update_doc_metadata(
    doc_id: int,
    payload: DocumentMetadataUpdate,
    current_user=Depends(require_admin),
):
    await repair_stale_processing_documents()
    existing = await get_document_or_404(doc_id)
    supplied_fields = getattr(payload, "model_fields_set", getattr(payload, "__fields_set__", set()))
    secondary_input = (
        payload.secondary_categories
        if "secondary_categories" in supplied_fields
        else existing.get("secondary_categories", [])
    )
    try:
        category, secondary = normalize_document_categories(
            payload.primary_category or payload.category or existing.get("category"),
            secondary_input,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # The metadata endpoint historically accepted a full form payload, but API
    # clients also use it for category-only edits. Preserve every omitted field
    # so adding a secondary category cannot silently clear legal metadata or
    # reactivate a historical document.
    def supplied_text(field_name: str, max_length: int) -> str:
        raw_value = (
            getattr(payload, field_name)
            if field_name in supplied_fields
            else existing.get(field_name)
        )
        return str(raw_value or "").strip()[:max_length]

    version_status = normalize_version_status(
        payload.version_status
        if "version_status" in supplied_fields
        else existing.get("version_status") or "active"
    )
    document_title = (
        supplied_text("document_title", 255)
        or Path(existing["original_filename"]).stem[:255]
    )
    source_name = supplied_text("source_name", 255)
    source_url = supplied_text("source_url", 500)
    law_number = supplied_text("law_number", 120)
    article_section = supplied_text("article_section", 255)
    issuing_agency = supplied_text("issuing_agency", 255)
    date_issued = supplied_text("date_issued", 120)
    effective_date = supplied_text("effective_date", 120)
    date_downloaded = supplied_text("date_downloaded", 120)
    superseding_document = supplied_text("superseding_document", 255)
    law_version = supplied_text("law_version", 120)
    document_type = str(
        payload.document_type
        if "document_type" in supplied_fields
        else existing.get("document_type") or ""
    ).strip()[:120]
    legal_role = str(
        payload.legal_role
        if "legal_role" in supplied_fields
        else existing.get("legal_role") or ""
    ).strip()[:120]
    implements = parse_string_list(
        payload.implements if "implements" in supplied_fields else existing.get("implements")
    )
    topics = parse_string_list(
        payload.topics if "topics" in supplied_fields else existing.get("topics")
    )
    sub_intents = parse_string_list(
        payload.sub_intents if "sub_intents" in supplied_fields else existing.get("sub_intents")
    )
    is_active = bool(
        payload.is_active
        if "is_active" in supplied_fields
        else existing.get("is_active", True)
    )
    document_year = infer_document_year(
        existing.get("original_filename") or "",
        {
            "document_title": document_title,
            "date_issued": date_issued,
            "effective_date": effective_date,
            "date_downloaded": date_downloaded,
            "law_version": law_version,
        },
    )

    update_sql = """
        UPDATE uploaded_documents
        SET document_title=%s,
            document_type=%s,
            legal_role=%s,
            implements_json=%s,
            topics_json=%s,
            sub_intents_json=%s,
            source_name=%s,
            source_url=%s,
            law_number=%s,
            article_section=%s,
            issuing_agency=%s,
            date_issued=%s,
            effective_date=%s,
            date_downloaded=%s,
            document_year=%s,
            superseding_document=%s,
            law_version=%s,
            category=%s,
            secondary_categories=%s,
            is_active=%s,
            version_status=%s
        WHERE id=%s
        """
    update_params = (
        document_title,
        document_type,
        legal_role,
        json_list(implements),
        json_list(topics),
        json_list(sub_intents),
        source_name,
        source_url,
        law_number,
        article_section,
        issuing_agency,
        date_issued,
        effective_date,
        date_downloaded,
        document_year,
        superseding_document,
        law_version,
        category,
        json.dumps(secondary, ensure_ascii=False),
        is_active,
        version_status,
        doc_id,
    )
    category_patch = category_metadata(category, secondary)
    index_patch = {
        "document_title": document_title,
        "document_type": document_type,
        "doc_type": document_type,
        "legal_role": legal_role,
        "implements": json_list(implements),
        "topics": json_list(topics),
        "sub_intents": json_list(sub_intents),
        "source_name": source_name,
        "source_url": source_url,
        "law_number": law_number,
        "article_section": article_section,
        "issuing_agency": issuing_agency,
        "date_issued": date_issued,
        "effective_date": effective_date,
        "date_downloaded": date_downloaded,
        "document_year": document_year,
        "superseding_document": superseding_document,
        "law_version": law_version,
        **category_patch,
        "is_active": is_active,
        "version_status": version_status,
        "legal_status": version_status,
        "superseded_by": superseding_document,
    }
    await update_document_record_with_index(
        doc_id,
        update_sql,
        update_params,
        index_patch,
        document_metadata_patch_from_row(existing),
    )
    await log_kb_admin_action(
        current_user["id"],
        "update_document_metadata",
        doc_id,
        {
            "document_title": document_title,
            "document_type": document_type,
            "legal_role": legal_role,
            "implements": implements,
            "topics": topics,
            "sub_intents": sub_intents,
            "source_name": source_name,
            "source_url": source_url,
            "law_number": law_number,
            "article_section": article_section,
            "issuing_agency": issuing_agency,
            "date_issued": date_issued,
            "effective_date": effective_date,
            "date_downloaded": date_downloaded,
            "document_year": document_year,
            "superseding_document": superseding_document,
            "law_version": law_version,
            "primary_category": category,
            "secondary_categories": secondary,
            "is_active": is_active,
            "version_status": version_status,
        },
    )
    return {"message": "Updated", "document": await get_document_or_404(doc_id)}


@router.put("/bulk/metadata")
async def bulk_update_document_metadata(
    payload: BulkDocumentMetadataUpdate,
    current_user=Depends(require_admin),
):
    document_ids = list(dict.fromkeys(int(value) for value in payload.document_ids if int(value) > 0))
    if not document_ids:
        raise HTTPException(status_code=400, detail="Select at least one document")
    if len(document_ids) > 200:
        raise HTTPException(status_code=400, detail="Bulk edits are limited to 200 documents")

    updated: list[int] = []
    skipped: list[dict[str, Any]] = []
    for document_id in document_ids:
        try:
            existing = await get_document_or_404(document_id)
            if existing.get("status") in {"pending", "processing"}:
                skipped.append({"id": document_id, "reason": "Document is processing"})
                continue
            primary, secondary = normalize_document_categories(
                payload.primary_category or existing.get("primary_category"),
                (
                    payload.secondary_categories
                    if payload.secondary_categories is not None
                    else existing.get("secondary_categories", [])
                ),
            )
            legal_status = (
                normalize_version_status(payload.version_status)
                if payload.version_status is not None
                else str(existing.get("version_status") or "active")
            )
            doc_type = (
                str(payload.document_type).strip()[:120]
                if payload.document_type is not None
                else str(existing.get("document_type") or "").strip()[:120]
            )
            update_sql = """
                UPDATE uploaded_documents
                SET category=%s, secondary_categories=%s,
                    version_status=%s, document_type=%s
                WHERE id=%s
                """
            update_params = (
                primary,
                json.dumps(secondary, ensure_ascii=False),
                legal_status,
                doc_type,
                document_id,
            )
            index_patch = {
                **category_metadata(primary, secondary),
                "version_status": legal_status,
                "legal_status": legal_status,
                "document_type": doc_type,
                "doc_type": doc_type,
            }
            await update_document_record_with_index(
                document_id,
                update_sql,
                update_params,
                index_patch,
                document_metadata_patch_from_row(existing),
            )
            updated.append(document_id)
        except (HTTPException, ValueError) as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            skipped.append({"id": document_id, "reason": str(detail)})

    await log_kb_admin_action(
        current_user["id"],
        "bulk_update_document_metadata",
        0,
        {
            "updated_document_ids": updated,
            "skipped": skipped,
            "primary_category": payload.primary_category,
            "secondary_categories": payload.secondary_categories,
            "version_status": payload.version_status,
            "document_type": payload.document_type,
        },
    )
    return {"updated": updated, "skipped": skipped, "updated_count": len(updated)}


@router.delete("/documents/{doc_id}")
async def delete_doc(doc_id: int, current_user=Depends(require_admin)):
    # Serialize deletes across API workers while the file is in recovery storage.
    pool = await get_pool()
    async with pool.acquire() as connection:
        async with connection.cursor() as cursor:
            lock_name = f"laborlens:delete-document:{doc_id}"
            await cursor.execute("SELECT GET_LOCK(%s, 0)", (lock_name,))
            acquired = await cursor.fetchone()
            if not acquired or acquired[0] != 1:
                raise HTTPException(409, "Document deletion is already in progress. Please wait and refresh.")
            try:
                return await _delete_doc_locked(doc_id, current_user)
            finally:
                try:
                    await cursor.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
                except Exception:
                    logger.exception("Could not release document deletion lock %s", doc_id)


async def _delete_doc_locked(doc_id: int, current_user):
    await repair_stale_processing_documents()
    doc = await execute_one("SELECT * FROM uploaded_documents WHERE id=%s", (doc_id,))
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    if doc["status"] in {"pending", "processing"}:
        raise HTTPException(
            status_code=409,
            detail="Document is still processing. Wait until indexing finishes before deleting it.",
        )
    active_job = await get_active_replacement_job(doc_id)
    if active_job:
        raise HTTPException(
            status_code=409,
            detail=f"Document replacement is still running. Job: {active_job['job_id']}",
        )
    upload_path = get_upload_path(doc["filename"])
    # Deterministic recovery location allows a retry after a process interruption.
    recovery_path = upload_path.parent / ".pending_deletions" / f"{doc_id}_{upload_path.name}"
    if recovery_path.exists() and not upload_path.exists():
        await asyncio.to_thread(recovery_path.replace, upload_path)
    # Never remove the source file while searchable passages may remain.
    try:
        deleted_chunks = await delete_document_chunks_by_db_id(doc_id)
    except Exception as exc:
        raise HTTPException(503, "Search index deletion failed. The document file was kept. Please retry Delete.") from exc
    try:
        if upload_path.exists():
            await asyncio.to_thread(recovery_path.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(upload_path.replace, recovery_path)
        await execute_insert("DELETE FROM uploaded_documents WHERE id=%s", (doc_id,))
    except Exception as exc:
        if recovery_path.exists() and not upload_path.exists():
            try:
                await asyncio.to_thread(recovery_path.replace, upload_path)
            except OSError:
                logger.exception("Document %s retained in deletion recovery storage", doc_id)
        raise HTTPException(503, "Deletion could not finish. The source file was retained. Retry Delete, or re-index the retained document to restore search.") from exc
    cleanup_pending = False
    try:
        await delete_uploaded_file(recovery_path)
    except Exception:
        # The record and passages are gone; retain the quarantined file rather
        # than misreporting a complete physical deletion or losing recovery data.
        cleanup_pending = True
        logger.warning("Document %s deleted from KB; quarantined file cleanup pending", doc_id)
    try:
        await log_kb_admin_action(
            current_user["id"], "delete_document", doc_id,
            {"original_filename": doc["original_filename"], "deleted_chunks": deleted_chunks,
             "file_cleanup_pending": cleanup_pending},
        )
    except Exception:
        logger.exception("Document %s deleted but audit logging failed", doc_id)
    return {"message": "Deleted", "deleted_chunks": deleted_chunks,
            "file_cleanup_pending": cleanup_pending}



@router.get("/stats")
async def stats(current_user=Depends(get_current_user)):
    await repair_stale_processing_documents()
    return {
        "chroma":    get_collection_stats(),
        "documents": await execute_query(
            "SELECT status, COUNT(*) as count FROM uploaded_documents GROUP BY status"
        ),
        "categories": await execute_query(
            """
            SELECT COALESCE(category, %s) as category, COUNT(*) as count
            FROM uploaded_documents
            GROUP BY COALESCE(category, %s)
            ORDER BY category
            """,
            (DEFAULT_KB_CATEGORY, DEFAULT_KB_CATEGORY),
        ),
        "versions": await execute_query(
            """
            SELECT COALESCE(version_status, 'active') as version_status, COUNT(*) as count
            FROM uploaded_documents
            GROUP BY COALESCE(version_status, 'active')
            ORDER BY version_status
            """
        ),
    }
