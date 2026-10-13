"""Published FAQ records and their approved supporting-document versions."""
import json

from fastapi import HTTPException
from routers.services.database import execute_query, execute_insert


FAQ_FIELDS = "id,question,answer,category,language,view_count,created_at,updated_at,publication_status,source_document_ids,source_versions,source_citation"


def json_value(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def source_eligible(doc):
    return bool(doc.get("is_active")) and doc.get("status") == "indexed" and int(doc.get("chunk_count") or 0) > 0 and str(doc.get("version_status") or "active").lower() in {"active", "current"} and not str(doc.get("processing_stage") or "").startswith("replacing")


def source_version(doc):
    # Replacing or reprocessing a source requires a new human review.
    return {key: str(doc.get(key) or "") for key in ("filename", "file_hash", "index_version", "last_indexed_at", "replaced_at", "document_title", "law_number", "article_section", "source_url")}


async def flag_source_faqs(doc_ids=None):
    if doc_ids is None:
        await execute_insert("UPDATE faqs SET publication_status='needs_review' WHERE publication_status='published'")
        return
    for doc_id in set(doc_ids):
        await execute_insert(
            "UPDATE faqs SET publication_status='needs_review' WHERE publication_status='published' AND JSON_CONTAINS(CASE WHEN JSON_VALID(source_document_ids) THEN source_document_ids ELSE '[]' END, %s)",
            (json.dumps(int(doc_id)),),
        )


async def source_documents(ids):
    if not ids:
        return {}
    placeholders = ",".join(["%s"] * len(ids))
    rows = await execute_query(
        f"SELECT id,document_title,original_filename,filename,status,chunk_count,is_active,version_status,processing_stage,file_hash,index_version,last_indexed_at,replaced_at,law_number,article_section,source_url FROM uploaded_documents WHERE id IN ({placeholders})",
        tuple(ids),
    )
    return {int(doc["id"]): doc for doc in rows}


async def prepare_faq_sources(faq):
    ids = list(dict.fromkeys(faq.source_document_ids))
    if any(doc_id <= 0 for doc_id in ids):
        raise HTTPException(422, "Supporting document IDs must be positive.")
    docs = await source_documents(ids)
    if len(docs) != len(ids):
        raise HTTPException(422, "A supporting document no longer exists. Select its replacement or remove it.")
    if faq.publication_status == "published" and (not ids or any(not source_eligible(doc) for doc in docs.values())):
        raise HTTPException(422, "Publishing requires at least one indexed, active supporting document. Finish processing or choose a current source.")
    versions = {str(doc_id): source_version(docs[doc_id]) for doc_id in ids}
    return json.dumps(ids), json.dumps(versions, sort_keys=True)


async def list_faq_records(public=False):
    where = " WHERE publication_status='published'" if public else ""
    rows = await execute_query(f"SELECT {FAQ_FIELDS} FROM faqs{where} ORDER BY updated_at DESC,id DESC")
    parsed = []
    all_ids = set()
    for row in rows:
        record = dict(row)
        ids = json_value(record.get("source_document_ids"), [])
        ids = [value for value in ids if isinstance(value, int) and not isinstance(value, bool) and value > 0] if isinstance(ids, list) else []
        record["source_document_ids"] = ids
        all_ids.update(ids)
        parsed.append(record)
    docs = await source_documents(sorted(all_ids))
    result = []
    for record in parsed:
        ids = record["source_document_ids"]
        snapshots = json_value(record.get("source_versions"), {})
        valid = bool(ids) and isinstance(snapshots, dict) and all(
            doc_id in docs and source_eligible(docs[doc_id]) and snapshots.get(str(doc_id)) == source_version(docs[doc_id])
            for doc_id in ids
        )
        if record.get("publication_status") == "published" and not valid:
            # Sticky review state: restoring a document does not silently republish an FAQ.
            await execute_insert(
                "UPDATE faqs SET publication_status='needs_review' WHERE id=%s AND publication_status='published' AND source_versions <=> %s",
                (record["id"], record.get("source_versions")),
            )
            record["publication_status"] = "needs_review"
        record.pop("source_versions", None)
        record["sources"] = [
            {"id": doc_id, "title": docs[doc_id].get("document_title") or docs[doc_id].get("original_filename") or "Supporting document",
             "law_number": docs[doc_id].get("law_number"), "article_section": docs[doc_id].get("article_section"), "source_url": docs[doc_id].get("source_url")}
            for doc_id in ids if doc_id in docs
        ]
        if not public or record.get("publication_status") == "published":
            if public:
                record = {key: record.get(key) for key in ("id", "question", "answer", "category", "language", "updated_at", "sources", "source_citation")}
            result.append(record)
    return result
