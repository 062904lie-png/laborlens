"""
LaborLens – ChromaDB Service
Handles document embedding storage and semantic search.
"""

import asyncio
import math
import chromadb
from chromadb.errors import InvalidDimensionException, NotFoundError
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import BACKEND_DIR, settings
import logging
import uuid
import re
import copy
import hashlib
import json
import time
import threading
from collections import OrderedDict

from routers.services.rag_policy import (
    authority_ids_for_chunk,
    enrich_chunk_metadata,
    is_regional_wage_query as policy_is_regional_wage_query,
    osh_chunk_supports_task,
    required_authorities_for_query,
    score_chunk,
    wage_order_query_details,
)
from routers.services.legal_topic_matrix import authority_search_terms, detect_supported_concepts, supported_passage_concepts
from routers.services.document_categories import (
    MAX_SECONDARY_CATEGORY_SLOTS,
    chroma_safe_metadata,
)
from routers.services.lda_topic_reranker import (
    invalidate_lda_topic_model,
    register_lda_document_loader,
    register_lda_chunk_page_loader,
    schedule_lda_topic_model_build,
    lda_training_chunk_limit,
    strip_chunk_metadata_header,
)
from routers.services.performance import measure, set_timing_metadata

logger = logging.getLogger(__name__)


class EmbeddingMismatchError(RuntimeError):
    """Incompatible embeddings must never trigger deletion of stored evidence."""


def embedding_mismatch(operation: str) -> EmbeddingMismatchError:
    message = (
        f"Knowledge-base embedding mismatch during {operation}. Existing index preserved. "
        "Restore the embedding model used to build this index, or perform a planned "
        "reindex into a separate collection."
    )
    logger.error(message)
    return EmbeddingMismatchError(message)


_client     = None
_collection = None
_embedder: Optional[SentenceTransformer] = None
_last_reset_reason: Optional[str] = None
_authority_lookup_cache: Dict[tuple, List[Dict[str, Any]]] = {}
_embedding_cache: OrderedDict[str, tuple[float, List[float]]] = OrderedDict()
_search_cache: OrderedDict[str, tuple[float, List[Dict[str, Any]]]] = OrderedDict()
_cache_generation = 0
_query_embedding_tasks = {}
_document_chunk_counts_cache: Optional[tuple[int, Dict[int, int]]] = None
_document_chunk_counts_cache_lock = threading.Lock()
EMBEDDING_BATCH_SIZE = 8
CHROMA_WRITE_BATCH_SIZE = 32
_embedding_inference_lock = threading.Lock()


def resolved_chroma_persist_dir() -> str:
    configured = Path(str(settings.CHROMA_PERSIST_DIR or "chroma_db")).expanduser()
    if not configured.is_absolute():
        configured = BACKEND_DIR / configured
    return str(configured.resolve())


def invalidate_retrieval_caches(*, warm_body_index=True) -> None:
    """Invalidate only document-level caches; never cache user conversations."""

    global _cache_generation, _document_chunk_counts_cache
    _authority_lookup_cache.clear()
    _embedding_cache.clear()
    _search_cache.clear()
    _cache_generation += 1
    _document_chunk_counts_cache = None
    from .body_text_index import invalidate
    invalidate()
    # The optional LDA model learns from the same stored chunk corpus.
    invalidate_lda_topic_model()
    if warm_body_index:
        schedule_body_text_index()


def schedule_body_text_index():
    from .body_text_index import schedule_build
    collection, generation = _collection, _cache_generation
    if collection is not None:
        schedule_build(collection, generation, lambda: (
            collection is _collection and generation == _cache_generation))


def _scan_document_chunk_counts() -> Dict[int, int]:
    """Build a per-document count snapshot from Chroma without loading chunk text."""
    collection = get_collection()
    generation = _cache_generation
    total = collection.count()
    counts: Dict[int, int] = {}
    offset = 0
    page_size = 512

    while offset < total:
        limit = min(page_size, total - offset)
        page = collection.get(offset=offset, limit=limit, include=["metadatas"])
        ids = page.get("ids") or []
        metadatas = page.get("metadatas") or []
        if not ids:
            raise RuntimeError("The vector index changed while document status was being read.")
        for index, metadata in enumerate(metadatas):
            if not isinstance(metadata, dict):
                continue
            try:
                doc_id = int(str(metadata.get("doc_db_id") or "").strip())
            except (TypeError, ValueError):
                continue
            if doc_id > 0:
                counts[doc_id] = counts.get(doc_id, 0) + 1
        offset += len(ids)

    if collection.count() != total or generation != _cache_generation:
        raise RuntimeError("The vector index changed while document status was being read.")
    return counts


def _get_document_chunk_counts_cached() -> Dict[int, int]:
    global _document_chunk_counts_cache
    with _document_chunk_counts_cache_lock:
        generation = _cache_generation
        cached = _document_chunk_counts_cache
        if cached is not None and cached[0] == generation:
            return dict(cached[1])
        counts = _scan_document_chunk_counts()
        if generation != _cache_generation:
            raise RuntimeError("The vector index changed while document status was being read.")
        _document_chunk_counts_cache = (generation, counts)
        return dict(counts)


async def get_document_chunk_counts_by_db_id() -> Dict[int, int]:
    """Return a cached, exact map of Chroma chunks grouped by database document ID."""
    return await asyncio.to_thread(_get_document_chunk_counts_cached)


def _cache_key(kind: str, query: str, **values: Any) -> str:
    normalized = re.sub(r"\s+", " ", query or "").strip().casefold()
    payload = json.dumps(
        {
            "kind": kind,
            "query": normalized,
            "generation": _cache_generation,
            "model": settings.EMBEDDING_MODEL,
            **values,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(cache: OrderedDict, key: str):
    item = cache.get(key)
    if item is None:
        return None
    created, value = item
    if time.monotonic() - created > float(settings.LABORLENS_RETRIEVAL_CACHE_TTL_SECONDS):
        cache.pop(key, None)
        return None
    cache.move_to_end(key)
    return copy.deepcopy(value)


def _cache_put(cache: OrderedDict, key: str, value: Any) -> None:
    cache[key] = (time.monotonic(), copy.deepcopy(value))
    cache.move_to_end(key)
    limit = max(16, min(int(settings.LABORLENS_RETRIEVAL_CACHE_SIZE), 2048))
    while len(cache) > limit:
        cache.popitem(last=False)


def uses_e5_model() -> bool:
    return "e5" in settings.EMBEDDING_MODEL.lower()


def prepare_embedding_texts(texts: List[str], input_type: str) -> List[str]:
    if not uses_e5_model():
        return texts
    prefix = "query: " if input_type == "query" else "passage: "
    return [
        text if text.lower().startswith(("query: ", "passage: ")) else f"{prefix}{text}"
        for text in texts
    ]


async def init_chroma():
    global _client, _collection, _embedder
    invalidate_retrieval_caches()
    _client = chromadb.PersistentClient(
        path=resolved_chroma_persist_dir(),
        settings=ChromaSettings(
            anonymized_telemetry=False,
            chroma_product_telemetry_impl="routers.services.chroma_telemetry.DisabledProductTelemetry",
        )
    )
    # Read existing metadata without replacing it with the configured model.
    _embedder = None
    try:
        _collection = _client.get_collection(name=settings.CHROMA_COLLECTION_NAME)
    except NotFoundError:
        _collection = _client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "embedding_model": settings.EMBEDDING_MODEL},
        )
    stored_model = (_collection.metadata or {}).get("embedding_model")
    if stored_model != settings.EMBEDDING_MODEL:
        raise embedding_mismatch("startup")
    _embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
    logger.info(f"✅ ChromaDB ready. Chunks in collection: {_collection.count()}")
    register_lda_document_loader(_load_lda_training_documents)
    register_lda_chunk_page_loader(_load_lda_chunk_page)
    # Build in the background: server startup and the first chat request stay
    # responsive even when the existing KB has many chunks.
    schedule_lda_topic_model_build()
    schedule_body_text_index()


def reset_collection(reason: str):
    """Explicit administrative rebuild only; never call for automatic recovery."""
    global _collection, _last_reset_reason
    if _client is None:
        raise RuntimeError("ChromaDB not initialized.")
    invalidate_retrieval_caches()
    logger.warning(
        "Resetting Chroma collection '%s': %s",
        settings.CHROMA_COLLECTION_NAME,
        reason,
    )
    try:
        _client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
    except Exception as exc:
        logger.warning("Could not delete old Chroma collection before reset: %s", exc)
    _collection = _client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": settings.EMBEDDING_MODEL,
        },
    )
    _last_reset_reason = reason
    invalidate_retrieval_caches()
    return _collection


def get_collection():
    if _collection is None:
        raise RuntimeError("ChromaDB not initialized.")
    return _collection

def _lda_training_chunk_limit() -> int:
    return lda_training_chunk_limit(settings)


def _load_lda_training_documents_sync(active_versions=None) -> List[str]:
    """Page through published document versions, without embedding calls/writes.

    Deduplicate normalized bodies, not differing metadata headers. Unregistered,
    inactive, staging and stale index versions never train the topic model.
    """

    collection = get_collection()
    total = max(0, int(collection.count() or 0))
    if not total:
        return []
    limit = _lda_training_chunk_limit()
    batch_size = max(16, min(512, int(settings.LDA_TRAINING_BATCH_SIZE)))
    active_versions = active_versions or {}

    documents: List[str] = []
    seen: set[str] = set()
    for offset in range(0, total, batch_size):
        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        bodies, metas = results.get('documents') or [], results.get('metadatas') or []
        if not bodies or len(bodies) != len(metas):
            raise ValueError('incomplete_lda_training_page')
        for document, meta in zip(bodies, metas):
            meta = meta or {}
            owner = str(meta.get('doc_db_id') or '')
            if owner not in active_versions or str(meta.get('index_version') or 1) != active_versions[owner]:
                continue
            if str(meta.get('is_active', True)).lower() in {'false', '0', 'no'}:
                continue
            status = str(meta.get('legal_status') or meta.get('version_status') or '').lower()
            if status not in {'', 'active', 'current', 'approved', 'indexed'}:
                continue
            text = strip_chunk_metadata_header(document)
            key = text.casefold()
            if len(re.findall(r'[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ-]{2,}', text)) >= 6 and key not in seen:
                seen.add(key)
                documents.append(text)
            if limit and len(documents) >= limit:
                return documents
    return documents


async def _load_lda_training_documents() -> List[str]:
    from .database import execute_query
    rows = await execute_query("""SELECT id,index_version FROM uploaded_documents
        WHERE is_active=1 AND status='indexed'
        AND COALESCE(version_status, 'active') IN ('active','current')""")
    versions = {str(row['id']): str(row.get('index_version') or 1) for row in rows}
    return await asyncio.to_thread(_load_lda_training_documents_sync, versions)


def _load_lda_chunk_page_sync(offset: int, limit: int) -> Dict[str, Any]:
    """Read text/IDs only: no embedding calls or Chroma metadata mutations."""
    collection = get_collection()
    total = max(0, int(collection.count() or 0))
    records = collection.get(offset=max(0, offset), limit=max(1, min(limit, 512)),
                             include=["documents"])
    ids, documents = records.get("ids") or [], records.get("documents") or []
    if len(ids) != len(documents):
        raise ValueError("incomplete_chunk_page")
    return {"total": total, "records": [{"id": identity, "text": text or ""}
                                         for identity, text in zip(ids, documents)]}


async def _load_lda_chunk_page(offset: int, limit: int) -> Dict[str, Any]:
    return await asyncio.to_thread(_load_lda_chunk_page_sync, offset, limit)


def embed_texts(texts: List[str], input_type: str = "passage") -> List[List[float]]:
    prepared = prepare_embedding_texts(texts, input_type)
    # Prevent concurrent encoder activation buffers during uploads and chat.
    with _embedding_inference_lock:
        return _embedder.encode(
            prepared,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
        ).tolist()


async def add_documents(
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    doc_id: str
) -> int:
    invalidate_retrieval_caches(warm_body_index=False)
    collection = get_collection()
    if not texts:
        return 0
    ids        = [f"{doc_id}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(texts))]
    metadatas = [
        chroma_safe_metadata({
            **(metadatas[i] if i < len(metadatas) and metadatas[i] else {}),
            "doc_id": doc_id,
            "document_id": doc_id,
            "chunk_id": ids[i],
        })
        for i in range(len(texts))
    ]
    # Encode and persist bounded groups instead of holding a large scanned PDF's
    # entire embedding matrix in memory. SentenceTransformer also uses a larger
    # inference batch inside each group to reduce per-batch CPU overhead.
    try:
        for start in range(0, len(texts), CHROMA_WRITE_BATCH_SIZE):
            end = min(start + CHROMA_WRITE_BATCH_SIZE, len(texts))
            batch_texts = texts[start:end]
            batch_metadatas = metadatas[start:end]
            batch_ids = ids[start:end]
            embeddings = await asyncio.to_thread(embed_texts, batch_texts, "passage")
            try:
                await asyncio.to_thread(
                    collection.add,
                    documents=batch_texts,
                    embeddings=embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )
            except InvalidDimensionException as exc:
                raise embedding_mismatch("document indexing") from exc
    finally:
        # A query during batching can cache a partial index. Invalidate on
        # completion/failure as well, so it cannot mask the finished upload.
        invalidate_retrieval_caches()
    logger.info(f"Added {len(texts)} chunks for doc_id={doc_id}")
    schedule_lda_topic_model_build()
    return len(texts)


async def query_embedding(query: str):
    """Share simultaneous query embeddings across vector/text/neighbor searches."""
    key = _cache_key('embedding', query)
    cached = _cache_get(_embedding_cache, key)
    if cached is not None:
        set_timing_metadata(embedding_cache_hit=True)
        return cached
    loop = asyncio.get_running_loop()
    task_key = (loop, key)
    task = _query_embedding_tasks.get(task_key)
    if task is None:
        generation = _cache_generation
        async def compute():
            with measure('embedding_ms'):
                vector = (await asyncio.to_thread(embed_texts, [query], 'query'))[0]
            if generation == _cache_generation:
                _cache_put(_embedding_cache, key, vector)
            return vector
        task = asyncio.create_task(compute())
        _query_embedding_tasks[task_key] = task
        def finished(done):
            if _query_embedding_tasks.get(task_key) is done:
                _query_embedding_tasks.pop(task_key, None)
            if not done.cancelled():
                done.exception()  # Consume failures even if all waiting requests cancelled.
        task.add_done_callback(finished)
    else:
        set_timing_metadata(embedding_work_shared=True)
    # One disconnected request must not cancel another request's shared work.
    return await asyncio.shield(task)


async def rescore_retrieval_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare probe discoveries on one full-question cosine scale.

    Short probes are for recall. Their high similarity is not comparable to a
    different query's similarity. Read existing vectors, never re-embed or
    modify documents. Registry/applicability and final checking remain separate.
    """
    if not chunks or _collection is None:
        return chunks
    import math
    try:
        identities = list(dict.fromkeys(str(c.get('id') or '') for c in chunks if c.get('id')))
        if not identities:
            return chunks
        records = await asyncio.to_thread(_collection.get, ids=identities, include=['embeddings'])
        vectors = records.get('embeddings')
        if vectors is None:
            return chunks
        by_id = dict(zip(records.get('ids') or [], vectors))
        key = _cache_key('embedding', query)
        query_vector = await query_embedding(query)
        norm = math.sqrt(sum(float(v) ** 2 for v in query_vector))
        if not norm:
            return chunks
        output = []
        for chunk in chunks:
            vector = by_id.get(str(chunk.get('id') or ''))
            if vector is None or len(vector) != len(query_vector):
                output.append(chunk)
                continue
            other = math.sqrt(sum(float(v) ** 2 for v in vector))
            if not other:
                output.append(chunk)
                continue
            scored = {**chunk, 'score': round(sum(float(a) * float(b) for a, b in
                      zip(query_vector, vector)) / (norm * other), 4)}
            # A lexical bonus from another probe cannot outweigh full-question
            # relevance either. Canonical literal-search bonuses remain valid.
            if chunk.get('score_query_key') != key:
                scored.pop('lexical_score', None)
            scored['score_query_key'] = key
            output.append(scored)
        return output
    except InvalidDimensionException as exc:
        raise embedding_mismatch('candidate rescoring') from exc
    except Exception as exc:
        logger.warning('Full-question candidate rescoring unavailable: %s', type(exc).__name__)
        return chunks


async def search_documents(
    query: str,
    n_results: int = 6,
    filter_metadata: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    collection = get_collection()
    search_key = _cache_key(
        "search",
        query,
        n_results=int(n_results),
        filter_metadata=filter_metadata or {},
    )
    cached_results = _cache_get(_search_cache, search_key)
    if cached_results is not None:
        set_timing_metadata(retrieval_cache_hit=True)
        return cached_results

    embedding_key = _cache_key('embedding', query)
    query_embed = await query_embedding(query)
    try:
        results = await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_embed],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )
    except InvalidDimensionException as exc:
        raise embedding_mismatch("search") from exc
    output = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            result_id = results.get("ids", [[]])[0][i] if results.get("ids") else ""
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            if metadata is None:
                metadata = {}
            if result_id and not metadata.get("chunk_id"):
                metadata = {**metadata, "chunk_id": result_id}
            output.append({
                "id":       result_id,
                "text":     doc,
                "metadata": metadata,
                "score":    round(1 - results["distances"][0][i], 4),
                "score_query_key": embedding_key,
            })
    _cache_put(_search_cache, search_key, output)
    return output


def document_text_probes(query: str) -> List[str]:
    """Derive bounded lexical probes from words actually supplied, not topics."""
    from .focused_retrieval_queries import literal_search_probes
    return literal_search_probes(query, limit=5)


async def search_document_text(query: str, n_results: int = 20, *, fast: bool = False) -> List[Dict[str, Any]]:
    """Supplement vectors with literal passage-body evidence.

    Metadata headers can dominate vector similarity. Lexical discovery checks
    the actual body and uses existing vectors for similarity; it never invents
    a cosine score or treats a keyword match as verified legal support.
    """
    if fast:
        return await search_lexical_documents(query, n_results)
    from .issue_evidence import issue_terms
    from .legal_topic_matrix import source_passage_text
    from .focused_retrieval_queries import question_focus
    if _collection is None or _embedder is None:
        return []
    probes = document_text_probes(query)
    if not probes:
        return []
    key = _cache_key('passage_text', query, n_results=n_results)
    cached = _cache_get(_search_cache, key)
    if cached is not None:
        return cached
    generation = _cache_generation
    focus = question_focus(query)
    query_terms = issue_terms(query)
    anchor_terms = issue_terms(' '.join(focus['anchors']))
    def lookup():
        candidates = {}
        from .body_text_index import candidates as body_candidates
        # Unlike substring probes in Chroma's stored text, this supplemental
        # index never matches injected titles/concept metadata. Fetch vectors
        # only for bounded hits, not for the whole corpus.
        body_hits = body_candidates(_collection, generation, query, limit=n_results)
        body_scores = dict(body_hits)
        if body_hits:
            records = _collection.get(ids=list(body_scores), include=['documents', 'metadatas', 'embeddings'])
            vectors = records.get('embeddings')
            for index, chunk in enumerate(_records_from_get_results(records, retrieval_method='question_text', enrich_metadata=False)):
                body = source_passage_text(chunk.get('text') or '').casefold()
                candidates[chunk['id']] = (chunk, vectors[index] if vectors is not None else None,
                                           issue_terms(body) & query_terms, body)
        # The full body index already searched every stored passage. Repeating
        # substring scans is useful only when it could recover a missing
        # explicit subject/location or an underfilled candidate set.
        from .evidence_selection import discovery_terms
        focus_terms = [discovery_terms(value) for value in [*focus['subjects'], *focus['anchors']]]
        has_focus = all(any(wanted <= discovery_terms(value[3]) for value in candidates.values())
                        for wanted in focus_terms if wanted)
        scan_probes = probes if len(candidates) < min(100, n_results) or not has_focus else []
        for probe in scan_probes:
            variants = list(dict.fromkeys((probe, probe.title(), probe.upper())))
            where = {'$or': [{'$contains': value} for value in variants]} if len(variants) > 1 else {'$contains': probe}
            records = _collection.get(where_document=where, limit=250,
                                      include=['documents', 'metadatas', 'embeddings'])
            vectors = records.get('embeddings')
            for index, chunk in enumerate(_records_from_get_results(
                    records, retrieval_method='question_text', enrich_metadata=False)):
                body = source_passage_text(chunk.get('text') or '').casefold()
                matched = issue_terms(body) & issue_terms(query)
                if not matched or not any(p in body for p in probes):
                    continue
                candidates[chunk['id']] = (chunk, vectors[index] if vectors is not None else None, matched, body)
        # Use specificity in this bounded candidate corpus. Clamping overlap
        # plus phrase bonuses to 1 made passages from unrelated regions tie
        # with an exact user location and win by collection insertion order.
        frequencies = {term: sum(term in value[2] for value in candidates.values()) for term in query_terms}
        weights = {term: 1 + math.log((len(candidates) + 1) / (count + 1))
                   for term, count in frequencies.items() if count}
        total = sum(weights.values()) or 1.0
        anchor_total = sum(weights.get(term, 0) for term in anchor_terms)
        phrases = [probe for probe in probes if ' ' in probe]
        scored = []
        for chunk, vector, matched, body in candidates.values():
            coverage = sum(weights.get(term, 0) for term in matched) / total
            phrase_coverage = sum(probe in body for probe in phrases) / max(1, len(phrases))
            if anchor_total:
                anchor_coverage = sum(weights.get(term, 0) for term in matched & anchor_terms) / anchor_total
                lexical = .65 * coverage + .20 * anchor_coverage + .15 * phrase_coverage
            else:
                lexical = .80 * coverage + .20 * phrase_coverage
            scored.append((chunk, vector, max(lexical, body_scores.get(chunk['id'], 0))))
        return sorted(scored, key=lambda value: value[2], reverse=True)[:max(1, int(n_results))]
    try:
        candidates = await asyncio.to_thread(lookup)
        if not candidates:
            return []
        embedding_key = _cache_key('embedding', query)
        query_vector = await query_embedding(query)
        norm = math.sqrt(sum(float(v) ** 2 for v in query_vector))
        output = []
        for chunk, vector, lexical in candidates:
            if vector is None or len(vector) != len(query_vector) or not norm:
                continue
            other_norm = math.sqrt(sum(float(v) ** 2 for v in vector))
            if not other_norm:
                continue
            chunk['score'] = round(sum(float(a) * float(b) for a, b in zip(query_vector, vector)) / (norm * other_norm), 4)
            chunk['lexical_score'] = round(min(1.0, lexical), 4)
            chunk['score_query_key'] = embedding_key
            output.append(chunk)
        if generation == _cache_generation:
            _cache_put(_search_cache, key, output)
        return output
    except Exception as exc:
        # Global vector discovery remains available if supplemental lookup fails.
        logger.warning('Supplemental document-text search unavailable: %s', type(exc).__name__)
        return []


async def search_lexical_documents(query: str, n_results: int = 20) -> List[Dict[str, Any]]:
    """Rank all indexed bodies; use separate probes while warm-up is pending."""
    if _collection is None:
        return []
    collection, generation = _collection, _cache_generation
    key = _cache_key('fast_body_text', query, n_results=n_results)
    cached = _cache_get(_search_cache, key)
    if cached is not None:
        return cached
    from .body_text_index import cached_candidates, terms, expanded_query
    from .legal_topic_matrix import source_passage_text
    probes = document_text_probes(query)

    def lookup():
        hits = cached_candidates(collection, generation, query, limit=n_results)
        if hits is not None:
            if not hits:
                return []
            records = collection.get(ids=[identity for identity, _ in hits], include=['documents', 'metadatas'])
        else:
            # A broad OR followed by LIMIT lets one frequent word crowd out
            # every other probe. Give each phrase/word its own candidate batch.
            pooled = {}
            for probe in probes:
                variants = list(dict.fromkeys((probe, probe.title(), probe.upper())))
                where = {'$or': [{'$contains': value} for value in variants]} if len(variants) > 1 else {'$contains': probe}
                batch = collection.get(where_document=where, limit=100, include=['documents', 'metadatas'])
                for identity, body, metadata in zip(batch.get('ids') or [], batch.get('documents') or [], batch.get('metadatas') or []):
                    pooled[identity] = (body, metadata)
            records = dict(ids=list(pooled), documents=[v[0] for v in pooled.values()],
                           metadatas=[v[1] for v in pooled.values()])
        wanted = terms(expanded_query(collection, generation, query))
        rows = _records_from_get_results(records, score=0.0, retrieval_method='question_text', enrich_metadata=False)
        selected = []
        for row in rows:
            body_terms = terms(source_passage_text(row.get('text') or ''))
            matched = wanted & body_terms
            if not matched:
                continue  # A match in injected headers is not passage evidence.
            row['lexical_score'] = len(matched) / max(1, len(wanted))
            selected.append(row)
        if hits is not None:
            # Chroma get(ids=...) does not promise the ranked request order.
            positions = {identity: rank for rank, (identity, _) in enumerate(hits)}
            return sorted(selected, key=lambda row: positions[row['id']])[:max(1, int(n_results))]
        return sorted(selected, key=lambda row: (-row['lexical_score'], row['id']))[:max(1, int(n_results))]

    rows = await asyncio.to_thread(lookup)
    if generation == _cache_generation and collection is _collection:
        _cache_put(_search_cache, key, rows)
    return rows


async def search_document_neighbors(query: str, seeds: List[Dict[str, Any]], n_results: int = 8) -> List[Dict[str, Any]]:
    """Read bounded surrounding pages of actual hits, without title/topic lists.

    Physical document ID and index version keep replacements isolated. Neighbor
    discovery is not evidence certification; publication/scope and ranking still
    apply in the caller. Scores use existing embeddings, not inherited scores.
    """
    if _collection is None or _embedder is None:
        return []
    owners, seen = [], set()
    for seed in seeds:
        meta = seed.get('metadata') or {}
        try:
            page = int(meta.get('page') or 0)
        except (TypeError, ValueError):
            continue
        owner = meta.get('doc_db_id')
        version = meta.get('index_version')
        identity = (str(owner), str(version))
        if owner is None or page < 1 or identity in seen:
            continue
        seen.add(identity)
        owners.append((owner, version, page))
        if len(owners) == 2:
            break
    if not owners:
        return []
    seed_ids = {str(c.get('id') or (c.get('metadata') or {}).get('chunk_id') or '') for c in seeds}

    def lookup():
        found = {}
        for owner, version, page in owners:
            pages = list(range(max(1, page - 1), page + 2))
            fields = [{'doc_db_id': str(owner)}, {'$or': [
                {'page': {'$in': pages}}, {'page': {'$in': [str(p) for p in pages]}}]}]
            if version not in (None, ''):
                fields.append({'index_version': str(version)})
            records = _collection.get(where={'$and': fields}, limit=48,
                                      include=['documents', 'metadatas', 'embeddings'])
            vectors = records.get('embeddings')
            for index, chunk in enumerate(_records_from_get_results(
                    records, score=0, retrieval_method='document_neighbor', enrich_metadata=False)):
                if chunk['id'] in seed_ids:
                    continue
                # Recheck identity even if a connector/mocked store ignores filters.
                meta = chunk['metadata']
                if str(meta.get('doc_db_id')) != str(owner) or (version not in (None, '')
                        and str(meta.get('index_version')) != str(version)):
                    continue
                try:
                    if int(meta.get('page') or 0) not in pages:
                        continue
                except (TypeError, ValueError):
                    continue
                found[chunk['id']] = (chunk, vectors[index] if vectors is not None else None)
        return list(found.values())

    try:
        found = await asyncio.to_thread(lookup)
        if not found:
            return []
        key = _cache_key('embedding', query)
        vector = await query_embedding(query)
        norm = math.sqrt(sum(float(v) ** 2 for v in vector))
        output = []
        for chunk, embedding in found:
            if embedding is None or len(embedding) != len(vector) or not norm:
                continue
            other = math.sqrt(sum(float(v) ** 2 for v in embedding))
            if not other:
                continue
            chunk['score'] = round(sum(float(a) * float(b) for a, b in zip(vector, embedding)) / (norm * other), 4)
            output.append(chunk)
        return sorted(output, key=lambda c: c['score'], reverse=True)[:max(1, min(n_results, 12))]
    except Exception as exc:
        logger.warning('Surrounding document search unavailable: %s', type(exc).__name__)
        return []


async def search_document_passages(
    query: str,
    seeds: List[Dict[str, Any]],
    n_results: int = 4,
    max_documents: int = 4,
) -> List[Dict[str, Any]]:
    """Search farther inside documents already matched to this question.

    A rule and its rate, exceptions, or night-work details may be separated by
    many pages. This bounded, query-scoped expansion searches only the first
    few actual hit documents, pinned to their current index version. It neither
    scans the whole knowledge base nor adds an LLM call.
    """
    if _collection is None or _embedder is None:
        return []
    owners, seen_owners = [], set()
    seed_ids = set()
    for seed in seeds or []:
        meta = seed.get('metadata') or {}
        identity = str(seed.get('id') or meta.get('chunk_id') or '')
        if identity:
            seed_ids.add(identity)
        owner = meta.get('doc_db_id')
        if owner in (None, ''):
            continue
        version = meta.get('index_version')
        owner_key = (str(owner), str(version or ''))
        if owner_key in seen_owners:
            continue
        seen_owners.add(owner_key)
        owners.append((str(owner), str(version) if version not in (None, '') else None))
        if len(owners) >= max(1, min(int(max_documents), 4)):
            break
    if not owners:
        return []

    per_document = max(1, min(int(n_results), 8))
    collection, generation = _collection, _cache_generation
    key = _cache_key('same_document_search', query, owners=owners,
                     per_document=per_document)
    cached = _cache_get(_search_cache, key)
    if cached is not None:
        return cached
    try:
        query_vector = await query_embedding(query)

        def lookup(owner, version):
            predicates = [{'doc_db_id': owner}]
            if version is not None:
                predicates.append({'index_version': version})
            where = {'$and': predicates} if len(predicates) > 1 else predicates[0]
            return collection.query(
                query_embeddings=[query_vector],
                n_results=per_document,
                where=where,
                include=['documents', 'metadatas', 'distances'],
            )

        batches = await asyncio.gather(*[
            asyncio.to_thread(lookup, owner, version) for owner, version in owners
        ])
        output, seen_ids = [], set(seed_ids)
        for (owner, version), batch in zip(owners, batches):
            batch = batch or {}
            ids = batch.get('ids') or [[]]
            docs = batch.get('documents') or [[]]
            metas = batch.get('metadatas') or [[]]
            distances = batch.get('distances') or [[]]
            for index, document_id in enumerate(ids[0] if ids else []):
                if not document_id or str(document_id) in seen_ids:
                    continue
                metadata = metas[0][index] if metas and metas[0] and index < len(metas[0]) else {}
                metadata = dict(metadata or {})
                if str(metadata.get('doc_db_id') or '') != owner:
                    continue
                if version is not None and str(metadata.get('index_version') or '') != version:
                    continue
                text = docs[0][index] if docs and docs[0] and index < len(docs[0]) else ''
                if not str(text or '').strip():
                    continue
                distance = distances[0][index] if distances and distances[0] and index < len(distances[0]) else 1.0
                metadata.setdefault('chunk_id', document_id)
                metadata['retrieval_method'] = 'same_document_search'
                output.append({
                    'id': document_id,
                    'text': text,
                    'metadata': metadata,
                    'score': round(1 - float(distance), 4),
                    'retrieval_method': 'same_document_search',
                })
                seen_ids.add(str(document_id))
        output.sort(key=lambda row: float(row.get('score') or 0), reverse=True)
        if generation == _cache_generation and collection is _collection:
            _cache_put(_search_cache, key, output)
        return output
    except InvalidDimensionException as exc:
        raise embedding_mismatch('same-document search') from exc
    except Exception as exc:
        logger.warning('Same-document passage search unavailable: %s', type(exc).__name__)
        return []


def is_regional_wage_query(query: str) -> bool:
    """Detect a wage question that names a Philippine region, province, or city."""

    return policy_is_regional_wage_query(query or "")


def _retrieval_identity(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    return str(
        chunk.get("id")
        or metadata.get("chunk_id")
        or f"{metadata.get('doc_id', '')}:{chunk.get('text', '')[:160]}"
    )


def category_any_filter(categories: List[str]) -> Dict[str, Any]:
    values: List[str] = []
    for category in categories:
        value = str(category or "").strip()
        if not value:
            continue
        values.extend([value, value.lower().replace(" & ", "_").replace(" / ", "_").replace(" ", "_")])
    values = list(dict.fromkeys(values))
    return {
        "$or": [
            {"category": {"$in": values}},
            *[
                {f"secondary_category_{index}": {"$in": values}}
                for index in range(1, MAX_SECONDARY_CATEGORY_SLOTS + 1)
            ],
        ]
    }


async def search_matching_regional_wage_documents(query: str) -> List[Dict[str, Any]]:
    """Read every indexed passage of matching RTWPB owners, across categories.

    SQL ownership/version state is authoritative. No fixed document IDs, wage
    values, source URLs, or LDA labels determine which documents are scanned.
    """
    from .database import execute_query
    from .regional_wage_evidence import document_regions, regional_wage_document, wage_scope_rejection, regional_wage_query
    details = wage_order_query_details(query)
    if not regional_wage_query(query):
        return []
    rows = await execute_query("""SELECT id,index_version,original_filename,document_title,
        source_url,category,version_status FROM uploaded_documents
        WHERE is_active=1 AND status='indexed'
        AND COALESCE(version_status,'active') IN ('active','current')""")
    matching = [row for row in rows
                if regional_wage_document(row)
                and details['region_code'] in document_regions(row)
                and not wage_scope_rejection(query, row)]
    collection = get_collection()
    output = []
    for row in matching:
        offset = 0
        while True:
            records = await asyncio.to_thread(collection.get,
                where={'$and': [{'doc_db_id': str(row['id'])},
                                {'index_version': str(row.get('index_version') or 1)}]},
                include=['documents', 'metadatas'], limit=128, offset=offset)
            batch = _records_from_get_results(records, score=0.6,
                retrieval_method='regional_document_scan', enrich_metadata=False)
            for chunk in batch:
                chunk['metadata'] = {**(chunk.get('metadata') or {}),
                    'region': details['region_code'], 'owning_region': details['region_code']}
            output.extend(batch)
            if len(records.get('ids') or []) < 128:
                break
            offset += 128
    return output


async def search_regional_wage_documents(query: str) -> List[Dict[str, Any]]:
    """Return four DOLE wage chunks first, followed by two general-law chunks.

    An empty result means no DOLE wage material was found and tells the caller
    to use its normal unrestricted retrieval path.
    """

    if not is_regional_wage_query(query):
        return []

    category_filter = category_any_filter(["DOLE Resources"])
    details = wage_order_query_details(query)
    region_source_url = str(details.get("region_source_url") or "").strip()
    regional_chunks: List[Dict[str, Any]] = []
    if region_source_url:
        regional_chunks = await search_documents(
            query=query,
            n_results=4,
            filter_metadata={
                "$and": [
                    category_filter,
                    {"source_url": region_source_url},
                ]
            },
        )
    category_chunks = await search_documents(
        query=query,
        n_results=4,
        filter_metadata=category_filter,
    )

    dole_chunks: List[Dict[str, Any]] = []
    seen_dole: set[str] = set()
    source_counts: Dict[str, int] = {}
    for chunk in [*regional_chunks, *category_chunks]:
        key = _retrieval_identity(chunk)
        if key in seen_dole:
            continue
        metadata = chunk.get("metadata") or {}
        source_key = str(
            metadata.get("source_url")
            or metadata.get("official_url")
            or metadata.get("document_title")
            or metadata.get("source")
            or key
        )
        if source_counts.get(source_key, 0) >= 2:
            continue
        dole_chunks.append(chunk)
        seen_dole.add(key)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        if len(dole_chunks) >= 4:
            break
    if not dole_chunks:
        return []

    general_chunks = await search_documents(
        query=query,
        n_results=2,
        filter_metadata=category_any_filter(["Core Labor Laws", "Employment Benefits"]),
    )

    output: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for route_name, chunks in (
        ("regional_wage_dole_boost", dole_chunks),
        ("regional_wage_supplement", general_chunks),
    ):
        for chunk in chunks:
            key = _retrieval_identity(chunk)
            if key in seen:
                continue
            metadata = {**(chunk.get("metadata") or {}), "retrieval_method": route_name}
            output.append({**chunk, "metadata": metadata})
            seen.add(key)
    return output


async def recover_deadline_passages(query: str, selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Read timing clauses from retrieved documents without supplying legal values."""
    # A date or duration in the facts (for example, a six-month probation)
    # does not itself ask for a deadline. Only run this broad same-document
    # lookup when the user is actually asking when or how long something takes.
    if not re.search(
        r"\b(?:when|deadline|how\s+(?:long|soon|many\s+(?:days?|weeks?|months?))|"
        r"within\s+how\s+many|by\s+what\s+date|kailan|gaano\s+katagal|san-o)\b",
        query,
        re.I,
    ):
        return []
    output = []
    seen = set()
    for seed in selected:
        meta = seed.get("metadata") or {}
        field = next((key for key in ("doc_id", "source_url", "source") if meta.get(key)), None)
        if not field or (field, meta[field]) in seen:
            continue
        seen.add((field, meta[field]))
        try:
            records = await asyncio.to_thread(
                get_collection().get, where={field: meta[field]},
                # Filter on the server before applying the result limit.
                # Chunk count does not indicate document length: legacy OCR
                # indexes may contain many overlapping definition chunks.
                where_document={"$or": [{"$contains": term} for term in (
                    "within", "Within", "not later than", "deadline",
                    "TIME OF PAYMENT", "time of payment",
                )]},
                include=["documents", "metadatas"], limit=48,
            )
        except Exception:
            logger.warning("Deadline context lookup failed", exc_info=True)
            continue
        for chunk in _records_from_get_results(records, score=float(seed.get("score") or 0.0), retrieval_method="same_document_deadline"):
            body = str(chunk.get("text") or "").split("\nText:\n")[-1]
            if re.search(r"\b(?:shall|must)\b[^.!?]{0,160}\b(?:within|not\s+later\s+than)\b|\bwithin\b[^.!?]{0,100}\bfrom\b", body, re.I):
                output.append(chunk)
        if len(seen) >= 3:
            break
    # Select by the user's subject within the timing clauses, rather than by
    # Chroma's insertion order. Suppress overlapping copies before the cap.
    stop = set('when should must shall within after before where what how long days months kailan dapat ibigay ang mga ako ko ng sa na pa rin the a an of my is it be to for from and'.split())
    terms = {t for t in re.findall(r'\w+', query.lower()) if len(t) > 2 and t not in stop}
    def relevance(chunk):
        body = str(chunk.get('text') or '').split('\nText:\n')[-1].lower()
        return sum(t in body for t in terms)
    best_match = max((relevance(c) for c in output), default=0)
    from routers.services.retrieval_router import _near_duplicate
    unique = []
    for chunk in sorted(output, key=relevance, reverse=True):
        if terms and not relevance(chunk):
            continue
        # A generic shared word such as "pay" must not reserve prompt space
        # for another benefit's deadline when a closer subject match exists.
        if relevance(chunk) < best_match:
            continue
        if not any(_near_duplicate(chunk, prior) for prior in unique):
            unique.append(chunk)
    return unique[:2]


def _records_from_get_results(
    results: Dict[str, Any],
    score: float = 1.0,
    retrieval_method: str = "metadata_lookup",
    enrich_metadata: bool = True,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    ids = results.get("ids") or []
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []
    for i, result_id in enumerate(ids):
        metadata = metas[i] if i < len(metas) and metas[i] else {}
        text = docs[i] if i < len(docs) else ""
        enriched = enrich_chunk_metadata(text, metadata, infer_concepts=False) if enrich_metadata else dict(metadata)
        if result_id and not enriched.get("chunk_id"):
            enriched["chunk_id"] = result_id
        enriched["retrieval_method"] = retrieval_method
        output.append({
            "id": result_id,
            "text": text,
            "metadata": enriched,
            "score": score,
        })
    return output


def record_explicitly_names_authority(
    record: Dict[str, Any],
    authority_id: str,
) -> bool:
    """Reject authority matches produced only by noisy inferred metadata.

    OCR fragments can mention an unrelated law and cause the ingestion
    enricher to assign that law as the chunk's canonical document.  Authority
    recovery must therefore require the document descriptor itself to name a
    numbered Act/Decree, or a matching Labor Code article identifier.
    """

    metadata = record.get("metadata") or {}
    authority_id = str(authority_id or "").strip().lower()
    descriptor = " ".join(
        str(metadata.get(key) or "")
        for key in (
            "document_title",
            "source",
            "source_name",
            "law_number",
            "citation",
            "official_url",
            "source_url",
        )
    )

    if authority_id == "sss_contribution_schedule_2025":
        return bool(
            re.search(r"\bsss\s+circular\s+2024[-\s]?006\b|\bci[-_/]2024[-_/]006\b", descriptor, re.I)
        )

    article_match = re.fullmatch(r"labor_code_article_(\d+)", authority_id)
    if article_match:
        article_number = article_match.group(1)
        exact_article = str(metadata.get("article_number") or "").strip() == article_number
        named_article = bool(
            re.search(
                rf"\b(?:art\.?|article)\s*{re.escape(article_number)}\b",
                f"{descriptor} {record.get('text', '')[:800]}",
                re.IGNORECASE,
            )
        )
        labor_code_source = (
            str(metadata.get("canonical_document_id") or "").strip().lower()
            == "labor_code_pd_442"
            or "labor code" in descriptor.lower()
        )
        return labor_code_source and (exact_article or named_article)

    if authority_id == "pd_851":
        return bool(
            re.search(
                r"\b(?:pd(?:\s+no\.?)?|presidential\s+decree(?:\s+no\.?)?)\s*851\b",
                descriptor,
                re.IGNORECASE,
            )
        )
    numbered_law = re.fullmatch(r"(ra|pd)_(\d+)(?:_irr)?", authority_id)
    if numbered_law:
        prefix, number = numbered_law.groups()
        descriptor = re.sub(r"[-_]+", " ", descriptor)
        label = (
            r"(?:ra(?:\s+no\.?)?|republic\s+act(?:\s+no\.?)?)"
            if prefix == "ra"
            else r"(?:pd(?:\s+no\.?)?|presidential\s+decree(?:\s+no\.?)?)"
        )
        return bool(
            re.search(
                rf"\b{label}\s*[-:]?\s*{re.escape(number)}\b",
                descriptor,
                re.IGNORECASE,
            )
        )

    return authority_id in authority_ids_for_chunk(record)


async def get_osh_documents_for_task(
    query_profile: Dict[str, Any],
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """Recover direct OSH propositions from the small OSH category.

    Several legacy RA 11058 and DO 198-18 chunks were indexed with hash-like
    authority IDs and inaccurate topic/concept metadata. Semantic search can
    therefore land on nearby definitions or committee passages while missing
    the controlling section. A bounded category lookup lets us re-normalize
    those records at read time and retain only passages that directly support
    the detected OSH task. This never broadens into unrelated KB categories.
    """

    task = str(query_profile.get("task") or "").strip().lower()
    sub_intent = str(query_profile.get("sub_intent") or "").strip().lower()
    cache_key = (_cache_generation, ('osh_task', task, sub_intent), int(limit),
                 str(query_profile.get('retrieval_query') or query_profile.get('normalized_query') or ''),
                 tuple(query_profile.get('required_concepts') or []),
                 str(query_profile.get('sector') or ''))
    cached = _authority_lookup_cache.get(cache_key)
    if cached is not None:
        return [
            {**item, "metadata": dict(item.get("metadata") or {})}
            for item in cached
        ]

    collection = get_collection()
    try:
        results = await asyncio.to_thread(
            collection.get,
            where={"category": "Occupational Safety & Health"},
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        logger.warning("Bounded OSH proposition lookup failed: %s", exc)
        return []

    records = _records_from_get_results(
        results,
        score=1.0,
        retrieval_method="osh_proposition_lookup",
    )
    responsive = [
        score_chunk(record, query_profile)
        for record in records
        if osh_chunk_supports_task(record, query_profile)
    ]
    responsive.sort(
        key=lambda item: (
            float(item.get("rerank_score", item.get("score", 0.0)) or 0.0),
            float(item.get("score") or 0.0),
        ),
        reverse=True,
    )

    requirements = required_authorities_for_query(query_profile)
    authority_order = list(dict.fromkeys(
        list(requirements.get("essential") or [])
        + list(requirements.get("supporting") or [])
    ))
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()

    # Preserve the authority hierarchy and ensure that each required
    # controlling source gets a direct, proposition-matched passage when one
    # is present in the OSH category.
    for authority_id in authority_order:
        match = next((
            record
            for record in responsive
            if str(record.get("id") or "") not in selected_ids
            and authority_id in authority_ids_for_chunk(record)
            and record_explicitly_names_authority(record, authority_id)
        ), None)
        if match is not None:
            selected.append(match)
            selected_ids.add(str(match.get("id") or ""))
        if len(selected) >= limit:
            break

    for record in responsive:
        record_id = str(record.get("id") or "")
        if record_id in selected_ids:
            continue
        selected.append(record)
        selected_ids.add(record_id)
        if len(selected) >= limit:
            break

    _authority_lookup_cache[cache_key] = [
        {**item, "metadata": dict(item.get("metadata") or {})}
        for item in selected
    ]
    return selected


async def get_departure_issue_documents(query: str) -> List[Dict[str, Any]]:
    """Bounded keyword lookup of uploaded operative departure provisions."""
    from routers.services.scenario_retrieval import is_employee_departure_query, scenario_evidence_gap
    if not is_employee_departure_query(query):
        return []
    collection = get_collection()
    records = []
    seen = set()
    # Text anchors, not predetermined answers; include notice exceptions as well.
    for phrase in ('abandonment of work', 'hold the employee liable for damages',
                   'Termination by Employee', 'without serving any notice'):
        result = await asyncio.to_thread(
            collection.get, where_document={"$contains": phrase},
            include=["documents", "metadatas"], limit=8,
        )
        for record in _records_from_get_results(result, score=0.6, retrieval_method="issue_text_lookup"):
            if record.get('id') not in seen and not scenario_evidence_gap(query, [record.get('text', '')]):
                seen.add(record.get('id'))
                records.append(record)
    return records


async def get_documents_for_concept_coverage(
    authority_ids: List[str],
    query_profile: Dict[str, Any],
    missing_concepts: List[str],
    limit: int = 6,
) -> List[Dict[str, Any]]:
    """Find missing answer concepts inside the governing source documents.

    Semantic search produces a compact first-pass set. When that set contains
    the right authority but misses a required answer clause, inspect the local
    chunks for that authority and retain only chunks that state the missing
    concepts. This is intentionally source-scoped, not a whole-KB scan.
    """

    normalized_ids = tuple(dict.fromkeys(
        str(authority_id or "").strip().lower()
        for authority_id in authority_ids or []
        if str(authority_id or "").strip()
    ))
    topic = str(query_profile.get("topic") or "").strip().lower()
    task = str(query_profile.get("task") or "").strip().lower()
    normalized_query = re.sub(
        r"\s+",
        " ",
        str(query_profile.get("retrieval_query") or query_profile.get("normalized_query") or ""),
    ).strip().lower()[:750]
    target_concepts = tuple(dict.fromkeys(
        str(concept or "").strip()
        for concept in missing_concepts or []
        if str(concept or "").strip()
    ))
    limit = max(1, int(limit))
    if not normalized_ids or not topic or not target_concepts:
        return []

    cache_key = (
        _cache_generation,
        "concept_coverage",
        normalized_ids,
        topic,
        task,
        normalized_query,
        target_concepts,
        limit,
    )
    cached = _authority_lookup_cache.get(cache_key)
    if cached is not None:
        return [
            {**item, "metadata": dict(item.get("metadata") or {})}
            for item in cached
        ]

    collection = get_collection()
    candidates: List[tuple[Dict[str, Any], set[str], int]] = []
    seen_ids: set[str] = set()
    target_set = set(target_concepts)
    lookup_results: Dict[str, Dict[str, Any]] = {}

    for authority_position, authority_id in enumerate(normalized_ids):
        lookups: List[Dict[str, Any]] = [
            {"normalized_authority_id": authority_id},
            {"canonical_document_id": authority_id},
        ]
        article_match = re.fullmatch(r"labor_code_article_(\d+)", authority_id)
        if article_match:
            lookups.extend([
                {"article_number": article_match.group(1)},
                {"canonical_document_id": "labor_code_pd_442"},
            ])

        for where in lookups:
            try:
                # Chroma does not rank metadata-get results. Read every local
                # chunk for the known authority so an answer clause on a later
                # page cannot be lost to the normal top-k cutoff.
                lookup_key = json.dumps(where, sort_keys=True)
                if lookup_key not in lookup_results:
                    lookup_results[lookup_key] = await asyncio.to_thread(
                        collection.get, where=where, include=["documents", "metadatas"],
                    )
                results = lookup_results[lookup_key]
            except Exception as exc:
                logger.debug(
                    "Concept coverage lookup failed for %s using %s: %s",
                    authority_id,
                    where,
                    exc,
                )
                continue

            for record in _records_from_get_results(
                results,
                score=1.0,
                retrieval_method="required_authority_lookup",
                enrich_metadata=False,
            ):
                record_id = str(
                    record.get("id")
                    or (record.get("metadata") or {}).get("chunk_id")
                    or record.get("text", "")[:160]
                )
                if record_id in seen_ids:
                    continue
                if not record_explicitly_names_authority(record, authority_id):
                    continue
                status = str(
                    (record.get("metadata") or {}).get("legal_status")
                    or (record.get("metadata") or {}).get("status")
                    or ""
                ).strip().lower()
                if status and status not in {"active", "current", "approved", "indexed"}:
                    continue

                # Cheap, question-specific body checks precede enrichment.
                # A raw metadata scan must not run every topic's regexes over
                # thousands of non-responsive paragraphs on the event loop.
                if not supported_passage_concepts(record, target_set):
                    continue
                from routers.services.legal_topic_matrix import bounded_evidence_passage
                record = {**record, "text": bounded_evidence_passage(record.get("text") or "", target_set)}
                supported = supported_passage_concepts(record, target_set)
                recovered = target_set.intersection(supported)
                if not recovered:
                    continue
                if authority_id not in authority_ids_for_chunk(record):
                    continue

                scored = score_chunk(record, query_profile)
                scored["metadata"] = {
                    **(scored.get("metadata") or {}),
                    "retrieval_method": "required_authority_lookup",
                    "concept_coverage_recovery": True,
                }
                candidates.append((scored, recovered, authority_position))
                seen_ids.add(record_id)

    selected: List[Dict[str, Any]] = []
    remaining = set(target_set)
    while candidates and remaining and len(selected) < limit:
        best_index = max(
            range(len(candidates)),
            key=lambda index: (
                len(candidates[index][1].intersection(remaining)),
                len(candidates[index][1]),
                str((candidates[index][0].get("metadata") or {}).get("legal_status") or "") == "current",
                float(candidates[index][0].get("rerank_score") or 0),
                -candidates[index][2],
                len(str(candidates[index][0].get("text") or "")),
            ),
        )
        record, supported, _authority_position = candidates.pop(best_index)
        if not supported.intersection(remaining):
            break
        selected.append(record)
        remaining.difference_update(supported)

    _authority_lookup_cache[cache_key] = [
        {**item, "metadata": dict(item.get("metadata") or {})}
        for item in selected
    ]
    return selected

async def get_documents_by_authority(
    authority_ids: List[str],
    limit_per_authority: int = 2,
    query: str = "",
) -> List[Dict[str, Any]]:
    """Fetch governing KB documents by metadata, then rank them for this query.

    ``authority_ids`` comes from the topic/source configuration.  The caller's
    query only chooses the most relevant passages within those documents; it
    never supplies a prewritten answer or legal conclusion.
    """

    normalized_ids = tuple(
        str(authority_id or "").strip().lower()
        for authority_id in authority_ids or []
        if str(authority_id or "").strip()
    )
    normalized_query = re.sub(r"\s+", " ", str(query or "")).strip().lower()[:750]
    cache_key = (_cache_generation, normalized_ids, int(limit_per_authority), normalized_query)
    cached = _authority_lookup_cache.get(cache_key)
    if cached is not None:
        return [
            {**item, "metadata": dict(item.get("metadata") or {})}
            for item in cached
        ]

    collection = get_collection()
    output: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for authority_id in authority_ids or []:
        authority_id = str(authority_id or "").strip().lower()
        if not authority_id:
            continue

        lookups: List[Dict[str, Any]] = [{"normalized_authority_id": authority_id}]
        if authority_id == "sss_contribution_schedule_2025":
            # Older OCR chunks have content-hash canonical IDs. Recover this
            # exact official schedule by its document heading, then normalize
            # and validate each result against the source descriptor.
            lookups.extend([
                {"__where_document": "SSS Circular 2024-006"},
                {"__where_document": "SCHEDULE OF SSS CONTRIBUTIONS"},
            ])
        article_match = re.fullmatch(r"labor_code_article_(\d+)", authority_id)
        if article_match:
            article_number = article_match.group(1)
            lookups.extend([
                {"article_number": article_number},
                {"canonical_document_id": "labor_code_pd_442"},
            ])
        else:
            lookups.append({"canonical_document_id": authority_id})
            order_match = re.fullmatch(r"dole_do_(\d{2,3})_(\d{2})", authority_id)
            if order_match:
                number, year = order_match.groups()
                # Some existing DOLE order chunks have stale canonical IDs.
                # Search their document text by the order's own number, then
                # re-identify candidates from title/source metadata below.
                order_passages = {
                    "dole_do_150_16": ("reserved status", "six (6) months", "after a period of six"),
                    "dole_do_215_20": ("payment of wages", "suspended employment"),
                }.get(authority_id)
                document_terms = order_passages or (f"{number}-{year}", f"{number} {year}")
                lookups.extend({"__where_document": term} for term in document_terms)

        search_terms = authority_search_terms(authority_id).lower()
        ranking_text = " ".join(part for part in (search_terms, normalized_query) if part)
        search_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", ranking_text)
            if len(token) > 2 or token.isdigit()
        }
        salary_match = re.search(
            r"\b(?:earns?|salary|wages?|compensation|monthly\s+pay|basic\s+pay)\b"
            r"[^.!?\n]{0,48}?(?:₱|php|p\s*)?\s*([\d,]+(?:\.\d{1,2})?)",
            normalized_query,
            re.IGNORECASE,
        )
        try:
            stated_salary = float(salary_match.group(1).replace(",", "")) if salary_match else None
        except (TypeError, ValueError):
            stated_salary = None

        def authority_candidate_score(record: Dict[str, Any]) -> tuple:
            metadata = record.get("metadata") or {}
            blob = " ".join(
                str(metadata.get(key) or "")
                for key in (
                    "citation",
                    "section",
                    "article_section",
                    "document_title",
                    "topic",
                    "subtopic",
                )
            )
            blob = f"{blob} {record.get('text', '')[:1800]}".lower()
            overlap = sum(1 for token in search_tokens if token in blob)
            salary_bracket_match = 0
            if authority_id == "sss_contribution_schedule_2025" and stated_salary is not None:
                body = str(record.get("text") or "").split("\nText:\n", 1)[-1]
                for match in re.finditer(
                    r"(?<!\d)(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*[-–]\s*"
                    r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
                    body,
                ):
                    try:
                        lower = float(match.group(1).replace(",", ""))
                        upper = float(match.group(2).replace(",", ""))
                    except ValueError:
                        continue
                    if lower <= stated_salary <= upper:
                        salary_bracket_match = 1
                        break
            explicit = int(record_explicitly_names_authority(record, authority_id))
            active = int(
                str(metadata.get("legal_status") or metadata.get("status") or "").lower()
                in {"active", "current", "approved", "indexed"}
            )
            return (explicit, salary_bracket_match, overlap, active, len(str(record.get("text") or "")))

        for where in lookups:
            try:
                lookup_args = {
                    "include": ["documents", "metadatas"],
                    "limit": max(limit_per_authority * 6, 12),
                }
                if "__where_document" in where:
                    lookup_args["where_document"] = {"$contains": where["__where_document"]}
                else:
                    lookup_args["where"] = where
                results = await asyncio.to_thread(
                    collection.get,
                    **lookup_args,
                )
            except Exception as exc:
                logger.debug("Authority lookup failed for %s using %s: %s", authority_id, where, exc)
                continue

            records = _records_from_get_results(
                results,
                score=1.0,
                retrieval_method="required_authority_lookup",
            )
            # Chroma ``get`` does not rank records. Rank the bounded authority
            # candidates locally so an exact section (for example RA 11058
            # Section 6) is chosen ahead of a generic chunk from the same law.
            records.sort(key=authority_candidate_score, reverse=True)
            for record in records:
                if record.get("id") in seen_ids:
                    continue
                if authority_id not in authority_ids_for_chunk(record):
                    continue
                if not record_explicitly_names_authority(record, authority_id):
                    continue
                status = str(
                    (record.get("metadata") or {}).get("legal_status")
                    or (record.get("metadata") or {}).get("status")
                    or ""
                ).strip().lower()
                if status and status not in {"active", "current", "approved", "indexed"}:
                    continue
                output.append(record)
                seen_ids.add(str(record.get("id") or ""))
                if len([
                    item for item in output
                    if authority_id in authority_ids_for_chunk(item)
                ]) >= limit_per_authority:
                    break
            if len([
                item for item in output
                if authority_id in authority_ids_for_chunk(item)
            ]) >= limit_per_authority:
                break

        # Older indexed OSH chunks may carry hash-like canonical IDs even
        # though their document title and URL clearly identify the authority.
        # Use one bounded semantic lookup and re-normalize metadata at read
        # time instead of scanning or reindexing the whole collection here.
        current_count = len([
            item for item in output
            if authority_id in authority_ids_for_chunk(item)
        ])
        if current_count < limit_per_authority:
            osh_authorities = {
                "ra_11058",
                "dole_do_198_18",
                "dole_do_252_25",
                "occupational_safety_health_standards",
            }
            semantic_records = await search_documents(
                query=ranking_text or search_terms,
                n_results=max(limit_per_authority * 8, 12),
                filter_metadata=(
                    {"category": "Occupational Safety & Health"}
                    if authority_id in osh_authorities and authority_id != "dole_do_252_25"
                    else None
                ),
            )
            semantic_records.sort(key=authority_candidate_score, reverse=True)
            for record in semantic_records:
                record_id = str(record.get("id") or "")
                if record_id in seen_ids:
                    continue
                if authority_id not in authority_ids_for_chunk(record):
                    continue
                if not record_explicitly_names_authority(record, authority_id):
                    continue
                record["metadata"] = {
                    **(record.get("metadata") or {}),
                    "retrieval_method": "required_authority_lookup",
                }
                output.append(record)
                seen_ids.add(record_id)
                current_count += 1
                if current_count >= limit_per_authority:
                    break

    # Semantic recovery can return legacy records whose authority was fixed
    # only in authority_ids_for_chunk's temporary enriched copy. Persist the
    # corrected identity on the returned record too, so answer citations use
    # the same authority that passed the source filter.
    normalized_output = []
    for item in output:
        item_metadata = enrich_chunk_metadata(
            str(item.get("text") or ""),
            item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
        )
        normalized_output.append({**item, "metadata": item_metadata})
    _authority_lookup_cache[cache_key] = [
        {**item, "metadata": dict(item.get("metadata") or {})}
        for item in normalized_output
    ]
    return normalized_output


async def delete_document_chunks(doc_id: str):
    invalidate_retrieval_caches()
    collection = get_collection()
    results    = collection.get(where={"doc_id": doc_id})
    if results and results["ids"]:
        collection.delete(ids=results["ids"])
        invalidate_retrieval_caches()
        logger.info(f"Deleted {len(results['ids'])} chunks for doc_id={doc_id}")
        schedule_lda_topic_model_build()


async def count_document_chunks_by_db_id(doc_db_id: int) -> int:
    collection = get_collection()
    results = collection.get(where={"doc_db_id": str(doc_db_id)})
    return len(results.get("ids") or [])


def _chunk_sort_key(item: Dict[str, Any]):
    chunk_id = item.get("id", "")
    meta = item.get("metadata", {}) or {}
    page = meta.get("page") or 0
    match = re.search(r"_(\d+)_[^_]+$", chunk_id)
    index = int(match.group(1)) if match else 0
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 0
    return (page, index, chunk_id)


async def get_document_chunks_by_db_id(
    doc_db_id: int,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    collection = get_collection()
    results = collection.get(
        where={"doc_db_id": str(doc_db_id)},
        include=["documents", "metadatas"],
    )
    items = []
    ids = results.get("ids") or []
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []
    for i, chunk_id in enumerate(ids):
        items.append({
            "id": chunk_id,
            "text": docs[i] if i < len(docs) else "",
            "metadata": metas[i] if i < len(metas) else {},
        })
    items.sort(key=_chunk_sort_key)
    total = len(items)
    offset = max(offset, 0)
    limit = max(min(limit, 200), 1)
    return {
        "total_chunks": total,
        "offset": offset,
        "limit": limit,
        "chunks": items[offset:offset + limit],
    }


async def get_all_document_chunks_by_db_id(doc_db_id: int) -> List[Dict[str, Any]]:
    collection = get_collection()
    results = collection.get(
        where={"doc_db_id": str(doc_db_id)},
        include=["documents", "metadatas"],
    )
    items = []
    ids = results.get("ids") or []
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []
    for i, chunk_id in enumerate(ids):
        items.append({
            "id": chunk_id,
            "text": docs[i] if i < len(docs) else "",
            "metadata": metas[i] if i < len(metas) else {},
        })
    items.sort(key=_chunk_sort_key)
    return items


async def get_document_chunk_ids_by_db_id(doc_db_id: int) -> List[str]:
    collection = get_collection()
    results = collection.get(where={"doc_db_id": str(doc_db_id)})
    return list(results.get("ids") or [])


async def update_document_chunk_metadata_by_db_id(doc_db_id: int, patch: Dict[str, Any]) -> int:
    invalidate_retrieval_caches(warm_body_index=False)
    collection = get_collection()
    results = collection.get(
        where={"doc_db_id": str(doc_db_id)},
        include=["metadatas"],
    )
    ids = results.get("ids") or []
    metadatas = results.get("metadatas") or []
    if not ids:
        return 0
    updated_metadatas = []
    for i, chunk_id in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        updated_metadatas.append(chroma_safe_metadata({**meta, **patch}))
    collection.update(ids=ids, metadatas=updated_metadatas)
    invalidate_retrieval_caches()
    logger.info("Updated metadata for %s chunks of doc_db_id=%s", len(ids), doc_db_id)
    schedule_lda_topic_model_build()
    return len(ids)


async def delete_document_chunks_by_ids(chunk_ids: List[str]) -> int:
    if not chunk_ids:
        return 0
    invalidate_retrieval_caches()
    collection = get_collection()
    collection.delete(ids=chunk_ids)
    invalidate_retrieval_caches()
    logger.info("Deleted %s chunks by explicit chunk id list", len(chunk_ids))
    schedule_lda_topic_model_build()
    return len(chunk_ids)


async def delete_document_chunks_by_db_id(doc_db_id: int) -> int:
    invalidate_retrieval_caches(warm_body_index=False)
    collection = get_collection()
    results = collection.get(where={"doc_db_id": str(doc_db_id)})
    ids = results.get("ids") if results else []
    if not ids:
        return 0
    collection.delete(ids=ids)
    invalidate_retrieval_caches()
    logger.info(f"Deleted {len(ids)} chunks for doc_db_id={doc_db_id}")
    schedule_lda_topic_model_build()
    return len(ids)


def get_collection_stats() -> Dict[str, Any]:
    collection = get_collection()
    return {
        "total_chunks":    collection.count(),
        "collection_name": settings.CHROMA_COLLECTION_NAME,
        "persist_dir":     settings.CHROMA_PERSIST_DIR,
        "embedding_model": settings.EMBEDDING_MODEL,
        "last_reset_reason": _last_reset_reason,
    }
