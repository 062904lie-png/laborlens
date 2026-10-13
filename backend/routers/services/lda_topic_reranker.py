"""Optional document-learned topic hints and context tie-breaking.

Hints supplement an unrestricted search; they never qualify sources, encode
legal rules, restrict topics, or replace the original question's retrieval.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import math
import re
import threading
from dataclasses import dataclass
from array import array
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence

from config import settings

try:  # Keep normal retrieval available if an optional deployment misses sklearn.
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:  # pragma: no cover - exercised by deployment fallback.
    LatentDirichletAllocation = None  # type: ignore[assignment,misc]
    CountVectorizer = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)

DocumentLoader = Callable[[], Awaitable[Sequence[str]] | Sequence[str]]
ChunkPageLoader = Callable[[int, int], Awaitable[Dict[str, Any]]]

# These common words occur in most Philippine legal materials and add noise to
# unsupervised topics.  The vectorizer still sees the substantive legal terms.
TOPIC_STOP_WORDS = {
    "a", "about", "after", "all", "an", "and", "any", "are", "as", "at", "be",
    "been", "by", "can", "chapter", "code", "department", "for", "from", "has",
    "have", "in", "is", "it", "law", "labor", "may", "of", "on", "or", "shall",
    "section", "such", "that", "the", "their", "this", "to", "under", "with",
    "without", "work", "worker", "workers", "act", "article", "philippine",
    "philippines", "provided", "rule", "rules", "thereof", "where", "which", "who",
    "ang", "at", "ay", "din", "ito", "mga", "na", "ng", "nang", "para", "sa",
    "siya", "the", "upon", "while", "within", "yon", "yung",
}


@dataclass(frozen=True)
class LdaTopicModel:
    """A small fitted model used only for topic-overlap scoring."""

    vectorizer: Any
    model: Any
    trained_chunk_count: int
    topic_count: int


_state_lock = threading.RLock()
_topic_model: Optional[LdaTopicModel] = None
_generation = 0
_build_task: Optional[asyncio.Task] = None
_build_loop: Optional[asyncio.AbstractEventLoop] = None
_document_loader: Optional[DocumentLoader] = None
_last_build_error: Optional[str] = None
_chunk_page_loader: Optional[ChunkPageLoader] = None
# Separate, process-local annotations: never modify legal metadata/embeddings.
# Compact float arrays retain ALL learned topic probabilities, not a top-N list.
_chunk_topics: Dict[str, tuple[str, Optional[array], tuple[int, ...]]] = {}
_assignment_total = 0
_assignment_classified = 0
_assignment_complete = False
_assignment_error: Optional[str] = None
_build_phase: Optional[str] = None


def _setting(config: Optional[object], name: str, default: Any) -> Any:
    return getattr(config or settings, name, default)


def _bounded_int(config: Optional[object], name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(_setting(config, name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _bounded_float(config: Optional[object], name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(_setting(config, name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def lda_enabled(config: Optional[object] = None) -> bool:
    # Missing configuration must fail closed: topic enrichment is optional and
    # must never become active merely because a deployment has an older config.
    return bool(_setting(config, "LDA_RERANK_ENABLED", False)) and CountVectorizer is not None and LatentDirichletAllocation is not None


def strip_chunk_metadata_header(text: str) -> str:
    """Use the legal passage body, not the metadata prepended for vector search."""

    value = str(text or "")
    marker = "\n\nText:\n"
    if marker in value:
        value = value.split(marker, 1)[1]
    return re.sub(r"\s+", " ", value).strip()


def _usable_texts(documents: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for document in documents or []:
        text = strip_chunk_metadata_header(str(document or ""))
        # A short label or page number does not produce a meaningful topic.
        if len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ-]{2,}", text)) < 6:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def lda_training_chunk_limit(config: Optional[object] = None) -> int:
    """Zero is an uncapped corpus, not a zero-sized or 100-chunk sample."""
    try:
        return max(0, int(_setting(config, 'LDA_RERANK_MAX_TRAINING_CHUNKS', 0)))
    except (TypeError, ValueError):
        return 0


def lda_topic_configuration(config: Optional[object] = None) -> Dict[str, Any]:
    requested = _bounded_int(config, "LDA_RERANK_TOPIC_COUNT", 0, 0, 512)
    maximum = _bounded_int(config, "LDA_RERANK_MAX_TOPICS", 128, 2, 512)
    return {"topic_selection": "Automatic" if requested == 0 else "Manual",
            "requested_topic_count": requested, "configured_topic_limit": maximum}


def _topic_count_for_corpus(documents: int, vocabulary: int, config: Optional[object]) -> int:
    options = lda_topic_configuration(config)
    # A reproducible sizing heuristic, NOT an estimate of the true number of
    # legal issues. More data supports more components, within resource limits.
    requested = options["requested_topic_count"] or max(2, math.ceil(math.sqrt(documents)))
    return max(2, min(requested, options["configured_topic_limit"],
                      max(2, documents // 2), max(2, vocabulary // 2)))


def build_lda_topic_model(
    documents: Iterable[str],
    config: Optional[object] = None,
) -> Optional[LdaTopicModel]:
    """Fit topics off the request path; use the full corpus unless explicitly capped."""

    if not lda_enabled(config):
        return None
    texts = _usable_texts(documents)
    minimum_documents = _bounded_int(config, "LDA_RERANK_MIN_CORPUS_CHUNKS", 80, 8, 1000)
    if len(texts) < minimum_documents:
        logger.info("LDA topic signal skipped: corpus has %s usable chunks (need %s).", len(texts), minimum_documents)
        return None

    max_documents = lda_training_chunk_limit(config)
    if max_documents:
        texts = texts[:max_documents]
    min_df = 2 if len(texts) >= 200 else 1
    max_features = _bounded_int(config, "LDA_RERANK_MAX_FEATURES", 12000, 500, 50000)

    try:
        vectorizer = CountVectorizer(
            lowercase=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b[^\W\d_][\w-]{2,}\b",
            stop_words=sorted(TOPIC_STOP_WORDS),
            min_df=min_df,
            max_df=0.95,
            max_features=max_features,
        )
        matrix = vectorizer.fit_transform(texts)
        if matrix.shape[1] < 8:
            logger.info("LDA topic signal skipped: corpus vocabulary is too small (%s terms).", matrix.shape[1])
            return None
        topic_count = _topic_count_for_corpus(len(texts), matrix.shape[1], config)
        model = LatentDirichletAllocation(
            n_components=topic_count,
            learning_method="online",
            batch_size=_bounded_int(config, 'LDA_TRAINING_BATCH_SIZE', 128, 16, 512),
            n_jobs=1,
            max_iter=_bounded_int(config, "LDA_RERANK_MAX_ITER", 8, 2, 25),
            random_state=17,
            evaluate_every=-1,
        )
        model.fit(matrix)
    except (TypeError, ValueError) as exc:
        logger.warning("LDA topic signal could not be trained: %s", exc)
        return None

    return LdaTopicModel(
        vectorizer=vectorizer,
        model=model,
        trained_chunk_count=len(texts),
        topic_count=topic_count,
    )


def get_lda_topic_model() -> Optional[LdaTopicModel]:
    with _state_lock:
        return _topic_model


def set_lda_topic_model(model: Optional[LdaTopicModel], *, generation: Optional[int] = None) -> bool:
    """Install a completed model unless a newer KB change superseded it."""

    global _topic_model, _last_build_error
    with _state_lock:
        if generation is not None and generation != _generation:
            return False
        _topic_model = model
        _last_build_error = None
        _clear_chunk_assignments()
        return True


def _clear_chunk_assignments() -> None:
    global _assignment_total, _assignment_classified, _assignment_complete, _assignment_error
    _chunk_topics.clear()
    _assignment_total = 0
    _assignment_classified = 0
    _assignment_complete = False
    _assignment_error = None


def invalidate_lda_topic_model() -> int:
    """Mark the in-memory model stale after an ingestion or deletion."""

    global _topic_model, _generation, _last_build_error
    with _state_lock:
        _generation += 1
        _topic_model = None
        _last_build_error = None
        _clear_chunk_assignments()
        return _generation


def register_lda_document_loader(loader: DocumentLoader) -> None:
    """Register the Chroma-backed loader without importing Chroma here."""

    global _document_loader
    with _state_lock:
        _document_loader = loader


def register_lda_chunk_page_loader(loader: ChunkPageLoader) -> None:
    """Read every stored chunk in batches, independently of the training sample."""
    global _chunk_page_loader
    with _state_lock:
        _chunk_page_loader = loader


def _body_fingerprint(text: str) -> str:
    return hashlib.sha256(strip_chunk_metadata_header(text).encode("utf-8")).hexdigest()


def _assign_chunk_batch(model: LdaTopicModel, records: Sequence[Dict[str, Any]]) -> Dict[str, tuple]:
    bodies = [strip_chunk_metadata_header(str(row.get("text") or "")) for row in records]
    matrix = model.vectorizer.transform(bodies)
    # Avoid expensive inference on empty/OOV/short passages and leave them
    # explicitly unclassified. Duplicate bodies still have separate chunk IDs.
    indexes = [i for i, body in enumerate(bodies)
               if len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ-]{2,}", body)) >= 6
               and matrix[i].nnz >= 3]
    distributions = model.model.transform(matrix[indexes]) if indexes else []
    by_index = dict(zip(indexes, distributions))
    output = {}
    baseline = 1.0 / model.topic_count
    for i, row in enumerate(records):
        identity = str(row.get("id") or "")
        if not identity:
            raise ValueError("missing_chunk_identity")
        distribution = by_index.get(i)
        probabilities = array("f", distribution) if distribution is not None else None
        # A near-uniform mixture is not a reliable label. No forced primary
        # topic, no top-10 truncation: retain every probability, mark only
        # stronger-than-background matches as labels.
        topics = tuple(j for j, value in enumerate(distribution) if value > baseline * 1.2) \
            if distribution is not None and max(distribution) >= baseline * 1.5 else ()
        output[identity] = (_body_fingerprint(bodies[i]), probabilities, topics)
    return output


def lda_chunk_topic_assignment(chunk_id: str, text: str) -> Optional[Dict[str, Any]]:
    """Return a current, exact-body annotation; IDs are local to model generation."""
    fingerprint = _body_fingerprint(text)
    with _state_lock:
        entry = _chunk_topics.get(str(chunk_id))
        if _topic_model is None or entry is None or entry[0] != fingerprint:
            return None
        _, probabilities, topics = entry
        return {"generation": _generation, "classified": bool(topics),
                "probabilities": list(probabilities) if probabilities is not None else [],
                "topics": [{"topic_id": index, "probability": float(probabilities[index])}
                           for index in topics]}


async def _assign_all_chunks(model: LdaTopicModel, generation: int,
                             loader: ChunkPageLoader, config: Optional[object]) -> None:
    global _assignment_total, _assignment_classified, _assignment_complete
    batch_size = _bounded_int(config, "LDA_CHUNK_ASSIGNMENT_BATCH_SIZE", 128, 16, 512)
    offset = 0
    expected_total = None
    while True:
        with _state_lock:
            if generation != _generation or _topic_model is not model:
                return
        page = await loader(offset, batch_size)
        records = page.get("records") or []
        total = max(0, int(page.get("total") or 0))
        if expected_total is not None and total != expected_total:
            raise ValueError("chunk_corpus_changed")
        expected_total = total
        annotations = await asyncio.to_thread(_assign_chunk_batch, model, records) if records else {}
        with _state_lock:
            if generation != _generation or _topic_model is not model:
                return  # Never publish assignments from an interrupted replacement.
            _assignment_total = total
            for identity, entry in annotations.items():
                previous = _chunk_topics.get(identity)
                _assignment_classified += int(bool(entry[2])) - int(bool(previous and previous[2]))
                _chunk_topics[identity] = entry
            if not records:
                if offset < total:
                    raise ValueError("incomplete_chunk_scan")
                _assignment_complete = True
                return
            offset += len(records)
            if offset >= total:
                if len(_chunk_topics) != total:
                    raise ValueError("incomplete_chunk_scan")
                _assignment_complete = True
                logger.info("LDA chunk assignments ready: checked=%s classified=%s topics=%s",
                            len(_chunk_topics), _assignment_classified, model.topic_count)
                return


async def _load_documents(loader: DocumentLoader) -> Sequence[str]:
    loaded = loader()
    if inspect.isawaitable(loaded):
        loaded = await loaded
    return loaded or []


async def _build_and_install(generation: int, loader: DocumentLoader, config: Optional[object]) -> None:
    global _build_task, _last_build_error, _build_phase, _assignment_error
    cancelled = False
    try:
        with _state_lock:
            _build_phase = "training"
        documents = await _load_documents(loader)
        model = await asyncio.to_thread(build_lda_topic_model, documents, config)
        if model is None:
            with _state_lock:
                if generation == _generation:
                    _last_build_error = "not_enough_usable_document_text"
            return
        if set_lda_topic_model(model, generation=generation):
            logger.info(
                "LDA topic reranker ready: topics=%s training_chunks=%s",
                model.topic_count,
                model.trained_chunk_count,
            )
            with _state_lock:
                chunk_loader = _chunk_page_loader
                _build_phase = "assigning" if chunk_loader else None
            if chunk_loader:
                try:
                    await _assign_all_chunks(model, generation, chunk_loader, config)
                except Exception as exc:
                    logger.warning("LDA chunk assignment incomplete: %s", type(exc).__name__)
                    with _state_lock:
                        if generation == _generation:
                            _assignment_error = type(exc).__name__
    except asyncio.CancelledError:
        cancelled = True
        raise
    except Exception as exc:  # Never let the optional topic signal break chat retrieval.
        logger.warning("LDA topic model build failed; retrieval will continue without it: %s", exc)
        with _state_lock:
            if generation == _generation:
                _last_build_error = type(exc).__name__
    finally:
        retry = False
        with _state_lock:
            _build_task = None
            _build_phase = None
            retry = not cancelled and generation != _generation and _document_loader is not None
        if retry:
            schedule_lda_topic_model_build(config=config)


def schedule_lda_topic_model_build(config: Optional[object] = None) -> bool:
    """Start one non-blocking model build when an event loop is available."""

    global _build_task, _build_loop
    if not lda_enabled(config):
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Metadata/deletion helpers can run in a worker thread. Queue the
        # refresh on the startup loop instead of silently waiting for a chat.
        with _state_lock:
            loop = _build_loop
            registered = _document_loader is not None
        if registered and loop is not None and loop.is_running() and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(lambda: schedule_lda_topic_model_build(config=config))
                return True
            except RuntimeError:  # Server shutdown raced the notification.
                pass
        return False
    with _state_lock:
        _build_loop = loop
        if _build_task is not None and not _build_task.done():
            return False
        if _document_loader is None:
            return False
        generation = _generation
        _build_task = loop.create_task(_build_and_install(generation, _document_loader, config))
    return True


def lda_topic_model_status() -> Dict[str, Any]:
    with _state_lock:
        model = _topic_model
        building = _build_task is not None and not _build_task.done()
        return {
            "enabled": lda_enabled(),
            "ready": model is not None,
            "building": building,
            "topic_count": model.topic_count if model else 0,
            "trained_chunk_count": model.trained_chunk_count if model else 0,
            "training_chunk_limit": lda_training_chunk_limit(),
            "training_corpus_mode": 'all_approved' if not lda_training_chunk_limit() else 'capped',
            "generation": _generation,
            "last_error": _last_build_error,
            "build_phase": _build_phase,
            "assignment_total_chunks": _assignment_total,
            "assignment_checked_chunks": len(_chunk_topics),
            "assignment_classified_chunks": _assignment_classified,
            "assignment_unclassified_chunks": len(_chunk_topics) - _assignment_classified,
            "assignment_complete": _assignment_complete,
            "assignment_error": _assignment_error,
        }


def lda_search_queries(query_text: str, config: Optional[object] = None) -> List[str]:
    """One additive probe using vocabulary learned from uploaded documents.

    Unfamiliar/short/ambiguous queries are a no-op. No topic names or document
    IDs are hardcoded. Call off the event loop: sklearn transforms are CPU work.
    """
    if not lda_enabled(config) or not _setting(config, 'LDA_SEARCH_HINTS_ENABLED', False):
        return []
    model = get_lda_topic_model()
    if model is None:
        return []
    query = strip_chunk_metadata_header(query_text)
    if len(query) > 2500:
        return []
    try:
        matrix = model.vectorizer.transform([query])
        # One keyword is not reliable evidence of a query's learned topic.
        if matrix.nnz < 2:
            return []
        distribution = model.model.transform(matrix)[0]
        topic = int(distribution.argmax())
        if float(distribution[topic]) < _bounded_float(config, 'LDA_RERANK_MIN_QUERY_CONFIDENCE', .42, .05, .99):
            return []
        vocabulary = model.vectorizer.get_feature_names_out()
        query_terms = set(model.vectorizer.build_analyzer()(query))
        weights = model.model.components_[topic]
        keywords = [str(vocabulary[index]) for index in weights.argsort()[::-1][:16]
                    if str(vocabulary[index]) not in query_terms
                    and not re.search(r'\d', str(vocabulary[index]))][:4]
        if not keywords or get_lda_topic_model() is not model:
            return []
        return [query + ' ' + ' '.join(keywords)]
    except Exception as exc:
        logger.debug('Optional LDA search hints skipped: %s', type(exc).__name__)
        return []


def apply_lda_topic_signals(
    chunks: List[Dict[str, Any]],
    query_text: str,
    query_profile: Optional[Dict[str, Any]] = None,
    config: Optional[object] = None,
) -> List[Dict[str, Any]]:
    """Annotate confident topic overlap without changing any retrieval rank.

    ``score`` remains vector similarity and ``rerank_score`` remains the legal
    policy score. The returned LDA fields are informational only; they are
    never allowed to decide source eligibility, deduplication, coverage, or
    source status.
    """

    output = [dict(chunk) for chunk in chunks or []]
    # Old requests can carry fields from an earlier model generation. Never let
    # a stale LDA score escape into a later ranking pass.
    for chunk in output:
        for key in ("retrieval_rank_score", "lda_topic_similarity", "lda_topic_boost"):
            chunk.pop(key, None)

    if not output or not lda_enabled(config):
        return output
    model = get_lda_topic_model()
    if model is None:
        # A request can start a rebuild after a future upload; it never waits.
        schedule_lda_topic_model_build(config=config)
        return output

    normalized_query = strip_chunk_metadata_header(query_text)
    if len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ-]{2,}", normalized_query)) < 3:
        return output
    try:
        query_matrix = model.vectorizer.transform([normalized_query])
        if not query_matrix.nnz:
            return output
        query_topics = model.model.transform(query_matrix)[0]
    except Exception as exc:  # An optional signal must remain fail-open.
        logger.debug("LDA query transform skipped: %s", exc)
        return output

    minimum_confidence = _bounded_float(config, "LDA_RERANK_MIN_QUERY_CONFIDENCE", 0.42, 0.05, 0.99)
    if float(max(query_topics)) < minimum_confidence:
        return output

    eligible_indexes: List[int] = []
    candidate_texts: List[str] = []
    minimum_base_score = _bounded_float(config, "LDA_RERANK_MIN_BASE_SCORE", 0.45, -1.0, 5.0)
    for index, chunk in enumerate(output):
        # Legal-policy boosts can elevate an authority for good reasons, but
        # they are not semantic evidence that an LDA topic matches. Require the
        # original vector similarity before annotating topic overlap.
        semantic_score = float(chunk.get("score", 0.0) or 0.0)
        text = strip_chunk_metadata_header(str(chunk.get("text") or ""))
        if semantic_score < minimum_base_score or not text:
            continue
        eligible_indexes.append(index)
        candidate_texts.append(text)
    if not candidate_texts:
        return output

    try:
        # Reuse background assignments only for the same model and exact body.
        # New/edited/excerpted passages fall back to inference, never exclusion.
        candidate_topics = [None] * len(candidate_texts)
        missing = []
        with _state_lock:
            if _topic_model is not model:
                return output
            for position, index in enumerate(eligible_indexes):
                chunk = output[index]
                identity = str(chunk.get("id") or (chunk.get("metadata") or {}).get("chunk_id") or "")
                entry = _chunk_topics.get(identity)
                if entry and entry[0] == _body_fingerprint(candidate_texts[position]):
                    candidate_topics[position] = entry[1] if entry[2] else []
                else:
                    missing.append(position)
        if missing:
            candidate_matrix = model.vectorizer.transform([candidate_texts[i] for i in missing])
            distributions = model.model.transform(candidate_matrix)
            for position, distribution in zip(missing, distributions):
                candidate_topics[position] = distribution
    except Exception as exc:
        logger.debug("LDA candidate transform skipped: %s", exc)
        return output

    baseline = 1.0 / max(1, model.topic_count)
    max_boost = _bounded_float(config, "LDA_RERANK_MAX_BOOST", 0.04, 0.0, 0.10)
    minimum_similarity = _bounded_float(config, "LDA_RERANK_MIN_SIMILARITY", 0.15, 0.0, 1.0)
    if get_lda_topic_model() is not model:
        return output
    for index, distribution in zip(eligible_indexes, candidate_topics):
        if distribution is None or not len(distribution):
            continue
        raw_overlap = float(sum(float(left) * float(right) for left, right in zip(query_topics, distribution)))
        # Uniform topic distributions should behave as no topical evidence.
        similarity = max(0.0, min(1.0, (raw_overlap - baseline) / max(1e-9, 1.0 - baseline)))
        boost = max_boost * similarity if similarity >= minimum_similarity else 0.0
        original = output[index]
        updated = dict(original)
        updated["lda_topic_similarity"] = round(similarity, 4)
        updated["lda_topic_boost"] = round(boost, 4)
        output[index] = updated
    return output


def order_approved_context(
    chunks: List[Dict[str, Any]],
    query_text: str,
    query_profile: Optional[Dict[str, Any]] = None,
    config: Optional[object] = None,
) -> List[Dict[str, Any]]:
    """Use LDA only to break exact ties within already approved context.

    The input list's membership and every non-tied legal/vector order are
    preserved. Call this only after source selection, authority coverage, and
    source-status checks have completed.
    """

    annotated = apply_lda_topic_signals(chunks, query_text, query_profile, config)
    if len(annotated) < 2:
        return annotated

    def approval_rank(chunk: Dict[str, Any]) -> tuple[float, float]:
        try:
            legal_score = float(chunk.get("rerank_score", chunk.get("score", 0.0)) or 0.0)
        except (TypeError, ValueError):
            legal_score = 0.0
        try:
            vector_score = float(chunk.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            vector_score = 0.0
        return legal_score, vector_score

    output: List[Dict[str, Any]] = []
    index = 0
    while index < len(annotated):
        rank = approval_rank(annotated[index])
        end = index + 1
        while end < len(annotated) and approval_rank(annotated[end]) == rank:
            end += 1
        # Python's sort is stable, so candidates without an LDA distinction
        # retain their prior approved order.
        output.extend(sorted(
            annotated[index:end],
            key=lambda chunk: float(chunk.get("lda_topic_boost", 0.0) or 0.0),
            reverse=True,
        ))
        index = end
    return output
