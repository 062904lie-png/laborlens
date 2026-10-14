"""LaborLens – Chat Router (RAG pipeline)
Unresolved = query where ChromaDB returned NO results OR
             all results had relevance score below MIN_RELEVANCE_SCORE.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import asyncio, uuid, time, json, re
import logging
from decimal import Decimal
from config import settings

from routers.auth import get_current_user
from routers.services.guest_usage import guest_usage
from routers.services.nlp_service import SUPPORTED_LANGUAGES, preprocess_query
from routers.services.chroma_service import (
    get_documents_by_authority,
    get_documents_for_concept_coverage,
    get_osh_documents_for_task,
    record_explicitly_names_authority,
    search_documents,
    search_regional_wage_documents,
)
from routers.services import gemini_service, openai_service, groq_provider
from routers.services.lda_topic_reranker import order_approved_context
from routers.services.document_evidence import is_document_driven
from routers.services.performance import add_timing, current_timing, set_timing_metadata
from routers.services.performance import begin_request, end_request
from routers.services.pipeline_request import bounded_answer, timed_stage, clamp_deadline
from routers.services.question_analysis_cache import with_question_analysis
from routers.services.provider_runtime import (
    CircuitOpenError,
    acquire_provider,
    abandon_provider,
    classify_provider_error,
    configured_provider_order,
    provider_timeout,
    generation_attempt_timeout,
    compress_answer_to_budget,
    check_final_answer_coverage,
    record_failure,
    record_success,
    retry_after_seconds,
)
from routers.services.openai_service import (
    build_sources_payload,
    answer_claims_unavailable,
    get_language_profile,
    language_name,
)
from routers.services.rag_policy import (
    SOURCE_STATUS_DIRECT,
    SOURCE_STATUS_CONFLICTING,
    SOURCE_STATUS_INSUFFICIENT,
    SOURCE_STATUS_NOT_REQUIRED,
    SOURCE_STATUS_PARTIAL,
    SOURCE_STATUS_PROVIDER_UNAVAILABLE,
    authority_ids_for_chunk,
    authority_query_terms_for_profile,
    classify_legal_query,
    build_wage_order_search_text,
    frontend_confidence,
    is_domestic_worker_overtime_query,
    log_rag_debug,
    missing_required_authorities,
    osh_chunk_supports_task,
    osh_evidence_is_responsive,
    required_authorities_for_query,
    select_generator_chunks,
    should_enqueue_review,
    source_match_status,
    wage_order_query_details,
)
from routers.services.kb_categories import get_sidebar_category, normalize_kb_category
from routers.services.document_categories import (
    category_match_type,
    secondary_categories_from_metadata,
)
from routers.services.source_history import (
    infer_source_followup_topic,
    select_relevant_assistant_for_source_followup,
)
from routers.services.database import execute_insert, execute_query, execute_one, get_pool
from routers.services.conversation_memory import (
    build_standalone_search_query,
    merge_conversation_state,
    parse_state,
    state_context_text,
    state_json,
    title_from_state_or_message,
)
from routers.services.calculation_state import (
    CALCULATION_TASKS,
    PENDING_CALCULATION_KEY,
    active_pending_calculation,
    apply_pending_calculation_reply,
    calculation_verification_query,
    infer_pending_field_from_answer,
    is_completed_calculation_state,
    parse_calculation_reply_value,
    supersede_calculation_state,
    update_pending_calculation_state,
)
from routers.services.calculators import (
    build_overtime_calculation,
    calculate_overtime_from_state,
    fmt_money,
    thirteenth_month_pay,
)
from routers.services.legal_answer_validator import (
    filter_display_sources_for_answer,
    normalize_known_source_filenames,
    remove_unsupported_authority_citations,
    remap_answer_source_citations,
    validate_legal_answer,
    validate_overtime_answer,
)
from routers.services.legal_topic_matrix import (
    collect_supported_concepts,
    detect_supported_concepts,
    missing_required_concepts,
)

router = APIRouter()
logger = logging.getLogger(__name__)

HIGH_TRAFFIC_MESSAGE = (
    "I am currently experiencing high traffic. "
    "Please try your question again in a moment."
)
SOURCES_FOUND_SERVICE_UNAVAILABLE_MESSAGE = (
    "LaborLens found relevant legal sources, but the response service is temporarily "
    "unavailable. Please try again shortly."
)
GENERATION_DEADLINE_NOT_REACHED_MESSAGE = (
    "LaborLens found legal sources, but the request deadline expired before answer generation started. "
    "Please try again."
)
GENERATION_ADMISSION_FAILED_MESSAGE = (
    "LaborLens found legal sources, but no answer model was available to start generation. Please try again."
)


def build_source_grounded_provider_fallback(
    language: str,
    context_chunks: List[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]] | None:
    """Return a useful source excerpt when every generation provider is down."""

    selected_chunks = [
        chunk
        for chunk in (context_chunks or [])[:8]
        if str(chunk.get("text") or "").strip()
    ][:2]
    sources = build_sources_payload(selected_chunks)
    if not sources:
        return None

    excerpt_lines: List[str] = []
    for index, source in enumerate(sources, 1):
        excerpt = str(source.get("excerpt") or "").strip()
        if not excerpt:
            continue
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+|\n+", excerpt)
            if part.strip()
        ]
        excerpt = " ".join(sentences[:2]) or excerpt
        if len(excerpt) > 650:
            excerpt = excerpt[:647].rsplit(" ", 1)[0].rstrip(".,;:") + "..."
        excerpt_lines.append(f"- {excerpt} (Source {index})")

    if not excerpt_lines:
        return None

    profile = get_language_profile(language)
    if language == "fil":
        direct = (
            "Pansamantalang hindi available ang response service, pero may nakuha akong "
            "relevanteng approved legal source para sa tanong mo. Narito ang direktang bahagi ng source:"
        )
        next_steps = "Subukan muli pagkalipas ng ilang sandali para makuha ang buong paliwanag na nakaangkop sa tanong mo. Para sa agarang tulong, tumawag sa DOLE Hotline 1349."
    elif language == "hil":
        direct = (
            "Temporaryo nga indi available ang response service, pero may nakuha ako nga "
            "relevant approved legal source para sa pamangkot mo. Ari ang direkta nga bahin sang source:"
        )
        next_steps = "Sulayi liwat pagkatapos sang pila ka tion para makuha ang kompleto nga sabat. Para sa gilayon nga bulig, tawag sa DOLE Hotline 1349."
    else:
        direct = (
            "The response service is temporarily unavailable, but I found relevant approved legal sources for your question. "
            "Here is the directly relevant part of the retrieved source:"
        )
        next_steps = "Please retry shortly for a complete explanation tailored to your question. For urgent help, contact the DOLE Hotline at 1349."

    answer = (
        f"**{profile['direct_label']}:**\n{direct}\n\n"
        + "\n".join(excerpt_lines)
        + f"\n\n**{profile['next_steps_label']}:**\n{next_steps}\n\n"
        f"{profile['disclaimer']}"
    )
    return answer, sources


async def _run_nonessential_write(query: str, params: tuple) -> None:
    try:
        await execute_insert(query, params)
    except Exception:
        logger.warning("Nonessential analytics write failed", exc_info=True)


def schedule_nonessential_write(query: str, params: tuple) -> None:
    """Keep analytics off the response path without risking chat history."""

    asyncio.create_task(_run_nonessential_write(query, params))


async def _run_nonessential_review(message_id: int, user_id: int, reason: str) -> None:
    try:
        await enqueue_answer_review(message_id, user_id, reason)
    except Exception:
        logger.warning("Nonessential answer-review enqueue failed", exc_info=True)


def schedule_nonessential_review(message_id: int, user_id: int, reason: str) -> None:
    asyncio.create_task(_run_nonessential_review(message_id, user_id, reason))


async def generate_response_with_fallback(
    *,
    query: str,
    context_chunks: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]],
    language: str,
    intent: str,
    query_profile: Dict[str, Any],
    source_status: str,
) -> Dict[str, Any]:
    """Prefer idle healthy models, then least busy, with bounded failover/waiting."""

    provider_args = {
        "query": query,
        "context_chunks": context_chunks,
        "chat_history": chat_history,
        "language": language,
        "intent": intent,
        "query_profile": query_profile,
        "source_status": source_status,
    }
    from routers.services import model_pools
    from routers.services.pipeline_request import clamp_deadline
    providers = (model_pools.generation_order() if settings.LABORLENS_MODEL_POOLS_ENABLED else []) or configured_provider_order()
    from routers.services.performance import current_timing
    timing = current_timing()
    initial_generation_calls = int((timing.metadata if timing else {}).get('generation_calls') or 0)
    set_timing_metadata(generation_pool_admission='pool_ready' if providers else 'no_configured_models',
                        generation_call_state='not_reached', generation_provider_attempts=0,
                        generation_skip_count=0)

    def skip_generation_candidate(candidate, reason):
        timing = current_timing()
        count = int((timing.metadata if timing else {}).get('generation_skip_count') or 0) + 1
        set_timing_metadata(generation_pool_admission='candidate_skipped', generation_skip_count=count,
                            generation_skip_reason=reason)
        if timing:
            from routers.services.backend_diagnostics import record_event
            record_event('generation', 'model_task_skipped', details={
                'role': 'generation', 'model': candidate, 'reason': reason})

    if not providers:
        set_timing_metadata(generation_outcome='admission_failed', generation_pool_admission='blocked')
        return {'answer': SOURCES_FOUND_SERVICE_UNAVAILABLE_MESSAGE,
                'sources': build_sources_payload(context_chunks), 'model': None,
                'intent': intent, 'language': language, 'kb_used': bool(context_chunks),
                'provider_unavailable': True, 'generation_outcome': 'admission_failed'}
    # Bound the complete provider cascade, not just each individual HTTP call.
    # A fallback must use only the time left after the primary attempt.
    complex_request = bool(
        len(query or "") > 500
        or (query_profile or {}).get("task") in {"comparison_question", "calculation_request", "jurisprudence_request"}
    )
    request_budget_seconds = max(
        1.0,
        float(
            settings.LABORLENS_COMPLEX_REQUEST_BUDGET_SECONDS
            if complex_request
            else settings.LABORLENS_NORMAL_REQUEST_BUDGET_SECONDS
        ),
    )
    request_deadline = clamp_deadline(time.monotonic() + request_budget_seconds)
    max_provider_attempts = (
        min(len(providers), max(1, int(getattr(settings, 'LABORLENS_SIMPLE_GENERATION_MAX_CALLS', 5))))
        if query_profile.get('simple_rag_mode') or query_profile.get('simple_fast_path') else len(providers)
    )
    primary = providers[0]
    fallback_reason: str | None = None
    last_error: BaseException | None = None
    gemini_timed_out = False

    from routers.services.pipeline_request import start_generation_attempt
    provider_attempts = 0
    pending = list(providers)
    while pending:
        if provider_attempts >= max_provider_attempts:
            break
        remaining = request_deadline - time.monotonic()
        if remaining <= 0.05:
            last_error = asyncio.TimeoutError('request budget exhausted before provider attempt')
            skip_generation_candidate(pending[0], 'generation_deadline')
            logger.warning('Request deadline reached; remaining models were not attempted')
            break
        if all('/' in model for model in pending):
            candidates = [model for model in pending
                          if not (model.split('/', 1)[0] == 'gemini' and gemini_timed_out)]
            candidate = await model_pools.reserve_generation_model(candidates, request_deadline)
            if candidate is None:
                for model in candidates:
                    reason = model_pools.local_limit_reason(model) or (
                        'busy' if model in model_pools._busy else 'provider_unavailable_or_unconfigured')
                    skip_generation_candidate(model, reason)
                break
        else:
            # Preserve the legacy provider-only path when pools are disabled.
            candidate = pending[0]
        pending.remove(candidate)
        provider = candidate.split('/', 1)[0]
        pooled = '/' in candidate
        if pooled and (not model_pools.provider_enabled(provider)
                       or not model_pools.provider_api_key(provider)
                       or (provider == 'gemini' and gemini_timed_out)):
            model_pools.release(candidate)
            skip_generation_candidate(candidate, 'provider_unavailable_or_unconfigured')
            continue
        remaining = request_deadline - time.monotonic()
        if remaining <= 0.05:
            if pooled:
                model_pools.release(candidate)
            last_error = asyncio.TimeoutError('request budget exhausted before provider attempt')
            skip_generation_candidate(candidate, 'generation_deadline')
            logger.warning('Request deadline reached; remaining models were not attempted')
            break
        remaining_configured = 1 + sum(
            1 for model in pending
            if model_pools.provider_enabled(model.split('/', 1)[0])
            and model_pools.provider_api_key(model.split('/', 1)[0])
            and not (model.split('/', 1)[0] == 'gemini' and gemini_timed_out)
        )
        attempts_left = max(1, min(remaining_configured, max_provider_attempts - provider_attempts))
        # Give the selected model useful time, reserving only one next attempt.
        # Pool size must not turn all attempts into ~3-second cancellations.
        attempt_timeout = generation_attempt_timeout(
            remaining,
            attempts_left,
            provider_timeout(provider),
            is_first_attempt=provider_attempts == 0,
        )
        selection_token = model_pools.selected_model.set(candidate if pooled else None)
        set_timing_metadata(generation_pool_admission='admitted', generation_selected_model=candidate)
        calls_token = model_pools.generation_request_count.set(0)
        is_fallback = candidate != primary
        timing_name = "fallback_model_ms" if is_fallback else "primary_model_ms"
        if provider != "gemini":
            try:
                await acquire_provider(candidate)
            except CircuitOpenError as exc:
                model_pools.selected_model.reset(selection_token)
                model_pools.generation_request_count.reset(calls_token)
                if pooled:
                    model_pools.release(candidate)
                last_error = exc
                if candidate == primary:
                    fallback_reason = "circuit_open"
                logger.warning("Skipping %s provider because its circuit is open", provider)
                skip_generation_candidate(candidate, 'circuit_open')
                continue
            except asyncio.CancelledError:
                # Cancellation while awaiting the circuit lock must not leak
                # a reservation or this request's selected-model context.
                model_pools.selected_model.reset(selection_token)
                model_pools.generation_request_count.reset(calls_token)
                if pooled:
                    model_pools.release(candidate)
                raise

        started = time.perf_counter()
        attempt_timer = None
        try:
            remaining = request_deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError("request budget exhausted before provider attempt")
            provider_attempts += 1
            set_timing_metadata(generation_provider_attempts=provider_attempts)
            start_generation_attempt()
            if provider == "groq":
                call = groq_provider.generate_response(**provider_args, raise_on_error=True)
            elif provider == "openai":
                call = openai_service.generate_response(**provider_args, raise_on_error=True)
            else:
                call = gemini_service.generate_response(**provider_args)
            attempt_timeout = min(attempt_timeout, remaining)
            async with asyncio.timeout(attempt_timeout) as attempt_timer:
                result = await call
            set_timing_metadata(generation_api_call_returned=True)
            if not isinstance(result, dict) or not str(result.get("answer") or "").strip():
                raise RuntimeError(f"{provider} returned an empty response.")
            # Enforce the hard response ceiling after continuation/repair paths;
            # ordinary responses are unchanged by this no-op fast path.
            bounded_answer, answer_compressed = compress_answer_to_budget(
                str(result["answer"]).strip(),
                settings.LABORLENS_HARD_OUTPUT_LIMIT,
            )
            if not bounded_answer.strip():
                raise RuntimeError(f"{provider} returned an empty response after compression.")
            if openai_service.answer_looks_truncated(bounded_answer):
                raise RuntimeError(f"{provider} returned an incomplete response after compression.")
            result["answer"] = bounded_answer
            set_timing_metadata(generation_outcome='succeeded', generation_call_state='returned',
                                generation_draft_created=True)
            elapsed_ms = (time.perf_counter() - started) * 1000
            add_timing(timing_name, elapsed_ms)
            if provider != "gemini":
                await record_success(candidate, elapsed_ms)
            result["provider"] = provider
            set_timing_metadata(
                provider_used=provider,
                model_used=result.get("model"),
                fallback_triggered=is_fallback or candidate != primary,
                fallback_reason=fallback_reason,
                provider_timeout=provider_timeout(provider),
                provider_error_type=None,
                retrieved_chunk_count=len(context_chunks),
                output_token_count=max(1, len(str(result.get("answer") or "")) // 4),
                answer_compressed=answer_compressed,
            )
            return result
        except model_pools.LocalModelLimitError as exc:
            # Local capacity was checked before HTTP. Do not label an uncalled
            # provider as failing or open its circuit because our budget is full.
            if provider != 'gemini':
                if model_pools.generation_request_count.get() == 0:
                    await abandon_provider(candidate, unattempted=True)
                else:
                    await record_failure(candidate, 'local_budget',
                                         (time.perf_counter() - started) * 1000, counts=False)
            last_error = exc
            fallback_reason = fallback_reason or 'local_budget'
            logger.info('Generation skipped/stopped: model=%s local_reason=%s', candidate, exc.reason)
            skip_generation_candidate(candidate, exc.reason)
            from routers.services.backend_diagnostics import record_event
            record_event('generation', 'model_task_skipped', details={
                'role': 'generation', 'model': candidate, 'reason': exc.reason})
            continue
        except asyncio.CancelledError:
            calls_for_candidate = model_pools.generation_request_count.get()
            if provider != 'gemini' and calls_for_candidate:
                await record_failure(candidate, 'timeout', (time.perf_counter() - started) * 1000, counts=False)
            elif provider != 'gemini':
                await abandon_provider(candidate, unattempted=True)
            timing = current_timing()
            dispatched = int((timing.metadata if timing else {}).get('generation_calls') or 0) > initial_generation_calls
            set_timing_metadata(generation_outcome='provider_interrupted' if dispatched else 'not_attempted',
                                generation_call_state='interrupted' if dispatched else 'not_reached')
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            set_timing_metadata(generation_exception=type(exc).__name__)
            add_timing(timing_name, elapsed_ms)
            reason, counts_for_circuit = classify_provider_error(exc)
            # Our attempt timer expiry is a local deadline, not proof that
            # the provider is unhealthy. HTTPX timeouts and provider errors keep
            # their existing circuit policy; local expiries remain reportable.
            local_attempt_timeout = bool(attempt_timer and attempt_timer.expired())
            if local_attempt_timeout:
                counts_for_circuit = False
            # Normal chat selects one Gemini model per pool attempt. A timeout
            # of that model must not disable the next Gemini backup; the shared
            # request deadline still bounds the entire chain. Legacy unpooled
            # cascades retain their existing provider-wide timeout policy.
            if pooled and provider == 'gemini' and reason == 'timeout' and not query_profile.get('simple_rag_mode'):
                gemini_timed_out = True
            if pooled and reason == 'rate_limit':
                model_pools.cooldown(candidate, retry_after_seconds(exc) or 60)
            if provider != "gemini":
                await record_failure(
                    candidate,
                    reason,
                    elapsed_ms,
                    counts=counts_for_circuit,
                    retry_after_seconds=retry_after_seconds(exc),
                )
            last_error = exc
            set_timing_metadata(generation_call_state='failed')
            fallback_reason = fallback_reason or reason
            from routers.services.backend_diagnostics import record_event
            record_event('generation', 'model_attempt_failed', level='warning', details={
                'role': 'generation', 'model': candidate, 'provider': provider,
                'reason': reason, 'latency_ms': elapsed_ms,
                'exception_type': type(exc).__name__,
                'timeout_scope': 'attempt_budget' if local_attempt_timeout else 'provider',
                'attempt_timeout_ms': attempt_timeout * 1000,
                'generation_remaining_ms': max(0, request_deadline - time.monotonic()) * 1000,
                'http_status': getattr(getattr(exc, 'response', None), 'status_code', None)})
            set_timing_metadata(
                fallback_triggered=True,
                fallback_reason=fallback_reason,
                provider_timeout=provider_timeout(provider),
                provider_error_type=reason,
            )
            logger.warning(
                "Provider %s failed after %.0f ms (%s); trying next healthy provider",
                candidate,
                elapsed_ms,
                reason,
            )

        finally:
            model_pools.selected_model.reset(selection_token)
            model_pools.generation_request_count.reset(calls_token)
            if pooled:
                model_pools.release(candidate)

    # The first sentences of retrieved chunks are not necessarily relevant to
    # the question. Keep sources available, but do not present excerpts as an answer.
    timing = current_timing()
    dispatched = int((timing.metadata if timing else {}).get('generation_calls') or 0) > initial_generation_calls
    outcome = 'provider_failed' if dispatched else 'admission_failed'
    set_timing_metadata(generation_outcome=outcome,
                        generation_pool_admission='blocked' if not dispatched else 'admitted')
    if dispatched:
        logger.error("All configured providers failed after generation was dispatched: %s", last_error)
    else:
        logger.warning("Generation was not dispatched; all candidates were skipped or blocked locally")
    if (query_profile or {}).get("task") == "conversation":
        unavailable_answer = openai_service.build_busy_answer(language)
    else:
        unavailable_answer = (
            SOURCES_FOUND_SERVICE_UNAVAILABLE_MESSAGE
            if context_chunks
            else HIGH_TRAFFIC_MESSAGE
        )
    return {
        "answer": unavailable_answer,
        "sources": build_sources_payload(context_chunks),
        "model": None,
        "intent": intent,
        "language": language,
        "kb_used": bool(context_chunks),
        "provider_unavailable": True,
        "generation_outcome": outcome,
    }


async def translate_answer_with_fallback(
    *,
    answer_text: str,
    target_language: str,
    chat_history: List[Dict[str, str]],
    source_language: str,
) -> str:
    """Use the configured healthy provider order for translation requests."""

    provider_args = {
        "answer_text": answer_text,
        "target_language": target_language,
        # The original answer is the entire translation target.  Passing the
        # surrounding conversation invites a provider to answer an earlier
        # question or to mistake a prior message for missing source text.
        "chat_history": [],
        "source_language": source_language,
    }
    providers = configured_provider_order()
    for index, provider in enumerate(providers):
        if provider != "gemini":
            try:
                await acquire_provider(provider)
            except CircuitOpenError:
                logger.warning("Skipping %s translation because its circuit is open", provider)
                continue
        started = time.perf_counter()
        try:
            call = (
                groq_provider.translate_answer_preserving_format(**provider_args)
                if provider == "groq"
                else gemini_service.translate_answer_preserving_format(**provider_args)
            )
            translated = await asyncio.wait_for(call, timeout=provider_timeout(provider))
            if not str(translated or "").strip():
                raise RuntimeError(f"{provider} returned an empty translation.")
            if translation_looks_like_missing_input_request(translated, answer_text):
                raise RuntimeError(f"{provider} returned a translation meta-response.")
            elapsed_ms = (time.perf_counter() - started) * 1000
            add_timing("fallback_model_ms" if index else "primary_model_ms", elapsed_ms)
            if provider != "gemini":
                await record_success(provider, elapsed_ms)
            set_timing_metadata(
                provider_used=provider,
                fallback_triggered=index > 0,
                provider_timeout=provider_timeout(provider),
            )
            return translated
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            add_timing("fallback_model_ms" if index else "primary_model_ms", elapsed_ms)
            reason, counts_for_circuit = classify_provider_error(exc)
            if provider != "gemini":
                await record_failure(
                    provider,
                    reason,
                    elapsed_ms,
                    counts=counts_for_circuit,
                    retry_after_seconds=retry_after_seconds(exc),
                )
            logger.warning("%s translation failed (%s); trying next provider", provider, reason)
    raise RuntimeError("translation providers unavailable")


TRANSLATION_MISSING_INPUT_RE = re.compile(
    r"(?:need|kailangan|kinahanglan).{0,45}(?:original|source).{0,35}(?:text|answer|content|teksto)",
    re.IGNORECASE | re.DOTALL,
)


def translation_looks_like_missing_input_request(translated: str, original: str) -> bool:
    """Reject a provider asking for text that was already supplied to it.

    This is deliberately content-neutral: it protects the translation action
    from a provider meta-response without inspecting or changing the legal
    answer itself.  A source that genuinely contains the same wording remains
    translatable because the marker is present in both texts.
    """

    return bool(
        TRANSLATION_MISSING_INPUT_RE.search(str(translated or ""))
        and not TRANSLATION_MISSING_INPUT_RE.search(str(original or ""))
    )

TRANSLATION_KEYWORDS_RE = re.compile(
    r"\b(translate|translation|translated|isalin|itranslate|i-translate|e-translate|etranslate)\b",
    re.IGNORECASE,
)
TRANSLATION_ACTION_RE = re.compile(
    r"(?:\b(?:say|make)\s+(?:this|that|it|your\s+answer|the\s+answer)\s+"
    r"(?:in|into)?\s*|\b(?:paki[-\s]?(?:translate|salin)|ihambal)\s+"
    r"(?:sa|into|in)?\s*|\bmahim[o]?\s+(?:ini|ina|it)\s+(?:sa|in)?\s*)"
    r"(?:english|eng|filipino|tagalog|hiligaynon|ilonggo)\b",
    re.IGNORECASE,
)
TRANSLATION_LANGUAGE_ALIASES = {
    "english": "en",
    "eng": "en",
    "filipino": "fil",
    "tagalog": "fil",
    "hiligaynon": "hil",
    "ilonggo": "hil",
}
TRANSLATION_TARGET_RE = re.compile(
    r"\b(?:english|eng|filipino|tagalog|hiligaynon|ilonggo)\b",
    re.IGNORECASE,
)
DISCLAIMER_MARKERS = (
    "not legal advice",
    "hindi legal na payo",
    "indi ini legal nga payo",
    "indi legal nga payo",
)
SENSITIVE_PATTERNS = (
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[email redacted]"),
    (re.compile(r"\b(?:\+?63|0)?9\d{9}\b"), "[phone redacted]"),
    (re.compile(r"\b\d{2,4}[- ]?\d{3,4}[- ]?\d{3,4}\b"), "[number redacted]"),
)
ORDINARY_CONVERSATION_INTENTS = {
    "greeting",
    "small_talk",
    "identity",
    "identity_question",
    "capability_question",
    "gratitude",
    "thanks",
    "farewell",
    "goodbye",
    "acknowledgement",
}
MODEL_GENERATED_CONVERSATION_INTENTS = {"greeting", "small_talk"}
CLARIFICATION_ELIGIBLE_INTENTS = {
    "general_inquiry",
    "wage_inquiry",
    "leave_benefits",
    "termination_rights",
    "13th_month",
    "employment_status",
    "occupational_safety",
    "holiday_pay",
}
# ── Relevance threshold ────────────────────────────────────────────────────────
# Candidate chunks below this are discarded before reranking. The stricter
# 0.65 threshold below controls whether LaborLens may answer at all.
MIN_RELEVANCE_SCORE = max(0.0, min(float(settings.RAG_MIN_RELEVANCE_SCORE), 1.0))
RELIABLE_SOURCE_THRESHOLD = max(
    MIN_RELEVANCE_SCORE,
    min(float(settings.RAG_RELIABLE_SOURCE_THRESHOLD), 1.0),
)
LIMITED_SOURCE_THRESHOLD = 0.70
OUTDATED_AMOUNT_TOPICS = {"minimum_wage", "social_security", "sss", "philhealth", "pagibig"}
GUEST_MESSAGE_LIMIT = 5
GUEST_SESSION_TTL_SECONDS = 60 * 60 * 24
GUEST_SESSIONS: Dict[str, Dict[str, Any]] = {}

FOLLOW_UPS = {
    "en": {
        "wage_inquiry":        ["How is overtime pay computed?", "What is night differential?", "What is COLA?"],
        "leave_benefits":      ["How many sick leave days am I entitled to?", "Can unused SIL be cashed out?", "What is maternity leave?"],
        "termination_rights":  ["What are just causes for termination?", "How is separation pay computed?", "What is illegal dismissal?"],
        "13th_month":          ["When must 13th month pay be released?", "Who is exempt from 13th month pay?", "Is bonus the same as 13th month?"],
        "employment_status":   ["When does probationary become regular?", "What rights do contractual workers have?", "What is project-based employment?"],
        "occupational_safety": ["What are employer OSH obligations?", "How do I report a workplace accident?", "What is the OSH Law?"],
        "holiday_pay":         ["What is double pay on regular holidays?", "Which days are regular vs special holidays?", "Do part-timers get holiday pay?"],
        "general_inquiry":     ["What are my basic rights as an employee?", "Where can I ask for help with a workplace concern?", "What benefits may apply to my job?"],
    },
    "fil": {
        "wage_inquiry":        ["Paano kinukuwenta ang overtime pay?", "Ano ang night differential?", "Ano ang COLA?"],
        "leave_benefits":      ["Ilang sick leave days ang puwede sa akin?", "Puwede bang i-cash ang unused SIL?", "Ano ang maternity leave?"],
        "termination_rights":  ["Ano ang just causes sa termination?", "Paano kinukuwenta ang separation pay?", "Ano ang illegal dismissal?"],
        "13th_month":          ["Kailan dapat ibigay ang 13th month pay?", "Sino ang exempted sa 13th month pay?", "Pareho ba ang bonus at 13th month?"],
        "employment_status":   ["Kailan nagiging regular ang probationary?", "Ano ang karapatan ng contractual workers?", "Ano ang project-based employment?"],
        "occupational_safety": ["Ano ang OSH duties ng employer?", "Paano mag-report ng workplace accident?", "Ano ang OSH Law?"],
        "holiday_pay":         ["Ano ang double pay sa regular holiday?", "Alin ang regular at special holidays?", "May holiday pay ba ang part-time?"],
        "general_inquiry":     ["Ano ang basic rights ko bilang empleyado?", "Saan ako puwedeng humingi ng tulong sa problema sa trabaho?", "Anong benefits ang maaaring naaangkop sa trabaho ko?"],
    },
    "hil": {
        "wage_inquiry":        ["Paano ginakuwenta ang overtime pay?", "Ano ang night differential?", "Ano ang COLA?"],
        "leave_benefits":      ["Pila ka sick leave days ang puwede sa akon?", "Puwede bala i-cash ang unused SIL?", "Ano ang maternity leave?"],
        "termination_rights":  ["Ano ang just causes sa termination?", "Paano ginakuwenta ang separation pay?", "Ano ang illegal dismissal?"],
        "13th_month":          ["San-o dapat ihatag ang 13th month pay?", "Sin-o ang exempted sa 13th month pay?", "Pareho bala ang bonus kag 13th month?"],
        "employment_status":   ["San-o nagaka-regular ang probationary?", "Ano ang katungod sang contractual workers?", "Ano ang project-based employment?"],
        "occupational_safety": ["Ano ang OSH duties sang employer?", "Paano mag-report sang workplace accident?", "Ano ang OSH Law?"],
        "holiday_pay":         ["Ano ang double pay sa regular holiday?", "Diin ang regular kag special holidays?", "May holiday pay bala ang part-time?"],
        "general_inquiry":     ["Ano ang basic rights ko bilang empleyado?", "Sa diin ako puwede mangayo sang bulig sa problema sa obra?", "Ano nga benepisyo ang mahimo mag-aplikar sa obra ko?"],
    },
}

STATIC_RESPONSES = {
    "en": {
        "identity": "I'm LaborLens. I help explain Philippine workplace rules for people in both private companies and government agencies, including pay, leave, benefits, job-related concerns, and civil service rules. The rules can differ depending on your job and sector.",
        "capability_question": (
            "I can help you understand Philippine workplace rules, whether you work for a private company or a government agency.\n\n"
            "You can ask about:\n"
            "- Pay, overtime, leave, benefits, and retirement\n"
            "- Employment contracts, government appointments, and job security\n"
            "- Resignation, dismissal, and workplace complaints\n"
            "- Workplace safety, harassment, and employee rights\n"
            "- Private-sector topics such as minimum wage, 13th-month pay, and kasambahay rights\n"
            "- Government topics such as civil service rules, leave credits, and employee discipline\n\n"
            "The rules can differ between the two sectors. Choose a topic or tell me about your situation and where you work."
        ),
        "gratitude": "You're welcome. If you have another question about work in a private company or government agency, just send it here.",
        "farewell": "Take care. You can come back anytime with a question about work in a private company or government agency.",
    },
    "fil": {
        "identity": "Ako si LaborLens. Tumutulong akong magpaliwanag ng mga patakaran sa trabaho sa Pilipinas para sa mga nagtatrabaho sa pribadong kumpanya at sa gobyerno. Maaari kang magtanong tungkol sa sahod, leave, benepisyo, problema sa trabaho, at civil service rules. Maaaring magkaiba ang patakaran depende sa trabaho at sektor mo.",
        "capability_question": (
            "Matutulungan kitang maintindihan ang mga patakaran sa trabaho sa Pilipinas, sa pribadong kumpanya man o sa ahensiya ng gobyerno.\n\n"
            "Maaari kang magtanong tungkol sa:\n"
            "- Sahod, overtime, leave, benepisyo, at pagreretiro\n"
            "- Kontrata sa trabaho, appointment sa gobyerno, at seguridad sa trabaho\n"
            "- Pagre-resign, pagkakatanggal, at reklamo sa trabaho\n"
            "- Kaligtasan, harassment, at karapatan ng empleyado\n"
            "- Para sa pribadong sektor: minimum wage, 13th-month pay, at karapatan ng kasambahay\n"
            "- Para sa gobyerno: civil service rules, leave credits, at pagdidisiplina sa empleyado\n\n"
            "Maaaring magkaiba ang patakaran sa dalawang sektor. Pumili ng paksa o ikuwento ang sitwasyon mo at kung saan ka nagtatrabaho."
        ),
        "gratitude": "Walang anuman. Kung may iba ka pang tanong tungkol sa trabaho sa pribadong kumpanya o sa gobyerno, sabihin mo lang.",
        "farewell": "Ingat. Puwede kang bumalik anumang oras kung may tanong ka tungkol sa trabaho sa pribadong kumpanya o sa gobyerno.",
    },
    "hil": {
        "identity": "Ako si LaborLens. Nagabulig ako magpaathag sang mga patakaran sa obra sa Pilipinas para sa mga nagaobra sa pribado nga kompanya kag sa gobyerno. Puwede ka mamangkot parte sa sweldo, leave, benepisyo, problema sa obra, kag civil service rules. Mahimo maglain ang patakaran suno sa imo obra kag sektor.",
        "capability_question": (
            "Mabuligan ta ikaw nga maintindihan ang mga patakaran sa obra sa Pilipinas, sa pribado nga kompanya man ukon sa ahensya sang gobyerno.\n\n"
            "Puwede ka mamangkot parte sa:\n"
            "- Sweldo, overtime, leave, benepisyo, kag pagretiro\n"
            "- Kontrata sa obra, appointment sa gobyerno, kag seguridad sa obra\n"
            "- Pag-resign, pagtanggal, kag reklamo sa obra\n"
            "- Kaluwasan, harassment, kag katungod sang empleyado\n"
            "- Para sa pribado nga sektor: minimum wage, 13th-month pay, kag katungod sang kasambahay\n"
            "- Para sa gobyerno: civil service rules, leave credits, kag pagdisiplina sa empleyado\n\n"
            "Mahimo maglain ang patakaran sa duha ka sektor. Pili sang topic ukon isugid ang imo sitwasyon kag kon sa diin ka nagaobra."
        ),
        "gratitude": "Walay sapayan. Kon may iban ka pa nga pamangkot parte sa obra sa pribado nga kompanya ukon sa gobyerno, isulat lang diri.",
        "farewell": "Halong. Puwede ka magbalik bisan san-o kon may pamangkot ka parte sa obra sa pribado nga kompanya ukon sa gobyerno.",
    },
}

CLARIFICATION_COPY = {
    "en": {
        "intro": "I can help with that. I just need a bit more detail first.",
        "ask": "Can you tell me:",
        "why": "Why I'm asking: labor answers can change depending on the worker's location, dates, employment status, pay basis, and documents.",
        "options_intro": "If it's easier, you can just reply with the closest topic:",
        "next": "Once you send a bit more detail, I'll give you a clearer answer based on the knowledge base.",
    },
    "fil": {
        "intro": "Matutulungan kita rito. Kailangan ko lang muna ng kaunting detalye.",
        "ask": "Puwede mo bang sabihin:",
        "why": "Bakit ko ito tinatanong: nagbabago ang sagot depende sa lokasyon, petsa, employment status, basehan ng sahod, at dokumento.",
        "options_intro": "Kung mas madali, puwede ka ring sumagot gamit ang pinakamalapit na topic:",
        "next": "Kapag may kaunti pang detalye na, ibibigay ko ang mas malinaw na sagot batay sa knowledge base.",
    },
    "hil": {
        "intro": "Mabuligan ta ikaw sini. Kinahanglan ko lang anay sing gamay pa nga detalye.",
        "ask": "Puwede mo bala isugid:",
        "why": "Ngaa ginapamangkot ko ini: naga depende ang sabat sa lokasyon, petsa, employment status, basehan sang sweldo, kag mga dokumento.",
        "options_intro": "Kon mas hapos, puwede ka man magsabat pinaagi sa pinakamalapit nga topic:",
        "next": "Kon may dugang na nga detalye, hatagan ta ikaw sing mas klaro nga sabat base sa knowledge base.",
    },
}

CLARIFICATION_QUESTIONS = {
    "en": {
        "wage_inquiry": [
            "Are you asking about minimum wage, unpaid salary, overtime, night differential, or another wage issue?",
            "What city or region does the worker work in?",
            "What dates, hours, or pay period are involved?",
        ],
        "leave_benefits": [
            "Which leave or benefit are you asking about: service incentive leave, sick leave, maternity/paternity leave, or another benefit?",
            "Has the employee worked for at least one year, and are they regular, probationary, or contractual?",
            "Was the leave denied, unpaid, unused, or are you asking how to apply?",
        ],
        "termination_rights": [
            "Was the worker fired, suspended, forced to resign, or did they resign voluntarily?",
            "What reason did the employer give, and was written notice or a hearing provided?",
            "How long has the worker been employed and what is their employment status?",
        ],
        "13th_month": [
            "Are you asking about eligibility, computation, deadline, or non-payment?",
            "What are the employee's work dates and salary basis?",
            "Has the employer already paid part or none of the 13th month pay?",
        ],
        "employment_status": [
            "Are you asking about regular, probationary, contractual, project-based, or seasonal employment?",
            "When did the worker start, and what work do they actually perform?",
            "Is there a written contract or repeated renewal?",
        ],
        "occupational_safety": [
            "Is this about an accident, unsafe condition, illness, or required safety equipment?",
            "When and where did it happen, and was it reported to the employer?",
            "What action or remedy do you want to know about?",
        ],
        "holiday_pay": [
            "Which holiday or date are you asking about?",
            "Did the employee work that day, or was it a rest day/no-work day?",
            "Is the employee daily-paid, monthly-paid, part-time, or regular full-time?",
        ],
        "general_inquiry": [
            "What specific workplace issue do you need help with?",
            "Are you asking as an employee, employer, applicant, or resigned worker?",
            "What dates, location/region, amount, or document details matter to the situation?",
        ],
    },
    "fil": {
        "wage_inquiry": [
            "Tungkol ba ito sa minimum wage, unpaid salary, overtime, night differential, o ibang sahod issue?",
            "Saang lungsod o rehiyon nagtatrabaho ang worker?",
            "Anong petsa, oras, o pay period ang involved?",
        ],
        "leave_benefits": [
            "Anong leave o benefit ang tinatanong mo: service incentive leave, sick leave, maternity/paternity leave, o iba pa?",
            "Nakapagtrabaho na ba ang empleyado ng at least one year, at regular, probationary, o contractual ba siya?",
            "Na-deny ba ang leave, hindi ba binayaran, unused ba, o nagtatanong ka kung paano mag-apply?",
        ],
        "termination_rights": [
            "Tinanggal ba ang worker, sinuspinde, pinilit mag-resign, o voluntary resignation ito?",
            "Anong dahilan ang ibinigay ng employer, at may written notice o hearing ba?",
            "Gaano na katagal nagtatrabaho ang worker at ano ang employment status niya?",
        ],
        "13th_month": [
            "Tungkol ba ito sa eligibility, computation, deadline, o hindi pagbabayad?",
            "Ano ang work dates at salary basis ng empleyado?",
            "May nabayaran na bang bahagi o wala pa talagang 13th month pay?",
        ],
        "employment_status": [
            "Regular, probationary, contractual, project-based, o seasonal employment ba ang tinatanong mo?",
            "Kailan nagsimula ang worker, at ano ang aktwal niyang trabaho?",
            "May written contract ba o paulit-ulit na renewal?",
        ],
        "occupational_safety": [
            "Tungkol ba ito sa accident, unsafe condition, illness, o required safety equipment?",
            "Kailan at saan ito nangyari, at na-report ba sa employer?",
            "Anong action o remedy ang gusto mong malaman?",
        ],
        "holiday_pay": [
            "Anong holiday o petsa ang tinatanong mo?",
            "Pumasok ba ang empleyado sa araw na iyon, o rest day/no-work day ba?",
            "Daily-paid, monthly-paid, part-time, o regular full-time ba ang empleyado?",
        ],
        "general_inquiry": [
            "Anong specific workplace issue ang kailangan mong tulong?",
            "Nagtatanong ka ba bilang employee, employer, applicant, o resigned worker?",
            "Anong petsa, lokasyon/rehiyon, halaga, o document details ang mahalaga sa sitwasyon?",
        ],
    },
    "hil": {
        "wage_inquiry": [
            "Parte bala ini sa minimum wage, unpaid salary, overtime, night differential, ukon iban nga suhol issue?",
            "Sa diin nga siyudad ukon rehiyon nagatrabaho ang worker?",
            "Ano nga petsa, oras, ukon pay period ang involved?",
        ],
        "leave_benefits": [
            "Ano nga leave ukon benefit ang ginapamangkot mo: service incentive leave, sick leave, maternity/paternity leave, ukon iban pa?",
            "Nakaubra na bala ang empleyado sang at least one year, kag regular, probationary, ukon contractual bala siya?",
            "Na-deny bala ang leave, wala nabayaran, unused, ukon gusto mo mahibaluan paano mag-apply?",
        ],
        "termination_rights": [
            "Gintanggal bala ang worker, gin-suspend, ginpilit mag-resign, ukon voluntary resignation ini?",
            "Ano nga rason ang ginhatag sang employer, kag may written notice ukon hearing bala?",
            "Pila na kadugay nagaubra ang worker kag ano iya employment status?",
        ],
        "13th_month": [
            "Parte bala ini sa eligibility, computation, deadline, ukon wala pagbayad?",
            "Ano ang work dates kag salary basis sang empleyado?",
            "May nabayaran na bala nga parte ukon wala pa gid 13th month pay?",
        ],
        "employment_status": [
            "Regular, probationary, contractual, project-based, ukon seasonal employment bala ang ginapamangkot mo?",
            "San-o nagsugod ang worker, kag ano ang aktwal niya nga obra?",
            "May written contract bala ukon sige-sige nga renewal?",
        ],
        "occupational_safety": [
            "Parte bala ini sa accident, unsafe condition, illness, ukon required safety equipment?",
            "San-o kag diin natabo, kag na-report bala sa employer?",
            "Ano nga action ukon remedy ang gusto mo mahibaluan?",
        ],
        "holiday_pay": [
            "Ano nga holiday ukon petsa ang ginapamangkot mo?",
            "Nagtrabaho bala ang empleyado sadto nga adlaw, ukon rest day/no-work day bala?",
            "Daily-paid, monthly-paid, part-time, ukon regular full-time bala ang empleyado?",
        ],
        "general_inquiry": [
            "Ano nga specific workplace issue ang kinahanglan mo bulig?",
            "Nagapamangkot ka bala bilang employee, employer, applicant, ukon resigned worker?",
            "Ano nga petsa, lokasyon/rehiyon, kantidad, ukon document details ang importante sa sitwasyon?",
        ],
    },
}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    language: Optional[str]   = None
    previous_intent: Optional[str] = None


class GuestChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    language: Optional[str] = None
    previous_intent: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)


def best_confidence_score(sources: Any) -> Optional[float]:
    """Return the strongest saved source similarity without inventing a zero."""
    if isinstance(sources, str):
        try:
            sources = json.loads(sources)
        except (TypeError, json.JSONDecodeError):
            sources = []
    scores: List[float] = []
    for source in sources if isinstance(sources, list) else []:
        if not isinstance(source, dict):
            continue
        raw_score = source.get("similarity_score", source.get("similarity", source.get("score")))
        if raw_score is None:
            continue
        try:
            scores.append(float(raw_score))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else None


def confidence_badge_label(
    confidence_score: Optional[float],
    confidence_level: Optional[str] = None,
    is_follow_up: bool = False,
    *,
    verified_partial: bool = False,
) -> Optional[str]:
    normalized_level = str(confidence_level or "").strip().lower().replace("_", " ")
    if normalized_level == 'unverified':
        return 'Unverified test answer'
    if normalized_level == "not required":
        return None
    if normalized_level == "insufficient":
        return "Insufficient legal source"
    if verified_partial and normalized_level == 'partial':
        # A successful independent partial check, not cosine similarity,
        # determines this label. The numeric retrieval score stays unchanged.
        return 'Partial source match'
    if is_follow_up and (confidence_score is None or confidence_score < RELIABLE_SOURCE_THRESHOLD):
        return "Partial source match"
    if confidence_score is None or confidence_score < RELIABLE_SOURCE_THRESHOLD:
        return "Insufficient legal source"
    if confidence_score < LIMITED_SOURCE_THRESHOLD or normalized_level in {"partial", "conflicting", "insufficient"}:
        return "Partial source match"
    if confidence_score < 0.90:
        return "Source match"
    return "High confidence match" if normalized_level in {"direct", "high", "source"} else "Partial source match"


class ChatResponse(BaseModel):
    session_id: str
    message_id: int
    answer: str
    intent: str
    language: str
    sources: List[dict]
    follow_up_questions: List[str]
    kb_found: bool
    source_match: Optional[str] = None
    topic: Optional[str] = None
    category: Optional[str] = None
    display_mode: Optional[str] = None
    calculation: Optional[Dict[str, Any]] = None
    quick_replies: List[str] = Field(default_factory=list)
    clarification_question: Optional[str] = None
    clarification_needed: bool = False
    confidence_level: str = "Insufficient"
    confidence_reason: str = "No source match details available."
    confidence_score: Optional[float] = None
    confidence_label: Optional[str] = None
    guest_usage: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
        self.answer = normalize_answer_disclaimers(self.answer)
        if not self.category and self.topic:
            self.category = get_sidebar_category(topic=self.topic)
        if not self.category:
            for source in self.sources or []:
                raw_category = (source or {}).get("category")
                if raw_category:
                    self.category = normalize_kb_category(raw_category)
                    break
        if not self.category:
            self.category = "Others"
        if self.confidence_score is None:
            self.confidence_score = best_confidence_score(self.sources)
        if 'independent verification disabled' in self.confidence_reason and self.sources:
            self.confidence_label = 'Source-based answer'
        if self.confidence_label is None:
            self.confidence_label = confidence_badge_label(
                self.confidence_score,
                self.confidence_level,
            )


async def save_assistant_message(
    *,
    session_id: str,
    user_id: int,
    content: str,
    language: str,
    sources: List[dict],
    intent: str,
    confidence_level: str,
    confidence_reason: str,
    is_follow_up: bool = False,
    verified_partial: bool = False,
) -> tuple[int, Optional[float], Optional[str]]:
    from routers.services.database import confidence_reason_for_storage
    content = normalize_answer_disclaimers(content)
    confidence_score = best_confidence_score(sources)
    confidence_label = confidence_badge_label(confidence_score, confidence_level, is_follow_up,
                                             verified_partial=verified_partial)
    if 'independent verification disabled' in confidence_reason and sources:
        confidence_label = 'Source-based answer'
    message_id = await execute_insert(
        """
        INSERT INTO chat_messages
            (session_id,user_id,role,content,language,sources,intent,
             confidence_level,confidence_reason,confidence_score,confidence_label)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            session_id,
            user_id,
            "assistant",
            content,
            language,
            json.dumps(sources),
            intent,
            confidence_level,
            confidence_reason_for_storage(confidence_reason),
            confidence_score,
            confidence_label,
        ),
    )
    # A child-message insert does not update the parent session's timestamp in
    # MySQL, so explicitly touch it after every successfully saved answer.
    await execute_insert(
        "UPDATE chat_sessions SET updated_at=CURRENT_TIMESTAMP WHERE id=%s AND user_id=%s",
        (session_id, user_id),
    )
    return message_id, confidence_score, confidence_label


class FeedbackRequest(BaseModel):
    message_id: int
    rating: str
    comment: Optional[str] = None


class SessionUpdateRequest(BaseModel):
    title: str


def is_kb_resolved(context_chunks: list) -> bool:
    """
    Returns True only if at least one chunk has a relevance score
    above MIN_RELEVANCE_SCORE. Otherwise the query is unresolved.
    """
    if not context_chunks:
        return False
    return any(chunk.get("score", 0) >= MIN_RELEVANCE_SCORE for chunk in context_chunks)


async def filter_searchable_chunks(context_chunks: list, question: str = '') -> list:
    if not context_chunks:
        return []

    doc_ids = []
    for chunk in context_chunks:
        doc_db_id = str((chunk.get("metadata") or {}).get("doc_db_id") or "").strip()
        if doc_db_id.isdigit():
            doc_ids.append(int(doc_db_id))
    doc_ids = sorted(set(doc_ids))
    if not doc_ids:
        # Unregistered/staging document IDs must never be treated like legacy
        # evidence without a database owner.
        return [c for c in context_chunks if not str((c.get('metadata') or {}).get('doc_db_id') or '').strip()]

    placeholders = ",".join(["%s"] * len(doc_ids))
    from routers.services.legal_version_ranking import historical_query
    legal_versions = "'active','current'"
    if historical_query(question):
        legal_versions += ",'superseded','replaced','obsolete','outdated','repealed'"
    active_rows = await execute_query(
        f"""
        SELECT id,index_version
        FROM uploaded_documents
        WHERE id IN ({placeholders})
          AND is_active=1
          AND status='indexed'
          AND COALESCE(version_status, 'active') IN ({legal_versions})
        """,
        tuple(doc_ids),
    )
    active_ids = {str(row["id"]): str(row.get("index_version") or 1) for row in active_rows}
    filtered = []
    for chunk in context_chunks:
        doc_db_id = str((chunk.get("metadata") or {}).get("doc_db_id") or "").strip()
        chunk_version = str((chunk.get('metadata') or {}).get('index_version') or 1)
        if not doc_db_id or (doc_db_id in active_ids and chunk_version == active_ids[doc_db_id]):
            filtered.append(chunk)
    return filtered


def choose_response_language(
    detected_language: str,
    requested_language: Optional[str],
    user_language: Optional[str],
    conversation_language: Optional[str] = None,
) -> str:
    """
    Honor an explicit UI language selection. In Auto-detect mode, retain the
    language already used for the current consultation so short or mixed
    follow-ups do not make the reply (and its reading voice) switch languages.
    The detected language is used to establish the first reply language.
    """
    requested = (requested_language or "").strip().lower()
    previous = (conversation_language or "").strip().lower()
    if requested in SUPPORTED_LANGUAGES:
        return requested
    if requested == "auto":
        if previous in SUPPORTED_LANGUAGES:
            return previous
        if detected_language in SUPPORTED_LANGUAGES:
            return detected_language
    if not requested and detected_language in SUPPORTED_LANGUAGES:
        return detected_language
    if user_language in SUPPORTED_LANGUAGES:
        return user_language
    if detected_language in SUPPORTED_LANGUAGES:
        return detected_language
    return "en"


def latest_assistant_language_from_history(history: List[Dict[str, Any]]) -> Optional[str]:
    """Return the response language from the latest assistant message, if known."""
    for message in reversed(history or []):
        if message.get("role") != "assistant":
            continue
        language = str(message.get("language") or "").strip().lower()
        if language in SUPPORTED_LANGUAGES:
            return language
    return None


def get_follow_ups(intent: str, language: str) -> List[str]:
    follow_ups = FOLLOW_UPS.get(language, FOLLOW_UPS["en"])
    return follow_ups.get(intent, follow_ups["general_inquiry"])[:3]


def parse_sources_payload(raw_sources: Any) -> List[dict]:
    if not raw_sources:
        return []
    if isinstance(raw_sources, list):
        return raw_sources
    if isinstance(raw_sources, str):
        try:
            parsed = json.loads(raw_sources)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def redact_sensitive_info(text: str) -> str:
    redacted = text or ""
    for pattern, replacement in SENSITIVE_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def contains_disclaimer(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in DISCLAIMER_MARKERS)


def normalize_answer_disclaimers(answer: str) -> str:
    from routers.services.answer_formatting import normalize_formula_markup
    text = normalize_formula_markup(answer).strip()
    if not text:
        return ""
    # Repair common model Markdown artifacts before answers are saved.
    text = re.sub(r"^\s*\*\*\s*$", "", text, flags=re.MULTILINE)
    if text.startswith("**") and text.count("**") % 2:
        text = text[2:].lstrip()
    if text.endswith("**") and text.count("**") % 2:
        text = text[:-2].rstrip()
    text = re.sub(
        r"\[([^\]]+)\]\(\[[^\]]*\]\((https?://[^)]+)\)\)",
        r"[\1](\2)",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(https?://[^\s)\]]+?)(?:/View|View)(?=[\s)\]]|$)", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*Source\s+(\d+)\s*,\s*source\s*\]", r"Source \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*Source\s+(\d+)\s*,\s*source\s*\)", r"(Source \1)", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\[?source\]?\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)

    heading_re = re.compile(
        r"^\s*(?:\*\*)?[^A-Za-z0-9\n]{0,4}(Direct Answer|Simple Explanation|"
        r"Requirements, Conditions, or Exceptions|Applicable Rates and Formula|Formula|"
        r"Legal Basis|Detailed Explanation|Practical Next Steps|Practical Guidance|"
        r"Disclaimer|Reminder)\s*:?(?:\*\*)?\s*$",
        re.IGNORECASE,
    )
    seen_headings: set[str] = set()
    normalized_lines: List[str] = []
    for line in text.splitlines():
        heading = heading_re.match(line)
        if heading:
            canonical = re.sub(r"[^a-z]", "", heading.group(1).lower())
            if canonical in seen_headings:
                continue
            seen_headings.add(canonical)
        normalized_lines.append(line.rstrip())
    text = "\n".join(normalized_lines).strip()

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    cleaned = []
    seen_disclaimer = False
    seen_amendment_warning = False
    seen_rate_warning = False
    for paragraph in paragraphs:
        if contains_disclaimer(paragraph):
            if seen_disclaimer:
                continue
            seen_disclaimer = True
        lower = paragraph.lower()
        if "may have been amended" in lower or "maaaring naamyendahan" in lower or "mahimo nga na-amiyendahan" in lower:
            if seen_amendment_warning:
                continue
            seen_amendment_warning = True
        if "rates and amounts may have been updated" in lower or "rates at amounts" in lower or "rates kag amounts" in lower:
            if seen_rate_warning:
                continue
            seen_rate_warning = True
        cleaned.append(paragraph)
    return "\n\n".join(cleaned).strip()


def remove_unsupported_answer_urls(
    answer: str,
    sources: List[Dict[str, Any]],
) -> str:
    """Remove provider-added links that are not present in displayed sources.

    Direct validator calls remain strict. On the response path, an otherwise
    useful grounded answer should not be replaced wholesale by a generic
    fallback merely because a provider added an unverified URL.
    """

    if not answer:
        return answer
    allowed = {
        str(source.get(key) or "").strip().rstrip(".,;")
        for source in sources or []
        for key in ("source_url", "official_url")
        if str(source.get(key) or "").strip()
    }

    def replace_link(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2).rstrip(".,;")
        return match.group(0) if url in allowed else label

    cleaned = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        replace_link,
        answer,
        flags=re.IGNORECASE,
    )

    def replace_plain(match: re.Match[str]) -> str:
        url = match.group(0).rstrip(".,;")
        return match.group(0) if url in allowed else ""

    cleaned = re.sub(r"https?://[^\s)\]]+", replace_plain, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def detect_conversation_action(query: str) -> str:
    """Classify lightweight response transformations before legal routing."""

    text = (query or "").strip()
    if not text:
        return "normal_legal_query"
    lowered = text.lower()
    if TRANSLATION_KEYWORDS_RE.search(lowered) or TRANSLATION_ACTION_RE.search(lowered):
        return "translation_request"
    if re.search(
        r"\b(?:in|into|sa)\s+(?:english|eng|filipino|tagalog|hiligaynon|ilonggo)\b|"
        r"\b(?:english|eng|filipino|tagalog|hiligaynon|ilonggo)\s+please\b|"
        r"\b(?:make|say)\s+(?:this|that|it)\s+(?:english|filipino|tagalog|hiligaynon|ilonggo)\b",
        lowered,
    ):
        return "translation_request"
    return "normal_legal_query"


def is_translation_request(query: str) -> bool:
    return detect_conversation_action(query) == "translation_request"


def extract_inline_translation_text(query: str) -> Optional[str]:
    """Return text explicitly supplied after a translation command, if any."""

    text = (query or "").strip()
    if not text:
        return None
    # A colon is the unambiguous inline form: ``translate this to X: text``.
    marker = re.search(
        r"\b(?:translate|isalin|paki[-\s]?(?:translate|salin)|say|make|ihambal)\b[^:\n]{0,120}:\s*(.+)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if marker:
        candidate = marker.group(1).strip().strip('\\\"“”‘’')
        return candidate or None
    # Also accept a dash separator used in chat shorthand.
    marker = re.search(
        r"\b(?:translate|isalin|say|make|ihambal)\b[^\n]{0,120}\s[-–—]\s+(.+)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if marker:
        candidate = marker.group(1).strip().strip('\\\"“”‘’')
        return candidate or None
    return None
    text = (query or "").strip().lower()
    if not text:
        return False
    if TRANSLATION_KEYWORDS_RE.search(text):
        return True
    if re.search(r"\b(in|to)\s+(english|filipino|tagalog|hiligaynon|ilonggo)\b", text):
        return True
    if re.search(r"\b(english|filipino|tagalog|hiligaynon|ilonggo)\s+please\b", text):
        return True
    return False


def requested_translation_language(query: str) -> Optional[str]:
    languages = requested_translation_languages(query)
    return languages[0] if languages else None


def requested_translation_languages(query: str) -> List[str]:
    text = (query or "").strip().lower()
    matches = []
    for phrase, code in TRANSLATION_LANGUAGE_ALIASES.items():
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", text):
            matches.append((match.start(), code))
    matches.sort(key=lambda item: item[0])
    languages: List[str] = []
    for _, code in matches:
        if code not in languages:
            languages.append(code)
    return languages


def build_translation_language_prompt(language: str) -> str:
    if language == "fil":
        return "Maaari kong isalin ang huling sagot. Sabihin mo lang kung gusto mo ito sa English, Filipino, o Hiligaynon."
    if language == "hil":
        return "Mahimo ko ma-translate ang pinakaulihi nga sabat. Isugid lang kung gusto mo ini sa English, Filipino, ukon Hiligaynon."
    return "I can translate the last answer. Just tell me if you want it in English, Filipino, or Hiligaynon."


def build_translation_missing_context(language: str) -> str:
    if language == "fil":
        return "Maaari kong isalin iyon, pero wala pa akong naunang sagot sa chat na ito na puwedeng i-translate."
    if language == "hil":
        return "Mahimo ko ina ma-translate, pero wala pa sing nauna nga sabat sa sini nga chat nga mahimo ko ma-translate."
    return "I can translate that, but there is no earlier answer in this chat yet for me to translate."


def confidence_for_translation(has_sources: bool) -> tuple[str, str]:
    if has_sources:
        return "Not Required", "This is a direct translation of the previous answer; its existing cited sources were reused without new legal retrieval."
    return "Not Required", "This is a direct translation of the previous assistant answer; no new legal-source check was needed."


def translation_unavailable_message(language: str) -> str:
    if language == "fil":
        return "Hindi ko makumpleto ang pagsasalin ngayon dahil pansamantalang hindi available ang translation service. Pakisubukan muli mamaya."
    if language == "hil":
        return "Indi ko makumpleto ang pag-translate subong kay temporaryo nga indi available ang translation service. Palihog sulayi liwat sa ulihi."
    return "I couldn’t complete the translation because the translation service is temporarily unavailable. Please try again shortly."


def is_substantive_assistant_message(message: Dict[str, Any]) -> bool:
    """Exclude loading/error/source-only assistant rows from translation context."""

    content = str(message.get("content") or "").strip()
    if not content:
        return False
    lowered = re.sub(r"[^a-z0-9 ]+", " ", content.lower())
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if lowered in {
        "checking legal sources",
        "sources verified",
        "source match",
        "answer found directly",
        "loading",
    }:
        return False
    if re.match(r"^(?:source|sources|source match|sources verified|citations?)\b", lowered):
        return False
    if re.match(r"^(?:error|unable to|failed to|request failed)\b", lowered):
        return False
    return True


def get_last_assistant_from_history(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for item in reversed(history or []):
        if item.get("role") == "assistant" and is_substantive_assistant_message(item):
            return item
    return None


async def get_last_assistant_from_session(session_id: str, user_id: int) -> Optional[Dict[str, Any]]:
    rows = await execute_query(
        """
        SELECT id, content, language, sources, intent, confidence_level, confidence_reason,
               confidence_score, confidence_label
        FROM chat_messages
        WHERE session_id=%s AND user_id=%s AND role='assistant'
        ORDER BY created_at DESC, id DESC
        LIMIT 10
        """,
        (session_id, user_id),
    )
    for row in rows or []:
        if is_substantive_assistant_message(row):
            return row
    return None


async def get_latest_assistant_language_from_session(
    session_id: str,
    user_id: int,
) -> Optional[str]:
    """Read the current consultation's last reply language for Auto-detect."""
    row = await execute_one(
        """
        SELECT language
        FROM chat_messages
        WHERE session_id=%s AND user_id=%s AND role='assistant'
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (session_id, user_id),
    )
    language = str((row or {}).get("language") or "").strip().lower()
    return language if language in SUPPORTED_LANGUAGES else None


async def get_relevant_assistant_for_source_followup(
    session_id: str,
    user_id: int,
    query: str,
) -> tuple[Optional[Dict[str, Any]], str, bool]:
    requested_topic = infer_source_followup_topic(query)
    rows = await execute_query(
        """
        SELECT id, content, language, sources, intent, confidence_level, confidence_reason,
               confidence_score, confidence_label
        FROM chat_messages
        WHERE session_id=%s AND user_id=%s AND role='assistant'
        ORDER BY created_at DESC, id DESC
        LIMIT 30
        """,
        (session_id, user_id),
    )
    return select_relevant_assistant_for_source_followup(rows or [], query, requested_topic)


def split_answer_body_and_disclaimer(answer: str) -> tuple[str, str]:
    text = (answer or "").strip()
    if not text:
        return "", ""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    body_parts = []
    disclaimer = ""
    for paragraph in paragraphs:
        if contains_disclaimer(paragraph):
            if not disclaimer:
                disclaimer = paragraph
            continue
        body_parts.append(paragraph)
    return "\n\n".join(body_parts).strip(), disclaimer.strip()


def translated_disclaimer(language: str, fallback: Optional[str] = None) -> str:
    profile = get_language_profile(language)
    disclaimer = (profile.get("disclaimer") or "").strip()
    return disclaimer or fallback or SAFETY_DISCLAIMER


def format_translated_answer(body: str, language: str) -> str:
    clean_body = (body or "").strip()
    disclaimer = translated_disclaimer(language)
    if not clean_body:
        return disclaimer
    return f"{clean_body}\n\n{disclaimer}"


def multi_translation_heading(language: str) -> str:
    if language == "fil":
        return "Tagalog"
    if language == "hil":
        return "Hiligaynon"
    return language_name(language)


async def translate_previous_answer(
    query: str,
    target_languages: List[str],
    fallback_language: str,
    previous_message: Optional[Dict[str, Any]],
    chat_history: List[Dict[str, str]],
    inline_text: Optional[str] = None,
) -> Dict[str, Any]:
    if not target_languages:
        return {
            "answer": build_translation_language_prompt(fallback_language),
            "language": fallback_language,
            "sources": [],
            "confidence_level": "Not Required",
            "confidence_reason": "The user asked for a translation but did not specify a target language; no legal retrieval was run.",
            "kb_found": False,
        }
    if not previous_message and not inline_text:
        return {
            "answer": build_translation_missing_context(fallback_language),
            "language": fallback_language,
            "sources": [],
            "confidence_level": "Not Required",
            "confidence_reason": "There was no previous assistant answer available to translate; no legal retrieval was run.",
            "kb_found": False,
        }

    source_message = previous_message or {}
    original_answer = normalize_answer_disclaimers(inline_text or source_message.get("content") or "")
    previous_language = str(source_message.get("language") or "").strip().lower()
    # Inline text is an independent translation target; never attach citations
    # from an unrelated previous answer to it.
    previous_sources = (
        parse_sources_payload(source_message.get("sources"))
        if previous_message and not inline_text
        else []
    )
    body_text, _ = split_answer_body_and_disclaimer(original_answer)
    if not body_text:
        return {
            "answer": build_translation_missing_context(fallback_language),
            "language": fallback_language,
            "sources": [],
            "confidence_level": "Not Required",
            "confidence_reason": "There was no substantive answer available to translate; no legal retrieval was run.",
            "kb_found": False,
        }

    translated_sections = []
    for target_language in target_languages:
        if previous_language == target_language and previous_language in SUPPORTED_LANGUAGES:
            translated_body = body_text
        else:
            try:
                translated_body = await translate_answer_with_fallback(
                    answer_text=body_text,
                    target_language=target_language,
                    chat_history=chat_history,
                    source_language=previous_language,
                )
            except Exception:
                logger.warning("Translation provider unavailable")
                return {
                    "answer": translation_unavailable_message(fallback_language),
                    "language": fallback_language,
                    "sources": previous_sources,
                    "confidence_level": "Not Required",
                    "confidence_reason": "The translation provider was unavailable; no legal retrieval was run.",
                    "kb_found": bool(previous_sources),
                }
        from routers.services.answer_presentation import remove_inline_source_attributions
        translated_answer = remove_inline_source_attributions(
            format_translated_answer(translated_body, target_language)
        )
        if len(target_languages) > 1:
            translated_sections.append(f"## {multi_translation_heading(target_language)}\n{translated_answer}")
        else:
            translated_sections.append(translated_answer)
    confidence_level, confidence_reason = confidence_for_translation(bool(previous_sources))
    final_answer = "\n\n".join(section.strip() for section in translated_sections if section.strip()).strip()
    return {
        "answer": final_answer,
        "language": target_languages[0],
        "sources": previous_sources,
        "confidence_level": confidence_level,
        "confidence_reason": confidence_reason,
        "kb_found": bool(previous_sources),
    }


def is_source_followup_request(query: str) -> bool:
    return bool(SOURCE_FOLLOWUP_RE.search(query or ""))


def source_display_name(source: Dict[str, Any]) -> str:
    article_number = str(source.get("article_number") or "").strip()
    article_title = str(source.get("article_title") or "").strip()
    citation = str(source.get("citation") or "").strip()
    document_title = str(source.get("document_title") or "").strip()
    source_name = str(source.get("source_name") or "").strip()
    if article_number and article_title:
        return f"Article {article_number} - {article_title}"
    if citation:
        return citation
    title_blob = f"{document_title} {source.get('source', '')}".lower()
    if "thirteenth" in title_blob or "13th" in title_blob:
        return "DOLE Guidelines on the Payment of Thirteenth-Month Pay"
    if str(source.get("canonical_document_id") or "").strip() == "pd_851":
        return "PD 851 - 13th-Month Pay"
    if document_title.lower().endswith((".pdf", ".docx")) or "_" in document_title:
        return source_name or "Knowledge base source"
    return document_title or source_name or str(source.get("source") or "Knowledge base source").strip()


def source_url(source: Dict[str, Any]) -> str:
    return str(source.get("source_url") or source.get("official_url") or "").strip()


def source_hierarchy_line(source: Dict[str, Any]) -> str:
    parts = []
    document = str(source.get("document_title") or source.get("source_name") or source.get("source") or "").strip()
    article_number = str(source.get("article_number") or "").strip()
    article_title = str(source.get("article_title") or "").strip()
    article_section = str(source.get("article_section") or "").strip()
    book = str(source.get("book") or "").strip()
    rule = str(source.get("rule") or "").strip()
    section = str(source.get("section") or "").strip()
    page = str(source.get("page") or "").strip()
    citation = str(source.get("citation") or "").strip()
    url = source_url(source)
    if document:
        parts.append(f"Document: {document}")
    if citation:
        parts.append(f"Citation: {citation}")
    if article_number or article_title:
        article = " ".join(part for part in [f"Article {article_number}" if article_number else "", article_title] if part)
        parts.append(article)
    if article_section:
        parts.append(f"Article/Section: {article_section}")
    if book:
        parts.append(f"Book: {book}")
    if rule:
        parts.append(f"Rule: {rule}")
    if section:
        parts.append(f"Section: {section}")
    if page:
        parts.append(f"Page: {page}")
    if url:
        parts.append(f"URL: {url}")
    return " | ".join(parts) if parts else source_display_name(source)


def infer_topic_from_previous_sources(previous_message: Optional[Dict[str, Any]], sources: List[Dict[str, Any]]) -> str:
    for item in sources or []:
        topic = str(item.get("topic") or "").strip()
        if topic:
            return topic
    text = (previous_message or {}).get("content") or ""
    if re.search(r"\b13th[-\s]*month|thirteenth\b", text, re.IGNORECASE):
        return "thirteenth_month_pay"
    if re.search(r"\bovertime|undertime|hours\s+of\s+work\b", text, re.IGNORECASE):
        return "overtime_pay"
    if re.search(r"\bholiday\b", text, re.IGNORECASE):
        return "holiday_pay"
    if re.search(r"\bkasambahay|domestic\s+worker\b", text, re.IGNORECASE):
        return "domestic_workers"
    return "general_labor"


def previous_source_status(previous_message: Optional[Dict[str, Any]], sources: List[Dict[str, Any]]) -> tuple[str, str]:
    if not sources:
        return SOURCE_STATUS_INSUFFICIENT, "The previous answer did not store source metadata."
    status = str((previous_message or {}).get("source_match") or "").strip().lower()
    reason = str((previous_message or {}).get("confidence_reason") or "").strip()
    if status in {SOURCE_STATUS_DIRECT, SOURCE_STATUS_PARTIAL, SOURCE_STATUS_INSUFFICIENT, SOURCE_STATUS_NOT_REQUIRED}:
        return status, reason or "This follows the previous answer's saved source-match status."
    confidence = str((previous_message or {}).get("confidence_level") or "").strip().lower()
    if confidence.startswith("direct"):
        return SOURCE_STATUS_DIRECT, reason or "The previous answer was marked as directly supported by saved source metadata."
    if confidence.startswith("partial"):
        return SOURCE_STATUS_PARTIAL, reason or "The previous answer was marked as partially supported by saved source metadata."
    return SOURCE_STATUS_PARTIAL, reason or "This uses the saved sources from the previous assistant answer."


def build_source_followup_response(
    query: str,
    language: str,
    previous_message: Optional[Dict[str, Any]],
    requested_topic: str = "",
    missing_requested_topic: bool = False,
) -> Optional[Dict[str, Any]]:
    if not is_source_followup_request(query):
        return None
    from routers.services.source_request_routing import legal_basis_explanation_requested
    if legal_basis_explanation_requested(query):
        return None
    query_text = query or ""
    is_contextual_request = bool(
        SOURCE_PASSAGE_RE.search(query_text)
        or SOURCE_MATCH_REASON_RE.search(query_text)
        or SOURCE_COMPARE_RE.search(query_text)
        or re.search(r"\b(your|previous|last|earlier|that|this)\s+(answer|response|source|sources)\b", query_text, re.IGNORECASE)
        or re.search(r"\bfor\s+your\s+answer\b", query_text, re.IGNORECASE)
    )
    if not is_contextual_request:
        return None
    if missing_requested_topic:
        readable_topic = (requested_topic or "that topic").replace("_", " ")
        return {
            "answer": (
                "**Direct Answer:**\n"
                f"I do not have a saved sourced answer about {readable_topic} in this conversation yet.\n\n"
                "**What I need:**\nPlease tell me which previous answer you mean, or ask the legal question again so I can retrieve and save the correct sources for it."
            ),
            "sources": [],
            "source_status": SOURCE_STATUS_INSUFFICIENT,
            "source_reason": f"The user requested source details for {readable_topic}, but no earlier assistant message with matching saved sources was found.",
            "intent": "citation_request",
            "topic": requested_topic or "general_labor",
            "display_mode": "simple_message",
            "kb_found": False,
        }
    previous_sources = parse_sources_payload((previous_message or {}).get("sources"))
    if not previous_sources:
        if SOURCE_MATCH_REASON_RE.search(query_text):
            previous_reason = str((previous_message or {}).get("confidence_reason") or "").strip()
            previous_level = str((previous_message or {}).get("confidence_level") or "").strip()
            reason = previous_reason or (
                "The previous answer was treated as partial when the retrieved source support was relevant "
                "but not complete enough to mark every required legal detail as a direct match."
            )
            label = previous_level or "Partial source match"
            return {
                "answer": (
                    f"**Direct Answer:**\nThe previous answer was labeled **{label}** because the source support was not treated as fully direct.\n\n"
                    f"**Reason:**\n{reason}\n\n"
                    "**What this means:**\nA partial match can happen when the retrieved excerpts address the topic, but the system still lacks a complete article, section, page, or required concept for every part of the answer."
                ),
                "sources": [],
                "source_status": SOURCE_STATUS_NOT_REQUIRED,
                "source_reason": "This explains a previous source-match label and does not require new retrieval.",
                "intent": "source_match_explanation",
                "topic": infer_topic_from_previous_sources(previous_message, []),
                "display_mode": "simple_message",
                "kb_found": False,
            }
        return {
            "answer": "I do not have saved source details from the previous answer in this conversation yet.",
            "sources": [],
            "source_status": SOURCE_STATUS_INSUFFICIENT,
            "source_reason": "The user asked for source details, but no previous source metadata was available.",
            "intent": "citation_request",
            "topic": "general_labor",
            "display_mode": "simple_message",
            "kb_found": False,
        }

    topic = requested_topic or infer_topic_from_previous_sources(previous_message, previous_sources)
    source_status, source_reason = previous_source_status(previous_message, previous_sources)
    lines: List[str] = []
    intent = "citation_request"
    display_mode = "legal_card"

    if SOURCE_MATCH_REASON_RE.search(query_text):
        intent = "source_match_explanation"
        display_mode = "legal_card"
        status_label = source_status.replace("_", " ").title()
        lines.append(f"**Direct Answer:**\nThe previous answer was labeled **{status_label}** based on the saved retrieval and validation metadata.")
        lines.append(
            "**Reason:**\n"
            f"{source_reason or 'The retrieved sources supported the topic, but the source set may not have covered every required legal detail.'}"
        )
        lines.append(
            "**What this means:**\n"
            "A partial match usually means the system found relevant legal sources, but the retrieved excerpts or metadata may be incomplete for every part of the answer."
        )
    elif SOURCE_COMPARE_RE.search(query_text):
        intent = "source_comparison"
        display_mode = "legal_card"
        first = previous_sources[0] if len(previous_sources) >= 1 else {}
        second = previous_sources[1] if len(previous_sources) >= 2 else {}
        key_fields = ("canonical_document_id", "article_number", "section", "citation", "content_hash")
        comparisons = []
        for field in key_fields:
            left = str(first.get(field) or "").strip()
            right = str(second.get(field) or "").strip()
            if left or right:
                comparisons.append(f"- {field}: Source 1 = {left or 'missing'}; Source 2 = {right or 'missing'}")
        same_canonical = bool(first and second and str(first.get("canonical_document_id") or "") == str(second.get("canonical_document_id") or ""))
        same_hash = bool(first and second and str(first.get("content_hash") or "") == str(second.get("content_hash") or ""))
        if same_hash:
            direct = "Yes. Source 1 and Source 2 appear to contain the same retrieved passage because their content_hash values match."
        elif same_canonical:
            direct = "They appear to come from the same legal document, but not necessarily the exact same passage because the content_hash values differ or are incomplete."
        else:
            direct = "They do not appear to be exact copies based on the saved canonical_document_id and content_hash metadata."
        lines.append(f"**Direct Answer:**\n{direct}")
        lines.append("**Metadata compared:**\n" + "\n".join(comparisons or ["- No comparable metadata was saved."]))
    elif SOURCE_PASSAGE_RE.search(query_text):
        lines.append("**Direct Answer:**\nHere are the exact retrieved passages saved from the previous answer.")
        passage_lines = []
        for index, item in enumerate(previous_sources[:5], 1):
            excerpt = str(item.get("excerpt") or "").strip()
            if not excerpt:
                continue
            passage_lines.append(f"{index}. {source_display_name(item)}\n{excerpt}")
        lines.append("**Retrieved Passage:**\n" + "\n\n".join(passage_lines or ["No excerpt text was saved for the displayed sources."]))
    else:
        lines.append("**Direct Answer:**\nHere are the article, rule, section, page, and link details saved for the previous answer.")
        lines.append(
            "**Legal Basis:**\n"
            + "\n".join(f"{index}. {source_hierarchy_line(item)}" for index, item in enumerate(previous_sources[:8], 1))
        )

    return {
        "answer": "\n\n".join(lines),
        "sources": previous_sources,
        "source_status": source_status,
        "source_reason": source_reason,
        "intent": intent,
        "topic": topic,
        "display_mode": display_mode,
        "kb_found": True,
    }


async def generate_conversational_reply(
    message: str,
    chat_history: List[Dict[str, Any]],
    language: str,
    intent: str,
) -> str:
    """Generate a fresh response for greetings and small talk without retrieval."""

    result = await generate_response_with_fallback(
        query=message,
        context_chunks=[],
        chat_history=chat_history,
        language=language,
        intent=intent,
        query_profile={"task": "conversation", "topic": "non_legal"},
        source_status=SOURCE_STATUS_NOT_REQUIRED,
    )
    answer = str(result.get("answer") or "").strip()
    return answer or openai_service.build_busy_answer(language)


def get_static_response(intent: str, language: str, message: str = "") -> Optional[str]:
    if intent in MODEL_GENERATED_CONVERSATION_INTENTS:
        return None
    responses = STATIC_RESPONSES.get(language, STATIC_RESPONSES["en"])
    return responses.get(intent)


def prompt_injection_intent(message: str) -> Optional[str]:
    text = message or ""
    if not (PROMPT_INJECTION_RE.search(text) or FORCED_UNSUPPORTED_CITATION_RE.search(text)):
        return None
    return "general_inquiry"


def build_prompt_injection_refusal(language: str) -> str:
    if language == "fil":
        return (
            "Hindi ko puwedeng i-ignore ang legal sources o system rules. "
            "Para maiwasan ang maling sagot, gagamitin ko lang ang approved knowledge-base sources para sa legal claims."
        )
    if language == "hil":
        return (
            "Indi ko puwede i-ignore ang legal sources ukon system rules. "
            "Para malikawan ang sala nga sabat, gamiton ko lang ang approved knowledge-base sources para sa legal claims."
        )
    return (
        "I cannot ignore the legal sources or system rules. "
        "To avoid an unsafe or unsupported answer, I will use only the approved knowledge-base sources for legal claims."
    )




def normalize_static_intent(
    message: str,
    detected_intent: str,
    previous_intent: Optional[str] = None,
) -> str:
    # The learned classifier owns the intent, including its uncertainty
    # fallback. Do not override it with keyword/greeting shortcuts.
    return detected_intent


SAFETY_DISCLAIMER = (
    "This is for informational purposes only and is not legal advice."
)


def ensure_safety_disclaimer(answer: str) -> str:
    cleaned = normalize_answer_disclaimers(answer)
    if contains_disclaimer(cleaned):
        return cleaned
    return f"{cleaned.rstrip()}\n\n{SAFETY_DISCLAIMER}"


def top_chunk_similarity(context_chunks: List[Dict[str, Any]]) -> Optional[float]:
    return best_confidence_score(context_chunks)


AUTHORITY_SCOPED_TOPICS = {
    "thirteenth_month_pay",
    "hours_of_work",
    "night_shift_differential",
    "service_incentive_leave",
    "solo_parent_leave",
    "service_charges",
    "separation_pay",
    "termination",
    "workplace_safety",
    "sexual_harassment",
    "safe_spaces",
    "labor_complaint",
    "jurisprudence_request",
    "telecommuting",
}


def scope_chunks_to_required_authorities(
    chunks: List[Dict[str, Any]],
    query_profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Keep exact governing sources ahead of semantically similar documents.

    A large Department Order collection contains recurring labor-law language.
    For topics with a known controlling law, only narrow the final LLM context
    after every essential authority has been recovered and explicitly verified.
    This avoids both irrelevant-source answers and accidental empty contexts.
    """

    if str(query_profile.get("topic") or "") not in AUTHORITY_SCOPED_TOPICS:
        return chunks

    requirements = required_authorities_for_query(query_profile)
    essential = list(dict.fromkeys(requirements.get("essential") or []))
    # Some current-rule questions retain historical material as supporting evidence
    # but restrict the generator to the authority configured for answer context.
    allowed = list(dict.fromkeys(
        requirements.get("context")
        or essential + list(requirements.get("supporting") or [])
    ))
    if not essential or not allowed:
        return chunks

    scoped: List[Dict[str, Any]] = []
    present: set[str] = set()
    seen: set[str] = set()
    for chunk in chunks or []:
        chunk_authorities = authority_ids_for_chunk(chunk)
        matching = [
            authority_id
            for authority_id in allowed
            if authority_id in chunk_authorities
            and record_explicitly_names_authority(chunk, authority_id)
        ]
        if not matching:
            continue
        key = str(
            (chunk.get("metadata") or {}).get("chunk_id")
            or chunk.get("id")
            or chunk.get("text", "")[:160]
        )
        if key in seen:
            continue
        seen.add(key)
        scoped.append(chunk)
        present.update(matching)

    if scoped and set(essential).issubset(present):
        # A verified governing Act need not crowd out approved guidance that
        # explicitly applies that Act and supplies a missing operative clause.
        # Such guidance supplements, never replaces, the essential authority.
        from routers.services.legal_topic_matrix import source_passage_text, supported_passage_concepts
        required = list(query_profile.get("required_concepts") or [])
        covered = set().union(*(supported_passage_concepts(c, required) for c in scoped))
        for chunk in chunks:
            if chunk in scoped:
                continue
            meta = chunk.get("metadata") or {}
            descriptor = " ".join(str(meta.get(k) or "") for k in ("document_title", "source", "source_name"))
            # A separate occupation-specific order quoting the Labor Code is
            # not general guidance. Admit broad guides/cases or an authority
            # already configured for this question, not arbitrary citing laws.
            broad_guidance = bool(re.search(r"\b(?:handbook|faqs?|manual|guide|guidance)\b", descriptor, re.I))
            broad_guidance |= normalize_kb_category(meta.get("category")) == "Jurisprudence"
            # Existing indexed court decisions may be filed under Labor
            # Relations rather than Jurisprudence. Their own docket identity
            # still identifies case guidance, not the law quoted in the body.
            broad_guidance |= bool(re.search(r"\bG\.?\s*R\.?\s*(?:No\.?\s*)?\d+", descriptor, re.I))
            independent = bool(set(query_profile.get('independent_evidence_components') or []))
            independent_guide = independent and normalize_kb_category(meta.get('category')) == 'Social Benefits'
            if not broad_guidance and not independent_guide and not set(allowed).intersection(authority_ids_for_chunk(chunk)):
                continue
            body = re.sub(r"\s+", " ", source_passage_text(chunk.get("text") or ""))
            body = re.sub(r"\bR\s+epublic\b", "Republic", body, flags=re.I)
            references = False
            for authority in allowed:
                law = re.fullmatch(r"(ra|pd)_(\d+)", authority)
                article = re.fullmatch(r"labor_code_article_(\d+)", authority)
                if law:
                    label = r"(?:RA|Republic\s+Act)" if law[1] == "ra" else r"(?:PD|Presidential\s+Decree)"
                    references |= bool(re.search(rf"\b{label}\s*(?:No\.?\s*)?{law[2]}\b", body, re.I))
                elif article:
                    references |= bool(re.search(rf"\b(?:Art\.?|Article)\s*{article[1]}\b", body, re.I))
            extra = supported_passage_concepts(chunk, required) - covered
            independent_extra = extra.intersection(query_profile.get('independent_evidence_components') or [])
            if extra and (references or (independent_guide and independent_extra)):
                scoped.append(chunk)
                covered.update(extra)
        query_profile["authority_scoped_context"] = {
            "essential": essential,
            "included": sorted(present),
        }
        # The caller applies the prompt cap after coverage-aware ranking.
        # An insertion-order cap here can discard a recovered authority or
        # complementary paragraph before it has been evaluated.
        return scoped
    return chunks


def missing_explicit_required_authorities(
    chunks: List[Dict[str, Any]],
    query_profile: Dict[str, Any],
    *,
    include_supporting: bool = False,
) -> List[str]:
    """Return required laws that are not explicitly identified by a source."""

    requirements = required_authorities_for_query(query_profile)
    required = list(requirements.get("essential") or [])
    if include_supporting:
        required.extend(requirements.get("supporting") or [])
    required = list(dict.fromkeys(required))
    return [
        authority_id
        for authority_id in required
        if not any(
            authority_id in authority_ids_for_chunk(chunk)
            and record_explicitly_names_authority(chunk, authority_id)
            for chunk in chunks or []
        )
    ]


def low_confidence_fallback(language: str) -> str:
    if language == "fil":
        return (
            "Ang partikular na impormasyong ito ay kasalukuyang wala sa aking knowledge base. "
            "Maaari kang tumawag sa DOLE Hotline 1349, pumunta sa pinakamalapit na DOLE Regional o Field Office, "
            "o kumonsulta sa lisensiyadong labor lawyer."
        )
    if language == "hil":
        return (
            "Ang sini nga partikular nga impormasyon wala subong sa akon knowledge base. "
            "Mahimo ka magtawag sa DOLE Hotline 1349, magkadto sa pinakamalapit nga DOLE Regional ukon Field Office, "
            "ukon magkonsulta sa lisensiyado nga labor lawyer."
        )
    return (
        "This specific information is not currently available in my knowledge base. "
        "Please contact the DOLE Hotline at 1349, visit the nearest DOLE Regional or Field Office, "
        "or consult a licensed labor attorney."
    )


def limited_sources_note(language: str) -> str:
    if language == "fil":
        return (
            "Paalala: Limitado ang mga source ko para sa sagot na ito. "
            "Paki-verify muna sa DOLE o sa isang legal professional bago kumilos batay sa impormasyong ito."
        )
    if language == "hil":
        return (
            "Pahanumdom: Limitado ang akon mga source para sa sini nga sabat. "
            "Palihog i-verify anay sa DOLE ukon sa isa ka legal professional antes maghimo sang aksyon."
        )
    return (
        "Note: My sources for this answer are limited. Please verify with DOLE or a legal professional "
        "before acting on this information."
    )


def outdated_amount_note(language: str) -> str:
    if language == "fil":
        return (
            "Paalala: Maaaring na-update na ang mga rate at halaga. Paki-verify ang pinakabagong figures "
            "sa kaukulang ahensiya o DOLE bago magkalkula."
        )
    if language == "hil":
        return (
            "Pahanumdom: Mahimo nga na-update na ang mga rate kag kantidad. Palihog i-verify ang pinakabag-o "
            "nga figures sa nagakaigo nga ahensya ukon DOLE antes magkalkula."
        )
    return (
        "Note: Rates and amounts may have been updated. Please verify the latest figures with the relevant "
        "agency or DOLE before computing."
    )


def amended_law_note(language: str) -> str:
    if language == "fil":
        return (
            "Paalala: Maaaring nabago na ang batas na ito. Paki-verify ang kasalukuyang bersyon sa DOLE "
            "o sa isang lisensiyadong labor lawyer."
        )
    if language == "hil":
        return (
            "Pahanumdom: Mahimo nga na-amiyendahan na ini nga kasuguan. Palihog i-verify ang kasamtangan "
            "nga bersyon sa DOLE ukon sa isa ka lisensiyado nga labor lawyer."
        )
    return (
        "Note: This law may have been amended. Please verify the current version with DOLE or a licensed "
        "labor attorney."
    )


def append_note_before_disclaimer(answer: str, note: str) -> str:
    if not note or note.lower() in (answer or "").lower():
        return answer
    body, disclaimer = split_answer_body_and_disclaimer(answer)
    parts = [part for part in (body, note, disclaimer) if part]
    return "\n\n".join(parts)


def apply_source_accuracy_notes(
    answer: str,
    top_score: Optional[float],
    query_profile: Dict[str, Any],
    language: str,
) -> str:
    if query_profile.get('simple_rag_mode'):
        return answer
    updated = answer
    if top_score is None or top_score < RELIABLE_SOURCE_THRESHOLD:
        return updated
    if top_score is not None and RELIABLE_SOURCE_THRESHOLD <= top_score < LIMITED_SOURCE_THRESHOLD:
        updated = append_note_before_disclaimer(updated, limited_sources_note(language))
    topic = str((query_profile or {}).get("topic") or "").strip().lower()
    if topic in OUTDATED_AMOUNT_TOPICS:
        updated = append_note_before_disclaimer(updated, outdated_amount_note(language))
    # Amendment cautions require evidence of a version concern; a retrieval
    # score alone does not justify adding one to every answer.
    return updated


def low_confidence_result(
    rag: Dict[str, Any],
    language: str,
) -> Optional[Dict[str, Any]]:
    plan = (rag.get('query_profile') or {}).get('request_understanding') or {}
    if plan.get('needs_clarification') and plan.get('question') and not rag.get('chunks'):
        # Missing user facts can be clarified without inventing a legal answer,
        # including when discovery found no passages or timed out.
        return dict(answer=plan['question'], sources=[], source_status=SOURCE_STATUS_NOT_REQUIRED,
                    source_reason='An essential user detail is missing.', kb_resolved=False,
                    validation={})
    if rag.get('retrieval_unavailable'):
        reason = 'Source discovery timed out. Please try again.'
        return dict(answer=reason, sources=[], provider_unavailable=True,
                    source_status=SOURCE_STATUS_PROVIDER_UNAVAILABLE, source_reason=reason,
                    kb_resolved=False, validation={})
    top_score = rag.get("top_similarity_score")
    if rag.get("source_status") in {SOURCE_STATUS_DIRECT, SOURCE_STATUS_PARTIAL} and rag.get("chunks"):
        # Exact authority metadata and direct concept coverage outrank a weak
        # embedding score. Keep verified recovered provisions available.
        return None
    if rag.get("source_status") not in {SOURCE_STATUS_INSUFFICIENT, SOURCE_STATUS_CONFLICTING} and top_score is not None and top_score >= RELIABLE_SOURCE_THRESHOLD:
        return None
    fallback_answer = low_confidence_fallback(language)
    if (rag.get("query_profile") or {}).get("topic") == "approved_leave_recall":
        fallback_answer = {
            "fil": "Ang tanong mo ay kung puwedeng pagtrabahuhin ang empleyado habang approved ang leave. Hindi sapat ang nakuha kong mga source para matukoy ang tuntunin sa pag-recall o pagkansela ng approved leave. Ang probisyon tungkol sa leave entitlement lamang ay hindi sumasagot dito. Makakatulong ang leave policy, approval, o kaugnay na kasunduan para sa karagdagang pagsusuri; hindi pa ako makapagbigay ng tiyak na oo o hindi.",
            "en": "Your question is whether an employer can require work during approved leave. The retrieved sources do not establish the applicable recall or cancellation rule. Leave-entitlement provisions alone do not answer that question. The leave policy, approval, or relevant agreement would help further review; I cannot establish a definite yes or no from the retrieved evidence.",
        }.get(language, fallback_answer)
    return {
        "answer": fallback_answer,
        "sources": [],
        "source_status": SOURCE_STATUS_INSUFFICIENT,
        "source_reason": rag.get("source_reason") or "Retrieved evidence did not meet the topic, scope, and relevance requirements for this question.",
        "kb_resolved": False,
        "validation": {},
    }


def confidence_for_static(intent: str) -> tuple[str, str]:
    if intent in ORDINARY_CONVERSATION_INTENTS:
        return "Not Required", "This is a system or small-talk message and does not require legal-source retrieval."
    return "Insufficient", "The answer was not based on a retrieved legal source."


def confidence_for_clarification() -> tuple[str, str]:
    return "Insufficient", "Important facts are missing, so the bot asked follow-up questions before giving a legal answer."


def confidence_for_sources(
    sources: List[dict],
    context_chunks: Optional[List[dict]] = None,
    structured: bool = False,
) -> tuple[str, str]:
    if structured:
        return "Direct", "The answer was found directly in the cited Labor Code source."
    scores = [
        float(item.get("score", 0) or 0)
        for item in (context_chunks or sources or [])
        if isinstance(item, dict)
    ]
    best_score = max(scores) if scores else 0
    if best_score >= 0.85:
        return "Direct", "The answer was found directly in retrieved legal source excerpts."
    if best_score >= MIN_RELEVANCE_SCORE:
        return "Partial", "The retrieved source addresses the topic, but additional facts may be needed for exact application."
    return "Insufficient", "The approved knowledge base did not return enough matching legal source material to answer safely."


async def enqueue_answer_review(message_id: int, user_id: int, reason: str) -> None:
    if not message_id:
        return
    await execute_insert(
        """
        UPDATE chat_messages
        SET admin_review_status='needs_review', admin_review_note=%s
        WHERE id=%s AND user_id=%s AND role='assistant'
          AND COALESCE(admin_review_status,'unreviewed')='unreviewed'
        """,
        (f"System review: {reason}"[:1000], message_id, user_id),
    )


def clarification_choices(language: str) -> List[str]:
    if language == "fil":
        return ["Sahod", "Termination", "Leave", "Overtime", "Harassment"]
    if language == "hil":
        return ["Sweldo", "Termination", "Leave", "Overtime", "Harassment"]
    return ["Wages", "Termination", "Leave", "Overtime", "Harassment"]


def prune_guest_sessions() -> None:
    cutoff = time.time() - GUEST_SESSION_TTL_SECONDS
    stale_ids = [
        session_id
        for session_id, usage in GUEST_SESSIONS.items()
        if usage.get("updated", 0) < cutoff
    ]
    for session_id in stale_ids:
        GUEST_SESSIONS.pop(session_id, None)


def get_guest_limit_answer(language: str) -> str:
    if language == "fil":
        return (
            "Naabot mo na ang guest limit. Gumawa o mag-sign in sa account para "
            "makapagpatuloy at ma-save ang chat history mo."
        )
    if language == "hil":
        return (
            "Naabot mo na ang guest limit. Maghimo ukon mag-sign in sa account "
            "para makapadayon kag ma-save ang imo chat history."
        )
    return (
        "You have reached the guest chat limit. Please create an account or sign "
        "in so you can continue and save your chat history."
    )


STOP_WORDS = {
    "a", "an", "and", "ang", "ano", "are", "about", "ba", "for", "how",
    "i", "in", "is", "it", "ko", "me", "my", "ng", "nga", "of", "on",
    "or", "po", "sa", "sang", "the", "this", "to", "what", "with",
}

EXACT_BROAD_QUERIES = {
    "benefit", "benefits", "dole", "employee", "employer", "employment",
    "employment status", "holiday", "holiday pay", "labor", "labor law",
    "leave", "leave benefit", "minimum wage", "overtime", "pay", "rights",
    "salary", "separation pay", "termination", "wage", "worker rights",
}

DEFINITION_QUERY_RE = re.compile(
    r"^\s*(what\s+(is|are)|define|meaning\s+of|ano\s+(ang|nga)|"
    r"ano\s+ang\s+kahulugan|ano\s+ang\s+ibig\s+sabihin|"
    r"ano\s+ang\s+buot\s+silingon)",
    re.IGNORECASE,
)

VAGUE_OPENER_RE = re.compile(
    r"^\s*(about|regarding|tell\s+me\s+about|what\s+about|"
    r"paano\s+yung|kumusta\s+ang|parte\s+sa|tungkol\s+sa)\b",
    re.IGNORECASE,
)

PRONOUN_ONLY_RE = re.compile(
    r"\b(is\s+(this|that|it)\s+legal|can\s+they\s+do\s+that|"
    r"what\s+should\s+i\s+do|ano\s+gagawin\s+ko|pwede\s+ba\s+yun|"
    r"puwede\s+bala\s+ina)\b",
    re.IGNORECASE,
)

FOLLOW_UP_CONTEXT_RE = re.compile(
    r"\b(what\s+about|how\s+about|what\s+if|how\s+much|how\s+many|"
    r"i\s+mean|also|same|that|this|it|them|those|previous|earlier|"
    r"can\s+they\s+do\s+that|can\s+my\s+employer|"
    r"paano\s+naman|kumusta\s+naman|eh\s+yung|ibig\s+kong\s+sabihin|"
    r"ano\s+naman|pwede\s+ba\s+yun|puwede\s+bala\s+ina|"
    r"kamusta\s+ang|kumusta\s+ang|kon\s+amo|kung\s+ganun)\b",
    re.IGNORECASE,
)

FOLLOW_UP_REPHRASE_RE = re.compile(
    r"\b(simple\s+terms?|in\s+tagalog|explain\s+more|what\s+does\s+that\s+mean|"
    r"paano|pano|ano\s+ibig\s+sabihin|i\s+mean|or\s+simple|elaborate|expand|"
    r"continue|what\s+about|and\s+then|what\s+else|pano\s+naman|paano\s+naman|"
    r"halimbawa|for\s+example|give\s+(?:me\s+an?\s+)?example|example\s+please)\b",
    re.IGNORECASE,
)

NATIONAL_LAW_QUERY_RE = re.compile(
    r"\b(?:what\s+is|what\s+does|explain|define|meaning\s+of|ano\s+ang|paano)\b.{0,40}"
    r"\b(?:article|art\.?|section|sec\.?|republic\s+act|ra\s*\d+|"
    r"presidential\s+decree|pd\s*\d+|labor\s+code|department\s+order|do\s*\d+)\b",
    re.IGNORECASE,
)

SHORT_DETAIL_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"hour|hours|oras|day|days|adlaw|month|months|bulan|year|years|tuig|"
    r"region|iloilo|manila|cebu|probationary|regular|contractual|"
    r"project[-\s]?based|part[-\s]?time|full[-\s]?time|salary|wage|"
    r"sweldo|suhol|from|to|since|worked|employed|hired)\b",
    re.IGNORECASE,
)

PERSONAL_ELIGIBILITY_RE = re.compile(
    r"\b(am\s+i|do\s+i|can\s+i|may\s+i|ako|ko|akon|eligible|entitled|"
    r"pwede|puwede|puede|bala|may\s+.*\s+bala|mayroon\s+ba|meron\s+ba)\b",
    re.IGNORECASE,
)

PERSONAL_AMOUNT_RE = re.compile(
    r"\b(how\s+much|how\s+many|amount|estimate|estimated|compute\s+my|"
    r"calculate\s+my|my\s+(pay|salary|wage|13th|overtime|holiday|separation)|"
    r"magkano|magkano\s+ang|pila|pila\s+ang|pila\s+ka|akon\s+(pay|sweldo|suhol))\b",
    re.IGNORECASE,
)

SOURCE_FOLLOWUP_RE = re.compile(
    r"\b("
    r"source|sources|citation|cite|legal\s+basis|article|rule|section|page|"
    r"retrieved\s+passage|exact\s+passage|exact\s+text|passage|"
    r"partial\s+source\s+match|direct\s+source\s+match|source\s+match|"
    r"source\s*1|source\s*2|copies\s+of\s+the\s+same|same\s+legal\s+provision"
    r")\b",
    re.IGNORECASE,
)

SOURCE_PASSAGE_RE = re.compile(
    r"\b(exact\s+retrieved\s+passage|retrieved\s+passage|exact\s+passage|exact\s+text|show\s+.*passage)\b",
    re.IGNORECASE,
)

SOURCE_HIERARCHY_RE = re.compile(
    r"\b(article|rule|section|page|legal\s+basis|citation)\b",
    re.IGNORECASE,
)

SOURCE_MATCH_REASON_RE = re.compile(
    r"\b(why|bakit|ngaa).{0,80}\b(partial|direct|insufficient|source\s+match)\b|"
    r"\b(partial|direct|insufficient)\s+source\s+match\b",
    re.IGNORECASE,
)

SOURCE_COMPARE_RE = re.compile(
    r"\b(source\s*1).{0,80}(source\s*2|same|copy|copies|duplicate|legal\s+provision)\b|"
    r"\b(same\s+legal\s+provision|copies\s+of\s+the\s+same)\b",
    re.IGNORECASE,
)

CONCEPTUAL_CALCULATION_GUARD_RE = re.compile(
    r"\b(undertime|offset|written\s+approval|authorization|threshold|"
    r"entitled|eligible|coverage|force\s+me|compulsory|guarantee|court\s+award)\b",
    re.IGNORECASE,
)

CALCULATION_CONTINUATION_RE = re.compile(
    r"\b(how\s+much|compute|calculate|recompute|recalculate|amount|estimate|estimated|"
    r"total|what\s+would\s+that\s+be|how\s+much\s+is\s+it|how\s+much\s+will|"
    r"if\s+(?:it\s+was|it'?s|this\s+was)\s+(?:on\s+)?(?:my\s+)?(?:ordinary|rest|special|regular\s+holiday|holiday))\b",
    re.IGNORECASE,
)


def is_pending_calculation_reply(message: str, state: Dict[str, Any]) -> bool:
    pending = active_pending_calculation(state)
    if not pending:
        return False
    field = str(pending.get("last_question_asked") or "")
    if not field:
        return False
    return parse_calculation_reply_value(message, field) is not None


def is_calculation_continuation_request(message: str, query_profile: Dict[str, Any]) -> bool:
    text = message or ""
    return bool(
        query_profile.get("task") in CALCULATION_TASKS
        or query_profile.get("calculation_requested")
        or PERSONAL_AMOUNT_RE.search(text)
        or CALCULATION_CONTINUATION_RE.search(text)
    )

PROMPT_INJECTION_RE = re.compile(
    r"\b(ignore|bypass|disregard|skip|forget)\b.{0,80}\b(source|sources|instruction|instructions|system|rule|rules|knowledge\s+base)\b|"
    r"\b(uploaded\s+document|document)\b.{0,80}\b(ignore|system\s+instructions|follow\s+it)\b|"
    r"\b(?:use|cite|apply)\b.{0,60}\b(?:article|section|law|case|citation)\s*(?:no\.?\s*)?[a-z0-9.-]+.{0,80}"
    r"\b(?:even\s+if|regardless|whether\s+or\s+not)\b.{0,50}\b(?:wrong|incorrect|unrelated|unsupported|inapplicable|not\s+applicable)\b|"
    r"\bjust\s+give\s+me\s+the\s+answer\s+i\s+want\b",
    re.IGNORECASE,
)

# Refuse a narrow class of requests that explicitly force a known-wrong or
# unrelated legal citation. Normal requests for an Article or section still
# use the retrieval path.
FORCED_UNSUPPORTED_CITATION_RE = re.compile(
    r"\b(?:use|cite|apply)\b.{0,80}\b(?:article|section|law|case|citation)\s*(?:no\.?\s*)?[a-z0-9.-]+"
    r".{0,100}\b(?:even\s+if|regardless|whether\s+or\s+not)\b.{0,50}\b(?:wrong|incorrect|unrelated|unsupported|inapplicable|not\s+applicable)\b",
    re.IGNORECASE,
)

GENERAL_FORMULA_RE = re.compile(
    r"\b(how\s+is|how\s+to|how\s+do\s+(?:i|you|we)\s+(?:calculate|compute)|"
    r"formula|computed|calculated|computation|"
    r"paano\s+(i-compute|kinukuwenta|kwentahin)|paano\s+ginakuwenta|"
    r"paano\s+kuwentahin|paano\s+kalkulahin)\b",
    re.IGNORECASE,
)

TOPIC_FACT_DEPENDENT_INTENTS = {
    "holiday_pay",
    "leave_benefits",
    "termination_rights",
    "wage_inquiry",
    "13th_month",
    "employment_status",
    "occupational_safety",
}

THEORETICAL_LEGAL_QUESTION_RE = re.compile(
    r"^\s*(?:under\s+(?:the\s+)?(?:osh\s+)?law\b|"
    r"can\s+(?:a|an|the)\s+(?:worker|employee|employer|company)\b|"
    r"is\s+it\s+legal\s+to\b)",
    re.IGNORECASE,
)

ACTIVE_CRISIS_RE = re.compile(
    r"\b(?:help(?:\s+me)?|i\s+am|i['’]m|we\s+are|my\s+(?:boss|employer)|"
    r"tulungan|saklolo|buligi|ako\s+ay|ako['’]y|ang\s+boss\s+ko)\b"
    r".{0,180}\b(?:in\s+danger|locked\s+me\s+in|trapped|cannot\s+leave|can['’]t\s+leave|"
    r"being\s+abused|being\s+attacked|being\s+(?:sexually\s+)?harassed|"
    r"threatening\s+me|hit\s+me|hurt\s+me|"
    r"seriously\s+injured|bleeding|panganib|peligro)\b",
    re.IGNORECASE,
)


def needs_case_escalation(query: str, nlp: dict) -> bool:
    if nlp.get("intent") in {"greeting", "small_talk", "identity", "gratitude", "farewell"}:
        return False
    if is_definition_query(query) or is_definition_query(nlp.get("normalized", "")):
        return False
    if THEORETICAL_LEGAL_QUESTION_RE.search(query or ""):
        return False
    return bool(ACTIVE_CRISIS_RE.search(query or ""))


def build_escalation_answer(language: str) -> str:
    profile = get_language_profile(language)
    if language == "fil":
        direct = (
            "Mukhang may posibleng urgent o sensitibong labor concern dito. "
            "Para mas ligtas, huwag muna mag-share ng personal IDs, exact address, o private contact details sa chat."
        )
        requirements = (
            "Kung gusto mong ipaliwanag pa ang sitwasyon, ibigay lang ang non-sensitive details: "
            "uri ng concern, petsa, rehiyon/lungsod, employment status, at kung may notice o dokumento."
        )
        next_steps = (
            "Kung may panganib, harassment, seryosong injury, child labor, o malapit na legal deadline, "
            "makipag-ugnayan agad sa DOLE Regional/Field Office, barangay/PNP para sa immediate safety, "
            "o sa licensed labor lawyer."
        )
    elif language == "hil":
        direct = (
            "Daw may posible nga urgent ukon sensitive nga labor concern diri. "
            "Para mas safe, indi anay mag-share sang personal IDs, exact address, ukon private contact details sa chat."
        )
        requirements = (
            "Kon gusto mo ipaathag pa ang sitwasyon, ihatag lang ang non-sensitive details: "
            "klase sang concern, petsa, rehiyon/siyudad, employment status, kag kon may notice ukon dokumento."
        )
        next_steps = (
            "Kon may peligro, harassment, seryoso nga injury, child labor, ukon lapit nga legal deadline, "
            "magpalapit dayon sa DOLE Regional/Field Office, barangay/PNP para sa immediate safety, "
            "ukon sa licensed labor lawyer."
        )
    else:
        direct = (
            "This sounds like it may involve an urgent or sensitive labor concern. "
            "For safety, please avoid sharing personal IDs, exact addresses, or private contact details in chat."
        )
        requirements = (
            "If you want to explain more, share only non-sensitive details: the type of concern, date, "
            "region/city, employment status, and whether there is a notice or document."
        )
        next_steps = (
            "If there is danger, harassment, serious injury, child labor, or a close legal deadline, "
            "contact the DOLE Regional/Field Office, barangay/PNP for immediate safety, or a licensed labor lawyer."
        )
    sections = [
        f"**{profile['direct_label']}:**\n{direct}",
        f"**{profile['requirements_label']}:**\n{requirements}",
        f"**{profile['next_steps_label']}:**\n{next_steps}",
        f"**{profile['reminder_label']}:**\n{profile['disclaimer']}",
    ]
    return normalize_answer_disclaimers("\n\n".join(sections))


def significant_words(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [word for word in words if word not in STOP_WORDS]


def is_definition_query(text: str) -> bool:
    return bool(DEFINITION_QUERY_RE.search(text))


def has_prior_context(history: List[dict]) -> bool:
    return any(msg.get("role") == "user" and msg.get("content", "").strip() for msg in history)


def is_contextual_follow_up(query: str, nlp: dict, history: List[dict]) -> bool:
    """True when a short/latest message depends on earlier chat context."""
    if not has_prior_context(history):
        return False
    text = (query or "").strip()
    if not text:
        return False
    if FOLLOW_UP_REPHRASE_RE.search(text) or FOLLOW_UP_CONTEXT_RE.search(text):
        return True
    if NATIONAL_LAW_QUERY_RE.search(text):
        return False
    word_count = len(re.findall(r"[\w'-]+", text, re.UNICODE))
    if word_count <= 5:
        return True
    words = significant_words(nlp.get("normalized", text))
    return len(words) <= 6 and bool(SHORT_DETAIL_RE.search(text))


def clean_history_for_search(text: str) -> str:
    clean = redact_sensitive_info(text or "")
    clean = re.sub(r"\[Source[^\]]*\]\([^)]+\)", " ", clean)
    clean = re.sub(r"https?://\S+", " ", clean)
    clean = re.sub(r"\*\*([^*]+):\*\*", r"\1:", clean)
    clean = re.split(
        r"\b(Sources used|Gin-gamit nga sources|Mga ginamit na source|"
        r"Confidence:|Was this helpful\?|This is for informational purposes|"
        r"This is for educational purposes|Para lamang ini sa impormasyon|"
        r"Ito ay para lamang sa impormasyon)\b",
        clean,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return re.sub(r"\s+", " ", clean).strip()


def build_contextual_search_text(
    query: str,
    nlp: dict,
    history: List[dict],
    conversation_state: Optional[Dict[str, Any]] = None,
    conversation_summary: str = "",
) -> str:
    """
    Expand follow-up questions for retrieval without changing the user-facing
    question. This helps FAISS/Chroma find legal sources for messages like
    "what about probationary employees?" or "two hours".
    """
    base = (nlp.get("search_text") or nlp.get("normalized") or query or "").strip()
    if not base:
        return base
    state_text = state_context_text(conversation_state or {})
    if not is_contextual_follow_up(query, nlp, history) and not state_text:
        return base

    parts = []
    for msg in (history or [])[-8:]:
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = clean_history_for_search(msg.get("content", ""))
        if not content:
            continue
        limit = 420 if role == "user" else 280
        parts.append(f"{role}: {content[:limit]}")

    history_text = " ".join(parts)
    if not parts and not state_text and not conversation_summary:
        return base
    return build_standalone_search_query(
        current_message=query,
        base_search_text=base,
        history_text=history_text,
        state=conversation_state or {},
        summary=conversation_summary,
    )


def regional_wage_follow_up_query(
    message: str,
    conversation_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Reapply the saved minimum-wage topic to a region-only follow-up."""

    original = (message or "").strip()
    if not original:
        return original
    state = conversation_state or {}
    last_topic = str(state.get("last_topic") or state.get("issue") or "").strip().lower()
    details = wage_order_query_details(original)
    if (
        last_topic in {"minimum_wage", "minimum wage"}
        and details.get("region_code")
        and not details.get("is_wage_order_query")
    ):
        return f"minimum wage {original}"
    return original


SOCIAL_AGENCY_FOLLOW_UP_RE = re.compile(
    r"\b(sss|social\s+security|philhealth|pag[-\s]?ibig|hdmf)\b",
    re.IGNORECASE,
)


def contextual_retrieval_query(
    message: str,
    conversation_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve topic-bearing follow-ups without appending the full transcript."""

    original = regional_wage_follow_up_query(message, conversation_state)
    state = conversation_state or {}
    last_topic = str(state.get("last_topic") or state.get("issue") or "").strip().lower()
    sector = str(state.get("sector") or "").strip().lower()

    if last_topic == "holiday_pay" and re.search(
        r"\b(holiday|rest\s+day|day\s+off|worked|unworked|absent)\b",
        original,
        re.IGNORECASE,
    ):
        return f"holiday pay {original}"
    if (
        last_topic == "domestic_workers"
        and sector == "domestic_workers"
        and SOCIAL_AGENCY_FOLLOW_UP_RE.search(original)
    ):
        return f"kasambahay coverage SSS PhilHealth Pag-IBIG RA 10361 {original}"
    return original


def _retrieval_chunk_key(chunk: Dict[str, Any]) -> str:
    return str(
        (chunk.get("metadata") or {}).get("chunk_id")
        or chunk.get("id")
        or chunk.get("text", "")[:160]
    )


def _merge_retrieval_chunks(
    current: List[Dict[str, Any]],
    additions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output = []
    seen = set()
    for chunk in [*(current or []), *(additions or [])]:
        key = _retrieval_chunk_key(chunk)
        if key in seen:
            continue
        output.append(chunk)
        seen.add(key)
    return output


@with_question_analysis
async def _targeted_retrieval_context(
    query: str, nlp: dict, history: List[dict], intent: str,
    conversation_state: Optional[Dict[str, Any]] = None,
    conversation_summary: str = "", retrieval_diagnostics=None, request_plan=None,
) -> Dict[str, Any]:
    """One global hybrid search; topic labels never exclude approved sources."""
    from routers.services.basic_rag import retrieve
    return await retrieve(query, nlp, history, intent, retrieval_diagnostics, request_plan=request_plan)


async def retrieve_scoped_context(
    query: str, nlp: dict, history: List[dict], intent: str,
    conversation_state: Optional[Dict[str, Any]] = None,
    conversation_summary: str = "", request_plan=None,
) -> Dict[str, Any]:
    from routers.services.answer_pipeline_budget import AnswerPipelineBudget
    from routers.services.performance import current_timing, RetrievalDiagnostics, use_retrieval_diagnostics
    from routers.services.basic_rag import RETRIEVAL_SECONDS
    started = time.perf_counter()
    timing = current_timing()
    budget = AnswerPipelineBudget(max(1.0, settings.LABORLENS_CHAT_REQUEST_SECONDS), 0)
    if timing:
        budget._deadline -= max(0, time.perf_counter() - timing.started)
        timing.pipeline_budget = budget
    diagnostics = RetrievalDiagnostics(budget)
    try:
        with use_retrieval_diagnostics(diagnostics):
            result = await budget.run(_targeted_retrieval_context, query, nlp, history, intent,
                                      retrieval_diagnostics=diagnostics, request_plan=request_plan, cap=RETRIEVAL_SECONDS)
        result['_answer_budget'] = budget
        set_timing_metadata(retrieval_completed=True, source_ranking_completed=False,
                            evidence_selected=bool(result.get('chunks')),
                            evidence_chunk_count=len(result.get('chunks') or []),
                            verification_state='not_run', simple_rag_mode=True)
        from routers.admin_tools import playground_trace
        trace = playground_trace.get()
        if trace is not None:
            trace.update(source_status=result.get('source_status'), source_reason=result.get('source_reason'),
                         query_profile=result.get('query_profile'),
                         retrieved_sources=build_sources_payload(result.get('chunks') or []))
        return result
    except asyncio.TimeoutError:
        diagnostics.cancel_active(deadline=True)
        set_timing_metadata(retrieval_completed=False, retrieval_cancelled=True,
                            generation_gate_reason='retrieval_deadline')
        return dict(chunks=[], query_profile={'simple_rag_mode': True, 'request_understanding': request_plan or {}},
                    _answer_budget=budget, source_status=SOURCE_STATUS_INSUFFICIENT,
                    source_reason='Source discovery timed out; this is not a conclusion about KB coverage.',
                    top_similarity_score=None, kb_resolved=False, retrieval_unavailable=True)
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        add_timing('retrieval_ms', elapsed_ms)
        set_timing_metadata(retrieval_total_ms=round(elapsed_ms, 2), retrieval_elapsed_ms=round(elapsed_ms, 2))



async def load_session_memory(session_id: str, user_id: int) -> tuple[str, Dict[str, Any]]:
    row = await execute_one(
        """
        SELECT conversation_summary, conversation_state
        FROM chat_sessions
        WHERE id=%s AND user_id=%s
        """,
        (session_id, user_id),
    )
    if not row:
        return "", {}
    return row.get("conversation_summary") or "", parse_state(row.get("conversation_state"))


async def load_recent_chat_history(
    session_id: str,
    user_id: int,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Load history only when both the session and authenticated owner match."""
    safe_limit = max(1, min(int(limit), 50))
    rows = await execute_query(
        """
        SELECT role, content, sources, intent, confidence_level AS source_match, user_id
        FROM chat_messages
        WHERE session_id=%s AND user_id=%s
        ORDER BY created_at DESC, id DESC
        LIMIT %s
        """,
        (session_id, user_id, safe_limit),
    )
    return [
        {
            "role": row["role"],
            "content": row["content"],
            "sources": parse_sources_payload(row.get("sources")),
            "intent": row.get("intent"),
            "source_match": row.get("source_match"),
        }
        for row in reversed(rows)
        if row.get("role") in {"user", "assistant"} and row.get("content")
           and row.get('user_id') == user_id
    ]


async def save_session_memory(
    session_id: str,
    user_id: int,
    summary: str,
    state: Dict[str, Any],
    title_hint: Optional[str] = None,
) -> None:
    # Keep the session's last meaningful legal category alongside its memory.
    # A missing/general topic produces None, so SQL preserves the prior value
    # for greetings, acknowledgements, and other non-topic follow-ups.
    session_category = get_sidebar_category(
        topic=(state or {}).get("last_topic"),
        sub_intent=(state or {}).get("last_sub_intent"),
        intent=(state or {}).get("last_intent"),
    )
    if session_category == "Others" and str((state or {}).get("last_topic") or "").strip().lower() in {"", "general_labor", "non_legal"}:
        session_category = None
    if title_hint:
        await execute_insert(
            """
            UPDATE chat_sessions
            SET conversation_summary=%s,
                conversation_state=%s,
                category=COALESCE(%s, category),
                title=CASE
                    WHEN title IS NULL OR title='' OR title='New Conversation' OR title='New Consultation'
                    THEN %s
                    ELSE title
                END
            WHERE id=%s AND user_id=%s
            """,
            (summary or "", state_json(state), session_category, title_hint[:255], session_id, user_id),
        )
    else:
        await execute_insert(
            """
            UPDATE chat_sessions
            SET conversation_summary=%s,
                conversation_state=%s,
                category=COALESCE(%s, category)
            WHERE id=%s AND user_id=%s
            """,
            (summary or "", state_json(state), session_category, session_id, user_id),
        )


def asks_for_personal_amount_without_values(query: str, nlp: dict) -> bool:
    if nlp["intent"] not in {"wage_inquiry", "13th_month", "holiday_pay", "termination_rights"}:
        return False
    if not PERSONAL_AMOUNT_RE.search(query):
        return False
    # "How much is holiday pay?" and "What is the 13th-month amount?" are
    # ordinary rate/formula questions, not requests for a personalized
    # calculation. Only enter the missing-input flow when the user clearly
    # describes their own pay or work.
    personal_context = re.search(
        r"\b(my|i\s+am|i['’]?m|i\s+worked|i\s+earn|i\s+earned|should\s+i\s+(?:get|receive|be\s+paid)|"
        r"how\s+much\s+(?:will|would)\s+i|magkano\s+ang\s+(?:matatanggap|makukuha)|akon|akon\s+nga)\b",
        query,
        re.IGNORECASE,
    )
    if not personal_context:
        return False
    entities = nlp.get("entities") or {}
    meaningful_numbers = [
        value for value in (entities.get("numbers") or [])
        if not re.search(r"\b(13th|13|thirteenth)\b", str(value).lower())
    ]
    has_values = bool(entities.get("money") or meaningful_numbers)
    return not has_values


def asks_for_fact_dependent_answer(query: str, nlp: dict) -> bool:
    if nlp["intent"] not in TOPIC_FACT_DEPENDENT_INTENTS:
        return False
    if is_definition_query(query) or GENERAL_FORMULA_RE.search(query):
        return False
    if asks_for_personal_amount_without_values(query, nlp):
        return True
    if nlp["intent"] == "termination_rights" and re.search(
        r"\b(legal|illegal|valid|allowed|pwede|puwede|puede|can\s+they|can\s+my|"
        r"gintanggal|tinanggal|fired|dismissed|forced\s+to\s+resign)\b",
        query,
        re.IGNORECASE,
    ):
        return True
    if nlp["intent"] in {"holiday_pay", "leave_benefits", "13th_month", "employment_status"}:
        return bool(PERSONAL_ELIGIBILITY_RE.search(query))
    if nlp["intent"] == "wage_inquiry" and re.search(
        r"\b(unpaid|underpaid|kulang|wala\s+bayad|wala\s+sweldo|not\s+paid|deducted|deduction)\b",
        query,
        re.IGNORECASE,
    ):
        if re.search(r"\b(deduct(?:s|ed|ion)?|withheld|withholding)\b", query, re.IGNORECASE):
            money_values = {
                re.sub(r"[^0-9.]", "", str(value))
                for value in ((nlp.get("entities") or {}).get("money") or [])
            }
            # A deduction check can proceed when both the underlying pay and
            # the deducted amount are supplied. If one is missing, ask for
            # that value rather than issuing a generic topic prompt.
            return len({value for value in money_values if value}) < 2
        return True
    if nlp["intent"] == "occupational_safety" and re.search(
        r"\b(accident|injury|unsafe|hazard|illness|nasaktan|aksidente|disgrasya)\b",
        query,
        re.IGNORECASE,
    ):
        return True
    return False


def is_answerable_fact_dependent_query(query: str, nlp: dict) -> bool:
    text = (query or "").lower()
    intent = nlp.get("intent")
    profile = classify_legal_query(query, nlp)
    sub_intent = str(profile.get("sub_intent") or "")
    if profile.get('requested_issues') and not profile.get('calculation_requested'):
        # A named issue is enough for a conditional rule explanation. Missing
        # source evidence must trigger retrieval, not an unrelated intake form.
        return True
    from routers.services.scenario_retrieval import is_answerable_work_injury_query
    if (is_answerable_work_injury_query(query)
            and not profile.get("calculation_requested")):
        # The incident and requested medical/reporting issue are already given.
        # Retrieve supporting rules, preserving conditional source-grounded
        # answers, instead of asking whether the stated injury happened.
        return True
    if (
        profile.get("topic") == "thirteenth_month_pay"
        and profile.get("task") in {"coverage_question", "eligibility_question"}
    ):
        return True
    if re.search(r'\b(tinanggal|natanggal|gintanggal|dismissed|terminated|terminate|dismiss|fired)\b', text) and re.search(
        r'\b(dahil|because|reason|rason|legal|rights|karapatan)\b', text,
    ):
        # The event and requested issue are already known. Retrieve the rules
        # and explain conditions; do not pronounce the dismissal unlawful.
        return True

    # These questions have a recognizable legal proposition even when the
    # user writes in mixed Filipino/Hiligaynon. They should retrieve the rule
    # and answer it directly instead of falling into the generic clarification
    # copy.
    if sub_intent in {
        "thirteenth_month_nonpayment",
        "thirteenth_month_resigned_employee",
        "thirteenth_month_payment_deadline",
        "maternity_standard_duration",
        "maternity_optional_extension",
        "maternity_solo_parent_additional",
        "solo_parent_leave",
        "codi_definition",
        "cba_definition",
        "right_to_self_organization",
        "safety_officer_requirement",
        "osh_program_requirement",
        "osh_committee_requirement",
        "sexual_harassment_reporting",
        "safe_spaces_workplace",
        "night_shift_differential",
        "rest_day_pay",
        "regular_holiday_pay",
        "regular_holiday_on_rest_day",
    }:
        return True
    if intent == "wage_inquiry" and wage_order_query_details(query).get("is_regional_wage_query"):
        # A location is enough to return the official sector range.  Missing
        # occupation/establishment details should be explained as conditions,
        # not used to block the answer with a clarification screen.
        return True
    if intent == "wage_inquiry" and re.search(
        r"\b(undertime|offset\s+my\s+undertime|offset\s+.*\s+overtime|unpaid\s+overtime|did\s+not\s+pay\s+my\s+overtime)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if intent in {"termination_rights", "general_inquiry"} and re.search(
        r"\b(kasambahay|domestic\s+worker|valid\s+reason|at\s+any\s+time|terminate)\b",
        text,
        re.IGNORECASE,
    ) and re.search(r"\b(kasambahay|domestic\s+worker)\b", text, re.IGNORECASE):
        return True
    if intent == "termination_rights" and re.search(
        r"\b(terminated|dismissed|fired|sacked|tinanggal|natanggal|gintanggal|ginpahalin|ginpahalinan|tanggalon|"
        r"forced\s+to\s+resign|pinilit\s+magresign)\b",
        text,
        re.IGNORECASE,
    ) and re.search(
        r"\b(no|without|walang|wala|indi|did\s+not|didn't)\b.{0,80}\b(written\s+notice|notice|sulat|hearing|reason|rason)\b|"
        r"\b(karapatan|rights?|due\s+process|ano\s+ang\s+mga)\b",
        text,
        re.IGNORECASE,
    ):
        # This already identifies termination plus a due-process concern. Give
        # the general rule first; any missing ground/status can be asked after.
        return True
    if intent == "13th_month" and re.search(
        r"\b(who\s+is\s+entitled|who\s+gets|coverage|probationary|resigned|separated|allowances?|overtime\s+pay|holiday\s+pay|"
        r"basic\s+salary|permanent|regular\s+employee|kasambahay|domestic\s+worker|"
        r"monthly\s+salary\s+limit|p\s*1[, ]?000|php\s*1[, ]?000)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if intent == "holiday_pay" and re.search(
        r"\b(regular\s+holiday|special\s+non[-\s]*working|special\s+holiday|"
        r"rest\s+day|work(?:ed)?\s+on|do\s+not\s+work|did\s+not\s+work|"
        r"absent\s+without\s+pay|day\s+before|falls\s+on|double\s+pay|"
        r"same\s+pay\s+rules|fewer\s+than\s+ten|small\s+establishment|"
        r"condition|entitled|eligible|successive\s+regular\s+holidays?)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if intent == "employment_status" and re.search(
        r"\b(probationary|regular|contractual|project[-\s]*based|security\s+of\s+tenure)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    return False


def unresolved_pay_type(query: str, history: Optional[List[dict]] = None) -> bool:
    """An employment-status reply cannot resolve an unspecified pay question."""
    vague = re.compile(r"^(?:how much (?:should|will|can|do) i (?:receive|get|earn)|magkano (?:ang )?(?:matatanggap|makukuha) ko|pila (?:ang )?(?:mabaton|makuha) ko)[?.!\s]*$", re.I)
    text = (query or "").strip()
    if vague.fullmatch(text):
        return True
    if not re.fullmatch(r"(?:i am |i'm )?(?:a )?(?:regular|probationary|contractual|project[- ]based|part[- ]time|full[- ]time)(?: employee| worker)?[.!\s]*", text, re.I):
        return False
    for message in reversed(history or []):
        if message.get("role") != "user":
            continue
        previous = str(message.get("content") or "").strip()
        if vague.fullmatch(previous):
            return True
        if not re.fullmatch(r"(?:regular|probationary|contractual|project[- ]based|part[- ]time|full[- ]time)[.!\s]*", previous, re.I):
            return False
    return False


def needs_clarification(query: str, nlp: dict, history: List[dict]) -> bool:
    """Return True when the user's latest message is too vague to answer safely."""
    from routers.services.scenario_retrieval import incomplete_issuance_query, unidentified_advisory_query
    from routers.services.scenario_retrieval import is_leave_recall_query, resolve_clarification_query
    resolved_question = resolve_clarification_query(query, history)
    from routers.services.clarification_intake import broad_intake_question, broad_benefits_query
    if broad_intake_question(query, history):
        return True
    if broad_benefits_query(resolved_question):
        return False
    if is_leave_recall_query(resolved_question):
        # The issue is already specific. Retrieve recall evidence rather than
        # repeating a generic intake form or asking for irrelevant salary data.
        return False
    if incomplete_issuance_query(query) or unidentified_advisory_query(query):
        return True
    from routers.services.evidence_limits import ambiguous_compound_reply
    if ambiguous_compound_reply(query, history):
        return True
    if unresolved_pay_type(query, history):
        return True
    if nlp["intent"] in ORDINARY_CONVERSATION_INTENTS:
        return False
    if nlp["intent"] not in CLARIFICATION_ELIGIBLE_INTENTS:
        return False

    original = query.strip()
    normalized = nlp.get("normalized", original).strip()
    original_lower = original.lower()
    normalized_lower = normalized.lower()
    words = significant_words(normalized)

    if not original:
        return True
    if is_definition_query(original) or is_definition_query(normalized):
        return False
    if NATIONAL_LAW_QUERY_RE.search(original) or NATIONAL_LAW_QUERY_RE.search(normalized):
        return False
    legal_profile = classify_legal_query(original, nlp)
    from routers.services.question_understanding import retrieval_ready_request, missing_separation_inputs
    if (legal_profile.get('topic') in {'separation_pay', 'termination'}
            and legal_profile.get('calculation_requested')
            and not retrieval_ready_request(resolved_question, calculation_requested=True)):
        return bool(missing_separation_inputs(resolved_question, history))
    if is_contextual_follow_up(original, nlp, history):
        return False
    # The overtime multiplier changes by day type.  Ask only for that missing
    # fact when the user supplied a rate and hours for a personal computation;
    # conceptual/formula questions remain answerable without it.
    if retrieval_ready_request(resolved_question, calculation_requested=bool(legal_profile.get('calculation_requested'))):
        return False
    if (
        legal_profile.get("topic") not in {"general_labor", "leave_benefits", "non_legal"}
        and legal_profile.get("task") in {"coverage_question", "eligibility_question"}
        and not legal_profile.get("calculation_requested")
    ):
        # A classified coverage/eligibility question has a bounded legal scope.
        # Explain the baseline and state any conditions instead of turning it
        # into an intake interview. Calculation and remaining-balance requests
        # still flow through the material-fact checks below.
        return False
    if (
        legal_profile.get("topic") == "overtime_pay"
        and legal_profile.get("calculation_requested")
        and re.search(r"\b(hourly\s+(?:rate|wage)|rate\s+is|₱|php|pesos?)\b", normalized, re.IGNORECASE)
        and re.search(r"\b(worked|overtime|ot|extra\s+hours?)\b", normalized, re.IGNORECASE)
        and not re.search(r"\b(ordinary\s+(?:working\s+)?day|rest\s+day|day\s+off|regular\s+holiday|special\s+(?:non[-\s]?)?working\s+day|holiday)\b", normalized, re.IGNORECASE)
        and not has_prior_context(history)
    ):
        return True
    if is_answerable_fact_dependent_query(original, nlp):
        return False
    from routers.services.document_evidence import complete_information_request
    if not legal_profile.get('calculation_requested') and complete_information_request(original):
        return False
    if not legal_profile.get("calculation_requested"):
        # A general entitlement/remedy can be explained conditionally. Do not
        # ask for the leave type or resignation status already stated here.
        if re.search(r'\bsick[-\s]+leave\b', original, re.I) and re.search(r'policy|polisiya|paid|bayaran|entitl|karapatan', original, re.I):
            return False
        if re.search(r'\bfinal[-\s]+pay\b', original, re.I) and re.search(r'\bdole\b|complain|reklamo|assistance|tulong', original, re.I):
            return False
    # Explanations and reporting options do not require an intake interview.
    # Keep missing-input checks above for actual personal computations.
    if not legal_profile.get("calculation_requested") and re.search(
        r"\b(general (?:rule|requirements)|resignation (?:rules|requirements)|"
        r"how (?:can|do) i (?:report|check)|where (?:can|do) i|"
        r"paano (?:ko )?(?:i[- ]?check|magreklamo)|saan (?:ako )?"
        r"(?:pwede|puwede)|ano ang (?:pwede|puwede) kong gawin)\b",
        original, re.IGNORECASE,
    ):
        return False
    if asks_for_fact_dependent_answer(original, nlp):
        return True
    if PRONOUN_ONLY_RE.search(original_lower):
        return not has_prior_context(history)
    if normalized_lower in EXACT_BROAD_QUERIES:
        return True
    if VAGUE_OPENER_RE.search(original_lower) and len(words) <= 4:
        return True
    if len(words) <= 1:
        return True
    if nlp["intent"] == "general_inquiry" and len(words) <= 3:
        return True
    if nlp["intent"] != "general_inquiry" and len(words) <= 2:
        return True
    return False


def is_semantic_intake_candidate(
    query: str,
    nlp: Dict[str, Any],
    history: List[Dict[str, Any]],
    profile: Dict[str, Any],
) -> bool:
    """Return whether an ambiguous, topic-free turn merits semantic intake.

    This deliberately relies on the normalized routing/profile state rather
    than matching a catalog of opening phrases.  The bounded planner below
    makes the final decision, and an unavailable or invalid planner response
    always leaves a possible legal question on the normal retrieval path.
    """

    from routers.services.question_understanding import retrieval_ready_request
    if retrieval_ready_request(query, calculation_requested=bool(profile.get('calculation_requested'))):
        return False
    if str(nlp.get("intent") or "") != "general_inquiry":
        return False
    if str(nlp.get("legal_intent") or ""):
        return False
    if profile.get("topic") != "general_labor" or profile.get("calculation_requested"):
        return False
    return not is_contextual_follow_up(query, nlp, history)


def get_clarification_questions(intent: str, language: str) -> List[str]:
    questions_by_language = CLARIFICATION_QUESTIONS.get(language, CLARIFICATION_QUESTIONS["en"])
    return questions_by_language.get(intent, questions_by_language["general_inquiry"])[:1]


def get_contextual_clarification_questions(
    query: str,
    nlp: dict,
    intent: str,
    language: str,
    history: Optional[List[dict]] = None,
) -> List[str]:
    """Return one question for the fact that is actually missing.

    The old clarification path selected a broad question from the top-level
    intent (for example, "Which leave are you asking about?") even when the
    query already identified the relevant issue.  Keep the public response
    shape backward compatible, but choose the question from the more precise
    legal topic/sub-intent profile.
    """
    from routers.services.scenario_retrieval import incomplete_issuance_query, unidentified_advisory_query
    from routers.services.clarification_intake import broad_intake_question
    intake = broad_intake_question(query, history, language)
    if intake:
        return [intake]
    from routers.services.scenario_retrieval import resolve_clarification_query
    resolved = resolve_clarification_query(query, history)
    if resolved != query:
        query = resolved
        nlp = preprocess_query(query)
    if unidentified_advisory_query(query):
        return [{
            'en': 'Please ask HR for the advisory number, date and full text or photo. Without identifying the advisory, I cannot confirm its coverage or applicability. Can you share those details?',
            'fil': 'Hingin sa HR ang advisory number, petsa at buong kopya o litrato. Hindi makukumpirma ang coverage o applicability ng advisory kung hindi ito matukoy. Maibabahagi mo ba ang mga detalyeng iyon?',
            'hil': 'Pangayua sa HR ang advisory number, petsa kag bug-os nga kopya ukon litrato. Indi pa makumpirma ang coverage sini kon wala natukoy ang advisory. Ma-share mo bala ini?',
        }.get(language, 'Please share the advisory number, date and full text before we assess its applicability.')]
    if incomplete_issuance_query(query):
        return [{
            'en': 'What is the complete Department Order number, including any letter suffix and year, or its full title?',
            'fil': 'Ano ang buong Department Order number, kasama ang titik at taon, o ang buong pamagat nito?',
            'hil': 'Ano ang kompleto nga Department Order number, upod ang letra kag tuig, ukon ang bug-os nga titulo sini?',
        }.get(language, 'What is the complete Department Order number and year, or its full title?')]
    if unresolved_pay_type(query, history):
        return [{
            "en": "Which type of pay do you want to check: basic salary, overtime pay, holiday pay, 13th-month pay, separation pay, or final pay?",
            "fil": "Anong uri ng bayad ang gusto mong i-check: basic salary, overtime pay, holiday pay, 13th-month pay, separation pay, o final pay?",
            "hil": "Ano nga klase sang bayad ang gusto mo ipa-check: basic salary, overtime pay, holiday pay, 13th-month pay, separation pay, ukon final pay?",
        }.get(language, "Which type of pay do you want to check: basic salary, overtime pay, holiday pay, 13th-month pay, separation pay, or final pay?")]
    from routers.services.evidence_limits import ambiguous_compound_reply
    if ambiguous_compound_reply(query, history):
        return [{
            "en": "Did you receive a written notice from your employer?",
            "fil": "Nakatanggap ka ba ng nakasulat na notice mula sa employer?",
            "hil": "Nakabaton ka bala sang sinulat nga notice halin sa employer?",
        }.get(language, "Did you receive a written notice from your employer?")]
    profile = classify_legal_query(query, nlp)
    topic = str(profile.get("topic") or "").strip().lower()
    sub_intent = str(profile.get("sub_intent") or "").strip().lower()
    normalized = str(profile.get("normalized_query") or query or "")
    # Short personal questions such as "Was I dismissed legally?" may be
    # routed by the NLP layer as a broad inquiry even though its top-level
    # intent already identifies the legal area.
    if topic in {"", "general_labor"}:
        topic = {
            "termination_rights": "termination",
            "13th_month": "thirteenth_month",
            "leave_benefits": "leave_benefits",
            "holiday_pay": "holiday_pay",
            "wage_inquiry": "wage_inquiry",
            "occupational_safety": "workplace_safety",
        }.get(intent, topic)
    has_rate = bool(
        re.search(
            r"(?:₱|\bphp\s*)\s*\d|\b\d[\d,.]*\s*(?:pesos?|per\s+(?:hour|day|month))\b|\b(?:rate|salary|wage)\s*(?:is|of|:|=)?\s*\d",
            normalized,
            re.IGNORECASE,
        )
    )
    from routers.services.conversation_memory import extract_hours
    has_hours = extract_hours(normalized).get('overtime_hours') is not None
    has_day_type = bool(
        re.search(
            r"\b(ordinary\s+(?:working\s+)?day|regular\s+workday|rest\s+day|day\s+off|"
            r"regular\s+holiday|special\s+(?:non[-\s]?)?working\s+day|holiday)\b",
            normalized,
            re.IGNORECASE,
        )
    )

    localized = {
        "overtime_day_type": {
            "fil": "Sa anong uri ng araw ginawa ang overtime: ordinaryong araw, rest day, special non-working day, o regular holiday?",
            "hil": "Sa ano nga klase sang adlaw ginhimo ang overtime: ordinaryo nga adlaw, rest day, special non-working day, ukon regular holiday?",
            "en": "Was the overtime on an ordinary workday, rest day, special non-working day, or regular holiday?",
        },
        "overtime_inputs": {
            "fil": "Ano ang hourly o daily rate, ilang overtime hours, at anong uri ng araw ginawa ang overtime?",
            "hil": "Ano ang hourly ukon daily rate, pila ka oras ang overtime, kag ano nga klase sang adlaw ini ginhimo?",
            "en": "What was the hourly or daily rate, how many overtime hours were worked, and what type of day was it?",
        },
        "minimum_wage_scope": {
            "fil": "Saang rehiyon o lungsod ka nagtatrabaho, at ano ang industriya o uri ng establisimyento?",
            "hil": "Sa diin nga rehiyon ukon siyudad ka nagaobra, kag ano ang industriya ukon klase sang establisimyento?",
            "en": "What region or city do you work in, and what industry or type of establishment is it?",
        },
        "thirteenth_month_nonpayment": {
            "fil": "Kailan dapat ibigay ang 13th-month pay, at may bahagi ba nito na naibigay na?",
            "hil": "San-o dapat ihatag ang 13th-month pay, kag may bahin bala sini nga nahatag na?",
            "en": "When was the 13th-month pay due, and was any part of it already paid?",
        },
        "thirteenth_month_inputs": {
            "fil": "Ano ang basic salary at mga petsa ng iyong employment na gagamitin sa pag-check o pag-compute?",
            "hil": "Ano ang basic salary kag mga petsa sang imo employment nga gamiton sa pag-check ukon pagkalkula?",
            "en": "What was your basic salary and what were your employment dates for the period in question?",
        },
        "termination_process": {
            "fil": "Anong dahilan ang ibinigay ng employer para sa pagtanggal sa iyo?",
            "hil": "Ano ang rason nga ginhatag sang employer sa pagtanggal sa imo?",
            "en": "What reason did the employer give for dismissing you?",
        },
        "resignation_status": {
            "en": "Have you already submitted a resignation letter, or has HR only asked you to resign?",
            "fil": "Nakapagsumite ka na ba ng resignation letter, o sinabihan ka pa lang ng HR na mag-resign?",
            "hil": "Nakapasa ka na bala sang resignation letter, ukon ginsilingan ka pa lang sang HR nga mag-resign?",
        },
        "resignation_notice": {
            "en": "On what date did you submit your resignation notice?",
            "fil": "Anong petsa mo isinumite ang iyong resignation notice?",
            "hil": "Ano nga petsa mo ginpasa ang imo resignation notice?",
        },
        "holiday_context": {
            "fil": "Anong holiday o petsa ito, pumasok ka ba, at rest day mo ba ang araw na iyon?",
            "hil": "Ano nga holiday ukon petsa ini, nagtrabaho ka bala, kag rest day mo bala sadto nga adlaw?",
            "en": "Which holiday or date was involved, did you work that day, and was it your rest day?",
        },
        "leave_type": {
            "fil": "Anong leave ang tinatanong mo: service incentive, sick, maternity, paternity, solo-parent, o iba pa?",
            "hil": "Ano nga leave ang ginapamangkot mo: service incentive, sick, maternity, paternity, solo-parent, ukon iban pa?",
            "en": "Which leave are you asking about: service incentive, sick, maternity, paternity, solo-parent, or another type?",
        },
        "unpaid_wage_period": {
            "fil": "Anong pay period o petsa ang hindi nabayaran, at magkano ang dapat mong natanggap?",
            "hil": "Ano nga pay period ukon petsa ang wala nabayaran, kag pila ang dapat mo tani mabaton?",
            "en": "Which pay period or dates were unpaid, and how much should you have received?",
        },
        "safety_context": {
            "fil": "May aksidente o injury ba, o unsafe condition lamang, at kailan at saan ito nangyari?",
            "hil": "May aksidente ukon injury bala, ukon unsafe condition lang, kag san-o kag diin ini natabo?",
            "en": "Was there an accident or injury, or only an unsafe condition, and when and where did it happen?",
        },
        "general": {
            "fil": "Anong partikular na workplace issue ang gusto mong masagot?",
            "hil": "Ano gid nga espesipiko nga workplace issue ang gusto mo nga sabton?",
            "en": "What specific workplace issue would you like answered?",
        },
    }

    key = "general"
    if re.search(r"resign", query, re.IGNORECASE):
        key = "resignation_notice" if re.search(r"nag[- ]?resign|submitted|handed|isinumite|ginpasa", query, re.IGNORECASE) else "resignation_status"
    elif topic == "minimum_wage" and not profile.get("region_code"):
        key = "minimum_wage_scope"
    elif topic == "overtime_pay" and profile.get("calculation_requested"):
        missing = []
        if not has_rate:
            missing.append({"en":"your hourly or daily pay rate", "fil":"iyong hourly o daily rate", "hil":"imo hourly ukon daily rate"}.get(language, "your hourly or daily pay rate"))
        if not has_hours:
            missing.append({"en":"the number of overtime hours", "fil":"bilang ng overtime hours", "hil":"kadamuon sang overtime hours"}.get(language, "the number of overtime hours"))
        if not has_day_type:
            missing.append({"en":"the day type (ordinary workday, rest day, or holiday)", "fil":"uri ng araw (ordinaryong araw, rest day, o holiday)", "hil":"klase sang adlaw (ordinaryo nga adlaw, rest day, ukon holiday)"}.get(language, "the day type (ordinary workday, rest day, or holiday)"))
        if missing and not (has_rate and has_hours and not has_day_type):
            prefix = {"en":"Please provide", "fil":"Pakibigay ang", "hil":"Palihog ihatag ang"}.get(language, "Please provide")
            return [prefix + " " + "; ".join(missing) + "."]
        key = "overtime_day_type" if has_rate and has_hours and not has_day_type else "overtime_inputs"
    elif sub_intent == "thirteenth_month_nonpayment":
        key = "thirteenth_month_nonpayment"
    elif topic == "thirteenth_month" and profile.get("calculation_requested"):
        key = "thirteenth_month_inputs"
    elif topic in {"separation_pay", "termination"} and profile.get('calculation_requested') and re.search(
        r"\b(separation\s+pay|how\s+much|magkano|pila)\b", normalized, re.IGNORECASE
    ):
        from routers.services.question_understanding import separation_intake_question
        question = separation_intake_question(query, language, history)
        return [question] if question else []
    elif sub_intent in {"termination_comparison", "termination_due_process"} or (
        intent == "termination_rights" and topic == "termination"
    ):
        key = "termination_process"
    elif topic == "holiday_pay":
        key = "holiday_context"
        if re.search(r'\bpumasok\b|\bi worked\b|nagtrabaho', query, re.I) and not re.search(r'regular\s+holiday|special\s+(?:non[-\s]?)?working', query, re.I):
            return [{
                'en': 'Which holiday or date was it (regular holiday or special non-working day)?',
                'fil': 'Anong holiday o petsa iyon—regular holiday o special non-working day?',
                'hil': 'Ano nga holiday ukon petsa ini—regular holiday ukon special non-working day?',
            }.get(language, 'Which holiday or date was it?')]
    elif topic in {"leave_benefits", "maternity_leave", "solo_parent_leave", "service_incentive_leave"}:
        key = "leave_type"
    elif topic in {"wage_inquiry", "payment_of_wages"} and re.search(
        r"\b(unpaid|underpaid|not\s+paid|wala\s+(?:bayad|sweldo)|hindi\s+(?:nabayaran|binayaran))\b",
        normalized,
        re.IGNORECASE,
    ):
        key = "unpaid_wage_period"
    elif topic in {"occupational_safety", "workplace_safety"}:
        key = "safety_context"
    elif intent == "leave_benefits":
        key = "leave_type"

    question = localized[key].get(language, localized[key]["en"])
    return [question]


async def decide_clarification(
    query: str,
    nlp: Dict[str, Any],
    intent: str,
    language: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Understand first, then retrieve even when a partial answer needs a fact.

    Only a request with no searchable subject stops for standalone clarification.
    All other missing details go into the generation plan, so a supported
    explanation and its focused follow-up can appear together.
    """
    from routers.services.scenario_retrieval import resolve_clarification_query
    from routers.services.document_evidence import resolve_subject_followup
    from routers.services.request_understanding import (
        plan_request,
        history_for_plan,
        ensure_elapsed_time_inputs,
    )

    history = list(history or [])
    resolved_query = resolve_clarification_query(query, history)
    if resolved_query == query:
        resolved_query = resolve_subject_followup(query, history)
    resolved_nlp = dict(nlp or {}, intent=(nlp or {}).get('intent') or intent)
    if resolved_query != query:
        resolved_nlp = await asyncio.to_thread(preprocess_query, resolved_query)
    resolved_intent = str(resolved_nlp.get("intent") or intent)
    profile = classify_legal_query(resolved_query, resolved_nlp)
    plan = await plan_request(query, history, language)
    if plan:
        # The semantic plan, not an earlier heuristic rewrite, controls all
        # downstream topic classification, retrieval and generation.
        resolved_query = plan['retrieval_query'] or query
        relevant = history_for_plan(query, history, plan)
        user_context = '\n'.join(str(m.get('content') or '') for m in relevant if m['role'] == 'user')
        plan = dict(plan, user_context=(user_context + '\n' + query).strip(), latest_question=query)
        resolved_nlp = await asyncio.to_thread(preprocess_query, resolved_query, topic_hints=plan.get('topic_hints'))
        resolved_intent = str(resolved_nlp.get('intent') or intent)
        profile = classify_legal_query(resolved_query, resolved_nlp)
        # Safety net for personal calculations: if the LLM overlooks a
        # required input that the existing intake recognizes, attach one
        # focused question while keeping retrieval enabled for a partial answer.
        # General explanations remain conditional instead of becoming an
        # unnecessary intake interview.
        if not plan['needs_clarification'] and plan.get('answer_kind') == 'personal_calculation':
            if needs_clarification(resolved_query, resolved_nlp, history):
                questions = await asyncio.to_thread(get_contextual_clarification_questions,
                    resolved_query, resolved_nlp, resolved_intent, language, history)
                if questions:
                    plan = dict(plan, needs_clarification=True,
                                missing_facts=['A detail needed for the personal calculation'],
                                question=questions[0])
                    set_timing_metadata(request_understanding_safety_net=True)
    else:
        # Fail soft without a second model intake call. The generator still
        # checks completeness from current user facts and available evidence.
        fallback_has_topic = not is_semantic_intake_candidate(
            query, resolved_nlp, history, profile,
        )
        plan = {'retrieval_query': resolved_query, 'has_topic': fallback_has_topic,
                'topic_hints': [],
                'answer_kind': 'personal_calculation' if profile.get('calculation_requested') else 'explanation',
                'needs_clarification': False, 'missing_facts': [], 'question': '',
                'user_context': resolved_query, 'latest_question': query}
        if needs_clarification(resolved_query, resolved_nlp, history):
            questions = await asyncio.to_thread(get_contextual_clarification_questions,
                resolved_query, resolved_nlp, resolved_intent, language, history)
            if questions:
                plan.update(needs_clarification=True, question=questions[0], missing_facts=['Essential user detail'])
    plan = ensure_elapsed_time_inputs(plan, query, history, language)
    decision: Dict[str, Any] = {
        # If the subject is clear, retrieve evidence and return a supported
        # partial answer with the follow-up underneath instead of stopping at
        # a question-only response when semantic planning is unavailable.
        "should_clarify": bool(plan['needs_clarification'] and not plan['has_topic']),
        "questions": [plan['question']] if plan['needs_clarification'] else [],
        "quick_replies": [],
        "query": resolved_query,
        "nlp": resolved_nlp,
        "intent": resolved_intent,
        "profile": profile,
        "request_plan": plan,
    }
    return decision


def build_clarification_answer(query: str, nlp: dict, intent: str, language: str, history: Optional[List[dict]] = None) -> str:
    copy = CLARIFICATION_COPY.get(language, CLARIFICATION_COPY["en"])
    questions = get_contextual_clarification_questions(query, nlp, intent, language, history)
    return f"{copy['intro']}\n\n{questions[0]}"


def should_handle_personalized_calculation(
    message: str,
    query_profile: Dict[str, Any],
    state: Dict[str, Any],
) -> bool:
    text = message or ""
    if is_source_followup_request(text):
        return False
    pending = active_pending_calculation(state)
    if pending:
        return is_pending_calculation_reply(text, state) or is_calculation_continuation_request(text, query_profile)
    topic = query_profile.get("topic")
    if query_profile.get("task") not in CALCULATION_TASKS and not query_profile.get("calculation_requested"):
        return False
    state_topic = state.get("calculation_topic") or state.get("issue")
    if topic != "overtime_pay" and state_topic not in {"overtime_pay", "unpaid_overtime"}:
        return False
    if not pending and not re.search(
        r"\b(my\s+(?:hourly|daily|monthly)\s+(?:rate|wage|pay|salary)|my\s+pay|my\s+salary|"
        r"i\s+(?:worked|earn|earned)|should\s+i\s+(?:get|receive|be\s+paid)|"
        r"magkano\s+ang\s+(?:matatanggap|makukuha)|akon\s+(?:nga\s+)?(?:pay|sweldo|suhol))\b",
        text,
        re.IGNORECASE,
    ):
        # A generic rate/formula question belongs to RAG. Calculation cards
        # are reserved for a worker's own figures or a pending calculation.
        return False
    if re.search(r"\b(eight|8)\s+hours?\s+per\s+day\b|\b(forty|40)\s+hours?\s+per\s+week\b", text, re.IGNORECASE):
        return False
    if CONCEPTUAL_CALCULATION_GUARD_RE.search(text) and not (
        state.get("hourly_rate") and state.get("overtime_hours")
    ):
        return False
    if is_calculation_continuation_request(text, query_profile):
        return True
    if state.get("hourly_rate") and state.get("overtime_hours") and parse_calculation_reply_value(text, "day_type"):
        return query_profile.get("calculation_requested") or query_profile.get("task") in CALCULATION_TASKS
    return bool(
        re.search(
            r"\b(timely\s+wage|hourly\s+(?:wage|rate|pay)|rate\s+per\s+hour|"
            r"\d+\s*(?:hrs?|hours?)\s*(?:ot|overtime)|ot\s+for\s+\d+|"
            r"rest\s+day|ordinary\s+day|holiday|special\s+day)\b",
            text,
            re.IGNORECASE,
        )
        and (state.get("hourly_rate") or state.get("daily_wage") or state.get("overtime_hours"))
    )


def is_complete_new_task(
    message: str,
    query_profile: Dict[str, Any],
    state: Dict[str, Any],
) -> bool:
    if is_source_followup_request(message):
        return False
    if active_pending_calculation(state) and is_pending_calculation_reply(message, state):
        return False
    if query_profile.get("task") in CALCULATION_TASKS or query_profile.get("calculation_requested"):
        return False
    if query_profile.get("intent") in {"greeting", "small_talk", "identity", "capability_question", "gratitude", "farewell"}:
        return False
    topic = str(query_profile.get("topic") or "")
    if topic in {"", "non_legal", "general_labor"}:
        return False
    text = message or ""
    return bool(
        re.search(r"\?\s*$|\b(what|who|when|where|why|how|can|does|do|is|are|difference|compare|explain)\b", text, re.IGNORECASE)
        or query_profile.get("task") in {
            "coverage_question",
            "eligibility_question",
            "exclusion_question",
            "authorization_question",
            "undertime_offset_question",
            "comparison_question",
            "general_explanation",
            "definition_question",
            "document_requirement",
            "legal_remedy",
        }
    )


def prepare_state_for_current_message(
    existing_state: Dict[str, Any],
    message: str,
    nlp: Dict[str, Any],
    language: str,
    query_profile: Dict[str, Any],
) -> Dict[str, Any]:
    state = dict(existing_state or {})
    superseded = False
    if (active_pending_calculation(state) or is_completed_calculation_state(state)) and is_complete_new_task(
        message,
        query_profile,
        state,
    ):
        state = supersede_calculation_state(state, "new_complete_task")
        superseded = True
    merged = merge_conversation_state(state, message, nlp, language)
    active_topic = str(query_profile.get("topic") or "").strip()
    if (
        str(state.get("last_topic") or "") == "holiday_pay"
        and re.search(r"\b(rest\s+day|day\s+off|holiday)\b", message or "", re.IGNORECASE)
    ):
        active_topic = "holiday_pay"
    if (
        str(state.get("last_topic") or "") == "domestic_workers"
        and str(merged.get("sector") or "") == "domestic_workers"
        and SOCIAL_AGENCY_FOLLOW_UP_RE.search(message or "")
    ):
        active_topic = "domestic_workers"
    if active_topic not in {"", "general_labor", "non_legal"}:
        merged["last_topic"] = active_topic
        if query_profile.get("sub_intent"):
            merged["last_sub_intent"] = query_profile["sub_intent"]
        if query_profile.get("intent"):
            merged["last_intent"] = query_profile["intent"]
    if superseded and not (query_profile.get("task") in CALCULATION_TASKS or query_profile.get("calculation_requested")):
        merged = supersede_calculation_state(merged, "new_complete_task")
    return merged


def should_handle_thirteenth_month_calculation(
    message: str,
    query_profile: Dict[str, Any],
    state: Dict[str, Any],
) -> bool:
    if is_source_followup_request(message):
        return False
    text = message or ""
    topic = query_profile.get("topic")
    if topic != "thirteenth_month_pay" and not re.search(r"\b13th[-\s]*month|thirteenth[-\s]*month\b", text, re.IGNORECASE):
        return False
    if not state.get("total_basic_salary_earned"):
        return False
    if query_profile.get("task") in CALCULATION_TASKS or query_profile.get("calculation_requested"):
        return True
    return bool(PERSONAL_AMOUNT_RE.search(text))


def withhold_unverified_calculation(calculation: Dict[str, Any]) -> None:
    """Do not expose unverified results through a secondary UI payload."""
    calculation["amount"] = None
    calculation["steps"] = []
    calculation["formula"] = None
    calculation["calculation_status"] = "unverified"
    if calculation.get("personalized_computation"):
        calculation["personalized_computation"]["amount"] = None


def calculation_intake_question(calculation: Dict[str, Any], language: str) -> str:
    """Localize input prompts, not legal answers or entitlements."""
    questions = {
        "fil": {
            "hourly_rate": "Magkano ang sahod mo kada oras?",
            "normal_work_hours": "Ilang regular na oras ng trabaho ang katumbas ng arawang sahod mo?",
            "overtime_hours": "Ilang oras kang nag-overtime?",
            "day_type": "Ang overtime ba ay sa ordinaryong araw, rest day, special non-working day, regular holiday, o holiday na rest day mo rin?",
        },
        "hil": {
            "hourly_rate": "Pila ang imo sweldo kada oras?",
            "normal_work_hours": "Pila ka regular nga oras sang trabaho ang katumbas sang imo inadlaw nga sweldo?",
            "overtime_hours": "Pila ka oras ka nag-overtime?",
            "day_type": "Ang overtime bala sa ordinaryo nga adlaw, rest day, special non-working day, regular holiday, ukon holiday nga rest day mo man?",
        },
    }
    localized = questions.get(language, {})
    state = calculation.get("state") or {}
    missing = calculation.get("missing_fields") or []
    if state.get("hours_worked") and not state.get("day_type") and "day_type" in missing:
        return localized.get("day_type", "Was this on an ordinary workday, rest day, special non-working holiday, regular holiday, or a holiday that was also your rest day?")
    if (state.get("hours_worked") and state.get("day_type")
            and not state.get("hourly_rate") and not state.get("daily_wage")
            and not state.get("daily_rate")):
        return {
            "fil": "Magkano ang sahod mo kada oras o kada araw?",
            "hil": "Pila ang imo sweldo kada oras ukon kada adlaw?",
        }.get(language, "What is your hourly rate or daily wage?")
    for field in ("hourly_rate", "normal_work_hours", "overtime_hours", "day_type"):
        if field in (calculation.get("missing_fields") or []) and field in localized:
            return localized[field]
    return calculation.get("clarification_question") or {
        "fil": "Ano ang kulang na detalye para makuwenta ang bayad?",
        "hil": "Ano ang kulang nga detalye para makwenta ang bayad?",
    }.get(language, "Which missing calculation detail can you provide?")


def source_supported_normal_work_hours(chunks: List[Dict[str, Any]]) -> Optional[str]:
    """Return a daily threshold only when an operative retrieved passage states it."""
    from routers.services.legal_topic_matrix import source_passage_text
    for chunk in chunks or []:
        body = re.sub(r"\s+", " ", source_passage_text(chunk.get("text") or "")).lower()
        explicit_eight_hour_rule = re.search(
            r"(?:beyond|in excess of|more than)\s+(?:eight\s*\(?\s*8\s*\)?|8)\s+hours?"
            r"|(?:normal|regular)\s+(?:hours?\s+of\s+work|working\s+hours?).{0,100}"
            r"(?:eight\s*\(?\s*8\s*\)?|8)\s+hours?",
            body,
        )
        if explicit_eight_hour_rule:
            return "8"
    return None


async def build_thirteenth_month_calculation_response(
    message: str,
    nlp: Dict[str, Any],
    intent: str,
    language: str,
    history: List[Dict[str, Any]],
    state: Dict[str, Any],
    request_plan=None,
) -> Optional[Dict[str, Any]]:
    if request_plan and request_plan.get('needs_clarification'):
        return None  # Retrieve and explain the supported part in normal chat.
    query_profile = classify_legal_query(message, nlp)
    if not should_handle_thirteenth_month_calculation(message, query_profile, state):
        return None

    calculation = thirteenth_month_pay(state["total_basic_salary_earned"])
    calculation_payload: Dict[str, Any] = {
        "topic": "thirteenth_month_pay",
        "inputs": calculation.get("inputs") or {},
        "missing_fields": [],
        "assumptions": calculation.get("assumptions") or [],
        "formula": calculation.get("formula"),
        "steps": calculation.get("steps") or [],
        "amount": calculation.get("estimated_result"),
        "currency": "PHP",
        "is_estimate": True,
        "calculation_status": "valid",
        "personalized_computation": {
            "amount": calculation.get("estimated_result"),
            "currency": "PHP",
            "is_estimate": True,
        },
    }
    state["calculation_topic"] = "thirteenth_month_pay"

    verification_query = (
        message + "\nSupporting provisions: "
        "13th-month pay formula total basic salary earned during the year divided by 12 "
        "PD 851 DOLE 13th month pay"
    )
    verification_nlp = {
        **nlp,
        "normalized": verification_query,
        "intent": "13th_month",
    }
    rag = await retrieve_scoped_context(
        verification_query,
        verification_nlp,
        history,
        "13th_month",
        conversation_state=state,
        request_plan=request_plan,
    )
    confidence_guard = low_confidence_result(rag, language)
    if confidence_guard:
        calculation_payload["legal_source_status"] = SOURCE_STATUS_INSUFFICIENT
        withhold_unverified_calculation(calculation_payload)
        return {
            "answer": confidence_guard["answer"],
            "sources": [],
            "source_status": confidence_guard["source_status"],
            "source_reason": confidence_guard["source_reason"],
            "kb_resolved": False,
            "validation": {},
            "calculation": calculation_payload,
            "state": state,
            "topic": "thirteenth_month_pay",
        }
    result, source_status, source_reason, resolved, validation = await generate_grounded_answer(
        rag, message, language, history, "13th_month", calculation=calculation_payload,
    )
    answer, sources = result["answer"], result["sources"]
    if not resolved:
        withhold_unverified_calculation(calculation_payload)
    calculation_payload["legal_source_status"] = source_status

    return {
        "answer": answer,
        "sources": sources,
        "source_status": source_status,
        "source_reason": source_reason,
        "kb_resolved": source_status in {SOURCE_STATUS_DIRECT, SOURCE_STATUS_PARTIAL},
        "validation": validation,
        "calculation": calculation_payload,
        "state": state,
        "topic": "thirteenth_month_pay",
    }


def calculation_state_from_history(
    history: List[Dict[str, Any]],
    current_message: str,
    nlp: Dict[str, Any],
    language: str,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    pending_field: Optional[str] = None
    for item in history or []:
        if item.get("role") == "user":
            state = merge_conversation_state(state, item.get("content", ""), {"normalized": item.get("content", "")}, language)
        elif item.get("role") == "assistant":
            pending_field = infer_pending_field_from_answer(item.get("content", "")) or pending_field
    if pending_field:
        state[PENDING_CALCULATION_KEY] = {
            "status": "awaiting_input",
            "topic": state.get("calculation_topic") or state.get("issue") or "overtime_pay",
            "fields": ["hourly_rate", "overtime_hours", "day_type", "daily_wage", "normal_work_hours"],
            "missing_fields": [pending_field],
            "last_question_asked": pending_field,
            "hourly_rate": state.get("hourly_rate"),
            "overtime_hours": state.get("overtime_hours"),
            "day_type": state.get("day_type"),
            "daily_wage": state.get("daily_wage"),
            "normal_work_hours": state.get("normal_work_hours"),
        }
    query_profile = classify_legal_query(current_message, nlp)
    return prepare_state_for_current_message(state, current_message, nlp, language, query_profile)


async def build_personalized_calculation_response(
    message: str,
    nlp: Dict[str, Any],
    intent: str,
    language: str,
    history: List[Dict[str, Any]],
    state: Dict[str, Any],
    request_plan=None,
) -> Optional[Dict[str, Any]]:
    if request_plan and request_plan.get('needs_clarification'):
        return None  # The LLM-written question follows the source-based prose.
    original_state = dict(state or {})
    state = apply_pending_calculation_reply(original_state, message)
    query_profile = classify_legal_query(message, nlp)
    if not should_handle_personalized_calculation(message, query_profile, state):
        return None
    if state.get("calculation_topic") == "overtime_pay" or active_pending_calculation(original_state) or active_pending_calculation(state):
        query_profile = {
            **query_profile,
            "topic": "overtime_pay",
            "task": "personalized_computation",
            "calculation_requested": True,
            "personal_facts_available": True,
        }

    state.setdefault("calculation_topic", "overtime_pay")
    calculation = build_overtime_calculation(state).to_payload()
    calculation["calculation_status"] = "valid" if calculation.get("amount") is not None else "awaiting_input"
    calculation["personalized_computation"] = {
        "amount": calculation.get("amount"),
        "currency": calculation.get("currency", "PHP"),
        "is_estimate": bool(calculation.get("is_estimate")),
    }
    calculation["state"] = state
    state = update_pending_calculation_state(state, calculation, "overtime_pay")
    logger.info(
        "Calculation flow | topic=%s | task=%s | pending=%s | inputs=%s | missing=%s | amount=%s",
        query_profile.get("topic"),
        query_profile.get("task"),
        active_pending_calculation(state),
        calculation.get("inputs"),
        calculation.get("missing_fields"),
        calculation.get("amount"),
    )
    answer = calculation_intake_question(calculation, language)
    if calculation.get("clarification_question"):
        calculation["clarification_question"] = answer
    sources: List[Dict[str, Any]] = []
    source_status = SOURCE_STATUS_INSUFFICIENT
    source_reason = "No approved legal source was retrieved for the applied calculation multiplier."
    validation: Dict[str, Any] = {}
    scenario_inputs_ready = bool(
        state.get("hours_worked") and state.get("day_type")
        and (state.get("hourly_rate") or state.get("daily_wage") or state.get("daily_rate"))
    )
    if request_plan and calculation.get('amount') is None and not scenario_inputs_ready:
        # Semantic intake may understand a typo/new wording the old card parser
        # does not. Let the normal generator use that interpretation instead of
        # asking another hardcoded calculator question for already supplied facts.
        return None

    if calculation.get("amount") is not None or scenario_inputs_ready:
        verification_query = calculation_verification_query(message, state, calculation)
        if scenario_inputs_ready:
            verification_query += f" Total hours worked: {state.get('hours_worked')}."
        verification_nlp = {
            **nlp,
            "normalized": verification_query,
            "intent": intent,
        }
        rag = await retrieve_scoped_context(
            verification_query,
            verification_nlp,
            history,
            intent,
            conversation_state=state,
            request_plan=request_plan,
        )
        confidence_guard = low_confidence_result(rag, language)
        if confidence_guard:
            answer = confidence_guard["answer"]
            withhold_unverified_calculation(calculation)
            sources = []
            source_status = confidence_guard["source_status"]
            source_reason = confidence_guard["source_reason"]
        else:
            if scenario_inputs_ready and not state.get("overtime_hours"):
                threshold = source_supported_normal_work_hours(rag.get("chunks") or [])
                if threshold:
                    state["normal_work_hours"] = threshold
                    try:
                        overtime = max(Decimal("0"), Decimal(str(state["hours_worked"])) - Decimal(threshold))
                    except Exception:
                        overtime = Decimal("0")
                    if overtime > 0:
                        state["overtime_hours"] = str(overtime.normalize())
                    calculation = build_overtime_calculation(state).to_payload()
                    calculation["calculation_status"] = "valid" if calculation.get("amount") is not None else "awaiting_input"
                    calculation["personalized_computation"] = {
                        "amount": calculation.get("amount"),
                        "currency": calculation.get("currency", "PHP"),
                        "is_estimate": bool(calculation.get("is_estimate")),
                    }
                    calculation["state"] = state
                    state = update_pending_calculation_state(state, calculation, "overtime_pay")
            result, source_status, source_reason, resolved, validation = await generate_grounded_answer(
                rag, message, language, history, intent, calculation=calculation,
            )
            answer, sources = result["answer"], result["sources"]
            if not resolved:
                withhold_unverified_calculation(calculation)
    else:
        source_status = "not_required"
        source_reason = "The bot asked for missing calculation values before applying a legal rate."
    calculation["legal_source_status"] = source_status

    return {
        "answer": answer,
        "sources": sources,
        "source_status": source_status,
        "source_reason": source_reason,
        "kb_resolved": source_status in {SOURCE_STATUS_DIRECT, SOURCE_STATUS_PARTIAL, "not_required"},
        "validation": validation,
        "calculation": calculation,
        "state": state,
    }


def validation_reason(validation: Dict[str, Any], default_reason: str) -> str:
    errors = validation.get("errors") or []
    missing = validation.get("missing_articles") or []
    missing_concepts = validation.get("missing_concepts") or []
    if errors:
        return "Answer verification failed: " + "; ".join(str(error) for error in errors[:3])
    if missing:
        return "The answer is legally correct, but some supporting provisions were not retrieved: Article " + ", Article ".join(missing) + "."
    if missing_concepts:
        return "The answer is supported by retrieved sources, but the source set is missing: " + ", ".join(missing_concepts) + "."
    return default_reason


def build_insufficient_source_answer(language: str, reason: str = "") -> str:
    from routers.services.response_style import official_guidance_referral
    return official_guidance_referral(language)


def verification_unavailable_answer(result: Dict[str, Any], language: str):
    """Withhold unchecked prose without pretending the documents are missing."""
    intros = {
        'en': 'I found relevant source material, but the answer check is temporarily unavailable, '
              'so I can’t verify a complete answer. Please review the source cards below or contact DOLE at 1349.',
        'fil': 'May nakuha akong posibleng may-kinalamang sanggunian, pero pansamantalang hindi gumagana '
               'ang pag-check ng sagot kaya hindi ko matiyak ang kumpletong sagot. Suriin ang mga source card sa ibaba o tumawag sa DOLE sa 1349.',
        'hil': 'May nakuha ako nga posible nga may kaangtanan nga mga source, pero temporaryo nga indi available '
               'ang pag-check sang sabat gani indi ko ma-verify ang kompleto nga sabat. Susiha ang mga source card sa idalom ukon tawagi ang DOLE sa 1349.',
    }
    sources = list(result.get('sources') or [])[:2]
    reason = 'Independent answer verification was unavailable. The unchecked legal draft was withheld.'
    result.update(answer=intros.get(language, intros['en']),
                  sources=sources, provider_unavailable=True, verification_unavailable=True,
                  final_status='answer_verification_unavailable')
    return (result, SOURCE_STATUS_PROVIDER_UNAVAILABLE, reason, False,
            {'source_match': SOURCE_STATUS_PROVIDER_UNAVAILABLE,
             'grounding_verification': {'state': 'unavailable'}, 'errors': [reason]})


def verification_failed_source_fallback(rag: Dict[str, Any], language: str, audit=None):
    """Withhold unverified conclusions and leave supporting material in source cards."""
    from routers.services.openai_service import build_sources_payload
    from routers.services.evidence_scope import scope_notice

    chunks = list(rag.get('chunks') or [])
    try:
        from routers.services.document_evidence import rank_question_candidates
        question = str((rag.get('query_profile') or {}).get('retrieval_query') or '')
        chunks = rank_question_candidates(chunks, question)[:2]
    except Exception:
        chunks = chunks[:2]
    sources = build_sources_payload(chunks)
    from routers.services.response_style import official_guidance_referral
    lead = official_guidance_referral(language)
    scope = (rag.get('query_profile') or {}).get('scope_completeness') or {}
    if scope.get('scope_gap'):
        lead += '\n\n' + scope_notice(scope, language)
    reason = 'The bounded repair and final verification did not establish a complete, supported answer.'
    result = {'answer': lead, 'sources': sources, 'verification_failed': True,
              'final_status': 'answer_verification_failed_source_limited',
              'grounding_verification': audit or {}}
    validation = {'source_match': SOURCE_STATUS_INSUFFICIENT, 'evidence_limitation': True,
                  'grounding_verification': audit or {}, 'errors': [reason]}
    return result, SOURCE_STATUS_INSUFFICIENT, reason, False, validation


def apply_source_validation(
    result: Dict[str, Any],
    query_profile: Dict[str, Any],
    source_reason: str,
    language: str,
    audit_phase: str = 'initial',
) -> tuple[Dict[str, Any], str, str, bool, Dict[str, Any]]:
    if is_document_driven(query_profile) and (result.get('grounding_verification') or {}).get('state') not in {
        'checked', 'unsupported', 'uncertain',
    }:
        set_timing_metadata(stage4_checks=[
            {'check': name, 'passed': False, 'reason': 'check_unavailable'}
            for name in ('question_coverage', 'claim_support', 'citation_identity',
                         'no_invented_claims', 'scope_and_facts')
        ])
        return verification_unavailable_answer(result, language)
    citation_started = time.perf_counter()
    result["answer"] = normalize_known_source_filenames(
        result.get("answer", ""),
        result.get("sources", []),
    )
    result["answer"] = remove_unsupported_authority_citations(
        result.get("answer", ""), result.get("sources", [])
    )
    verification = result.get('grounding_verification') or {}
    if is_document_driven(query_profile) and verification.get('state') in {'unsupported', 'uncertain'}:
        # Operational diagnostics only: never log draft/user text or quotations.
        logger.warning('Answer audit rejected: state=%s checked_units=%s supported_units=%s '
                       'citation_mismatches=%s missing_points=%s evidence_gaps=%s',
                       verification.get('state'), verification.get('checked_claim_count', 0),
                       verification.get('supported_claim_count', 0),
                       sum(c.get('issue') == 'citation_mismatch' for c in verification.get('claim_checks') or []),
                       len(verification.get('missing_points') or []), len(verification.get('evidence_gaps') or []))
        from routers.services.answer_audit_diagnostics import audit_diagnostics
        logger.warning('Answer audit diagnostics: phase=%s details=%s',
                       audit_phase if audit_phase in {'initial', 'repair', 'supported_portion'} else 'other',
                       audit_diagnostics(verification))
    display_profile = {**query_profile, 'supporting_source_origins': verification.get('supporting_source_origins', [])
                       if verification.get('state') == 'checked' else []}
    result["sources"] = filter_display_sources_for_answer(
        result.get("answer", ""),
        result.get("sources", []),
        display_profile,
    )
    result["answer"] = remap_answer_source_citations(result.get("answer", ""), result["sources"])
    result["answer"] = remove_unsupported_answer_urls(
        result.get("answer", ""),
        result["sources"],
    )
    add_timing("citation_validation_ms", (time.perf_counter() - citation_started) * 1000)
    validation = validate_legal_answer(result["answer"], result["sources"], {
        **query_profile, 'answer_language': language,
        'final_grounding_checked': is_document_driven(query_profile) and verification.get('state') == 'checked',
    })
    from routers.services.evidence_scope import scope_completeness, scope_notice
    final_scope = scope_completeness(query_profile.get('retrieval_query') or query_profile.get('normalized_query') or '', result['sources'])
    validation['scope_completeness'] = final_scope
    if is_document_driven(query_profile):
        verification = result.get('grounding_verification')
        validation['grounding_verification'] = verification or {'state': 'unavailable'}
        if verification and verification.get('state') in {'unsupported', 'uncertain'}:
            validation.setdefault('errors', []).extend(verification.get('errors') or ['The draft contains a claim whose source support is uncertain.'])
            validation['source_match'] = SOURCE_STATUS_INSUFFICIENT
        elif not verification.get('missing_points') and not validation.get('errors') and not validation.get('warnings'):
            validation['source_match'] = SOURCE_STATUS_DIRECT
        if verification.get('state') == 'checked':
            from routers.services.document_evidence import has_evidence_limitation
            evidence_gap = bool(verification.get('evidence_gaps') or has_evidence_limitation(result['answer']))
            if evidence_gap:
                # A checked explanation of a general rule is not proof that
                # the worker's requested conclusion was actually resolved.
                validation['source_match'] = (SOURCE_STATUS_PARTIAL
                    if verification.get('supported_claim_count', 0) > 0 else SOURCE_STATUS_INSUFFICIENT)
                validation['evidence_limitation'] = True
        if (verification.get('state') == 'checked' and final_scope['scope_gap']
                and not validation.get('errors')):
            # A checker can certify a scoped statement without resolving the
            # requested general rule. Never publish that as a sufficient answer.
            result['answer'] = scope_notice(final_scope, language)
            validation.update(source_match=SOURCE_STATUS_PARTIAL, evidence_limitation=True)
    claim_checks = [item for item in (verification.get('claim_checks') or []) if isinstance(item, dict)]
    rejected_claims = any(item.get('verdict') in {'unsupported', 'uncertain'} for item in claim_checks)
    citation_mismatch = any(item.get('issue') == 'citation_mismatch' for item in claim_checks)
    checked = verification.get('state') == 'checked'
    coverage_gap = bool(verification.get('missing_points') or verification.get('evidence_gaps'))
    stage4_checks = [
        {'check': 'question_coverage', 'passed': checked and not coverage_gap,
         'reason': 'verified' if checked and not coverage_gap else 'requested_point_gap' if coverage_gap else 'check_unavailable'},
        {'check': 'claim_support', 'passed': checked and not rejected_claims,
         'reason': 'supported' if checked and not rejected_claims else 'unsupported_claim' if rejected_claims else 'check_unavailable'},
        {'check': 'citation_identity', 'passed': checked and not citation_mismatch,
         'reason': 'verified' if checked and not citation_mismatch else 'citation_mismatch' if citation_mismatch else 'check_unavailable'},
        {'check': 'no_invented_claims', 'passed': checked and not rejected_claims and not verification.get('unchecked_claims'),
         'reason': 'verified' if checked and not rejected_claims and not verification.get('unchecked_claims') else 'unsupported_claim' if rejected_claims else 'unchecked_claim' if verification.get('unchecked_claims') else 'check_unavailable'},
        {'check': 'scope_and_facts', 'passed': checked and not final_scope.get('scope_gap') and not rejected_claims,
         'reason': 'verified' if checked and not final_scope.get('scope_gap') and not rejected_claims else 'scope_gap' if final_scope.get('scope_gap') else 'fact_application_unverified' if rejected_claims else 'check_unavailable'},
    ]
    set_timing_metadata(stage4_checks=stage4_checks,
                        verification_failed_due_to_missing_evidence=bool(
                            verification.get('evidence_gaps') or final_scope.get('scope_gap')))
    source_status = validation["source_match"]
    checked_partial_answer = bool(
        is_document_driven(query_profile)
        and (result.get('grounding_verification') or {}).get('state') == 'checked'
        and (result.get('grounding_verification') or {}).get('supported_claim_count', 0) > 0
    )
    if answer_claims_unavailable(result.get("answer", "")) and not checked_partial_answer:
        source_status = SOURCE_STATUS_INSUFFICIENT
        validation["source_match"] = SOURCE_STATUS_INSUFFICIENT
        validation.setdefault("errors", []).append(
            "The generated answer states that the requested information is unavailable."
        )
    resolved = source_status in {SOURCE_STATUS_DIRECT, SOURCE_STATUS_PARTIAL}
    if is_document_driven(query_profile) and verification.get('state') == 'checked':
        source_reason = ('Some points were verified, but a requested conclusion remains unresolved in the retrieved evidence.'
                         if validation.get('evidence_limitation') else 'The explanation was checked against the retrieved sources.')
    reason = validation_reason(validation, source_reason)
    if source_status == SOURCE_STATUS_INSUFFICIENT:
        if is_document_driven(query_profile) and verification.get('state') in {'unsupported', 'uncertain'}:
            # A rejected draft is not proof that a document is missing. Keep
            # that distinction visible instead of asking an admin to reindex
            # sources which retrieval already found.
            messages = {
                'en': 'I found source passages, but I could not verify the explanation against them. '
                      'I have withheld the draft instead of presenting it as a reliable answer. '
                      'Please try again, or ask the administrator to review the answer and its source passages.',
                'fil': 'May nakuha akong mga bahagi ng source, pero hindi ko ma-verify ang paliwanag gamit ang mga ito. '
                       'Hindi ko ipapakita ang draft bilang maaasahang sagot. Subukan muli, '
                       'o ipa-review sa admin ang sagot at ang mga source nito.',
                'hil': 'May nakuha ako nga mga bahin sang source, pero indi ko ma-verify ang paliwanag gamit ini. '
                       'Indi ko ipakita ang draft bilang masaligan nga sabat. Sulayi liwat, '
                       'ukon ipa-review sa admin ang sabat kag ang mga source sini.',
            }
            result.update(answer=messages.get(language, messages['en']), verification_failed=True,
                          final_status='answer_verification_failed')
        else:
            result["answer"] = build_insufficient_source_answer(language, reason)
        # Retrieved candidates are not verified evidence for the fallback.
        result["sources"] = []
    if checked_partial_answer and resolved:
        from routers.services.document_evidence import partial_evidence_notice
        # Ranking maps are discovery hints, not a second final answer checker.
        # Display filtering can remove an unused, incidental mapped passage.
        # That does not make its topic an unanswered part of the user's request.
        missing = list(dict.fromkeys(verification.get('missing_points') or []))
        if missing:
            result['answer'] += partial_evidence_notice(missing, language)
            source_status = SOURCE_STATUS_PARTIAL
            validation['source_match'] = SOURCE_STATUS_PARTIAL
    return result, source_status, reason, resolved, validation


TERMINATION_COMPARISON_COMPONENTS = (
    "just_cause_definition",
    "just_cause_grounds",
    "just_cause_due_process",
    "authorized_cause_definition",
    "authorized_cause_grounds",
    "authorized_cause_notice",
    "separation_pay",
    "examples",
    "legal_basis",
)


def enforce_final_answer_coverage(
    result: Dict[str, Any],
    query_profile: Dict[str, Any],
    language: str,
) -> Dict[str, Any]:
    """Validate the rendered answer after all provider/compaction steps."""
    # A provider-unavailable message is intentionally not a legal answer;
    # running legal component checks on it would incorrectly relabel a
    # service outage as "insufficient legal source."
    if result.get("provider_unavailable"):
        return result
    if query_profile.get('simple_rag_mode'):
        return result
    if is_document_driven(query_profile):
        # Coverage for these answers comes from the requested question points
        # and the independent draft check, not a fixed termination checklist.
        return result
    if str(query_profile.get("sub_intent") or "") != "termination_comparison":
        return result
    answer = str(result.get("answer") or "")
    answer, compacted = compress_answer_to_budget(answer, settings.LABORLENS_HARD_OUTPUT_LIMIT)
    coverage = check_final_answer_coverage(answer, TERMINATION_COMPARISON_COMPONENTS)
    result["answer"] = answer
    result["final_answer_coverage"] = coverage
    set_timing_metadata(
        final_answer_coverage=coverage.get("coverage"),
        missing_after_compaction=coverage.get("missing"),
        compaction_triggered=compacted,
        unsafe_due_process_phrase=coverage.get("unsafe_due_process_phrase"),
        empty_heading_detected=coverage.get("empty_heading"),
        finalization_allowed=coverage.get("complete"),
    )
    if not coverage.get("complete"):
        logger.error(
            "[Final Answer Coverage] sub_intent=termination_comparison missing_after_compaction=%s "
            "unsafe_due_process_phrase=%s empty_heading_detected=%s finalization_allowed=false",
            coverage.get("missing"),
            coverage.get("unsafe_due_process_phrase"),
            coverage.get("empty_heading"),
        )
        # Keep a complete-sentence supported portion visible when the provider
        # stopped early. A generic insufficient-source answer would discard the
        # useful comparison the model already produced.
        if coverage.get("complete_sentence") and not coverage.get("unsafe_due_process_phrase") and len(answer.strip()) >= 180:
            result["final_status"] = "answer_partial_after_compaction"
        else:
            result["answer"] = build_insufficient_source_answer(
                language,
                "The generated answer did not preserve every requested legal component after validation.",
            )
            result["final_status"] = "answer_incomplete_after_compaction"
    return result


@timed_stage('recovery_ms', recovery_called=True)
async def recover_answer_evidence(rag, question, missing_points, profile):
    """One bounded discovery pass for gaps identified by the answer checker."""
    from routers.services.chroma_service import search_document_text, search_document_neighbors, rescore_retrieval_chunks
    from routers.services.document_evidence import (
        eligible_candidates, candidate_pool, rank_question_candidates,
        select_question_chunks, question_search_queries, normalized,
    )
    requested = []
    for point in missing_points or []:
        # A rejected claim without a named gap previously just repeated the
        # broad question twice. Recover its actual components independently.
        requested.extend(question_search_queries(question) or [question]
                         if normalized(point) == normalized(question) else [str(point)])
    probes = [str(p)[:240] + ' | ' + question for p in list(dict.fromkeys(requested))[:2]]
    if not probes:
        return rag, profile
    set_timing_metadata(recovery_triggered=True)

    async def discover(probe):
        vector, literal = await asyncio.gather(
            search_documents(query=probe, n_results=20, filter_metadata=None),
            search_document_text(probe, 20))
        return eligible_candidates(await filter_searchable_chunks(
            _merge_retrieval_chunks(vector, literal), question), profile)

    async def recover():
        groups = await asyncio.gather(*(discover(p) for p in probes))
        additions = [group[i] for i in range(max(map(len, groups), default=0))
                     for group in groups if i < len(group)]
        known = list(rag.get('all_chunks') or rag['chunks'])
        known_keys = {_retrieval_chunk_key(c) for c in known}
        additions = [c for c in _merge_retrieval_chunks(additions, []) if _retrieval_chunk_key(c) not in known_keys]
        neighbors = eligible_candidates(await filter_searchable_chunks(
            await search_document_neighbors(question, additions + rank_question_candidates(known, question)), question), profile)
        neighbors = [c for c in neighbors if _retrieval_chunk_key(c) not in known_keys]
        if not additions and not neighbors:
            return rag, profile
        new_chunks = await rescore_retrieval_chunks(question, _merge_retrieval_chunks(additions, neighbors))
        chunks = _merge_retrieval_chunks(known, new_chunks)
        from routers.services.ranking_policy import integrate_recovery, conflict_reason, recovery_introduces_conflict
        candidates = candidate_pool(new_chunks, new_chunks, profile)
        ranking_local_scores = dict(rag.get('ranking_local_scores') or {})
        prior_ranked = rag.get('ranked_candidates') or rag['chunks']
        ranked = integrate_recovery(question, prior_ranked, candidates, profile,
                                    prior_scores=ranking_local_scores)
        from routers.services.document_evidence import duplicate_key, question_candidate_scorer
        scorer = question_candidate_scorer(question)
        ranking_local_scores.update({duplicate_key(c): scorer(c) for c in candidates})
        llm_ranked = bool(rag.get('llm_ranked'))
        semantic_ranked_keys = frozenset(rag.get('semantic_ranked_keys') or ())
        set_timing_metadata(recovery_new_candidates=len(new_chunks), prior_llm_ranking_preserved=llm_ranked)
        if recovery_introduces_conflict(question, prior_ranked, candidates):
            from routers.services.model_pools import rank_evidence, last_ranking_succeeded
            conflict_ranked, _ = await rank_evidence(question, ranked, profile=profile)
            if last_ranking_succeeded.get():
                ranked, llm_ranked = conflict_ranked, True
                semantic_ranked_keys = frozenset(duplicate_key(c) for c in ranked)
                set_timing_metadata(prior_llm_ranking_preserved=False)
        from routers.services.evidence_selection import select_evidence, context_budget, refresh_selection_diagnostics
        selected, selection = select_evidence(ranked, question, preserve_order=True,
            additional_points=missing_points or (), semantic_ranked_keys=semantic_ranked_keys)
        from routers.services.complete_sections import expand_complete_sections
        selected = eligible_candidates(await filter_searchable_chunks(
            await expand_complete_sections(selected, limit=None, token_budget=context_budget(question)), question), profile)
        if not selected:
            return rag, profile
        updated = {**profile, 'candidate_reranked': True,
                   'llm_ranking_used': llm_ranked,
                   'evidence_selection': refresh_selection_diagnostics(selection, selected)}
        from routers.services.evidence_scope import scope_completeness
        updated['scope_completeness'] = scope_completeness(question, selected)
        updated.pop('document_evidence', None)
        updated.pop('document_evidence_coverage', None)
        set_timing_metadata(selected_passages=len(selected), evidence_tokens=updated['evidence_selection']['estimated_evidence_tokens'])
        return {**rag, 'chunks': selected, 'all_chunks': chunks, 'query_profile': updated,
                'ranked_candidates': ranked, 'llm_ranked': llm_ranked,
                'semantic_ranked_keys': semantic_ranked_keys,
                'ranking_local_scores': ranking_local_scores}, updated

    try:
        return await asyncio.wait_for(recover(), timeout=10)
    except Exception as exc:
        # Discovery is optional. A connector failure must not replace usable
        # earlier evidence; source publication is rechecked before any answer.
        logger.warning('Answer evidence recovery unavailable (%s); retaining previously retrieved evidence', type(exc).__name__)
        return rag, profile


@with_question_analysis
async def generate_grounded_answer(
    rag: Dict[str, Any], query: str, language: str,
    history: List[Dict[str, Any]], intent: str,
    calculation: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], str, str, bool, Dict[str, Any]]:
    """Source-based chat without model ranking, independent audit or repair loops."""
    from routers.services.basic_rag import generate
    set_timing_metadata(generation_eligibility_checked=True,
                        generation_eligible=bool(rag.get('chunks')),
                        generation_gate_reason='passages_available' if rag.get('chunks') else 'no_passages')
    return await generate(rag, query, language, history, intent)


async def generate_named_issuance_answer(query: str, language: str, history: List[dict]):
    from routers.services.named_issuance import answer_named_issuance
    lookup = await answer_named_issuance(query, language)
    if not lookup or not lookup.get("kb_found"):
        # A formatting-specific exact lookup is not a veto on global search.
        return None
    # Verbatim text is appropriate only when explicitly requested.
    if re.search(r"\b(?:quote|verbatim|exact wording|original text)\b|eksaktong teksto|orihinal nga teksto", query, re.I):
        return lookup
    # Non-verbatim questions use the same global, bounded search as ordinary
    # chat. The exact-issuance lookup above is not a second model-ranking path.
    lookup_nlp = await asyncio.to_thread(preprocess_query, query)
    rag = await retrieve_scoped_context(query, lookup_nlp, history, 'document_section_lookup')
    result, status, reason, resolved, validation = await generate_grounded_answer(
        rag, query, language, history, "document_section_lookup",
    )
    return {**result, "intent": "document_section_lookup", "kb_found": resolved,
            "source_status": status, "source_reason": reason, "validation": validation}


def build_source_listing_answer(
    query: str,
    language: str,
    query_profile: Dict[str, Any],
    chunks: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    text = query or ""
    from routers.services.source_request_routing import source_records_requested
    if not source_records_requested(text):
        return None
    sources = filter_display_sources_for_answer(
        text,
        build_sources_payload(chunks),
        query_profile,
    )
    if not sources:
        return None
    profile = get_language_profile(language)
    if language == "fil":
        direct = "Ito ang naka-save na source metadata na sumusuporta sa sagot."
    elif language == "hil":
        direct = "Amo ini ang naka-save nga source metadata nga nagasuporta sa sabat."
    else:
        direct = "These are the saved source records that support the answer."
    basis = []
    for index, source in enumerate(sources[:5], 1):
        name = source_display_name(source)
        hierarchy = source_hierarchy_line(source)
        url = source_url(source)
        url_part = f"\n  URL: {url}" if url else ""
        basis.append(f"{index}. {name}\n  {hierarchy}{url_part}")
    answer = (
        f"**{profile['direct_label']}:**\n{direct}\n\n"
        f"**{profile['legal_basis_label']}:**\n" + "\n".join(basis)
    )
    source_status = query_profile.get("source_match") or SOURCE_STATUS_PARTIAL
    if source_status == SOURCE_STATUS_INSUFFICIENT:
        source_status = SOURCE_STATUS_PARTIAL
    return {
        "answer": answer,
        "sources": sources,
        "source_status": source_status,
        "source_reason": "The answer lists stored source metadata from retrieved knowledge-base records.",
        "kb_resolved": True,
        "validation": {},
    }


def log_final_chat_response(
    response: ChatResponse,
    query_profile: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
) -> None:
    validation = validation or {}
    query_profile = query_profile or {}
    audit = validation.get('grounding_verification') or {}
    set_timing_metadata(source_match=response.source_match or response.confidence_level,
                        intent=response.intent, final_status=getattr(response, 'final_status', None),
                        verification_state=audit.get('state'),
                        supported_claim_count=audit.get('supported_claim_count'),
                        checked_claim_count=audit.get('checked_claim_count'),
                        validation_error_count=len(audit.get('errors') or []),
                        missing_point_count=len(audit.get('missing_points') or []))
    source_ids = [
        source.get("label")
        or source.get("chunk_id")
        or source.get("id")
        or source.get("citation")
        or source.get("source")
        for source in response.sources
    ]
    logger.info(
        "FINAL CHAT RESPONSE source_match=%s topic=%s source_ids=%s essential_concepts=%s supported_concepts=%s missing_concepts=%s unsupported_claims=%s removed_citations=%s reason=%s",
        response.source_match or response.confidence_level,
        response.topic or query_profile.get("topic"),
        source_ids,
        validation.get("essential_concepts", []),
        validation.get("supported_concepts", []),
        validation.get("missing_concepts", []),
        validation.get("errors", []),
        validation.get("removed_citations", []),
        response.confidence_reason,
    )


@router.post("/guest-message", response_model=ChatResponse)
async def send_guest_message(req: GuestChatRequest, request: Request):
    allowance = await guest_usage(request, consume=True)
    if not allowance["allowed"]:
        language = choose_response_language(
            detected_language=(await asyncio.to_thread(preprocess_query, req.message))["language"],
            requested_language=req.language, user_language=None,
        )
        result = ChatResponse(
            session_id=req.session_id if req.session_id in GUEST_SESSIONS else str(uuid.uuid4()),
            message_id=allowance["used"], answer=get_guest_limit_answer(language),
            intent="guest_limit_reached", language=language, sources=[],
            follow_up_questions=[], kb_found=False,
        )
    else:
        result = await run_guest_message(req, request)
        result.message_id = allowance["used"]
    result.guest_usage = allowance
    return result


@router.get("/guest-status")
async def guest_status(request: Request):
    return await guest_usage(request)


async def run_guest_message(req: GuestChatRequest, request: Request, *, sandbox: bool = False):
    """Shared chat path; admin sandbox has no guest session or persisted history."""
    if not sandbox:
        prune_guest_sessions()
    # Guest history stays in that browser. Only continue a server-issued,
    # unguessable session ID that is already known to this process.
    session_id = req.session_id if req.session_id in GUEST_SESSIONS else str(uuid.uuid4())
    usage = {"count": 0, "updated": time.time()} if sandbox else GUEST_SESSIONS.setdefault(
        session_id,
        {"count": 0, "updated": time.time(), "client": request.client.host if request.client else ""},
    )

    routing_started = time.perf_counter()
    nlp = await asyncio.to_thread(preprocess_query, req.message)
    language = choose_response_language(
        detected_language=nlp["language"],
        requested_language=req.language,
        user_language=None,
        conversation_language=latest_assistant_language_from_history(req.history),
    )
    action = detect_conversation_action(req.message)
    intent = "translation_request" if action == "translation_request" else normalize_static_intent(
        req.message, nlp["intent"], req.previous_intent
    )
    add_timing("routing_ms", (time.perf_counter() - routing_started) * 1000)
    logger.debug(
        "Conversation routing | normalized=%s | conversational_phrases=%s | legal_intent=%s | final_intent=%s | rag_executed=pending | display_mode=pending | quick_replies=pending | source_match=pending | guest=true",
        nlp.get("normalized"),
        nlp.get("conversational_phrases", []),
        nlp.get("legal_intent") or "none",
        intent,
    )

    # The public endpoint reserves the durable client budget before entering
    # this pipeline. This local counter is only a conversation sequence number;
    # admin sandbox calls intentionally do not consume the guest allowance.
    usage["count"] += 1
    usage["updated"] = time.time()
    message_id = usage["count"]
    history = [
        {
            "role": msg.get("role", ""),
            "content": msg.get("content", ""),
            "sources": parse_sources_payload(msg.get("sources")),
            "intent": msg.get("intent"),
        }
        for msg in req.history[-20:]
        if msg.get("role") in {"user", "assistant"} and msg.get("content")
    ]
    contextual_follow_up = is_contextual_follow_up(req.message, nlp, history)

    if action == "translation_request":
        inline_text = extract_inline_translation_text(req.message)
        # Prefer the sanitized history used by the rest of the guest route so
        # a malformed client-side record cannot make a translation lose its
        # immediately preceding assistant answer.
        previous_message = get_last_assistant_from_history(history) or get_last_assistant_from_history(req.history)
        translation_result = await translate_previous_answer(
            query=req.message,
            target_languages=requested_translation_languages(req.message),
            fallback_language=language,
            previous_message=previous_message,
            chat_history=history,
            inline_text=inline_text,
        )
        logger.info(
            "[Conversation Action] message=%r detected_action=translation_request "
            "target_language=%s source_message_id=%s source_message_role=%s "
            "retrieval_skipped=true existing_category=%s final_category=%s guest=true",
            req.message,
            translation_result["language"],
            (previous_message or {}).get("id"),
            "assistant" if previous_message else "inline",
            None,
            None,
        )
        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            answer=translation_result["answer"],
            intent="translation_request",
            language=translation_result["language"],
            sources=translation_result["sources"],
            follow_up_questions=[],
            kb_found=translation_result["kb_found"],
            confidence_level=translation_result["confidence_level"],
            confidence_reason=translation_result["confidence_reason"],
        )

    static_answer = (
        await generate_conversational_reply(req.message, history, language, intent)
        if intent in MODEL_GENERATED_CONVERSATION_INTENTS
        else get_static_response(intent, language, req.message)
    )
    if static_answer:
        confidence_level, confidence_reason = confidence_for_static(intent)
        if confidence_level != "Not Required":
            static_answer = ensure_safety_disclaimer(static_answer)
        logger.info(
            "Chat route | intent=%s | language=%s | rag_executed=false | display_mode=simple_message | quick_replies=[] | source_match=%s | guest=true",
            intent,
            language,
            SOURCE_STATUS_NOT_REQUIRED,
        )
        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            answer=static_answer,
            intent=intent,
            language=language,
            sources=[],
            follow_up_questions=[],
            kb_found=False,
            source_match=SOURCE_STATUS_NOT_REQUIRED,
            display_mode="simple_message",
            quick_replies=[],
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )

    refusal_intent = prompt_injection_intent(req.message)
    if refusal_intent:
        confidence_level, confidence_reason = (
            "Not Required",
            "The user requested bypassing source or system rules, so no legal retrieval was needed.",
        )
        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            answer=build_prompt_injection_refusal(language),
            intent=refusal_intent,
            language=language,
            sources=[],
            follow_up_questions=[],
            kb_found=False,
            source_match=SOURCE_STATUS_NOT_REQUIRED,
            display_mode="simple_message",
            quick_replies=[],
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )

    guest_state = calculation_state_from_history(history, req.message, nlp, language)
    source_followup = build_source_followup_response(
        req.message,
        language,
        get_last_assistant_from_history(req.history),
    )
    if source_followup:
        confidence_level = frontend_confidence(source_followup["source_status"])
        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            answer=ensure_safety_disclaimer(source_followup["answer"]),
            intent=source_followup["intent"],
            language=language,
            sources=source_followup["sources"],
            follow_up_questions=[],
            kb_found=source_followup["kb_found"],
            source_match=source_followup["source_status"],
            topic=source_followup["topic"],
            display_mode=source_followup["display_mode"],
            quick_replies=[],
            confidence_level=confidence_level,
            confidence_reason=source_followup["source_reason"],
        )

    clarification_decision = await decide_clarification(req.message, nlp, intent, language, history)
    thirteenth_calculation_result = await build_thirteenth_month_calculation_response(
        req.message,
        nlp,
        intent,
        language,
        history,
        guest_state,
        request_plan=clarification_decision['request_plan'],
    )
    if thirteenth_calculation_result:
        confidence_level = frontend_confidence(thirteenth_calculation_result["source_status"])
        response = ChatResponse(
            session_id=session_id,
            message_id=message_id,
            answer=ensure_safety_disclaimer(thirteenth_calculation_result["answer"]),
            intent="calculation_request",
            language=language,
            sources=thirteenth_calculation_result["sources"],
            follow_up_questions=[],
            kb_found=thirteenth_calculation_result["kb_resolved"],
            source_match=thirteenth_calculation_result["source_status"],
            topic="thirteenth_month_pay",
            display_mode="calculation_card",
            calculation=thirteenth_calculation_result["calculation"],
            quick_replies=[],
            confidence_level=confidence_level,
            confidence_reason=thirteenth_calculation_result["source_reason"],
        )
        log_final_chat_response(
            response,
            {"topic": "thirteenth_month_pay", "task": "personalized_computation"},
            thirteenth_calculation_result.get("validation", {}),
        )
        return response

    calculation_result = await build_personalized_calculation_response(
        req.message,
        nlp,
        intent,
        language,
        history,
        guest_state,
        request_plan=clarification_decision['request_plan'],
    )
    if calculation_result:
        confidence_level = frontend_confidence(calculation_result["source_status"])
        confidence_reason = calculation_result["source_reason"]
        response = ChatResponse(
            session_id=session_id,
            message_id=message_id,
            answer=ensure_safety_disclaimer(calculation_result["answer"]),
            intent="calculation_request",
            language=language,
            sources=calculation_result["sources"],
            follow_up_questions=[],
            kb_found=calculation_result["kb_resolved"],
            source_match=calculation_result["source_status"],
            topic="overtime_pay",
            display_mode="calculation_card",
            calculation=calculation_result["calculation"],
            quick_replies=calculation_result["calculation"].get("quick_replies") or [],
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )
        log_final_chat_response(
            response,
            {"topic": "overtime_pay", "task": "personalized_computation"},
            calculation_result.get("validation", {}),
        )
        return response

    if needs_case_escalation(req.message, nlp):
        confidence_level, confidence_reason = (
            "Insufficient",
            "The query may require urgent action or legal review, so the bot escalated before giving a legal answer.",
        )
        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            answer=build_escalation_answer(language),
            intent="case_escalation",
            language=language,
            sources=[],
            follow_up_questions=[],
            kb_found=False,
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )

    structured_answer = await generate_named_issuance_answer(req.message, language, history)
    if structured_answer:
        structured_kb_found = structured_answer.get("kb_found", True)
        structured_status = structured_answer.get("source_status") or (
            SOURCE_STATUS_PARTIAL if structured_kb_found else SOURCE_STATUS_INSUFFICIENT
        )
        confidence_level = frontend_confidence(structured_status)
        confidence_reason = structured_answer.get("source_reason") or (
            "The requested passage was retrieved from an uploaded document."
            if structured_kb_found else "The requested section could not be verified in the retrieved material."
        )
        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            answer=ensure_safety_disclaimer(structured_answer["answer"]),
            intent=structured_answer["intent"],
            language=language,
            sources=structured_answer["sources"],
            follow_up_questions=structured_answer.get("follow_up_questions", []),
            kb_found=structured_kb_found,
            source_match=structured_status,
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )

    if clarification_decision["should_clarify"]:
        clarification_questions = clarification_decision["questions"]
        answer = clarification_questions[0]
        confidence_level, confidence_reason = confidence_for_clarification()
        clarification_profile = clarification_decision["profile"]
        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            answer=answer,
            intent="needs_clarification",
            language=language,
            sources=[],
            follow_up_questions=clarification_questions,
            quick_replies=clarification_decision["quick_replies"],
            kb_found=False,
            source_match=SOURCE_STATUS_NOT_REQUIRED,
            topic=clarification_profile.get("topic"),
            display_mode="clarification",
            clarification_question=clarification_questions[0] if clarification_questions else None,
            clarification_needed=True,
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )

    rag = await retrieve_scoped_context(req.message, nlp, history, intent,
                                       request_plan=clarification_decision['request_plan'])
    good_chunks = rag["chunks"]
    confidence_guard = low_confidence_result(rag, language)
    source_listing = None if confidence_guard else build_source_listing_answer(
        req.message, language, rag["query_profile"], good_chunks
    )
    if confidence_guard:
        result = {
            "answer": confidence_guard["answer"],
            "sources": confidence_guard["sources"],
        }
        source_status = confidence_guard["source_status"]
        source_reason = confidence_guard["source_reason"]
        kb_resolved = confidence_guard["kb_resolved"]
        validation = confidence_guard["validation"]
    elif source_listing:
        result = {
            "answer": source_listing["answer"],
            "sources": source_listing["sources"],
        }
        source_status = source_listing["source_status"]
        source_reason = source_listing["source_reason"]
        kb_resolved = source_listing["kb_resolved"]
        validation = source_listing.get("validation", {})
    else:
        result, source_status, source_reason, kb_resolved, validation = await generate_grounded_answer(
            rag, nlp["normalized"], language, history, intent,
        )
    result = enforce_final_answer_coverage(result, rag["query_profile"], language)
    if result.get("final_status") == "answer_incomplete_after_compaction":
        source_status = SOURCE_STATUS_PARTIAL
        source_reason = "The answer was withheld because required requested components did not survive final validation."
        kb_resolved = False
    confidence_level = frontend_confidence(source_status)
    confidence_reason = source_reason
    confidence_score = best_confidence_score(result["sources"])
    confidence_label = confidence_badge_label(confidence_score, confidence_level, contextual_follow_up,
        verified_partial=bool(result.get('partial_after_claim_check')))
    provider_unavailable = bool(result.get("provider_unavailable"))
    final_answer = result["answer"] if provider_unavailable else apply_source_accuracy_notes(
        result["answer"],
        rag.get("top_similarity_score"),
        rag["query_profile"],
        language,
    )
    if not (provider_unavailable or result.get("verification_failed") or confidence_guard or source_listing):
        from routers.services.answer_presentation import remove_inline_source_attributions
        final_answer = remove_inline_source_attributions(final_answer)

    response = ChatResponse(
        session_id=session_id,
        message_id=message_id,
        answer=final_answer if provider_unavailable else ensure_safety_disclaimer(final_answer),
        intent=intent,
        language=language,
        sources=result["sources"],
        follow_up_questions=result.get('follow_up_questions', clarification_decision['questions']),
        kb_found=kb_resolved,
        source_match=source_status,
        topic=rag["query_profile"].get("topic"),
        confidence_level=confidence_level,
        confidence_reason=confidence_reason,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
    )
    log_final_chat_response(response, rag["query_profile"], validation)
    return response


@router.post("/message", response_model=ChatResponse)
async def send_message(req: ChatRequest, current_user=Depends(get_current_user)):
    t0 = time.time()
    stored_user_message = redact_sensitive_info(req.message)

    # ── Session ────────────────────────────────────────────────────────────────
    if req.session_id:
        session = await execute_one(
            "SELECT * FROM chat_sessions WHERE id=%s AND user_id=%s AND is_active=1",
            (req.session_id, current_user["id"]),
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        session_id = req.session_id
    else:
        session_id = str(uuid.uuid4())
        # New sessions have no persisted category yet; translation requests
        # below still read session metadata on this first message.
        session = {"category": None}
        await execute_insert(
            "INSERT INTO chat_sessions (id, user_id, title) VALUES (%s,%s,%s)",
            (session_id, current_user["id"], stored_user_message[:60]),
        )

    # ── NLP ────────────────────────────────────────────────────────────────────
    routing_started = time.perf_counter()
    nlp      = await asyncio.to_thread(preprocess_query, req.message)
    previous_response_language = None
    if (req.language or "").strip().lower() == "auto":
        previous_response_language = await get_latest_assistant_language_from_session(
            session_id,
            current_user["id"],
        )
    language = choose_response_language(
        detected_language=nlp["language"],
        requested_language=req.language,
        user_language=current_user.get("language_preference"),
        conversation_language=previous_response_language,
    )
    action = detect_conversation_action(req.message)
    intent = "translation_request" if action == "translation_request" else normalize_static_intent(
        req.message, nlp["intent"], req.previous_intent
    )
    add_timing("routing_ms", (time.perf_counter() - routing_started) * 1000)
    logger.debug(
        "Conversation routing | normalized=%s | conversational_phrases=%s | legal_intent=%s | final_intent=%s | rag_executed=pending | display_mode=pending | quick_replies=pending | source_match=pending | guest=false",
        nlp.get("normalized"),
        nlp.get("conversational_phrases", []),
        nlp.get("legal_intent") or "none",
        intent,
    )
    conversation_summary, conversation_state = await load_session_memory(
        session_id,
        current_user["id"],
    )
    if action == "translation_request":
        inline_text = extract_inline_translation_text(req.message)
        history = await load_recent_chat_history(session_id, current_user["id"])
        previous_message = await get_last_assistant_from_session(session_id, current_user["id"])
        # Use the bounded in-memory history as a secondary source if a just
        # persisted answer is not yet visible to the session lookup.  This
        # keeps a translation request attached to the answer it follows.
        previous_message = previous_message or get_last_assistant_from_history(history)
        translation_result = await translate_previous_answer(
            query=req.message,
            target_languages=requested_translation_languages(req.message),
            fallback_language=language,
            previous_message=previous_message,
            chat_history=history,
            inline_text=inline_text,
        )
        session_category = session.get("category")
        logger.info(
            "[Conversation Action] message=%r detected_action=translation_request "
            "target_language=%s source_message_id=%s source_message_role=%s "
            "retrieval_skipped=true existing_category=%s final_category=%s inline=%s",
            req.message,
            translation_result["language"],
            (previous_message or {}).get("id"),
            "assistant" if previous_message else "inline_or_missing",
            session_category,
            session_category,
            bool(inline_text),
        )
        translation_answer = translation_result["answer"]
        assistant_language = translation_result["language"]
        confidence_level = translation_result["confidence_level"]
        confidence_reason = translation_result["confidence_reason"]
        translated_sources = translation_result["sources"]
        kb_found = translation_result["kb_found"]
        await execute_insert(
            "INSERT INTO chat_messages (session_id,user_id,role,content,language,intent) VALUES (%s,%s,%s,%s,%s,%s)",
            (session_id, current_user["id"], "user", stored_user_message, language, "translation_request"),
        )
        msg_id, confidence_score, confidence_label = await save_assistant_message(
            session_id=session_id,
            user_id=current_user["id"],
            content=translation_answer,
            language=assistant_language,
            sources=translated_sources,
            intent="translation_request",
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        schedule_nonessential_write(
            """INSERT INTO query_logs
               (user_id, session_id, query_text, intent, response_time_ms, resolved)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (current_user["id"], session_id, stored_user_message, "translation_request", elapsed_ms, True),
        )
        return ChatResponse(
            session_id=session_id,
            message_id=msg_id,
            answer=translation_answer,
            intent="translation_request",
            language=assistant_language,
            sources=translated_sources,
            follow_up_questions=[],
            kb_found=kb_found,
            source_match=SOURCE_STATUS_NOT_REQUIRED,
            display_mode="simple_message",
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
            confidence_score=confidence_score,
            confidence_label=confidence_label,
        )

    current_query_profile = classify_legal_query(req.message, nlp)
    updated_state = prepare_state_for_current_message(
        conversation_state,
        req.message,
        nlp,
        language,
        current_query_profile,
    )
    title_hint = title_from_state_or_message(req.message, updated_state)

    if intent in MODEL_GENERATED_CONVERSATION_INTENTS:
        conversation_history = await load_recent_chat_history(session_id, current_user["id"])
        static_answer = await generate_conversational_reply(
            req.message,
            conversation_history,
            language,
            intent,
        )
    else:
        static_answer = get_static_response(intent, language, req.message)
    if static_answer:
        confidence_level, confidence_reason = confidence_for_static(intent)
        if confidence_level != "Not Required":
            static_answer = ensure_safety_disclaimer(static_answer)
        logger.info(
            "Chat route | intent=%s | language=%s | rag_executed=false | display_mode=simple_message | quick_replies=[] | source_match=%s | guest=false",
            intent,
            language,
            SOURCE_STATUS_NOT_REQUIRED,
        )
        await execute_insert(
            "INSERT INTO chat_messages (session_id,user_id,role,content,language,intent) VALUES (%s,%s,%s,%s,%s,%s)",
            (session_id, current_user["id"], "user", stored_user_message, language, intent),
        )
        msg_id, confidence_score, confidence_label = await save_assistant_message(
            session_id=session_id,
            user_id=current_user["id"],
            content=static_answer,
            language=language,
            sources=[],
            intent=intent,
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        schedule_nonessential_write(
            """INSERT INTO query_logs
               (user_id, session_id, query_text, intent, response_time_ms, resolved)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (current_user["id"], session_id, stored_user_message, intent, elapsed_ms, True),
        )
        return ChatResponse(
            session_id=session_id,
            message_id=msg_id,
            answer=static_answer,
            intent=intent,
            language=language,
            sources=[],
            follow_up_questions=[],
            kb_found=False,
            source_match=SOURCE_STATUS_NOT_REQUIRED,
            display_mode="simple_message",
            quick_replies=[],
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )

    refusal_intent = prompt_injection_intent(req.message)
    if refusal_intent:
        answer = build_prompt_injection_refusal(language)
        confidence_level, confidence_reason = (
            "Not Required",
            "The user requested bypassing source or system rules, so no legal retrieval was needed.",
        )
        await execute_insert(
            "INSERT INTO chat_messages (session_id,user_id,role,content,language,intent) VALUES (%s,%s,%s,%s,%s,%s)",
            (session_id, current_user["id"], "user", stored_user_message, language, refusal_intent),
        )
        msg_id, confidence_score, confidence_label = await save_assistant_message(
            session_id=session_id,
            user_id=current_user["id"],
            content=answer,
            language=language,
            sources=[],
            intent=refusal_intent,
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        schedule_nonessential_write(
            """INSERT INTO query_logs
               (user_id, session_id, query_text, intent, response_time_ms, resolved)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (current_user["id"], session_id, stored_user_message, refusal_intent, elapsed_ms, True),
        )
        await save_session_memory(
            session_id,
            current_user["id"],
            conversation_summary,
            updated_state,
            title_hint,
        )
        return ChatResponse(
            session_id=session_id,
            message_id=msg_id,
            answer=answer,
            intent=refusal_intent,
            language=language,
            sources=[],
            follow_up_questions=[],
            kb_found=False,
            source_match=SOURCE_STATUS_NOT_REQUIRED,
            display_mode="simple_message",
            quick_replies=[],
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )

    history = await load_recent_chat_history(session_id, current_user["id"])

    source_message, requested_source_topic, missing_requested_source_topic = await get_relevant_assistant_for_source_followup(
        session_id,
        current_user["id"],
        req.message,
    )
    source_followup = build_source_followup_response(
        req.message,
        language,
        source_message,
        requested_topic=requested_source_topic,
        missing_requested_topic=missing_requested_source_topic,
    )
    if source_followup:
        answer = ensure_safety_disclaimer(source_followup["answer"])
        source_status = source_followup["source_status"]
        confidence_level = frontend_confidence(source_status)
        confidence_reason = source_followup["source_reason"]
        await execute_insert(
            "INSERT INTO chat_messages (session_id,user_id,role,content,language,intent) VALUES (%s,%s,%s,%s,%s,%s)",
            (session_id, current_user["id"], "user", stored_user_message, language, source_followup["intent"]),
        )
        msg_id, confidence_score, confidence_label = await save_assistant_message(
            session_id=session_id,
            user_id=current_user["id"],
            content=answer,
            language=language,
            sources=source_followup["sources"],
            intent=source_followup["intent"],
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        schedule_nonessential_write(
            """INSERT INTO query_logs
               (user_id, session_id, query_text, intent, response_time_ms, resolved)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (current_user["id"], session_id, stored_user_message, source_followup["intent"], elapsed_ms, source_followup["kb_found"]),
        )
        await save_session_memory(
            session_id,
            current_user["id"],
            conversation_summary,
            updated_state,
            title_hint,
        )
        return ChatResponse(
            session_id=session_id,
            message_id=msg_id,
            answer=answer,
            intent=source_followup["intent"],
            language=language,
            sources=source_followup["sources"],
            follow_up_questions=[],
            kb_found=source_followup["kb_found"],
            source_match=source_status,
            topic=source_followup["topic"],
            display_mode=source_followup["display_mode"],
            quick_replies=[],
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )

    clarification_decision = await decide_clarification(req.message, nlp, intent, language, history)
    thirteenth_calculation_result = await build_thirteenth_month_calculation_response(
        req.message,
        nlp,
        intent,
        language,
        history,
        updated_state,
        request_plan=clarification_decision['request_plan'],
    )
    if thirteenth_calculation_result:
        answer = ensure_safety_disclaimer(thirteenth_calculation_result["answer"])
        source_status = thirteenth_calculation_result["source_status"]
        confidence_level = frontend_confidence(source_status)
        confidence_reason = thirteenth_calculation_result["source_reason"]
        await execute_insert(
            "INSERT INTO chat_messages (session_id,user_id,role,content,language,intent) VALUES (%s,%s,%s,%s,%s,%s)",
            (session_id, current_user["id"], "user", stored_user_message, language, "calculation_request"),
        )
        msg_id, confidence_score, confidence_label = await save_assistant_message(
            session_id=session_id,
            user_id=current_user["id"],
            content=answer,
            language=language,
            sources=thirteenth_calculation_result["sources"],
            intent="calculation_request",
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        schedule_nonessential_write(
            """INSERT INTO query_logs
               (user_id, session_id, query_text, intent, response_time_ms, resolved)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (current_user["id"], session_id, stored_user_message, "calculation_request", elapsed_ms, thirteenth_calculation_result["kb_resolved"]),
        )
        await save_session_memory(
            session_id,
            current_user["id"],
            conversation_summary,
            thirteenth_calculation_result.get("state", updated_state),
            title_hint,
        )
        response = ChatResponse(
            session_id=session_id,
            message_id=msg_id,
            answer=answer,
            intent="calculation_request",
            language=language,
            sources=thirteenth_calculation_result["sources"],
            follow_up_questions=[],
            kb_found=thirteenth_calculation_result["kb_resolved"],
            source_match=source_status,
            topic="thirteenth_month_pay",
            display_mode="calculation_card",
            calculation=thirteenth_calculation_result["calculation"],
            quick_replies=[],
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )
        log_final_chat_response(
            response,
            {"topic": "thirteenth_month_pay", "task": "personalized_computation"},
            thirteenth_calculation_result.get("validation", {}),
        )
        return response

    calculation_result = await build_personalized_calculation_response(
        req.message,
        nlp,
        intent,
        language,
        history,
        updated_state,
        request_plan=clarification_decision['request_plan'],
    )
    if calculation_result:
        answer = ensure_safety_disclaimer(calculation_result["answer"])
        source_status = calculation_result["source_status"]
        confidence_level = frontend_confidence(source_status)
        confidence_reason = calculation_result["source_reason"]
        await execute_insert(
            "INSERT INTO chat_messages (session_id,user_id,role,content,language,intent) VALUES (%s,%s,%s,%s,%s,%s)",
            (session_id, current_user["id"], "user", stored_user_message, language, "calculation_request"),
        )
        msg_id, confidence_score, confidence_label = await save_assistant_message(
            session_id=session_id,
            user_id=current_user["id"],
            content=answer,
            language=language,
            sources=calculation_result["sources"],
            intent="calculation_request",
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        schedule_nonessential_write(
            """INSERT INTO query_logs
               (user_id, session_id, query_text, intent, response_time_ms, resolved)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (current_user["id"], session_id, stored_user_message, "calculation_request", elapsed_ms, calculation_result["kb_resolved"]),
        )
        await save_session_memory(
            session_id,
            current_user["id"],
            conversation_summary,
            calculation_result.get("state", updated_state),
            title_hint,
        )
        response = ChatResponse(
            session_id=session_id,
            message_id=msg_id,
            answer=answer,
            intent="calculation_request",
            language=language,
            sources=calculation_result["sources"],
            follow_up_questions=[],
            kb_found=calculation_result["kb_resolved"],
            source_match=source_status,
            topic="overtime_pay",
            display_mode="calculation_card",
            calculation=calculation_result["calculation"],
            quick_replies=calculation_result["calculation"].get("quick_replies") or [],
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )
        log_final_chat_response(
            response,
            {"topic": "overtime_pay", "task": "personalized_computation"},
            calculation_result.get("validation", {}),
        )
        return response

    if needs_case_escalation(req.message, nlp):
        answer = build_escalation_answer(language)
        confidence_level, confidence_reason = (
            "Insufficient",
            "The query may require urgent action or legal review, so the bot escalated before giving a legal answer.",
        )
        await execute_insert(
            "INSERT INTO chat_messages (session_id,user_id,role,content,language,intent) VALUES (%s,%s,%s,%s,%s,%s)",
            (session_id, current_user["id"], "user", stored_user_message, language, "case_escalation"),
        )
        msg_id, confidence_score, confidence_label = await save_assistant_message(
            session_id=session_id,
            user_id=current_user["id"],
            content=answer,
            language=language,
            sources=[],
            intent="case_escalation",
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        schedule_nonessential_write(
            """INSERT INTO query_logs
               (user_id, session_id, query_text, intent, response_time_ms, resolved)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (current_user["id"], session_id, stored_user_message, "case_escalation", elapsed_ms, True),
        )
        await save_session_memory(
            session_id,
            current_user["id"],
            conversation_summary,
            updated_state,
            title_hint,
        )
        return ChatResponse(
            session_id=session_id,
            message_id=msg_id,
            answer=answer,
            intent="case_escalation",
            language=language,
            sources=[],
            follow_up_questions=[],
            kb_found=False,
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )

    structured_answer = await generate_named_issuance_answer(req.message, language, history)
    if structured_answer:
        structured_intent = structured_answer["intent"]
        structured_response = ensure_safety_disclaimer(structured_answer["answer"])
        structured_kb_found = structured_answer.get("kb_found", True)
        structured_status = structured_answer.get("source_status") or (
            SOURCE_STATUS_PARTIAL if structured_kb_found else SOURCE_STATUS_INSUFFICIENT
        )
        confidence_level = frontend_confidence(structured_status)
        confidence_reason = structured_answer.get("source_reason") or (
            "The requested passage was retrieved from an uploaded document."
            if structured_kb_found else "The requested section could not be verified in the retrieved material."
        )
        await execute_insert(
            "INSERT INTO chat_messages (session_id,user_id,role,content,language,intent) VALUES (%s,%s,%s,%s,%s,%s)",
            (session_id, current_user["id"], "user", stored_user_message, language, structured_intent),
        )
        msg_id, confidence_score, confidence_label = await save_assistant_message(
            session_id=session_id,
            user_id=current_user["id"],
            content=structured_response,
            language=language,
            sources=structured_answer["sources"],
            intent=structured_intent,
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        schedule_nonessential_write(
            """INSERT INTO query_logs
               (user_id, session_id, query_text, intent, response_time_ms, resolved)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (current_user["id"], session_id, stored_user_message, structured_intent, elapsed_ms, structured_kb_found),
        )
        await save_session_memory(
            session_id,
            current_user["id"],
            conversation_summary,
            updated_state,
            title_hint,
        )
        if not structured_kb_found:
            schedule_nonessential_review(
                msg_id,
                current_user["id"],
                "The structured lookup did not find the requested legal reference in the knowledge base.",
            )
        return ChatResponse(
            session_id=session_id,
            message_id=msg_id,
            answer=structured_response,
            intent=structured_intent,
            language=language,
            sources=structured_answer["sources"],
            follow_up_questions=structured_answer.get("follow_up_questions", []),
            kb_found=structured_kb_found,
            source_match=structured_status,
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )

    # ── RAG: ChromaDB search ───────────────────────────────────────────────────
    history = await load_recent_chat_history(session_id, current_user["id"])

    if clarification_decision["should_clarify"]:
        clarification_questions = clarification_decision["questions"]
        answer = clarification_questions[0]
        confidence_level, confidence_reason = confidence_for_clarification()
        clarification_profile = clarification_decision["profile"]
        await execute_insert(
            "INSERT INTO chat_messages (session_id,user_id,role,content,language,intent) VALUES (%s,%s,%s,%s,%s,%s)",
            (session_id, current_user["id"], "user", stored_user_message, language, "needs_clarification"),
        )
        msg_id, confidence_score, confidence_label = await save_assistant_message(
            session_id=session_id,
            user_id=current_user["id"],
            content=answer,
            language=language,
            sources=[],
            intent="needs_clarification",
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        schedule_nonessential_write(
            """INSERT INTO query_logs
               (user_id, session_id, query_text, intent, response_time_ms, resolved)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (current_user["id"], session_id, stored_user_message, "needs_clarification", elapsed_ms, True),
        )
        await save_session_memory(
            session_id,
            current_user["id"],
            conversation_summary,
            updated_state,
            title_hint,
        )
        return ChatResponse(
            session_id=session_id,
            message_id=msg_id,
            answer=answer,
            intent="needs_clarification",
            language=language,
            sources=[],
            follow_up_questions=clarification_questions,
            quick_replies=clarification_decision["quick_replies"],
            kb_found=False,
            source_match=SOURCE_STATUS_NOT_REQUIRED,
            topic=clarification_profile.get("topic"),
            display_mode="clarification",
            clarification_question=clarification_questions[0] if clarification_questions else None,
            clarification_needed=True,
            confidence_level=confidence_level,
            confidence_reason=confidence_reason,
        )

    rag = await retrieve_scoped_context(
        req.message,
        nlp,
        history,
        intent,
        conversation_state=updated_state,
        conversation_summary=conversation_summary,
        request_plan=clarification_decision['request_plan'],
    )
    good_chunks = rag["chunks"]
    kb_resolved = rag["kb_resolved"]

    # ── Chat history ───────────────────────────────────────────────────────────
    history = await load_recent_chat_history(session_id, current_user["id"])
    contextual_follow_up = is_contextual_follow_up(req.message, nlp, history)

    # ── Generate response (pass only good chunks to LLM) ──────────────────────
    confidence_guard = low_confidence_result(rag, language)
    source_listing = None if confidence_guard else build_source_listing_answer(
        req.message, language, rag["query_profile"], good_chunks
    )
    if confidence_guard:
        result = {
            "answer": confidence_guard["answer"],
            "sources": confidence_guard["sources"],
        }
        source_status = confidence_guard["source_status"]
        source_reason = confidence_guard["source_reason"]
        kb_resolved = confidence_guard["kb_resolved"]
        validation = confidence_guard["validation"]
    elif source_listing:
        result = {
            "answer": source_listing["answer"],
            "sources": source_listing["sources"],
        }
        source_status = source_listing["source_status"]
        source_reason = source_listing["source_reason"]
        kb_resolved = source_listing["kb_resolved"]
        validation = source_listing.get("validation", {})
    else:
        result, source_status, source_reason, kb_resolved, validation = await generate_grounded_answer(
            rag, nlp["normalized"], language, history, intent,
        )
    result = enforce_final_answer_coverage(result, rag["query_profile"], language)
    if result.get("final_status") == "answer_incomplete_after_compaction":
        source_status = SOURCE_STATUS_PARTIAL
        source_reason = "The answer was withheld because required requested components did not survive final validation."
        kb_resolved = False
    provider_unavailable = bool(result.get("provider_unavailable"))
    if provider_unavailable:
        answer_text = result["answer"]
    else:
        calculated_answer = result["answer"]
        warned_answer = apply_source_accuracy_notes(
            calculated_answer,
            rag.get("top_similarity_score"),
            rag["query_profile"],
            language,
        )
        answer_text = ensure_safety_disclaimer(warned_answer)
    if not (provider_unavailable or result.get("verification_failed") or confidence_guard or source_listing):
        from routers.services.answer_presentation import remove_inline_source_attributions
        answer_text = remove_inline_source_attributions(answer_text)
    confidence_level = frontend_confidence(source_status)
    confidence_reason = source_reason

    # ── Save messages ──────────────────────────────────────────────────────────
    await execute_insert(
        "INSERT INTO chat_messages (session_id,user_id,role,content,language,intent) VALUES (%s,%s,%s,%s,%s,%s)",
        (session_id, current_user["id"], "user", stored_user_message, language, intent),
    )
    msg_id, confidence_score, confidence_label = await save_assistant_message(
        session_id=session_id,
        user_id=current_user["id"],
        content=answer_text,
        language=language,
        sources=result["sources"],
        intent=intent,
        confidence_level=confidence_level,
        confidence_reason=confidence_reason,
        is_follow_up=contextual_follow_up,
        verified_partial=bool(result.get('partial_after_claim_check')),
    )

    # ── Log query with correct resolved status ─────────────────────────────────
    # resolved=True  → KB had a good answer (score >= threshold)
    # resolved=False → KB had no good answer → appears in Unresolved Queries tab
    elapsed_ms = int((time.time() - t0) * 1000)
    schedule_nonessential_write(
        """INSERT INTO query_logs
           (user_id, session_id, query_text, intent, response_time_ms, resolved)
           VALUES (%s,%s,%s,%s,%s,%s)""",
        (current_user["id"], session_id, stored_user_message, intent,
         elapsed_ms, kb_resolved),        # <-- correct resolved flag
    )
    if should_enqueue_review(source_status):
        schedule_nonessential_review(
            msg_id,
            current_user["id"],
            "The retrieved sources were not strong enough for a direct answer.",
        )
    await save_session_memory(
        session_id,
        current_user["id"],
        conversation_summary,
        updated_state,
        title_hint,
    )

    response = ChatResponse(
        session_id=session_id,
        message_id=msg_id,
        answer=answer_text,
        intent=intent,
        language=language,
        sources=result["sources"],
        follow_up_questions=result.get('follow_up_questions', clarification_decision['questions']),
        kb_found=kb_resolved,
        source_match=source_status,
        topic=rag["query_profile"].get("topic"),
        confidence_level=confidence_level,
        confidence_reason=confidence_reason,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
    )
    log_final_chat_response(response, rag["query_profile"], validation)
    return response


def _sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _validated_stream_chunks(answer: str, size: int = 160):
    """Split only the final validated answer, keeping one UI message card."""

    text = str(answer or "")
    cursor = 0
    while cursor < len(text):
        end = min(len(text), cursor + size)
        if end < len(text):
            boundary = max(text.rfind("\n", cursor, end), text.rfind(" ", cursor, end))
            if boundary > cursor + size // 2:
                end = boundary + 1
        yield text[cursor:end]
        cursor = end


@router.post("/message/stream")
async def stream_message(req: ChatRequest, current_user=Depends(get_current_user)):
    """SSE transport that never exposes an unvalidated legal draft.

    The assistant card/status is visible immediately. Answer deltas begin only
    after the normal endpoint has completed all citation and legal validation,
    so streaming cannot permanently surface a fabricated citation.
    """

    async def event_stream():
        timing, timing_token = begin_request()
        try:
            start_message = (
                "Preparing translation…"
                if detect_conversation_action(req.message) == "translation_request"
                else "Checking legal sources…"
            )
            yield _sse_event({"type": "start", "message": start_message})
            response = await send_message(req, current_user)
            first_delta_at: float | None = None
            for chunk in _validated_stream_chunks(response.answer):
                if first_delta_at is None:
                    first_delta_at = time.perf_counter()
                    set_timing_metadata(
                        time_to_first_validated_token_ms=round(
                            (first_delta_at - timing.started) * 1000,
                            2,
                        )
                    )
                yield _sse_event({"type": "delta", "text": chunk})
                await asyncio.sleep(0)
            payload = response.model_dump(mode="json")
            yield _sse_event({"type": "done", **payload})
        except Exception as exc:
            logger.exception("Streaming chat request failed")
            yield _sse_event({"type": "error", "message": str(exc)})
        finally:
            end_request(timing, timing_token)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/sessions")
async def get_sessions(current_user=Depends(get_current_user)):
    rows = await execute_query(
        "SELECT s.id,s.title,s.category,s.conversation_state,s.created_at,s.updated_at, "
        "(SELECT m.content FROM chat_messages m WHERE m.session_id=s.id "
        "AND m.user_id=s.user_id AND m.role='user' ORDER BY m.created_at,m.id LIMIT 1) AS opening_question "
        "FROM chat_sessions s WHERE s.user_id=%s AND s.is_active=1 ORDER BY s.updated_at DESC, s.id DESC",
        (current_user["id"],),
    )
    # The title describes the opening question, not the last topic discussed.
    # Recompute for old rows too: a nonempty stored category can be wrong.
    # Do not mutate storage merely by listing the sidebar.
    for row in rows:
        opening = str(row.pop("opening_question", "") or "").strip()
        if opening:
            row["category"] = await asyncio.to_thread(opening_question_category, opening)
        elif not row.get("category"):
            state = parse_state(row.get("conversation_state"))
            row["category"] = get_sidebar_category(
                topic=state.get("last_topic"),
                sub_intent=state.get("last_sub_intent"),
                intent=state.get("last_intent"),
            )
        row.pop("conversation_state", None)
    return rows


def opening_question_category(message: str) -> str:
    # Reuse legal classification, not truncated titles or assistant citations.
    if re.search(r'\b(?:DOLE|labor|department)\s+advisory\b', message, re.I):
        return 'Labor Advisories'
    if re.search(r'\bdepartment\s+order\b', message, re.I):
        return 'DOLE Resources'
    if re.search(r'\b(kasambahay|domestic\s+workers?)\b', message, re.I):
        return 'Domestic Workers'
    nlp = preprocess_query(message)
    profile = classify_legal_query(message, nlp)
    return get_sidebar_category(topic=profile.get('topic'), sub_intent=profile.get('sub_intent'),
                                message=message, intent=nlp.get('intent'))


@router.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str, current_user=Depends(get_current_user)):
    s = await execute_one(
        "SELECT id FROM chat_sessions WHERE id=%s AND user_id=%s AND is_active=1",
        (session_id, current_user["id"]),
    )
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    rows = await execute_query(
        "SELECT id,role,content,language,sources,intent,confidence_level,confidence_reason,"
        "confidence_score,confidence_label,created_at "
        "FROM chat_messages WHERE session_id=%s AND user_id=%s ORDER BY created_at ASC, id ASC",
        (session_id, current_user["id"]),
    )
    for row in rows:
        if row.get("intent") != "needs_clarification":
            continue
        # Older messages predate the explicit clarification fields. Recover
        # the one question from the saved answer so the improved UI still
        # works after a conversation is reopened.
        compact_answer = re.sub(r"\s+", " ", str(row.get("content") or "")).strip()
        match = re.search(
            r"(?:Please tell me:|Puwede mo bang sabihin:|Puwede mo bala isugid:)\s*(.*?\?)",
            compact_answer,
            re.IGNORECASE,
        )
        question = match.group(1).strip() if match else ""
        row["display_mode"] = "clarification"
        row["clarification_needed"] = True
        row["clarification_question"] = question or None
        row["follow_up_questions"] = [question] if question else []
    return rows


@router.patch("/sessions/{session_id}")
async def update_session_title(
    session_id: str,
    req: SessionUpdateRequest,
    current_user=Depends(get_current_user),
):
    title = (req.title or "").strip()[:120]
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")
    session = await execute_one(
        "SELECT id FROM chat_sessions WHERE id=%s AND user_id=%s AND is_active=1",
        (session_id, current_user["id"]),
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await execute_insert(
        "UPDATE chat_sessions SET title=%s WHERE id=%s AND user_id=%s",
        (redact_sensitive_info(title), session_id, current_user["id"]),
    )
    return {"message": "Renamed", "title": redact_sensitive_info(title)}


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest, current_user=Depends(get_current_user)):
    rating = (req.rating or "").strip().lower()
    if rating not in {"yes", "no"}:
        raise HTTPException(status_code=400, detail="Rating must be yes or no")
    message = await execute_one(
        """
        SELECT id FROM chat_messages
        WHERE id=%s AND user_id=%s AND role='assistant'
        """,
        (req.message_id, current_user["id"]),
    )
    if not message:
        raise HTTPException(status_code=404, detail="Assistant message not found")
    await execute_insert(
        """
        INSERT INTO chat_feedback (message_id,user_id,rating,comment)
        VALUES (%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            rating=VALUES(rating),
            comment=VALUES(comment),
            created_at=CURRENT_TIMESTAMP
        """,
        (req.message_id, current_user["id"], rating, redact_sensitive_info((req.comment or "").strip())[:1000]),
    )
    return {"message": "Feedback saved", "rating": rating}


@router.delete("/sessions/all")
async def delete_all_sessions(current_user=Depends(get_current_user)):
    pool = await get_pool()
    async with pool.acquire() as connection:
        await connection.begin()
        try:
            async with connection.cursor() as cursor:
                await cursor.execute("SELECT id FROM chat_sessions WHERE user_id=%s FOR UPDATE", (current_user['id'],))
                await cursor.fetchall()
                # Explicit cleanup also covers older installations and logs
                # containing question copies without a session foreign key.
                for table in ('query_logs', 'chat_messages', 'chat_sessions'):
                    await cursor.execute(f"DELETE FROM {table} WHERE user_id=%s", (current_user['id'],))
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
    return {"message": "All sessions deleted"}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, current_user=Depends(get_current_user)):
    pool = await get_pool()
    async with pool.acquire() as connection:
        await connection.begin()
        try:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT id FROM chat_sessions WHERE id=%s AND user_id=%s FOR UPDATE",
                    (session_id, current_user["id"]),
                )
                if not await cursor.fetchone():
                    raise HTTPException(404, "Session not found")
                # Logs contain copies of the question and have no session FK.
                await cursor.execute(
                    "DELETE FROM query_logs WHERE session_id=%s AND user_id=%s",
                    (session_id, current_user["id"]),
                )
                await cursor.execute(
                    "DELETE FROM chat_messages WHERE session_id=%s AND user_id=%s",
                    (session_id, current_user["id"]),
                )
                # Message feedback/reviews use ON DELETE CASCADE.
                await cursor.execute(
                    "DELETE FROM chat_sessions WHERE id=%s AND user_id=%s",
                    (session_id, current_user["id"]),
                )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
    return {"message": "Deleted"}
