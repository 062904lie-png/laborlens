"""Legacy Groq transport; shared prompts and sources live in openai_service."""

import logging
import hashlib
import json
import re
import time
from routers.services.document_evidence import is_document_driven
from typing import Any, Dict, List

import httpx

from config import settings
from routers.services.database import execute_query
from routers.services.document_text_cleaner import strip_document_footnotes
from routers.services.hiligaynon_style import hiligaynon_guidance
from routers.services.kb_categories import normalize_kb_category
from routers.services.rag_policy import enrich_chunk_metadata, legal_scope_prompt
from routers.services.provider_runtime import (
    answer_ends_at_sentence_boundary,
    compress_answer_to_budget,
    is_usable_partial_answer,
    merge_continuation,
    output_token_limit,
)
from routers.services.performance import current_timing, set_timing_metadata
from routers.services.request_budget import compact_messages, estimate_request_tokens

logger = logging.getLogger(__name__)

from .openai_service import (
    ACCURACY_GUARDRAILS_PROMPT,
    BASE_PROMPTS,
    CONCISE_ANSWER_PROMPT,
    CONTEXT_VISIBILITY_PROMPT,
    CONVERSATION_QUALITY_PROMPT,
    DOCUMENT_FOUND_TEMPLATE,
    DOCUMENT_GROUNDING_CORE,
    DOCUMENT_SCOPE_GUIDANCE,
    KB_FOUND_TEMPLATE,
    KB_NOT_FOUND_TEMPLATE,
    LANGUAGE_PROFILES,
    LIGHT_PRESENTATION_GUIDE,
    MAX_PROMPT_HISTORY_CHARS,
    MAX_PROMPT_HISTORY_MESSAGES,
    MAX_RESPONSE_TOKENS,
    ORIGINAL_WORDING_PROMPT,
    PLAIN_LANGUAGE_PROMPT,
    SITUATION_APPLICATION_PROMPT,
    STRUCTURAL_CLASSIFICATION_RE,
    UNAVAILABLE_ANSWER_MARKERS,
    answer_claims_unavailable,
    answer_has_scope_violation,
    answer_looks_truncated,
    build_busy_answer,
    build_conversation_messages,
    build_messages,
    build_scope_safe_fallback,
    build_sources_payload,
    compact_history_content,
    evidence_profile,
    extract_retry_after_seconds,
    format_compact_source_label,
    format_source_label,
    get_hiligaynon_glossary_prompt,
    get_language_guidance,
    get_language_profile,
    language_name,
    prompt_evidence_chunks,
    strip_response_code_fences,
    verified_structural_metadata,
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_TEMPERATURE = 0.7

DEFAULT_TOP_P = 0.9

MIN_RESPONSE_TOKENS = 400

class GroqServiceError(Exception):
    """Raised when Groq returns a non-success response."""

class GroqRateLimitError(GroqServiceError):
    def __init__(self, message: str, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds

def get_sampling_parameters() -> Dict[str, float]:
    """Return non-deterministic sampling values for every Groq completion."""
    try:
        temperature = float(getattr(settings, "GROQ_TEMPERATURE", DEFAULT_TEMPERATURE))
    except (TypeError, ValueError):
        temperature = DEFAULT_TEMPERATURE
    if temperature <= 0:
        temperature = DEFAULT_TEMPERATURE

    try:
        top_p = float(getattr(settings, "GROQ_TOP_P", DEFAULT_TOP_P))
    except (TypeError, ValueError):
        top_p = DEFAULT_TOP_P
    if not 0 < top_p <= 1:
        top_p = DEFAULT_TOP_P

    return {"temperature": temperature, "top_p": top_p}

def get_reasoning_parameters(model_name: str | None = None) -> Dict[str, str]:
    """Configure GPT-OSS reasoning without sending unsupported model options."""

    model = str(model_name or getattr(settings, "GROQ_MODEL", "") or "").lower()
    if "gpt-oss" not in model:
        return {}
    effort = str(getattr(settings, "GROQ_REASONING_EFFORT", "low") or "low").lower()
    if effort not in {"low", "medium", "high"}:
        effort = "low"
    return {"reasoning_effort": effort}

def get_token_limit_parameter(model_name: str, token_limit: int) -> Dict[str, int]:
    """Use Groq's GPT-OSS completion-limit field; retain the legacy field elsewhere."""

    parameter = "max_completion_tokens" if "gpt-oss" in str(model_name or "").lower() else "max_tokens"
    return {parameter: int(token_limit)}

async def generate_response(
    query: str,
    context_chunks: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]],
    language: str = "en",
    intent: str = "general_inquiry",
    query_profile: Dict[str, Any] | None = None,
    source_status: str = "insufficient",
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """Call Groq API and return a KB-grounded response."""
    from .model_pools import selected_model
    selected = selected_model.get()
    model_name = selected.split('/', 1)[1] if selected and selected.startswith('groq/') else settings.GROQ_MODEL

    context_chunks = prompt_evidence_chunks(context_chunks, query_profile)
    messages = build_messages(
        query,
        context_chunks,
        chat_history,
        language,
        query_profile=query_profile,
        source_status=source_status,
    )
    glossary_prompt = await get_hiligaynon_glossary_prompt(language)
    if glossary_prompt:
        messages.insert(1, {"role": "system", "content": glossary_prompt})
    response_token_limit = output_token_limit(query, query_profile)
    request_budget = max(1500, int(settings.GROQ_REQUEST_TOKEN_BUDGET))
    # Room for a source/scope correction and a complete-answer retry. Output
    # allowance is unchanged; never append a rejected draft without re-budgeting.
    prompt_budget = request_budget - 512
    # Rebuilding already regenerates scope, coverage and style instructions.
    # Only the asynchronously loaded glossary is external to build_messages.
    supplemental_system_messages = [{"role": "system", "content": glossary_prompt}] if glossary_prompt else []
    context_chunks = list(context_chunks)
    messages = compact_messages(messages, response_token_limit, prompt_budget, language_name(language))
    while (not is_document_driven(query_profile)
           and estimate_request_tokens(messages, response_token_limit) > prompt_budget and len(context_chunks) > 1):
        # Keep whole, highest-ranked passages. Never cut an exception halfway.
        context_chunks.pop()
        messages = build_messages(query, context_chunks, [], language,
                                  query_profile=query_profile, source_status=source_status)
        messages[1:1] = supplemental_system_messages
        messages = compact_messages(messages, response_token_limit, prompt_budget, language_name(language))
    evidence_index = max((i for i, m in enumerate(messages) if m['role'] == 'user'), default=-1)
    set_timing_metadata(groq_estimated_request_tokens=estimate_request_tokens(messages, response_token_limit),
                        groq_context_chunks_sent=len(context_chunks))
    continuation_max_tokens = 500 if (
        (query_profile or {}).get("task") in {"comparison_question", "calculation_request", "jurisprudence_request"}
        or (query_profile or {}).get("sub_intent") == "termination_comparison"
    ) else 250

    async def run_completion(
        message_list: List[Dict[str, str]],
        max_tokens: int = response_token_limit,
    ) -> tuple[str, str]:
        message_list = compact_messages(
            message_list, max_tokens, request_budget, language_name(language),
            evidence_index=evidence_index,
        )
        if estimate_request_tokens(message_list, max_tokens) > request_budget:
            raise GroqServiceError('Request exceeds configured local token budget; provider call skipped.')
        from .provider_runtime import completion_token_limit
        # Preserve evidence while spending unused request capacity on reasoning.
        available_output = request_budget - estimate_request_tokens(message_list, 0)
        completion_limit = min(completion_token_limit(model_name, max_tokens), available_output)
        set_timing_metadata(groq_completion_token_limit=completion_limit)
        payload = {
            "model": model_name,
            "messages": message_list,
            **get_token_limit_parameter(model_name, completion_limit),
            **get_sampling_parameters(),
            **get_reasoning_parameters(model_name),
            "stream": bool(settings.LABORLENS_STREAMING_ENABLED),
        }
        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        from .pipeline_request import call_allowed, record_llm_call
        if not call_allowed('generation'):
            from .model_pools import LocalModelLimitError
            raise LocalModelLimitError('simple_request_call_budget')
        if selected:
            from .model_pools import charge_request
            charge_request('groq/' + model_name,
                           estimate_request_tokens(message_list, completion_limit))
        record_llm_call('generation')
        async with httpx.AsyncClient(
            timeout=max(1.0, min(float(settings.LABORLENS_GROQ_TIMEOUT_SECONDS), 30.0))
        ) as client:
            if settings.LABORLENS_STREAMING_ENABLED and hasattr(client, "stream"):
                provider_started = time.perf_counter()
                chunks: List[str] = []
                finish_reason = "stop"
                first_token_seen = False
                reasoning_chars = 0
                usage: Dict[str, Any] = {}
                async with client.stream("POST", GROQ_URL, headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", errors="replace")
                        if resp.status_code == 429 or (resp.status_code == 413 and 'rate_limit_exceeded' in body):
                            retry_after = resp.headers.get("retry-after")
                            retry_after_seconds = None
                            if retry_after:
                                try:
                                    retry_after_seconds = float(retry_after)
                                except ValueError:
                                    retry_after_seconds = None
                            raise GroqRateLimitError(
                                f"Groq rate limit reached: {body}",
                                retry_after_seconds=retry_after_seconds or extract_retry_after_seconds(body) or 60.0,
                            )
                        raise GroqServiceError(f"Groq API error {resp.status_code}: {body}")
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        event = json.loads(data)
                        if isinstance(event.get("usage"), dict):
                            usage = event["usage"]
                        choice = (event.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        reasoning_chars += len(str(delta.get("reasoning") or delta.get("reasoning_content") or ""))
                        piece = str(delta.get("content") or "")
                        if piece:
                            chunks.append(piece)
                            if not first_token_seen:
                                first_token_seen = True
                                request_timing = current_timing()
                                elapsed_ms = (
                                    (time.perf_counter() - request_timing.started) * 1000
                                    if request_timing
                                    else (time.perf_counter() - provider_started) * 1000
                                )
                                if not request_timing or "time_to_first_token_ms" not in request_timing.metadata:
                                    set_timing_metadata(time_to_first_token_ms=round(elapsed_ms, 2))
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
                answer = "".join(chunks).strip()
                if not answer:
                    safe_usage = {
                        key: usage.get(key)
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        if usage.get(key) is not None
                    }
                    logger.warning(
                        "Groq returned empty visible content | model=%s finish_reason=%s "
                        "reasoning_chars=%d usage=%s",
                        model_name,
                        finish_reason,
                        reasoning_chars,
                        safe_usage,
                    )
                    raise GroqServiceError(
                        f"Groq returned an empty response (finish_reason={finish_reason})."
                    )
                return answer, finish_reason

            payload["stream"] = False
            resp = await client.post(GROQ_URL, headers=headers, json=payload)
            if resp.status_code != 200:
                if resp.status_code == 429 or (resp.status_code == 413 and 'rate_limit_exceeded' in resp.text):
                    logger.warning("Groq rate limit %s: %s", resp.status_code, resp.text)
                    retry_after = resp.headers.get("retry-after")
                    retry_after_seconds = None
                    if retry_after:
                        try:
                            retry_after_seconds = float(retry_after)
                        except ValueError:
                            retry_after_seconds = None
                    retry_after_seconds = retry_after_seconds or extract_retry_after_seconds(resp.text)
                    raise GroqRateLimitError(
                        f"Groq rate limit reached: {resp.text}",
                        retry_after_seconds=retry_after_seconds or 60.0,
                    )
                logger.error("Groq API error %s: %s", resp.status_code, resp.text)
                raise GroqServiceError(f"Groq API error {resp.status_code}: {resp.text}")
            data = resp.json()
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = str(message.get("content") or "").strip()
            if not content:
                usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                safe_usage = {
                    key: usage.get(key)
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if usage.get(key) is not None
                }
                finish_reason = str(choice.get("finish_reason") or "unknown")
                reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
                logger.warning(
                    "Groq returned empty visible content | model=%s finish_reason=%s "
                    "reasoning_chars=%d usage=%s",
                    model_name,
                    finish_reason,
                    len(str(reasoning)),
                    safe_usage,
                )
                raise GroqServiceError(
                    f"Groq returned an empty response (finish_reason={finish_reason})."
                )
            return content, str(choice.get("finish_reason") or "stop")

    async def run_complete_generation(message_list: List[Dict[str, str]]) -> str:
        answer, finish_reason = await run_completion(message_list)
        if (query_profile or {}).get('simple_rag_mode'):
            answer = strip_response_code_fences(answer)
            if answer_looks_truncated(answer):
                raise GroqServiceError('Groq returned an incomplete answer; no repair requested.')
            return answer
        initial_answer = answer
        cut_off = finish_reason == "length" or answer_looks_truncated(answer)
        safe_partial = is_usable_partial_answer(answer) and not answer_looks_truncated(answer)
        # `length` means the model hit its allowance, not that the provider
        # failed. A complete-looking grounded draft should be compacted and
        # returned without spending a second request on continuation.
        if cut_off and safe_partial:
            answer, compacted = compress_answer_to_budget(
                answer, settings.LABORLENS_HARD_OUTPUT_LIMIT
            )
            logger.info(
                "Groq draft retained | finish_reason=%s draft_chars=%d safe_partial=true compaction_triggered=%s continuation_requested=false",
                finish_reason,
                len(initial_answer or ""),
                compacted,
            )
            return strip_response_code_fences(answer)
        if cut_off:
            from .provider_runtime import complete_answer_retry_messages

            logger.warning("Incomplete Groq draft; requesting a concise complete replacement")
            answer, finish_reason = await run_completion(
                complete_answer_retry_messages(message_list),
                max_tokens=response_token_limit,
            )
        answer = strip_response_code_fences(answer)
        if answer_looks_truncated(answer):
            raise GroqServiceError("Groq returned an incomplete answer after repair.")
        return answer

    try:
        answer = await run_complete_generation(messages)
    except GroqRateLimitError as exc:
        logger.warning("Groq rate limit during response generation: %s", exc)
        if raise_on_error:
            raise
        return {
            "answer": build_busy_answer(language, exc.retry_after_seconds),
            "sources": build_sources_payload(context_chunks),
            "model": model_name,
            "intent": intent,
            "language": language,
            "kb_used": bool(context_chunks),
        }
    except GroqServiceError as exc:
        logger.exception("Groq service error during response generation: %s", exc)
        if raise_on_error:
            raise
        return {
            "answer": build_busy_answer(language),
            "sources": build_sources_payload(context_chunks),
            "model": model_name,
            "intent": intent,
            "language": language,
            "kb_used": bool(context_chunks),
        }
    if (query_profile or {}).get('simple_rag_mode'):
        return dict(answer=answer, sources=build_sources_payload(context_chunks), model=model_name,
                    intent=intent, language=language, kb_used=bool(context_chunks))
    if context_chunks and not is_document_driven(query_profile) and answer_claims_unavailable(answer):
        repair_messages = messages + [
            {"role": "assistant", "content": answer},
            {
                "role": "user",
                "content": (
                    "The previous draft incorrectly treated the answer as unavailable.\n"
                    "Using only the retrieved excerpts already shown above, rewrite the answer directly.\n"
                    "If the user's wording is broader than the legal text, explain the closest supported rule from the excerpts.\n"
                    "Do not say the answer is unavailable when the excerpts already address the topic.\n"
                    "Keep the same relevant-section rules and omit empty or irrelevant sections."
                ),
            },
        ]
        try:
            answer = await run_complete_generation(repair_messages)
        except GroqRateLimitError as exc:
            logger.warning("Groq rate limit during unavailable-answer repair: %s", exc)
        except GroqServiceError as exc:
            logger.warning("Groq service error during unavailable-answer repair: %s", exc)
    if context_chunks and answer_has_scope_violation(answer, query_profile):
        repair_messages = messages + [
            {"role": "assistant", "content": answer},
            {
                "role": "user",
                "content": (
                    "The previous draft failed source-scope verification. Rewrite it before returning it.\n"
                    "For a general overtime question, do not mention or apply any 40-hour weekly comparison. "
                    "That rule belongs only to covered hospital/clinic personnel and was not selected as the main scope.\n"
                    "Use only the general private-employment overtime sources retrieved above. "
                    "Keep the answer concise, omit inline citations and source names, and omit unsupported special-sector rules."
                ),
            },
        ]
        try:
            answer = await run_complete_generation(repair_messages)
        except GroqRateLimitError as exc:
            logger.warning("Groq rate limit during source-scope repair: %s", exc)
            answer = build_scope_safe_fallback(language, query_profile)
        except GroqServiceError as exc:
            logger.warning("Groq service error during source-scope repair: %s", exc)
            answer = build_scope_safe_fallback(language, query_profile)

    sources = build_sources_payload(context_chunks)

    logger.info(
        "Response generated | intent=%s | language=%s | kb_chunks_used=%s | kb_found=%s",
        intent,
        language,
        len(context_chunks),
        bool(context_chunks),
    )

    return {
        "answer": answer,
        "sources": sources,
        "model": model_name,
        "intent": intent,
        "language": language,
        "kb_used": bool(context_chunks),
    }

async def translate_answer_preserving_format(
    answer_text: str,
    target_language: str,
    chat_history: List[Dict[str, str]],
    source_language: str = "",
) -> str:
    target_name = language_name(target_language)
    source_name = language_name(source_language) if source_language else "the original language"
    profile = get_language_profile(target_language)
    translation_token_limit = max(
        240,
        min(
            int(settings.LABORLENS_MAX_OUTPUT_TOKENS_COMPLEX),
            MAX_RESPONSE_TOKENS,
            int(len(answer_text.split()) * 2.2) + 80,
        ),
    )
    system_prompt = (
        "You are a careful translator for LaborLens.\n"
        "Translate the assistant answer exactly into the target language.\n"
        "Preserve the same order, section layout, bullets, numbering, markdown emphasis, "
        "line breaks, and overall structure as closely as possible.\n"
        "Do not summarize, shorten, expand, explain, or reorder anything.\n"
        "Complete every section fully. Do not leave any heading, label, bullet, or sentence unfinished.\n"
        "Keep article numbers, amounts and dates accurate, but omit citations, source labels and document titles "
        "from the translated reply because source records are displayed separately. Do not add source URLs.\n"
        "The complete original answer is between the markers in the final user message. "
        "Never ask the user to provide the original answer, source text, or more context. "
        "Translate only the marked answer.\n"
        f'When the answer contains the section label "Definition:", use the exact translated label "{profile["definition_label"]}:".\n'
        'When the answer contains "Legal Basis:", translate it naturally for the target language.\n'
        f'Use "{profile["direct_label"]}:", "{profile["explanation_label"]}:", '
        f'"{profile["requirements_label"]}:", "{profile["computation_label"]}:", '
        f'"{profile["legal_basis_label"]}:", "{profile["next_steps_label"]}:", and '
        f'"{profile["reminder_label"]}:" as the translated section labels when matching those sections.\n'
        "Do not add a source list; the interface displays retrieved sources separately.\n"
        "Return only the translated answer text."
    )
    messages = [{"role": "system", "content": system_prompt}]
    glossary_prompt = await get_hiligaynon_glossary_prompt(target_language)
    if glossary_prompt:
        messages.append({"role": "system", "content": glossary_prompt})
    for msg in (chat_history or [])[-MAX_PROMPT_HISTORY_MESSAGES:]:
        role = msg.get("role", "")
        content = compact_history_content(msg.get("content", ""))
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    user_prompt = (
        f"Translate this answer from {source_name} into {target_name}.\n\n"
        "Answer to translate:\n"
        "<<<\n"
        f"{answer_text}\n"
        ">>>"
    )
    request_messages = messages + [{"role": "user", "content": user_prompt}]
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(
        timeout=max(1.0, min(float(settings.LABORLENS_GROQ_TIMEOUT_SECONDS), 30.0))
    ) as client:
        async def request_translation(
            message_list: List[Dict[str, str]],
            max_tokens: int,
        ) -> tuple[str, str]:
            payload = {
                "model": settings.GROQ_MODEL,
                "messages": message_list,
                **get_token_limit_parameter(settings.GROQ_MODEL, max_tokens),
                **get_sampling_parameters(),
                "stream": False,
            }
            resp = await client.post(GROQ_URL, headers=headers, json=payload)
            if resp.status_code != 200:
                logger.error("Groq translation error %s: %s", resp.status_code, resp.text)
                raise Exception(f"Groq translation error {resp.status_code}: {resp.text}")
            choice = resp.json()["choices"][0]
            return choice["message"]["content"], str(choice.get("finish_reason") or "stop")

        translated, finish_reason = await request_translation(
            request_messages,
            translation_token_limit,
        )
        if finish_reason == "length" or answer_looks_truncated(translated):
            logger.warning(
                "Incomplete Groq translation detected | finish_reason=%s; requesting continuation",
                finish_reason,
            )
            tail = (translated or "")[-3500:]
            continuation_messages = [
                {
                    "role": "system",
                    "content": "Finish only the incomplete translation. Add no facts; omit citations and source labels, and preserve numbers.",
                },
                {
                    "role": "user",
                    "content": f"Translation tail:\n<<<\n{tail}\n>>>\nReturn only the missing ending.",
                },
            ]
            continuation, continuation_finish_reason = await request_translation(
                continuation_messages,
                max(200, min(int(settings.LABORLENS_CONTINUATION_TOKENS), 250)),
            )
            translated = merge_continuation(translated, continuation)
            if continuation_finish_reason == "length" or answer_looks_truncated(continuation):
                logger.error(
                    "Groq translation continuation may still be incomplete | finish_reason=%s",
                    continuation_finish_reason,
                )
        return strip_response_code_fences(translated)
