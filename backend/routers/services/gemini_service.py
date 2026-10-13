"""Google Gemini LLM service with an ordered, multi-model fallback cascade.

The retrieval, grounding, citation, language, and conversation guardrails are
shared with the established LaborLens prompt builder. Groq is deliberately not
called here; the chat router owns the final cross-provider fallback.
"""

from __future__ import annotations

import asyncio
from routers.services.document_evidence import is_document_driven
from dataclasses import dataclass, field
import logging
import time
from datetime import date
from typing import Any, Dict, List, Sequence

import google.generativeai as genai

from config import settings
from routers.services.openai_service import (
    MAX_PROMPT_HISTORY_MESSAGES,
    answer_claims_unavailable,
    answer_has_scope_violation,
    answer_looks_truncated,
    build_messages,
    build_sources_payload,
    prompt_evidence_chunks,
    compact_history_content,
    get_hiligaynon_glossary_prompt,
    get_language_profile,
    language_name,
    strip_response_code_fences,
)
from routers.services.provider_runtime import (
    CircuitOpenError,
    acquire_provider,
    abandon_provider,
    classify_provider_error,
    health_for,
    is_usable_partial_answer,
    merge_continuation,
    output_token_limit,
    completion_token_limit,
    ranked_health_keys,
    record_failure,
    record_success,
    retry_after_seconds,
)
from routers.services.performance import current_timing, set_timing_metadata

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]
GEMINI_MODELS = DEFAULT_GEMINI_MODELS

MAX_OUTPUT_TOKENS = max(400, min(int(settings.LABORLENS_HARD_OUTPUT_LIMIT), 2000))
TEMPERATURE = 0.7
TOP_P = 0.9
REQUEST_TIMEOUT_SECONDS = max(1.0, min(float(settings.LABORLENS_GEMINI_TIMEOUT_SECONDS), 30.0))


def configured_gemini_models() -> List[str]:
    """Return unique configured IDs, preserving the configured preference order."""

    from .model_pools import selected_model
    selected = selected_model.get()
    if selected and selected.startswith('gemini/'):
        return [selected.split('/', 1)[1]]
    raw = str(settings.LABORLENS_GEMINI_MODELS or "")
    models: List[str] = []
    for value in raw.split(","):
        model = value.strip()
        if model and model not in models:
            models.append(model)
    return models or list(DEFAULT_GEMINI_MODELS)


def gemini_health_key(model_name: str) -> str:
    return f"gemini:{model_name}"


@dataclass
class GeminiRequestBudget:
    """One wall-clock and distinct-model budget shared by all Gemini work."""

    started: float = field(default_factory=time.monotonic)
    attempted_models: set[str] = field(default_factory=set)
    preferred_model: str | None = None
    attempt_number: int = 0

    @property
    def deadline(self) -> float:
        total = max(1.0, min(float(settings.LABORLENS_GEMINI_TOTAL_BUDGET_SECONDS), 30.0))
        return self.started + total

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def may_try(self, model_name: str) -> bool:
        maximum = max(
            1,
            min(int(settings.LABORLENS_GEMINI_MAX_MODELS_PER_REQUEST), len(configured_gemini_models())),
        )
        return model_name in self.attempted_models or len(self.attempted_models) < maximum

GENERAL_RETRIEVED_CONTEXT_INSTRUCTION = """\
Rules for describing source availability:
- You only see the excerpts retrieved for the current question. Never claim,
  list, or guess what documents, laws, topics, or regions exist in the complete
  knowledge base or database.
- If an answer is not supported by the current excerpts, say that the specific
  rule is not available in the current context. Do not say that it is absent
  from the database.
- When a retrieved official source directly addresses the user's question,
  extract and explain that rule instead of returning an insufficient-source
  response. OCR text, headings, numbered clauses, and table-like fragments are
  valid evidence when their labels and values are clearly associated.
- Use only excerpts that actually match the question. Do not cite an unrelated
  Department Order merely because it is newer or shares generic labor words.
"""

REGIONAL_WAGE_SYSTEM_INSTRUCTION = """\
When answering minimum wage questions:
1. If a DOLE Regional Wage Order or RTWPB summary is present in the context,
   give the exact daily or monthly rate. Do not add a source citation or Wage
   Order identifier unless the user asks for it; the interface displays
   retrieved source records separately.
2. Do not claim the rate is missing when a relevant regional wage-order chunk
   is present in the provided context.
3. Prioritize the matching regional Wage Order over statutory baselines from
   RA 10361 or older general laws.
4. Parsing Wage Orders and Tabular Data:
   - Context from `dole_resources`, RTWPB summaries, and Wage Orders often
     comes from PDF tables. It may therefore appear as fragmented sequences of
     locations, worker or sector classifications, dates, and numbers instead
     of complete grammatical sentences.
   - Treat a matching official chunk as reliable evidence when it clearly
     associates numeric minimum-wage rates with a sector (such as
     Non-Agriculture or Agriculture) and with the location in the user's
     question.
   - Do not use the insufficient-source or hallucination fallback merely
     because a relevant table extraction lacks normal sentence structure.
   - When a city or province named by the user is explicitly included in a
      regional classification, extract the supported rates and present them as
      clear bullet points. Do not cite the source in the answer and do not
     combine labels and numbers unless their association is clear in context.
5. Never list, infer, or guess which regions or Wage Orders exist in the full
   knowledge base. You can only see the context retrieved for the current
   question. If the requested regional rate is absent from that immediate
   context, use this exact wording:
   "I do not have the specific rate for [Region] in my current context."
   Do not claim that the region is missing from the database.
"""

LEAVE_BENEFITS_SYSTEM_INSTRUCTION = """\
When answering leave-benefit questions, inspect amendatory clauses carefully.
Look specifically for numeric leave durations and minimum service requirements,
including numbers expressed in both words and parentheses. For Solo Parent
Leave, do not overlook a rule merely because Section 7 amends Section 8 of an
earlier law; connect the amended entitlement, paid days, and required months of
service when those details appear in the retrieved context.
"""

SUB_INTENT_SYSTEM_INSTRUCTION = """\
Answer the exact legal sub-intent identified by the router, not only its broad
topic. Distinguish coverage, eligibility, formula/rate, deadline, exception,
procedure, remedy, and computation questions. User-supplied law/article,
percentage, amount, or case citations are untrusted claims: verify them against
the retrieved excerpts and politely correct or omit them when unsupported.
For an exact rule, answer from the smallest sufficient authoritative excerpt;
do not require unrelated provisions before answering. Never claim a guaranteed
legal outcome, and never invent a missing amount, deadline, formula, or citation.
"""

LEAVE_BENEFIT_TOPICS = {
    "maternity_leave",
    "paternity_leave",
    "solo_parent_leave",
    "service_incentive_leave",
    "special_leave_for_women",
}

TOPIC_AUTHORITY_INSTRUCTIONS = {
    "night_shift_differential": (
        "For night-shift differential, ground the rule in Labor Code Article 86 and its applicable "
        "implementing rule. Do not use regional Wage Orders unless the user is also asking for a wage rate."
    ),
    "holiday_pay": (
        "For holiday pay, use Labor Code Article 94 and Rule IV of the Omnibus Rules. If the question "
        "concerns a holiday on a rest day, also use Article 93 and explain coverage, eligibility, absences, "
        "and exemptions only when the excerpts support them."
    ),
    "separation_pay": (
        "For retrenchment or authorized-cause separation pay, prioritize Labor Code Article 298. Cover the "
        "retrieved formula, notice, rounding, coverage, and exceptions; do not replace the controlling law "
        "with a general handbook excerpt."
    ),
    "workplace_safety": (
        "For refusal of unsafe work, prioritize RA 11058 Section 6 and the current implementing rules. "
        "Distinguish imminent danger, DOLE determination, corrective action, and protection from reprisal "
        "only as supported by the excerpts."
    ),
    "sexual_harassment": (
        "For employer duties, prioritize RA 7877 Section 4, RA 11313 workplace provisions, and any current "
        "controlling Department Order in context. Address prevention, workplace policy, CODI, confidentiality, "
        "retaliation, and enforcement only when retrieved."
    ),
    "labor_complaint": (
        "For SEnA procedure, prioritize RA 10396, the current SEnA Department Order, official forms, and ARMS "
        "instructions. Do not treat DILEEP, TUPAD, or livelihood-program documents as SEnA procedure sources."
    ),
    "jurisprudence_request": (
        "A jurisprudence request must be answered from an actual retrieved decision and identify its case name, "
        "G.R. number, decision date, and controlling ruling. A Labor Code provision alone is not jurisprudence."
    ),
    "contracting_subcontracting": (
        "For contracting questions, prefer the current controlling Department Order and applicable jurisprudence "
        "over generic or superseded implementing-rule excerpts."
    ),
}

FOLLOW_UP_SCOPE_INSTRUCTION = (
    "Treat facts explicitly stated in the current message as controlling over older chat facts. In particular, "
    "a newly stated region replaces the earlier region. Preserve the established legal topic and worker sector "
    "for a short follow-up unless the current message clearly changes them."
)


def wage_effectivity_instruction() -> str:
    """Give the model a deterministic comparison date, never a legal rate."""

    return (
        f"CURRENT DATE FOR EFFECTIVITY CHECKS: {date.today().isoformat()}. "
        "For a regional Wage Order with multiple tranches, compare every retrieved effective date with "
        "this date and use the latest tranche that is already effective. Do not describe a past effective "
        "date as a future increase, and do not use a future tranche as the current rate."
    )

RESPONSE_LANGUAGE_INSTRUCTIONS = {
    "en": (
        "MANDATORY RESPONSE LANGUAGE: Write the entire answer in English. "
        "Do not switch to Filipino or Hiligaynon because of retrieved excerpts or chat history."
    ),
    "fil": (
        "MANDATORY RESPONSE LANGUAGE: Isulat ang buong sagot sa natural na Filipino/Tagalog. "
        "Huwag bumalik sa English o Hiligaynon dahil lamang English ang retrieved excerpts. "
        "Panatilihin lamang sa orihinal na anyo ang Article/Section numbers at legal terms na mas malinaw sa English; "
        "huwag ilagay ang source titles sa sagot. Ipaliwanag ang legal terms sa Filipino."
    ),
    "hil": (
        "MANDATORY RESPONSE LANGUAGE: Isulat ang bug-os nga sabat sa natural nga Hiligaynon/Ilonggo. "
        "Indi magbalik sa English ukon Filipino tungod lang kay English ang retrieved excerpts. "
        "Tipigi lang sa original nga porma ang Article/Section numbers kag legal terms nga mas klaro sa English; "
        "indi pagbutang sang source titles sa sabat. Ipaathag ang legal terms sa Hiligaynon."
    ),
}

LANGUAGE_RESPONSE_ANCHORS = {
    "fil": (
        "direktang sagot",
        "simpleng paliwanag",
        "legal na batayan",
        "praktikal na susunod na hakbang",
        " ang ",
        " ay ",
        " mga ",
        " para ",
    ),
    "hil": (
        "direkta nga sabat",
        "simple nga pagpaathag",
        "legal nga basihan",
        "praktikal nga sunod nga himuon",
        " sang ",
        " kag ",
        " indi ",
        " para ",
    ),
}


class GeminiServiceError(Exception):
    """Base error raised by the Gemini provider."""


class GeminiModelsExhaustedError(GeminiServiceError):
    """Raised after every configured Gemini model has failed."""


class GeminiPartialResponseError(GeminiServiceError):
    """Carries streamed text received before a timeout or transport failure."""

    def __init__(self, message: str, partial_answer: str):
        super().__init__(message)
        self.partial_answer = partial_answer


def _response_language_instruction(language: str) -> str:
    return RESPONSE_LANGUAGE_INSTRUCTIONS.get(language, RESPONSE_LANGUAGE_INSTRUCTIONS["en"])


def _response_uses_expected_language(answer: str, language: str) -> bool:
    """Detect obvious Filipino/Hiligaynon drift before returning a Gemini answer."""

    if language == "en":
        return True
    padded = f" {(answer or '').strip().lower()} "
    anchors = LANGUAGE_RESPONSE_ANCHORS.get(language, ())
    return sum(1 for anchor in anchors if anchor in padded) >= 2


def _is_rate_limit_or_quota_error(exc: Exception) -> bool:
    """Recognize Gemini quota failures without depending on one SDK exception type."""

    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    message = str(exc).lower()
    return (
        status_code == 429
        or response_status == 429
        or "429" in message
        or "rate limit" in message
        or "ratelimit" in message
        or "quota" in message
        or "resource_exhausted" in message
    )


def _gemini_request_parts(
    messages: Sequence[Dict[str, str]],
) -> tuple[str, List[Dict[str, Any]]]:
    """Convert OpenAI-style prompt messages to Gemini roles and parts."""

    system_parts: List[str] = []
    contents: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"][0] += f"\n\n{content}"
        else:
            contents.append({"role": gemini_role, "parts": [content]})

    if not contents:
        contents.append({"role": "user", "parts": ["Please answer the request."]})
    return "\n\n".join(system_parts), contents


def _finish_reason(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "unknown"
    reason = getattr(candidates[0], "finish_reason", "unknown")
    name = getattr(reason, "name", None)
    return str(name or reason)


def _response_text(response: Any) -> str:
    try:
        text = str(response.text or "").strip()
    except Exception as exc:
        raise GeminiServiceError(f"Gemini returned no readable response: {exc}") from exc
    if not text:
        raise GeminiServiceError("Gemini returned an empty response.")
    return text


async def _call_model(
    model_name: str,
    messages: Sequence[Dict[str, str]],
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
) -> tuple[str, str]:
    from .model_pools import selected_model, charge_request
    from .pipeline_request import call_allowed, record_llm_call
    if not call_allowed('generation'):
        from .model_pools import LocalModelLimitError
        raise LocalModelLimitError('simple_request_call_budget')
    system_instruction, contents = _gemini_request_parts(messages)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction or None,
    )
    generation_config = {
        "max_output_tokens": completion_token_limit(model_name, min(max_output_tokens, MAX_OUTPUT_TOKENS)),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    if selected_model.get():
        from .request_budget import estimate_request_tokens
        # The cascade may call a different Gemini model than the selected one.
        # Account for the actual model and reasoning-inclusive output allowance.
        charge_request('gemini/' + model_name,
                       estimate_request_tokens(messages, generation_config['max_output_tokens']))
    record_llm_call('generation')
    if settings.LABORLENS_STREAMING_ENABLED:
        pieces: List[str] = []

        async def consume_stream() -> tuple[str, str]:
            provider_started = time.perf_counter()
            stream = await model.generate_content_async(
                contents,
                generation_config=generation_config,
                stream=True,
                request_options={"retry": None, "timeout": timeout_seconds},
            )
            last_chunk: Any = None
            first_token_seen = False
            async for chunk in stream:
                last_chunk = chunk
                try:
                    piece = str(chunk.text or "")
                except Exception:
                    piece = ""
                if not piece:
                    continue
                pieces.append(piece)
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
                    set_timing_metadata(
                        gemini_provider_ttft_ms=round(
                            (time.perf_counter() - provider_started) * 1000,
                            2,
                        )
                    )
            answer = "".join(pieces).strip()
            if not answer:
                raise GeminiServiceError("Gemini returned an empty response.")
            return answer, _finish_reason(last_chunk)

        try:
            return await asyncio.wait_for(consume_stream(), timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            partial = "".join(pieces).strip()
            if partial:
                raise GeminiPartialResponseError(
                    "Gemini timed out after returning a partial response.",
                    partial,
                ) from exc
            raise

    response = await asyncio.wait_for(
        model.generate_content_async(contents, generation_config=generation_config,
                                     request_options={"retry": None, "timeout": timeout_seconds}),
        timeout=timeout_seconds,
    )
    return _response_text(response), _finish_reason(response)


async def _generate_with_cascade(
    messages: Sequence[Dict[str, str]],
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    budget: GeminiRequestBudget | None = None,
) -> tuple[str, str, str]:
    """Try at most two healthy models inside one shared wall-clock budget."""
    from .model_pools import LocalModelLimitError

    if not settings.LABORLENS_GEMINI_ENABLED:
        raise GeminiModelsExhaustedError("Gemini is disabled by configuration.")
    if not settings.GEMINI_API_KEY:
        raise GeminiModelsExhaustedError("GEMINI_API_KEY is not configured.")

    genai.configure(api_key=settings.GEMINI_API_KEY)
    budget = budget or GeminiRequestBudget()
    failures: List[str] = []
    last_error: Exception | None = None
    configured = configured_gemini_models()
    keys = [gemini_health_key(model) for model in configured]
    ranked_keys = ranked_health_keys(keys)
    if budget.preferred_model:
        preferred_key = gemini_health_key(budget.preferred_model)
        if preferred_key in ranked_keys:
            ranked_keys.remove(preferred_key)
            ranked_keys.insert(0, preferred_key)

    for health_key in ranked_keys:
        model_name = health_key.split(":", 1)[1]
        if not budget.may_try(model_name):
            continue
        remaining = budget.remaining_seconds()
        if remaining <= 0.05:
            failures.append("total Gemini budget exhausted")
            break
        circuit_state_before = health_for(health_key).state
        try:
            await acquire_provider(health_key)
        except CircuitOpenError:
            logger.info("Skipping Gemini %s because circuit is OPEN", model_name)
            failures.append(f"{model_name}: circuit OPEN")
            continue

        budget.attempted_models.add(model_name)
        budget.attempt_number += 1
        attempt_number = budget.attempt_number
        started = time.perf_counter()
        timeout_seconds = min(
            max(1.0, min(float(settings.LABORLENS_GEMINI_TIMEOUT_SECONDS), 30.0)),
            remaining,
        )
        try:
            answer, finish_reason = await _call_model(
                model_name,
                messages,
                max_output_tokens=max_output_tokens,
                timeout_seconds=timeout_seconds,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            incomplete = answer_looks_truncated(answer)
            if incomplete:
                await record_failure(health_key, 'incomplete_response', latency_ms, counts=False)
            else:
                await record_success(health_key, latency_ms)
            budget.preferred_model = model_name
            timing = current_timing()
            provider_ttft_ms = timing.metadata.get("gemini_provider_ttft_ms") if timing else None
            set_timing_metadata(
                gemini_model=model_name,
                gemini_attempt_number=attempt_number,
                gemini_finish_reason=finish_reason,
            )
            logger.info(
                "Gemini model attempt | provider=gemini model=%s attempt_number=%d "
                "circuit_state_before=%s model_latency_ms=%.2f provider_ttft_ms=%s "
                "finish_reason=%s success=%s failure_reason=%s circuit_state_after=%s",
                model_name,
                attempt_number,
                circuit_state_before,
                latency_ms,
                provider_ttft_ms,
                finish_reason,
                not incomplete,
                'incomplete_response' if incomplete else 'none',
                health_for(health_key).state,
            )
            return answer, finish_reason, model_name
        except LocalModelLimitError as exc:
            await abandon_provider(health_key, unattempted=True)
            last_error = exc
            failures.append(f'{model_name}: local {exc.reason} budget')
            logger.info('Gemini generation skipped: model=%s local_reason=%s', model_name, exc.reason)
            continue
        except asyncio.CancelledError:
            await record_failure(
                health_key, 'cancelled', (time.perf_counter() - started) * 1000,
                counts=False,
            )
            raise
        except Exception as exc:
            last_error = exc
            latency_ms = (time.perf_counter() - started) * 1000
            reason, counts_for_circuit = classify_provider_error(exc)
            retry_after = retry_after_seconds(exc)
            await record_failure(
                health_key,
                reason,
                latency_ms,
                counts=counts_for_circuit,
                retry_after_seconds=retry_after,
            )
            partial = getattr(exc, "partial_answer", "")
            error_kind = "rate limit/quota" if _is_rate_limit_or_quota_error(exc) else reason
            failures.append(f"{model_name}: {error_kind} ({type(exc).__name__})")
            timing = current_timing()
            provider_ttft_ms = timing.metadata.get("gemini_provider_ttft_ms") if timing else None
            logger.warning(
                "Gemini model attempt | provider=gemini model=%s attempt_number=%d "
                "circuit_state_before=%s model_latency_ms=%.2f provider_ttft_ms=%s "
                "finish_reason=%s success=false failure_reason=%s circuit_state_after=%s",
                model_name,
                attempt_number,
                circuit_state_before,
                latency_ms,
                provider_ttft_ms,
                "TIMEOUT_PARTIAL" if partial else "unknown",
                reason,
                health_for(health_key).state,
            )
            if partial and _partial_answer_is_salvageable(partial):
                budget.preferred_model = model_name
                return partial, "TIMEOUT_PARTIAL", model_name

    summary = "; ".join(failures)
    if isinstance(last_error, LocalModelLimitError):
        raise last_error
    raise GeminiModelsExhaustedError(
        f"All configured Gemini models failed: {summary}"
    ) from last_error


def _was_cut_off(finish_reason: str, answer: str) -> bool:
    normalized = str(finish_reason or "").upper()
    return "MAX_TOKENS" in normalized or normalized == "2" or answer_looks_truncated(answer)


def _partial_answer_is_salvageable(answer: str) -> bool:
    """Keep a substantial grounded core rather than discarding it on a late failure."""

    return is_usable_partial_answer(answer)


def _compact_continuation_messages(answer: str) -> List[Dict[str, str]]:
    """Continue from the grounded draft tail without resending history/RAG context."""

    tail = (answer or "")[-3500:]
    return [
        {
            "role": "system",
            "content": (
                "Continue an existing LaborLens answer. Use only facts and rules already present in "
                "the supplied draft tail. Do not add source labels or new rules. Preserve the draft "
                "language and return only the missing ending. "
                "Keep short sentences and simple everyday wording."
            ),
        },
        {
            "role": "user",
            "content": (
                "Draft tail:\n<<<\n"
                f"{tail}\n"
                ">>>\nContinue from the last complete sentence or unfinished section only. "
                "Do not repeat or rewrite earlier content."
            ),
        },
    ]


async def _complete_generation(
    messages: List[Dict[str, str]],
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    budget: GeminiRequestBudget | None = None,
    continuation_max_tokens: int = 250,
    retry_incomplete: bool = True,
) -> tuple[str, str]:
    budget = budget or GeminiRequestBudget()
    answer, finish_reason, model_name = await _generate_with_cascade(
        messages,
        max_output_tokens=max_output_tokens,
        budget=budget,
    )
    if retry_incomplete and _was_cut_off(finish_reason, answer) and answer_looks_truncated(answer):
        from .provider_runtime import complete_answer_retry_messages

        logger.warning("Incomplete Gemini draft; requesting a concise complete replacement")
        answer, finish_reason, model_name = await _generate_with_cascade(
            complete_answer_retry_messages(messages),
            max_output_tokens=max_output_tokens,
            budget=budget,
        )
    answer = strip_response_code_fences(answer)
    if answer_looks_truncated(answer):
        raise GeminiServiceError("Gemini returned an incomplete answer after repair.")
    return answer, model_name


async def generate_response(
    query: str,
    context_chunks: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]],
    language: str = "en",
    intent: str = "general_inquiry",
    query_profile: Dict[str, Any] | None = None,
    source_status: str = "insufficient",
) -> Dict[str, Any]:
    """Generate a guarded RAG answer through the ordered Gemini model cascade."""

    conversation_mode = (query_profile or {}).get("task") == "conversation"
    context_chunks = prompt_evidence_chunks(context_chunks, query_profile)
    messages = build_messages(
        query,
        context_chunks,
        chat_history,
        language,
        query_profile=query_profile,
        source_status=source_status,
    )
    if conversation_mode:
        messages.insert(1, {"role": "system", "content": _response_language_instruction(language)})
        budget = GeminiRequestBudget()
        answer, model_name = await _complete_generation(
            messages,
            max_output_tokens=160,
            budget=budget,
            continuation_max_tokens=120,
        )
        if not _response_uses_expected_language(answer, language):
            language_repair_messages = messages + [
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": (
                        "Rewrite the reply entirely in the selected response language. "
                        "Keep it brief, natural, and non-legal. Do not add citations or legal claims."
                    ),
                },
            ]
            answer, model_name = await _complete_generation(
                language_repair_messages,
                max_output_tokens=160,
                budget=budget,
                continuation_max_tokens=120,
            )
        return {
            "answer": answer,
            "sources": [],
            "model": model_name,
            "intent": intent,
            "language": language,
            "kb_used": False,
        }

    if (query_profile or {}).get('simple_rag_mode'):
        answer, model_name = await _complete_generation(
            messages, max_output_tokens=output_token_limit(query, query_profile),
            budget=GeminiRequestBudget(), retry_incomplete=False,
        )
        return dict(answer=answer, sources=build_sources_payload(context_chunks), model=model_name,
                    intent=intent, language=language, kb_used=bool(context_chunks))

    messages.insert(1, {"role": "system", "content": _response_language_instruction(language)})
    messages.insert(2, {"role": "system", "content": GENERAL_RETRIEVED_CONTEXT_INSTRUCTION})
    messages.insert(3, {"role": "system", "content": FOLLOW_UP_SCOPE_INSTRUCTION})
    if not is_document_driven(query_profile):
        messages.insert(3, {"role": "system", "content": SUB_INTENT_SYSTEM_INSTRUCTION})
    topic_instruction = TOPIC_AUTHORITY_INSTRUCTIONS.get((query_profile or {}).get("topic"))
    if topic_instruction and not is_document_driven(query_profile):
        messages.insert(3, {"role": "system", "content": topic_instruction})
    if not is_document_driven(query_profile) and (query_profile or {}).get("is_wage_order_query"):
        messages.insert(3, {"role": "system", "content": REGIONAL_WAGE_SYSTEM_INSTRUCTION})
        messages.insert(4, {"role": "system", "content": wage_effectivity_instruction()})
    if not is_document_driven(query_profile) and (query_profile or {}).get("topic") in LEAVE_BENEFIT_TOPICS:
        messages.insert(3, {"role": "system", "content": LEAVE_BENEFITS_SYSTEM_INSTRUCTION})
    glossary_prompt = await get_hiligaynon_glossary_prompt(language)
    if glossary_prompt:
        messages.insert(2, {"role": "system", "content": glossary_prompt})

    token_limit = output_token_limit(query, query_profile)
    continuation_max_tokens = 500 if (
        (query_profile or {}).get("task") in {"comparison_question", "calculation_request", "jurisprudence_request"}
        or (query_profile or {}).get("sub_intent") == "termination_comparison"
    ) else 250
    budget = GeminiRequestBudget()
    answer, model_name = await _complete_generation(
        messages,
        max_output_tokens=token_limit,
        budget=budget,
        continuation_max_tokens=continuation_max_tokens,
    )

    if context_chunks and not is_document_driven(query_profile) and answer_claims_unavailable(answer):
        repair_messages = messages + [
            {"role": "assistant", "content": answer},
            {
                "role": "user",
                "content": (
                    "The previous draft incorrectly treated supported information as unavailable. "
                    "Rewrite it using only the retrieved excerpts. Do not add source labels, "
                    "document titles, or facts that the excerpts do not support."
                ),
            },
        ]
        answer, model_name = await _complete_generation(
            repair_messages,
            max_output_tokens=token_limit,
            budget=budget,
            continuation_max_tokens=continuation_max_tokens,
        )

    if context_chunks and answer_has_scope_violation(answer, query_profile):
        repair_messages = messages + [
            {"role": "assistant", "content": answer},
            {
                "role": "user",
                "content": (
                    "The previous draft exceeded the retrieved legal scope. Rewrite it using only "
                    "the selected source scope, omit inline citations and source names, and omit unsupported special-sector rules."
                ),
            },
        ]
        answer, model_name = await _complete_generation(
            repair_messages,
            max_output_tokens=token_limit,
            budget=budget,
            continuation_max_tokens=continuation_max_tokens,
        )

    if not _response_uses_expected_language(answer, language):
        logger.warning("Gemini response language drift detected; regenerating in %s", language)
        language_repair_messages = messages + [
            {"role": "assistant", "content": answer},
            {
                "role": "user",
                "content": (
                    "The previous draft used the wrong response language. Rewrite the entire answer "
                    "from the beginning and obey this requirement exactly:\n"
                    f"{_response_language_instruction(language)}\n"
                    "Preserve the supported legal meaning and numbers, but omit source labels and citations."
                ),
            },
        ]
        answer, model_name = await _complete_generation(
            language_repair_messages,
            max_output_tokens=token_limit,
            budget=budget,
            continuation_max_tokens=continuation_max_tokens,
        )

    return {
        "answer": answer,
        "sources": build_sources_payload(context_chunks),
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
    """Translate through Gemini while preserving LaborLens answer structure."""

    target_name = language_name(target_language)
    source_name = language_name(source_language) if source_language else "the original language"
    profile = get_language_profile(target_language)
    system_prompt = (
        "You are a careful translator for LaborLens. Translate only the supplied answer. "
        "Preserve its meaning, article numbers, amounts, and dates. Omit citations, source tags, "
        "source titles, and URLs because the interface displays retrieved sources separately. "
        "Translate it naturally; do not shorten, expand, explain, or add facts. "
        "The complete original answer is supplied between markers in the final user message; "
        "never ask for the original text, source text, or more context. Complete every section "
        "and return only the translated answer. "
        f"Use these section labels where applicable: {profile['direct_label']}, "
        f"{profile['explanation_label']}, {profile['requirements_label']}, "
        f"{profile['computation_label']}, {profile['legal_basis_label']}, "
        f"{profile['next_steps_label']}, and {profile['reminder_label']}."
    )
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.append(
        {"role": "system", "content": _response_language_instruction(target_language)}
    )
    glossary_prompt = await get_hiligaynon_glossary_prompt(target_language)
    if glossary_prompt:
        messages.append({"role": "system", "content": glossary_prompt})
    for message in (chat_history or [])[-MAX_PROMPT_HISTORY_MESSAGES:]:
        role = str(message.get("role") or "")
        content = compact_history_content(message.get("content", ""))
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Translate this answer from {source_name} into {target_name}.\n\n"
                f"Answer to translate:\n<<<\n{answer_text}\n>>>"
            ),
        }
    )
    # Keep translation generation proportional to the supplied answer rather
    # than spending the full legal-answer allowance on a short transformation.
    translation_token_limit = max(
        240,
        min(MAX_OUTPUT_TOKENS, int(len(answer_text.split()) * 2.2) + 80),
    )
    budget = GeminiRequestBudget()
    translated, model_name = await _complete_generation(
        messages,
        max_output_tokens=translation_token_limit,
        budget=budget,
    )
    if not _response_uses_expected_language(translated, target_language):
        logger.warning(
            "Gemini translation language drift detected; regenerating in %s",
            target_language,
        )
        repair_messages = messages + [
            {"role": "assistant", "content": translated},
            {
                "role": "user",
                "content": (
                    "Rewrite the complete translation in the required target language. "
                    f"{_response_language_instruction(target_language)}"
                ),
            },
        ]
        translated, model_name = await _complete_generation(
            repair_messages,
            max_output_tokens=translation_token_limit,
            budget=budget,
        )
    return translated
