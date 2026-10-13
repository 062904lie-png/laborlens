"""Provider health, circuit breaking, error classification, and token policy."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import re
import time
from typing import Any, Dict, Iterable, List

from config import settings


TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class CircuitOpenError(RuntimeError):
    """Raised when a known-unhealthy provider must be skipped immediately."""


@dataclass
class ProviderHealth:
    state: str = "CLOSED"
    consecutive_failures: int = 0
    opened_at: float | None = None
    retry_at: float | None = None
    half_open_probe_in_flight: bool = False
    requests: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    rate_limits: int = 0
    total_latency_ms: float = 0.0
    recent_latency_ms: float | None = None
    last_failure_at: float | None = None
    last_success_at: float | None = None
    last_error_type: str | None = None
    last_error_detail: str | None = None
    last_http_status: int | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


_health: Dict[str, ProviderHealth] = {}


def normalize_provider(value: str) -> str:
    return str(value or "").strip().lower()


def configured_provider_order() -> List[str]:
    ordered: List[str] = []
    for value in (settings.LABORLENS_PRIMARY_PROVIDER, settings.LABORLENS_FALLBACK_PROVIDER):
        provider = normalize_provider(value)
        if provider in {"groq", "gemini"} and provider not in ordered:
            ordered.append(provider)
    return ordered or ["groq", "gemini"]


def provider_timeout(provider: str) -> float:
    specific = {
        "groq": settings.LABORLENS_GROQ_TIMEOUT_SECONDS,
        "gemini": settings.LABORLENS_GEMINI_TIMEOUT_SECONDS,
        "openai": getattr(settings, "LABORLENS_OPENAI_TIMEOUT_SECONDS", settings.LABORLENS_PROVIDER_TIMEOUT_SECONDS),
    }.get(provider, settings.LABORLENS_PROVIDER_TIMEOUT_SECONDS)
    # Keep an individual provider attempt short enough that the request-wide
    # deadline still leaves time for a genuine fallback attempt.
    return max(1.0, min(float(specific or settings.LABORLENS_PROVIDER_TIMEOUT_SECONDS), 30.0))


def generation_attempt_timeout(
    remaining_seconds: float,
    attempts_left: int,
    provider_cap_seconds: float,
    *,
    is_first_attempt: bool,
    first_attempt_target_seconds: float | None = None,
    backup_reserve_seconds: float | None = None,
) -> float:
    """Give each attempt useful time without dividing by the whole model pool.

    Reserve time for one next attempt, not every configured fallback. If that
    would leave only a short retry, spend the remainder on the current attempt.
    Fast failures return unused time to the caller's unchanged deadline.
    The first_attempt arguments remain accepted for existing callers; the
    target now applies to subsequent attempts too.
    """

    remaining = max(0.0, float(remaining_seconds))
    attempts = max(1, int(attempts_left))
    cap = max(0.0, float(provider_cap_seconds))
    target = max(.001, float(
        getattr(settings, 'LABORLENS_GENERATION_ATTEMPT_SECONDS', 12.0)
        if first_attempt_target_seconds is None else first_attempt_target_seconds
    ))
    reserve = (
        float(settings.LABORLENS_MODEL_BACKUP_RESERVE_SECONDS)
        if backup_reserve_seconds is None
        else float(backup_reserve_seconds)
    )
    reserve = max(0.0, reserve)
    if attempts <= 1 or remaining <= target + reserve:
        return min(cap, remaining)
    return min(cap, target, remaining - reserve)


def health_for(provider: str) -> ProviderHealth:
    return _health.setdefault(normalize_provider(provider), ProviderHealth())


async def acquire_provider(provider: str) -> None:
    health = health_for(provider)
    cooldown = max(1.0, float(settings.LABORLENS_PROVIDER_COOLDOWN_SECONDS))
    async with health.lock:
        if health.state == "OPEN":
            retry_at = health.retry_at
            if retry_at is None and health.opened_at is not None:
                retry_at = health.opened_at + cooldown
            if retry_at is not None and time.monotonic() >= retry_at:
                health.state = "HALF_OPEN"
            else:
                raise CircuitOpenError(f"{provider} circuit is OPEN")
        if health.state == "HALF_OPEN":
            if health.half_open_probe_in_flight:
                raise CircuitOpenError(f"{provider} circuit probe is already in flight")
            health.half_open_probe_in_flight = True
        health.requests += 1


async def abandon_provider(provider: str, *, unattempted: bool = False) -> None:
    """Release a cancelled probe without reporting a fabricated provider failure."""
    health = health_for(provider)
    async with health.lock:
        health.half_open_probe_in_flight = False
        if unattempted:
            health.requests = max(0, health.requests - 1)


async def record_success(provider: str, latency_ms: float) -> None:
    health = health_for(provider)
    async with health.lock:
        health.successes += 1
        health.last_success_at = time.time()
        health.total_latency_ms += latency_ms
        health.recent_latency_ms = latency_ms
        health.consecutive_failures = 0
        health.opened_at = None
        health.retry_at = None
        health.half_open_probe_in_flight = False
        health.state = "CLOSED"


async def record_failure(
    provider: str,
    reason: str,
    latency_ms: float,
    *,
    counts: bool = True,
    retry_after_seconds: float | None = None,
    detail: str | None = None,
    http_status: int | None = None,
) -> None:
    health = health_for(provider)
    async with health.lock:
        health.failures += 1
        health.total_latency_ms += latency_ms
        health.recent_latency_ms = latency_ms
        health.last_failure_at = time.time()
        health.last_error_type = reason
        # Accept machine-generated diagnostic labels only, never exception
        # messages, headers, API keys, prompts, or provider response bodies.
        health.last_error_detail = detail if isinstance(detail, str) and re.fullmatch(r'[a-z_]{1,64}', detail) else None
        health.last_http_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        health.half_open_probe_in_flight = False
        if reason == "timeout":
            health.timeouts += 1
        if reason == "rate_limit":
            health.rate_limits += 1
        if counts:
            health.consecutive_failures += 1
            threshold = max(1, int(settings.LABORLENS_PROVIDER_FAILURE_THRESHOLD))
            retry_after = (
                max(1.0, min(float(retry_after_seconds), 86400.0))
                if retry_after_seconds is not None
                else None
            )
            if health.state == "HALF_OPEN" or health.consecutive_failures >= threshold or retry_after:
                health.state = "OPEN"
                health.opened_at = time.monotonic()
                health.retry_at = health.opened_at + retry_after if retry_after else None


def retry_delay_from_text(text: str) -> float | None:
    """Read provider durations such as '3m31.68s', not only plain seconds."""
    number = r'\d+(?:\.\d+)?'
    unit = r'(?:days?|hours?|minutes?|seconds?|mins?|secs?|[dhms])'
    match = re.search(r'(?:try\s+again|retry)\s+(?:in|after)\s+('
                      + number + r'\s*' + unit + r'(?:\s*' + number + r'\s*' + unit + r')*)',
                      str(text or ''), re.I)
    if match:
        total = 0.0
        for value, label in re.findall('(' + number + r')\s*(' + unit + ')', match[1], re.I):
            factor = {'d': 86400, 'h': 3600, 'm': 60, 's': 1}[label[0].lower()]
            total += float(value) * factor
        return max(1.0, min(total, 86400.0))
    match = re.search(r'retry_delay[^0-9]*(?:seconds[^0-9]*)?(' + number + ')', str(text or ''), re.I)
    return max(1.0, min(float(match[1]), 86400.0)) if match else None


def retry_after_seconds(exc: BaseException) -> float | None:
    """Extract a bounded provider retry delay without logging response secrets."""

    explicit = getattr(exc, "retry_after_seconds", None)
    if explicit is not None:
        try:
            return max(1.0, min(float(explicit), 86400.0))
        except (TypeError, ValueError):
            pass
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None) or {}
    for key in ("retry-after", "Retry-After"):
        value = headers.get(key) if hasattr(headers, "get") else None
        if value is not None:
            try:
                return max(1.0, min(float(value), 86400.0))
            except (TypeError, ValueError):
                pass
    delay = retry_delay_from_text(str(exc))
    if delay is None and response is not None:
        try:
            delay = retry_delay_from_text(response.text)
        except (RuntimeError, AttributeError):
            pass  # A streaming response may not have been read yet.
    return delay


def ranked_health_keys(keys: Iterable[str]) -> List[str]:
    """Prefer closed, successful, low-latency candidates while preserving order ties."""

    now = time.monotonic()
    cooldown = max(1.0, float(settings.LABORLENS_PROVIDER_COOLDOWN_SECONDS))

    def score(item: tuple[int, str]) -> tuple[float, ...]:
        index, key = item
        health = health_for(key)
        retry_at = health.retry_at
        if retry_at is None and health.opened_at is not None:
            retry_at = health.opened_at + cooldown
        if health.state == "CLOSED":
            state_rank = 0.0
        elif health.state == "HALF_OPEN":
            state_rank = 1.0
        elif retry_at is not None and now >= retry_at:
            state_rank = 1.0
        else:
            state_rank = 3.0
        failure_rate = health.failures / max(1, health.requests)
        latency = health.recent_latency_ms if health.recent_latency_ms is not None else float("inf")
        never_measured = 1.0 if health.requests == 0 else 0.0
        return state_rank, failure_rate, never_measured, latency, float(index)

    return [key for _, key in sorted(enumerate(keys), key=score)]


def classify_provider_error(exc: BaseException) -> tuple[str, bool]:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout", True
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    response = getattr(exc, "response", None)
    status = status or getattr(response, "status_code", None)
    message = str(exc).lower()
    if "timed out" in message or "timeout" in message:
        return "timeout", True
    if status == 429 or "429" in message or "rate limit" in message or "quota" in message:
        return "rate_limit", True
    if status in TRANSIENT_STATUS_CODES or any(f" {code}" in message for code in TRANSIENT_STATUS_CODES):
        return "provider_error", True
    if any(term in message for term in ("connect", "network", "overload", "unavailable", "temporar")):
        return "provider_error", True
    if "empty response" in message or "no readable response" in message:
        return "empty_response", True
    if "invalid" in message or "json" in message:
        return "invalid_response_format", False
    return "provider_error", False


def completion_token_limit(model: str, visible_tokens: int) -> int:
    """Allow reasoning without inflating the user-visible answer length policy."""
    if 'gemini' in model.lower() or 'gpt-oss' in model.lower():
        allowance = max(1024, min(int(settings.LABORLENS_REASONING_COMPLETION_TOKENS), 8192))
        return max(visible_tokens, allowance)
    return visible_tokens


def output_token_limit(query: str, query_profile: Dict[str, Any] | None = None) -> int:
    profile = query_profile or {}
    complex_request = bool(
        len(query or "") > 500
        or profile.get("task") in {"comparison_question", "calculation_request", "jurisprudence_request"}
        or re.search(r"\b(detailed|comprehensive|in depth|elaborate|all exceptions)\b", query or "", re.I)
    )
    # Deterministic/static requests should stay cheap. Normal legal answers
    # use the normal budget; only explicitly complex requests get the ceiling.
    simple_request = bool(
        profile.get("task") in {"greeting", "small_talk", "identity", "capability_question"}
        or re.fullmatch(r"\s*(what is|define|who is)\b.{0,120}", query or "", re.I)
    )
    if complex_request:
        configured = settings.LABORLENS_MAX_OUTPUT_TOKENS_COMPLEX
    elif simple_request:
        configured = settings.LABORLENS_SIMPLE_OUTPUT_TOKENS
    else:
        configured = settings.LABORLENS_MAX_OUTPUT_TOKENS
    hard_limit = max(400, min(int(settings.LABORLENS_HARD_OUTPUT_LIMIT), 2048))
    return max(400, min(int(configured), hard_limit))


def merge_continuation(initial: str, continuation: str) -> str:
    """Append only new continuation text and trim accidental prefix overlap."""

    left = (initial or "").rstrip()
    right = (continuation or "").strip()
    if not right:
        return left
    max_overlap = min(300, len(left), len(right))
    overlap = 0
    for size in range(max_overlap, 19, -1):
        if left[-size:].casefold() == right[:size].casefold():
            overlap = size
            break
    return f"{left}\n{right[overlap:].lstrip()}".strip()


def answer_ends_at_sentence_boundary(answer: str) -> bool:
    """Return whether a generated reply ends cleanly, including source tags.

    This is deliberately topic-neutral. It prevents any legal answer from being
    retained after a provider limit if its final thought is cut off.
    """

    lines = [line.strip() for line in str(answer or "").splitlines() if line.strip()]
    if not lines:
        return False
    ending = lines[-1].strip("*_` \t\"'")
    # A normal inline citation follows a complete sentence, not the other way
    # around. Remove it before checking the preceding punctuation.
    ending = re.sub(
        r"(?:\s*\((?:source\s*)?\d+(?:\s*,\s*(?:source\s*)?\d+)*\))+\s*$",
        "",
        ending,
        flags=re.IGNORECASE,
    ).rstrip("*_` \t\"'")
    return bool(re.search(r"[.!?](?:[\"'”’\)\]]*)$", ending))


def compress_answer_to_budget(answer: str, max_tokens: int = 1200) -> tuple[str, bool]:
    """Bound runaway continuation output without inventing or rephrasing facts.

    This is a loss-minimizing safety valve: keep the beginning (direct answer
    and explanation) and the ending (exceptions/citations), cutting only at a
    sentence/line boundary. Normal responses are returned byte-for-byte.
    """
    text = (answer or "").strip()
    limit = max(400, min(int(max_tokens or 1200), 2048))
    if len(text) <= limit * 4:
        return text, False
    char_limit = limit * 4
    # Prefer semantic section preservation whenever the draft has labeled
    # legal sections. This avoids the old head+tail cut dropping a notice or
    # separation-pay qualification from the middle of a comparison.
    lines = text.splitlines()
    heading_indexes = [
        index for index, line in enumerate(lines)
        if re.match(r"^\s*(?:#{1,6}\s+|\*\*[^*]+\*\*:|[A-Z][A-Za-z /,&-]{2,60}:)", line)
        and re.search(r"\b(direct answer|just cause|authorized cause|requirements?|conditions?|exceptions?|notice|due process|separation pay|legal basis|examples?|practical next)", line, re.I)
        and len(line.strip()) <= 120
    ]
    if len(heading_indexes) >= 2:
        sections: list[str] = []
        for position, start in enumerate(heading_indexes):
            end = heading_indexes[position + 1] if position + 1 < len(heading_indexes) else len(lines)
            block = "\n".join(lines[start:end]).strip()
            if block:
                sections.append(block)
        per_section = max(180, char_limit // max(1, len(sections)))
        compact_sections: list[str] = []
        for block in sections:
            if len(block) <= per_section:
                compact_sections.append(block)
                continue
            block_lines = block.splitlines()
            heading = block_lines[0]
            body = "\n".join(block_lines[1:])
            important = [
                line.strip() for line in block_lines[1:]
                if re.search(r"\b(exception|except|notice|dole|employee|formula|rate|article|section|source|must|shall|required)\b", line, re.I)
            ]
            selected = []
            for line in important:
                if line and line not in selected:
                    selected.append(line)
            candidate = "\n".join([heading, *selected])
            if len(candidate) < per_section:
                candidate = (candidate + "\n" + body[: max(0, per_section - len(candidate))]).rstrip()
            compact_sections.append(candidate[:per_section].rsplit(" ", 1)[0].rstrip())
        semantic = "\n\n".join(compact_sections).strip()
        if len(semantic) <= char_limit:
            return semantic, True
    head = int(char_limit * 0.72)
    tail = char_limit - head
    head_text = text[:head].rsplit(".", 1)[0].strip()
    tail_text = text[-tail:].lstrip()
    if not head_text:
        head_text = text[:head].rstrip()
    compressed = f"{head_text}.\n\n{tail_text}".strip()
    return compressed[:char_limit].rstrip(), True


def check_final_answer_coverage(answer: str, required_components: Iterable[str]) -> Dict[str, Any]:
    """Check user-visible component coverage and structural completeness."""
    text = (answer or "").strip()
    lowered = text.casefold()
    aliases = {
        "just_cause_definition": ("just cause",),
        "just_cause_grounds": ("serious misconduct", "willful disobedience", "gross neglect", "fraud", "grounds"),
        "just_cause_due_process": ("due process", "opportunity to explain", "notice to explain"),
        "authorized_cause_definition": ("authorized cause",),
        "authorized_cause_grounds": ("redundancy", "retrenchment", "closure", "disease", "labor-saving", "grounds"),
        "authorized_cause_notice": ("authorized cause", "notice", "dole"),
        "separation_pay": ("separation pay",),
        "examples": ("example", "for instance"),
        "legal_basis": ("legal basis", "article", "section", "source"),
    }
    coverage: Dict[str, bool] = {}
    for component in required_components or []:
        key = str(component)
        terms = aliases.get(key, (key.replace("_", " "),))
        if key == "authorized_cause_notice":
            coverage[key] = "authorized cause" in lowered and any(term in lowered for term in terms[1:])
        else:
            coverage[key] = any(term in lowered for term in terms)
    nonempty_headings = True
    for index, line in enumerate(text.splitlines()):
        if re.match(r"^\s*(?:#{1,6}\s*|[*_]{1,3})?[A-Za-z][^:]{0,100}(?:\([^)]*\))?\s*$", line) and re.search(r"\b(just cause|authorized cause|legal basis|separation pay|notice|examples?)\b", line, re.I):
            following = "\n".join(text.splitlines()[index + 1:]).strip()
            if not following or re.match(r"^(?:#{1,6}\s*)?[A-Za-z][^:]{0,100}\s*$", following.splitlines()[0]):
                nonempty_headings = False
    unsafe_due_process_phrase = bool(re.search(
        r"\b(?:fire|fired|dismiss|terminate(?:d)?)\b[^.\n]{0,100}\b(?:right away|immediately|on the spot|without (?:notice|a chance to explain)|no hearing required|without a special notice period)\b",
        lowered,
    ))
    complete_sentence = not answer_looks_structurally_incomplete(text)
    return {
        "coverage": coverage,
        "missing": [key for key, present in coverage.items() if not present],
        "empty_heading": not nonempty_headings,
        "unsafe_due_process_phrase": unsafe_due_process_phrase,
        "complete_sentence": complete_sentence,
        "complete": bool(text) and all(coverage.values()) and nonempty_headings and not unsafe_due_process_phrase and complete_sentence,
    }


def answer_looks_structurally_incomplete(text: str) -> bool:
    value = (text or "").rstrip()
    if not value:
        return True
    last = value.splitlines()[-1].strip().strip("*_` ")
    return bool(re.search(r"(?:[:=;,/\\+\-]|\b(?:and|or|to|if|is|are|the))\s*$", last, re.I))


def is_usable_partial_answer(answer: str) -> bool:
    """Conservative salvage check for a draft that stopped at the token limit."""
    text = (answer or "").strip()
    if len(text) < 160:
        return False
    lowered = text.casefold()
    has_conclusion = bool(re.search(r"[.!?](?:\s|$)", text)) or any(
        marker in lowered
        for marker in ("direct answer", "direktang sagot", "direkta nga sabat")
    )
    has_basis = any(
        marker in lowered
        for marker in ("legal basis", "legal na batayan", "legal nga basihan", "source 1")
    )
    # Obvious unfinished critical propositions must never be salvaged.
    if re.search(r"(?:entitled to|applicable rate is|may terminate.*if|formula\s*:?)\s*$", lowered):
        return False
    return (
        has_conclusion
        and has_basis
        and answer_ends_at_sentence_boundary(text)
        and not re.search(r"[=:;,(/+\-]\s*$", text)
    )


def provider_health_snapshot() -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for provider, health in _health.items():
        retry_in = None
        if health.state == "OPEN":
            retry_at = health.retry_at
            if retry_at is None and health.opened_at is not None:
                retry_at = health.opened_at + max(
                    1.0, float(settings.LABORLENS_PROVIDER_COOLDOWN_SECONDS)
                )
            if retry_at is not None:
                retry_in = round(max(0.0, retry_at - time.monotonic()), 2)
        output[provider] = {
            "requests": health.requests,
            "successes": health.successes,
            "failures": health.failures,
            "timeouts": health.timeouts,
            "rate_limits": health.rate_limits,
            "success_rate": round(health.successes / max(1, health.requests), 4),
            "average_latency_ms": round(health.total_latency_ms / max(1, health.requests), 2),
            "recent_latency_ms": health.recent_latency_ms,
            "circuit_state": health.state,
            "last_failure_at": health.last_failure_at,
            "last_success_at": health.last_success_at,
            "last_error_type": health.last_error_type,
            "last_error_detail": health.last_error_detail,
            "last_http_status": health.last_http_status,
            "retry_in_seconds": retry_in,
        }
    return output


def reset_provider_health(providers: Iterable[str] = ("groq", "gemini")) -> None:
    for provider in providers:
        _health.pop(normalize_provider(provider), None)


def complete_answer_retry_messages(messages):
    """Regenerate from original evidence, never guess an unfinished formula's tail."""
    return [*messages, {
        "role": "user",
        "content": (
            "Your previous attempt was cut off. Write the entire answer again, concisely, "
            "using only the supplied evidence. Aim for at most 150 words. Prioritize the "
            "direct answer and essential citations. Include only calculations you can finish "
            "with clear units and a final result. Avoid repeating summaries. End with a "
            "complete sentence; do not leave any formula, list item, or bracket unfinished."
        ),
    }]
