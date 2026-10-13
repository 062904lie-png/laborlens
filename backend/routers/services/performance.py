"""Request-scoped latency measurements for the LaborLens critical path.

Timing data is logged and saved as privacy-safe admin diagnostics. It is not
included in ordinary chat responses and never stores user message text.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import asyncio
import logging
import re
import time
import uuid
from typing import Any, Dict, Iterator, Optional


logger = logging.getLogger(__name__)


@dataclass
class RequestTiming:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started: float = field(default_factory=time.perf_counter)
    values: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    pipeline_budget: Any = None

    def add_ms(self, name: str, elapsed_ms: float) -> None:
        self.values[name] = self.values.get(name, 0.0) + max(0.0, elapsed_ms)

    def set_meta(self, **values: Any) -> None:
        self.metadata.update({key: value for key, value in values.items() if value is not None})

    def finish(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "request_total_ms": round((time.perf_counter() - self.started) * 1000, 2),
            "routing_ms": 0.0,
            "retrieval_ms": 0.0,
            "embedding_ms": 0.0,
            "rerank_ms": 0.0,
            "primary_model_ms": 0.0,
            "fallback_model_ms": 0.0,
            "validation_ms": 0.0,
            "citation_validation_ms": 0.0,
            "database_read_ms": 0.0,
            "database_write_ms": 0.0,
            "session_update_ms": 0.0,
            "time_to_first_token_ms": None,
        }
        payload.update({key: round(value, 2) for key, value in self.values.items()})
        payload.update(self.metadata)
        payload['generation_ms'] = round(self.values.get('primary_model_ms', 0) + self.values.get('fallback_model_ms', 0), 2)
        payload.setdefault('total_answer_ms', payload['request_total_ms'])
        return payload


_current: ContextVar[Optional[RequestTiming]] = ContextVar("laborlens_request_timing", default=None)
_retrieval_diagnostics: ContextVar[Any] = ContextVar('retrieval_diagnostics', default=None)


def _retrieval_slug(value):
    return re.sub(r'[^a-zA-Z0-9_.:/-]+', '_', str(value or '')).strip('_')[:100]


class RetrievalDiagnostics:
    """Request-local stage timings that survive cancellation of retrieval."""

    def __init__(self, budget=None):
        self.budget = budget
        self.started = time.perf_counter()
        self.stages = []
        self.active = {}
        self.counts = {}

    def remaining_ms(self):
        try:
            return round(max(0.0, self.budget.remaining()) * 1000, 2) if self.budget else None
        except Exception:
            return None

    def _publish(self, active_stage=None):
        active_names = [entry['stage'] for entry in self.active.values()]
        if active_stage is None:
            active_stage = active_names[-1] if active_names else 'not_active'
        operation_counts = [{'stage': name, 'operation_count': count}
                            for name, count in self.counts.items()]
        set_timing_metadata(
            retrieval_active_stage=active_stage,
            retrieval_active_stages=active_names,
            retrieval_stage_details=self.stages[-200:],
            retrieval_operation_counts=operation_counts[-64:],
            retrieval_remaining_deadline_ms=self.remaining_ms(),
            retrieval_elapsed_ms=round((time.perf_counter() - self.started) * 1000, 2),
            retrieval_active_stage_count=len(active_names),
            retrieval_stage_count=len(self.stages),
            retrieval_candidates_produced=sum(int(entry.get('candidate_count') or 0) for entry in self.stages),
            retrieval_selected_chunk_count=next((int(entry.get('input_candidate_count') or 0)
                for entry in reversed(self.stages) if entry.get('stage') == 'section_expansion'), 0),
            retrieval_parents_reconstructed=sum(int(entry.get('candidate_count') or 0)
                for entry in self.stages if entry.get('stage') == 'section_parent_assembly'),
            retrieval_sections_expanded=next((int(entry.get('candidate_count') or 0)
                for entry in reversed(self.stages) if entry.get('stage') == 'section_expansion'), 0),
            retrieval_section_expansion_calls=sum(count for name, count in self.counts.items()
                if name in {'parent_location_reconstruction_read', 'section_part_reconstruction_read',
                            'referenced_section_expansion_read'}),
        )

    async def run(self, stage, operation, *, category=None, categories=None,
                  candidate_count=None, input_count=None):
        started = time.perf_counter()
        entry = {
            'stage': _retrieval_slug(stage), 'category': _retrieval_slug(category),
            'categories': [_retrieval_slug(value) for value in (categories or [])[:13]],
            'started_at_ms': round(time.time() * 1000),
            'start_elapsed_ms': round((started - self.started) * 1000, 2),
            'remaining_deadline_ms': self.remaining_ms(), 'operation_count': 1,
            'input_candidate_count': int(input_count) if input_count is not None else None,
            'candidate_count': 0, 'completed': False, 'cancelled': False,
            'cancelled_by_deadline': False, 'elapsed_ms': 0.0, 'ended_at_ms': None,
        }
        self.stages.append(entry)
        token = id(entry)
        self.active[token] = entry
        self.counts[entry['stage']] = self.counts.get(entry['stage'], 0) + 1
        self._publish(entry['stage'])
        try:
            result = await operation()
            if candidate_count:
                entry['candidate_count'] = int(candidate_count(result) or 0)
            elif isinstance(result, (list, tuple, set)):
                entry['candidate_count'] = len(result)
            entry['completed'] = True
            return result
        except asyncio.CancelledError:
            remaining = self.remaining_ms()
            entry['cancelled'] = True
            entry['cancelled_by_deadline'] = remaining is not None and remaining <= 50
            raise
        except BaseException as exc:
            entry['failed'] = True
            entry['failure_type'] = type(exc).__name__
            raise
        finally:
            ended = time.perf_counter()
            entry['ended_at_ms'] = round(time.time() * 1000)
            entry['remaining_at_end_ms'] = self.remaining_ms()
            entry['elapsed_ms'] = round((ended - started) * 1000, 2)
            self.active.pop(token, None)
            self._publish()

    def measure_sync(self, stage, operation, *, category=None, categories=None,
                     candidate_count=None, input_count=None):
        started = time.perf_counter()
        entry = {
            'stage': _retrieval_slug(stage), 'category': _retrieval_slug(category),
            'categories': [_retrieval_slug(value) for value in (categories or [])[:13]],
            'started_at_ms': round(time.time() * 1000),
            'start_elapsed_ms': round((started - self.started) * 1000, 2),
            'remaining_deadline_ms': self.remaining_ms(), 'operation_count': 1,
            'input_candidate_count': int(input_count) if input_count is not None else None,
            'candidate_count': 0, 'completed': False, 'cancelled': False,
            'cancelled_by_deadline': False, 'elapsed_ms': 0.0, 'ended_at_ms': None,
        }
        self.stages.append(entry)
        token = id(entry)
        self.active[token] = entry
        self.counts[entry['stage']] = self.counts.get(entry['stage'], 0) + 1
        self._publish(entry['stage'])
        try:
            result = operation()
            if candidate_count:
                entry['candidate_count'] = int(candidate_count(result) or 0)
            elif isinstance(result, (list, tuple, set)):
                entry['candidate_count'] = len(result)
            entry['completed'] = True
            return result
        except BaseException as exc:
            entry['failed'] = True
            entry['failure_type'] = type(exc).__name__
            raise
        finally:
            ended = time.perf_counter()
            entry['ended_at_ms'] = round(time.time() * 1000)
            entry['remaining_at_end_ms'] = self.remaining_ms()
            entry['elapsed_ms'] = round((ended - started) * 1000, 2)
            self.active.pop(token, None)
            self._publish()

    def cancel_active(self, *, deadline=False):
        now = time.perf_counter()
        for entry in self.active.values():
            entry['cancelled'] = True
            entry['cancelled_by_deadline'] = bool(deadline)
            entry['elapsed_ms'] = round((now - self.started) * 1000 - entry['start_elapsed_ms'], 2)
            entry['ended_at_ms'] = round(time.time() * 1000)
            entry['remaining_at_end_ms'] = self.remaining_ms()
        self._publish()


@contextmanager
def use_retrieval_diagnostics(diagnostics):
    token = _retrieval_diagnostics.set(diagnostics)
    try:
        yield diagnostics
    finally:
        _retrieval_diagnostics.reset(token)


def current_retrieval_diagnostics():
    return _retrieval_diagnostics.get()


def begin_request() -> tuple[RequestTiming, Any]:
    timing = RequestTiming()
    timing.set_meta(llm_rerank_called=False, recovery_triggered=False,
                    recovery_new_candidates=0, prior_llm_ranking_preserved=False,
                    generation_provider_calls=0, verification_called=False,
                    repair_called=False, total_model_calls=0, estimated_model_tokens=0)
    timing.set_meta(simple_fast_path=False, retrieval_fast_exit=False, rerank_called=False,
                    rerank_skipped=False, verification_failed_due_to_missing_evidence=False,
                    generation_calls=0, verification_calls=0, recovery_called=False, total_llm_calls=0)
    timing.set_meta(retrieval_discovery_complete=False)
    timing.set_meta(retrieval_completed=False, source_ranking_completed=False,
                    evidence_selected=False, evidence_sufficient=False,
                    generation_eligibility_checked=False, generation_eligible=False,
                    generation_pool_admission='not_reached', generation_call_state='not_reached',
                    verification_eligibility_checked=False, verification_eligible=False)
    timing.set_meta(simple_legal_rag=False, retrieval_cancelled=False,
                    retrieval_active_stage='not_started', retrieval_active_stages=[],
                    retrieval_stage_details=[], retrieval_operation_counts=[],
                    retrieval_fallback_categories_attempted=[], retrieval_total_ms=0.0,
                    retrieval_elapsed_ms=0.0, retrieval_remaining_deadline_ms=None,
                    retrieval_active_stage_count=0, retrieval_cancelled_stage_count=0)
    timing.values.update(generation_ms=0.0, verification_ms=0.0, recovery_ms=0.0)
    return timing, _current.set(timing)


def end_request(timing: RequestTiming, token: Any) -> Dict[str, Any]:
    payload = timing.finish()
    logger.info("[LaborLens Timing] %s", json.dumps(payload, sort_keys=True, default=str))
    try:
        from .backend_diagnostics import record_event
        record_event('chat', 'request_complete', request_id=timing.request_id, details=payload)
    finally:
        _current.reset(token)
    return payload


def current_timing() -> Optional[RequestTiming]:
    return _current.get()


async def parallel_discovery(*jobs):
    """Own independent searches, including cancellation if any one fails."""
    tasks = [asyncio.create_task(job) for job in jobs]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def add_timing(name: str, elapsed_ms: float) -> None:
    timing = current_timing()
    if timing:
        timing.add_ms(name, elapsed_ms)


def set_timing_metadata(**values: Any) -> None:
    timing = current_timing()
    if timing:
        timing.set_meta(**values)


@contextmanager
def measure(name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        add_timing(name, (time.perf_counter() - started) * 1000)
