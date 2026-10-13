"""One monotonic deadline shared by retrieval, generation and checking."""
import asyncio
import math
import time
from collections.abc import Callable
from typing import Any


def _finite_seconds(value: Any, name: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a finite duration') from None
    if not math.isfinite(seconds):
        raise ValueError(f'{name} must be a finite duration')
    return seconds


class AnswerPipelineBudget:
    """One bounded initial/repair deadline, never a new budget per retry.

    ``run`` accepts an async callable rather than a created coroutine so work
    cannot be scheduled after the deadline. Reserves leave time for later work
    without creating a second deadline or changing provider timeout settings.
    """

    def __init__(self, generation_seconds: float, verify_seconds: float, *,
                 recovery_seconds: float = 0,
                 clock: Callable[[], float] | None = None):
        self.generation_seconds = _finite_seconds(generation_seconds, 'generation_seconds')
        self.verify_seconds = _finite_seconds(verify_seconds, 'verify_seconds')
        self.recovery_seconds = _finite_seconds(recovery_seconds, 'recovery_seconds')
        if self.generation_seconds < 0 or self.verify_seconds < 0 or self.recovery_seconds < 0:
            raise ValueError('Pipeline durations must not be negative')
        self._clock = clock or time.monotonic
        self.final_check_reserve = self.verify_seconds if self.recovery_seconds > 0 else 0
        self._total = (self.generation_seconds + self.verify_seconds +
                       self.recovery_seconds + self.final_check_reserve)
        if not math.isfinite(self._total):
            raise ValueError('The total pipeline duration must be finite')
        self._deadline = _finite_seconds(self._clock(), 'clock') + self._total
        if not math.isfinite(self._deadline):
            raise ValueError('The pipeline deadline must be finite')

    @classmethod
    def from_settings(cls, settings, query: str, query_profile=None, *, clock=None):
        profile = query_profile or {}
        complex_request = bool(
            len(query or '') > 500
            or profile.get('task') in {'comparison_question', 'calculation_request', 'jurisprudence_request'}
        )
        generation = getattr(settings, 'LABORLENS_COMPLEX_REQUEST_BUDGET_SECONDS', 60.0) if complex_request else \
                     getattr(settings, 'LABORLENS_NORMAL_REQUEST_BUDGET_SECONDS', 45.0)
        verification = getattr(settings, 'LABORLENS_VERIFY_TIMEOUT_SECONDS',
                               getattr(settings, 'LABORLENS_RERANK_TIMEOUT_SECONDS', 24.0))
        # Legacy configurations without the new allowance retain their deadline.
        # Production adds one recovery allowance and final check, not per retry.
        return cls(max(1.0, _finite_seconds(generation, 'generation_seconds')),
                   max(.01, _finite_seconds(verification, 'verify_seconds')),
                   recovery_seconds=max(0, _finite_seconds(getattr(
                       settings, 'LABORLENS_ANSWER_RECOVERY_SECONDS', 0), 'recovery_seconds')), clock=clock)

    def remaining(self) -> float:
        now = _finite_seconds(self._clock(), 'clock')
        return min(self._total, max(0.0, self._deadline - now))

    def restrict_to_simple(self, *, minimum_remaining=0):
        """Remove unused recovery time without extending the original deadline."""
        floor = max(0, _finite_seconds(minimum_remaining, 'minimum_remaining'))
        total = min(self._total, max(self.generation_seconds + self.verify_seconds,
                                    self._total - self.remaining() + floor))
        self._deadline -= self._total - total
        self._total = total
        self.recovery_seconds = 0
        self.final_check_reserve = 0

    def timeout(self, cap=None, reserve=0) -> float:
        allowance = max(0.0, self.remaining() - max(0.0, _finite_seconds(reserve, 'reserve')))
        if cap is not None:
            allowance = min(allowance, max(0.0, _finite_seconds(cap, 'cap')))
        return allowance

    async def run(self, fn, *args, cap=None, reserve=0, **kwargs):
        allowance = self.timeout(cap=cap, reserve=reserve)
        if allowance <= 0:
            raise TimeoutError('Answer pipeline time budget exhausted before starting work')
        # wait_for owns cancellation of the newly created awaitable. Do not
        # swallow external CancelledError or relabel it as provider success.
        return await asyncio.wait_for(fn(*args, **kwargs), timeout=allowance)
