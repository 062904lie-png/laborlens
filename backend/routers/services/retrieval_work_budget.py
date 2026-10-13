"""Bound optional retrieval work while retaining evidence for the checked answer."""
import asyncio
from .answer_pipeline_budget import _finite_seconds
from .performance import set_timing_metadata, current_timing


class RetrievalWorkBudget:
    def __init__(self, request, settings):
        self.request = request
        self._clock = request._clock
        complex_request = request.generation_seconds > float(getattr(
            settings, 'LABORLENS_NORMAL_REQUEST_BUDGET_SECONDS', 45))
        cap = getattr(settings, 'LABORLENS_RETRIEVAL_COMPLEX_SECONDS', 75) if complex_request else getattr(
            settings, 'LABORLENS_RETRIEVAL_NORMAL_SECONDS', 60)
        generation = min(request.generation_seconds, max(0, _finite_seconds(getattr(
            settings, 'LABORLENS_RETRIEVAL_GENERATION_RESERVE_SECONDS', 15), 'generation_reserve')))
        self.answer_reserve = generation + request.verify_seconds
        # Tiny offline/test budgets retain a finite retrieval window as well.
        self.answer_reserve = min(self.answer_reserve, request.remaining() * .6)
        self.allowance = request.timeout(cap=cap, reserve=self.answer_reserve)
        self.deadline = self._clock() + self.allowance
        self.finalization_reserve = min(self.allowance * .2, max(0, _finite_seconds(getattr(
            settings, 'LABORLENS_RETRIEVAL_FINALIZATION_RESERVE_SECONDS', 8), 'finalization_reserve')))
        self.ranking_remaining = max(0, _finite_seconds(getattr(
            settings, 'LABORLENS_RETRIEVAL_RANKING_SECONDS', 12), 'ranking_seconds'))
        set_timing_metadata(retrieval_budget_ms=round(self.allowance * 1000, 2),
                            answer_time_reserved_ms=round(self.answer_reserve * 1000, 2))

    def remaining(self):
        return max(0, min(self.deadline - self._clock(),
                         self.request.timeout(reserve=self.answer_reserve)))

    async def optional(self, stage, operation, fallback, *, ranking=False):
        # Discovery must also leave room for the first mandatory rerank. After
        # ranking, its unused allowance remains available for conflict recovery.
        reserved_ranking = 0 if ranking else self.ranking_remaining
        allowance = max(0, self.remaining() - self.finalization_reserve - reserved_ranking)
        if ranking:
            allowance = min(allowance, self.ranking_remaining)
        started = self._clock()
        try:
            if allowance <= 0:
                self._limited(stage)
                return fallback
            return await asyncio.wait_for(operation(), timeout=allowance)
        except TimeoutError:
            # Optional work cannot erase the already searchable candidate pool.
            # External cancellation still propagates; no support is fabricated.
            self._limited(stage)
            return fallback
        finally:
            if ranking:
                self.ranking_remaining = max(0, self.ranking_remaining - max(0, self._clock() - started))

    @staticmethod
    def _limited(stage):
        timing = current_timing()
        count = int(timing.metadata.get('retrieval_optional_limit_count', 0)) if timing else 0
        set_timing_metadata(retrieval_optional_limited=True,
                            retrieval_optional_limit_count=count + 1,
                            retrieval_last_limited_stage=stage)
