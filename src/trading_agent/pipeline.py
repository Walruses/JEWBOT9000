"""Slow loop: poll data sources, de-duplicate, have the analyst score new items."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from typing import Protocol

from .models import NewsItem, Signal
from .signals.hub import SignalHub
from .sources.base import NewsSource

log = logging.getLogger(__name__)


class Analyst(Protocol):
    async def analyze(
        self, symbol: str, items: list[NewsItem], now: float | None = None
    ) -> Signal | None: ...


class EventRecorder(Protocol):
    def news(self, item: NewsItem) -> None: ...

    def signal(self, signal: Signal) -> None: ...


class NewsPipeline:
    def __init__(
        self,
        sources: list[NewsSource],
        analyst: Analyst,
        hub: SignalHub,
        symbols: list[str],
        poll_interval: float = 60.0,
        max_item_age: float = 3600.0,
        max_concurrent_analyses: int = 4,
        clock: Callable[[], float] = time.time,
        recorder: EventRecorder | None = None,
    ):
        self.sources = sources
        self.analyst = analyst
        self.hub = hub
        self.symbols = symbols
        self.poll_interval = poll_interval
        self.max_item_age = max_item_age
        self._clock = clock
        self.recorder = recorder
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._seen_cap = 50_000
        self._sem = asyncio.Semaphore(max_concurrent_analyses)

    async def run(self) -> None:
        while True:
            try:
                await self.poll_once()
            except Exception:
                log.exception("news pipeline poll failed")
            await asyncio.sleep(self.poll_interval)

    async def poll_once(self) -> list[Signal]:
        results = await asyncio.gather(
            *(s.fetch(self.symbols) for s in self.sources), return_exceptions=True
        )
        new_by_symbol: dict[str, list[NewsItem]] = defaultdict(list)
        cutoff = self._clock() - self.max_item_age
        for source, result in zip(self.sources, results, strict=True):
            if isinstance(result, BaseException):
                log.warning("source %s failed: %r", source.name, result)
                continue
            for item in result:
                if item.id in self._seen:
                    continue
                self._mark_seen(item.id)
                if self.recorder:
                    self.recorder.news(item)
                # Old items are already priced in; this also stops a restart from
                # trading on the backlog.
                if item.published_at >= cutoff:
                    new_by_symbol[item.symbol].append(item)

        signals = await asyncio.gather(
            *(self._analyze(sym, items) for sym, items in new_by_symbol.items())
        )
        published = [s for s in signals if s is not None]
        for sig in published:
            self.hub.publish(sig)
            if self.recorder:
                self.recorder.signal(sig)
            log.info(
                "signal %s %s score=%+.2f conf=%.2f ttl=%.0fs: %s",
                sig.symbol,
                sig.source,
                sig.score,
                sig.confidence,
                sig.ttl,
                sig.rationale,
            )
        return published

    async def _analyze(self, symbol: str, items: list[NewsItem]) -> Signal | None:
        async with self._sem:
            try:
                return await self.analyst.analyze(symbol, items, now=self._clock())
            except Exception:
                log.exception("analysis failed for %s", symbol)
                return None

    def _mark_seen(self, item_id: str) -> None:
        self._seen[item_id] = None
        if len(self._seen) > self._seen_cap:
            self._seen.popitem(last=False)
