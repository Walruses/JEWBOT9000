"""Slow loop: poll data sources, de-duplicate, have the analyst score new items."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import time as time_of_day
from enum import Enum
from typing import Protocol

from .models import NewsItem, Signal
from .session import NEW_YORK
from .signals.hub import SignalHub
from .sources.base import NewsSource

log = logging.getLogger(__name__)


class Analyst(Protocol):
    async def analyze(
        self, symbol: str, items: list[NewsItem], now: float | None = None, note: str = ""
    ) -> Signal | None: ...


class NewsMode(Enum):
    LIVE = "live"  # analyse new items as they arrive
    EXTENDED = "extended"  # pre-market / after-hours: collect for the briefing
    QUIET = "quiet"  # overnight / weekend: collect, rarely


BRIEFING_NOTE = (
    "Pre-open briefing: these items were published since the last regular session "
    "closed. The regular session opens at 09:30 ET and the opening price may already "
    "reflect some of them. Judge the view for the first hours of trading."
)


@dataclass(frozen=True)
class NewsSchedule:
    """When to poll and whether to analyse. Off-hours news isn't analysed item by item:
    nothing can trade on it until the open, and by then it's partly priced in. It is
    collected and analysed once per symbol in a pre-open briefing instead."""

    briefing: time_of_day = time_of_day(9, 20)
    live_end: time_of_day = time_of_day(15, 50)  # no new positions after this anyway
    extended_start: time_of_day = time_of_day(4, 0)
    extended_end: time_of_day = time_of_day(20, 0)
    live_interval: float = 60.0
    extended_interval: float = 300.0
    quiet_interval: float = 1800.0

    def mode(self, ts: float) -> NewsMode:
        dt = datetime.fromtimestamp(ts, NEW_YORK)
        if dt.weekday() >= 5:
            return NewsMode.QUIET
        t = dt.time()
        if self.briefing <= t < self.live_end:
            return NewsMode.LIVE
        if self.extended_start <= t < self.extended_end:
            return NewsMode.EXTENDED
        return NewsMode.QUIET

    def sleep_seconds(self, ts: float) -> float:
        mode = self.mode(ts)
        if mode is NewsMode.LIVE:
            return self.live_interval
        interval = self.extended_interval if mode is NewsMode.EXTENDED else self.quiet_interval
        # Wake up in time for the briefing rather than oversleeping it.
        return max(1.0, min(interval, self.next_briefing(ts) - ts))

    def next_briefing(self, ts: float) -> float:
        day = datetime.fromtimestamp(ts, NEW_YORK).date()
        for offset in range(8):
            candidate = datetime.combine(day + timedelta(days=offset), self.briefing, NEW_YORK)
            if candidate.weekday() < 5 and candidate.timestamp() > ts:
                return candidate.timestamp()
        raise AssertionError("unreachable")


class EventRecorder(Protocol):
    def news(self, item: NewsItem) -> None: ...

    def signal(self, signal: Signal) -> None: ...


class SignalJournal(Protocol):
    def record_signal(self, signal: Signal, items: list[NewsItem]) -> None: ...


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
        journal: SignalJournal | None = None,
        schedule: NewsSchedule | None = None,
        buffer_max_age: float = 80 * 3600.0,
        buffer_max_items: int = 100,
    ):
        self.sources = sources
        self.analyst = analyst
        self.hub = hub
        self.symbols = symbols
        self.poll_interval = poll_interval
        self.max_item_age = max_item_age
        self._clock = clock
        self.recorder = recorder
        self.journal = journal
        self.schedule = schedule
        self.buffer_max_age = buffer_max_age  # covers Friday close to Monday open
        self.buffer_max_items = buffer_max_items
        self._buffer: dict[str, list[NewsItem]] = defaultdict(list)
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._seen_cap = 50_000
        self._sem = asyncio.Semaphore(max_concurrent_analyses)

    async def run(self) -> None:
        while True:
            now = self._clock()
            live = self.schedule is None or self.schedule.mode(now) is NewsMode.LIVE
            try:
                if live:
                    if any(self._buffer.values()):
                        await self.briefing()
                    await self.poll_once()
                else:
                    await self.poll_once(analyze=False)
            except Exception:
                log.exception("news pipeline poll failed")
            sleep = self.schedule.sleep_seconds(now) if self.schedule else self.poll_interval
            await asyncio.sleep(sleep)

    async def poll_once(self, analyze: bool = True) -> list[Signal]:
        """Fetch from every source. With analyze=False, new items are buffered for the
        next briefing instead of being analysed now."""
        results = await asyncio.gather(
            *(s.fetch(self.symbols) for s in self.sources), return_exceptions=True
        )
        new_by_symbol: dict[str, list[NewsItem]] = defaultdict(list)
        now = self._clock()
        cutoff = now - (self.max_item_age if analyze else self.buffer_max_age)
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

        if not analyze:
            for sym, items in new_by_symbol.items():
                buf = self._buffer[sym]
                buf.extend(items)
                # Keep the most recent items; very old ones age out.
                buf.sort(key=lambda i: i.published_at)
                del buf[: max(0, len(buf) - self.buffer_max_items)]
            return []
        return await self._analyze_batches(list(new_by_symbol.items()))

    async def briefing(self) -> list[Signal]:
        """Analyse everything collected since the close: one call per symbol."""
        cutoff = self._clock() - self.buffer_max_age
        batches = [
            (sym, [i for i in items if i.published_at >= cutoff])
            for sym, items in self._buffer.items()
        ]
        self._buffer.clear()
        batches = [(sym, items) for sym, items in batches if items]
        log.info("pre-open briefing: %s", {sym: len(items) for sym, items in batches})
        return await self._analyze_batches(batches, note=BRIEFING_NOTE)

    async def _analyze_batches(
        self, batches: list[tuple[str, list[NewsItem]]], note: str = ""
    ) -> list[Signal]:
        signals = await asyncio.gather(*(self._analyze(sym, items, note) for sym, items in batches))
        published = []
        for (_, items), sig in zip(batches, signals, strict=True):
            if sig is None:
                continue
            published.append(sig)
            if self.journal:
                self.journal.record_signal(sig, items)
            self.hub.publish(sig)
            if self.recorder:
                self.recorder.signal(sig)
            log.info(
                "signal %s %s [%s] score=%+.2f conf=%.2f ttl=%.0fs: %s",
                sig.symbol,
                sig.source,
                sig.model,
                sig.score,
                sig.confidence,
                sig.ttl,
                sig.rationale,
            )
        return published

    async def _analyze(self, symbol: str, items: list[NewsItem], note: str = "") -> Signal | None:
        async with self._sem:
            try:
                if note:
                    return await self.analyst.analyze(symbol, items, now=self._clock(), note=note)
                return await self.analyst.analyze(symbol, items, now=self._clock())
            except Exception:
                log.exception("analysis failed for %s", symbol)
                return None

    def _mark_seen(self, item_id: str) -> None:
        self._seen[item_id] = None
        if len(self._seen) > self._seen_cap:
            self._seen.popitem(last=False)
