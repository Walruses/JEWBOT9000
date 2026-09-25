"""Holds the latest signal from each source for each symbol.

Slow producers (LLM news analysis, polled every minute or so) and fast producers
(microstructure, every tick) both publish here; the strategy reads a consistent view.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..models import Signal


class SignalHub:
    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._latest: dict[str, dict[str, Signal]] = {}

    def publish(self, signal: Signal) -> None:
        self._latest.setdefault(signal.symbol, {})[signal.source] = signal

    def active(self, symbol: str) -> list[tuple[Signal, float]]:
        """Non-expired signals for symbol, each paired with its decay weight in (0, 1]."""
        now = self._clock()
        out = []
        for sig in self._latest.get(symbol, {}).values():
            age = now - sig.ts
            if 0 <= age < sig.ttl:
                out.append((sig, 1.0 - age / sig.ttl))
        return out
