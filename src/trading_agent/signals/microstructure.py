"""Fast quantitative signals computed on every tick."""

from __future__ import annotations

import math
from collections import deque

from ..models import Signal, Tick


class MicrostructureSignals:
    def __init__(self, window: int = 100, ttl: float = 5.0):
        self.window = window
        self.ttl = ttl
        self._mids: dict[str, deque[float]] = {}

    def update(self, tick: Tick) -> list[Signal]:
        out = []
        depth = tick.bid_size + tick.ask_size
        if depth > 0:
            # Top-of-book imbalance: more resting bids than offers tends to precede upticks.
            score = (tick.bid_size - tick.ask_size) / depth
            out.append(Signal(tick.symbol, "micro:imbalance", score, 0.5, tick.ts, self.ttl))

        mids = self._mids.setdefault(tick.symbol, deque(maxlen=self.window))
        mids.append(tick.mid)
        z = _zscore(mids) if len(mids) == self.window else None
        if z is not None:
            # Short-horizon mean reversion: fade stretched moves.
            score = max(-1.0, min(1.0, -z / 3))
            out.append(Signal(tick.symbol, "micro:reversion", score, 0.5, tick.ts, self.ttl))
        return out


def _zscore(values: deque[float]) -> float | None:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    if var <= 0:
        return None
    return (values[-1] - mean) / math.sqrt(var)
