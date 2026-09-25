"""In-process simulated broker for tests and dry runs. No network, no real money.

Fills are conservative: a resting buy only fills once the ask trades through its limit
(and vice versa), which ignores queue position but never flatters the strategy.
"""

from __future__ import annotations

import asyncio
import itertools
import random
import time
from dataclasses import dataclass

from ..models import Fill, OrderIntent, Side, Tick
from .base import DoneCallback, FillCallback, TickCallback


@dataclass
class _SimOrder:
    intent: OrderIntent
    on_fill: FillCallback
    on_done: DoneCallback


class SimBroker:
    def __init__(
        self,
        synthetic_feed: bool = False,
        tick_interval: float = 0.01,
        seed: int = 0,
        commission_per_share: float = 0.0,
        min_commission: float = 0.0,
    ):
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.fees_paid = 0.0
        self.fill_count = 0
        self.shares_traded = 0
        self._positions: dict[str, int] = {}
        self._avg: dict[str, float] = {}
        self._orders: dict[str, _SimOrder] = {}
        self._ids = itertools.count(1)
        self._on_tick: TickCallback | None = None
        self._synthetic = synthetic_feed
        self._tick_interval = tick_interval
        self._rng = random.Random(seed)
        self._feed_task: asyncio.Task | None = None

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        if self._feed_task:
            self._feed_task.cancel()

    async def positions(self) -> dict[str, tuple[int, float]]:
        return {s: (q, self._avg.get(s, 0.0)) for s, q in self._positions.items() if q}

    def is_connected(self) -> bool:
        return True

    async def reconnect(self) -> None:
        pass

    async def resubscribe(self) -> None:
        pass

    async def account(self) -> None:
        return None  # the engine falls back to starting equity + PnL

    async def subscribe(self, symbols: list[str], on_tick: TickCallback) -> None:
        self._on_tick = on_tick
        if self._synthetic:
            self._feed_task = asyncio.create_task(self._random_walk(symbols))

    def place_limit(self, intent: OrderIntent, on_fill: FillCallback, on_done: DoneCallback) -> str:
        oid = f"sim-{next(self._ids)}"
        self._orders[oid] = _SimOrder(intent, on_fill, on_done)
        return oid

    def cancel(self, order_id: str) -> None:
        order = self._orders.pop(order_id, None)
        if order:
            order.on_done(order_id, order.intent.qty)

    def cancel_all(self) -> None:
        for oid in list(self._orders):
            self.cancel(oid)

    @property
    def open_orders(self) -> dict[str, OrderIntent]:
        return {oid: o.intent for oid, o in self._orders.items()}

    def push_tick(self, tick: Tick) -> None:
        """Match resting orders against the new quote, then deliver it to the subscriber."""
        for oid, order in list(self._orders.items()):
            it = order.intent
            if it.symbol != tick.symbol:
                continue
            crossed = (it.side is Side.BUY and tick.ask <= it.limit_price) or (
                it.side is Side.SELL and tick.bid >= it.limit_price
            )
            if crossed:
                del self._orders[oid]
                fee = max(self.min_commission, self.commission_per_share * it.qty)
                self._book(it)
                self.fees_paid += fee
                self.fill_count += 1
                self.shares_traded += it.qty
                order.on_fill(Fill(oid, it.symbol, it.side, it.qty, it.limit_price, tick.ts, fee))
                order.on_done(oid, 0)
        if self._on_tick:
            self._on_tick(tick)

    def _book(self, it: OrderIntent) -> None:
        old = self._positions.get(it.symbol, 0)
        new = old + it.side.sign * it.qty
        if new == 0:
            self._avg[it.symbol] = 0.0
        elif old == 0 or (old > 0) != (new > 0):
            self._avg[it.symbol] = it.limit_price
        elif abs(new) > abs(old):
            avg = self._avg.get(it.symbol, 0.0)
            self._avg[it.symbol] = (avg * abs(old) + it.limit_price * it.qty) / abs(new)
        self._positions[it.symbol] = new

    async def _random_walk(self, symbols: list[str]) -> None:
        mids = {s: 100.0 for s in symbols}
        while True:
            for s in symbols:
                mids[s] = max(0.01, mids[s] + self._rng.gauss(0, 0.02))
                bid, ask = round(mids[s] - 0.01, 2), round(mids[s] + 0.01, 2)
                sizes = self._rng.randint(1, 20) * 100, self._rng.randint(1, 20) * 100
                self.push_tick(Tick(s, bid, ask, mids[s], time.time(), *sizes))
            await asyncio.sleep(self._tick_interval)
