"""Wires market data -> strategy -> risk -> broker, and enforces the kill switch."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from .broker.base import Broker
from .models import Fill, OrderIntent, Tick
from .risk import RiskManager
from .strategy.base import Strategy

log = logging.getLogger(__name__)


@dataclass
class WorkingOrder:
    intent: OrderIntent
    placed_at: float
    cancel_requested: bool = False


class Engine:
    def __init__(
        self,
        broker: Broker,
        strategy: Strategy,
        risk: RiskManager,
        order_ttl: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.broker = broker
        self.strategy = strategy
        self.risk = risk
        self.order_ttl = order_ttl
        self._clock = clock
        self.working: dict[str, WorkingOrder] = {}
        self._stop = asyncio.Event()
        # Set once connected and subscribed, so dependent tasks (e.g. IBKR news) can start.
        self.ready = asyncio.Event()
        self._halt_handled = False

    async def run(self) -> None:
        await self.broker.connect()
        try:
            await self.broker.subscribe(self.strategy.symbols, self.on_tick)
            self.ready.set()
            await self._stop.wait()
        finally:
            self.broker.cancel_all()
            await self.broker.disconnect()

    def stop(self) -> None:
        self._stop.set()

    def on_tick(self, tick: Tick) -> None:
        if not tick.valid:
            return
        self.risk.update_mark(tick.symbol, tick.mid)
        if self._enforce_halt():
            return
        self._cancel_stale()

        # One working order per symbol keeps behaviour simple and avoids self-crossing.
        if any(w.intent.symbol == tick.symbol for w in self.working.values()):
            return
        for intent in self.strategy.on_tick(tick, self.risk.position(tick.symbol)):
            self._submit(intent, tick.mid)

    def _submit(self, intent: OrderIntent, mid: float) -> None:
        ok, reason = self.risk.check(intent, mid)
        if not ok:
            log.debug("rejected %s: %s", intent, reason)
            return
        self.risk.on_submit(intent)
        closed = False

        def on_done(order_id: str, unfilled_qty: int) -> None:
            nonlocal closed
            closed = True
            self.working.pop(order_id, None)
            self.risk.on_order_closed(intent.symbol, intent.side, unfilled_qty)

        try:
            oid = self.broker.place_limit(intent, self._on_fill, on_done)
        except Exception:
            log.exception("order placement failed: %s", intent)
            self.risk.on_order_closed(intent.symbol, intent.side, intent.qty)
            return
        # The simulator may complete an order synchronously inside place_limit.
        if not closed:
            self.working[oid] = WorkingOrder(intent, self._clock())

    def _on_fill(self, fill: Fill) -> None:
        log.info("fill %s %s %d @ %.4f", fill.symbol, fill.side.value, fill.qty, fill.price)
        self.risk.on_fill(fill)
        self.strategy.on_fill(fill)
        self._enforce_halt()

    def _cancel_stale(self) -> None:
        now = self._clock()
        for oid, w in list(self.working.items()):
            if not w.cancel_requested and now - w.placed_at >= self.order_ttl:
                w.cancel_requested = True
                self.broker.cancel(oid)

    def _enforce_halt(self) -> bool:
        if not self.risk.halted:
            return False
        if not self._halt_handled:
            self._halt_handled = True
            log.critical("risk halt -> cancelling all orders: %s", self.risk.halt_reason)
            self.broker.cancel_all()
        return True
