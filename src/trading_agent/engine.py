"""Wires market data -> strategy -> risk -> broker, and enforces the kill switch.

Also owns everything that has to survive the real world: reconciling positions with the
broker on startup and while running, persisting the day's PnL/halt state, and working
every position back to flat before the close.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Protocol

from .broker.base import Broker
from .models import Fill, OrderIntent, Side, Tick
from .risk import RiskManager
from .session import Phase, TradingSession
from .state import DailyState, StateStore
from .strategy.base import Strategy

log = logging.getLogger(__name__)


class TickRecorder(Protocol):
    def tick(self, tick: Tick) -> None: ...


class TradeJournal(Protocol):
    def record_decision(
        self, intent: OrderIntent, mid: float, reason: str, context: dict
    ) -> int: ...

    def on_fill(self, fill: Fill, decision_id: int | None) -> None: ...

    def on_tick(self, tick: Tick) -> None: ...

    def seed_position(self, symbol: str, qty: int, avg_price: float) -> None: ...


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
        clock: Callable[[], float] = time.time,
        session: TradingSession | None = None,
        state_store: StateStore | None = None,
        recorder: TickRecorder | None = None,
        reconcile_interval: float = 15.0,
        journal: TradeJournal | None = None,
    ):
        self.broker = broker
        self.strategy = strategy
        self.risk = risk
        self.order_ttl = order_ttl
        self.session = session
        self.state_store = state_store
        self.recorder = recorder
        self.journal = journal
        self.reconcile_interval = reconcile_interval
        self._clock = clock
        self.working: dict[str, WorkingOrder] = {}
        self._stop = asyncio.Event()
        # Set once connected and subscribed, so dependent tasks (e.g. IBKR news) can start.
        self.ready = asyncio.Event()
        self._halt_handled = False
        self._day: date | None = None
        self._mismatch_strikes: dict[str, int] = {}

    # ---- lifecycle -------------------------------------------------------------------

    async def run(self) -> None:
        await self.broker.connect()
        try:
            await self.startup()
            await self.broker.subscribe(self.strategy.symbols, self.on_tick)
            self.ready.set()
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._stop.wait(), self.reconcile_interval)
                except TimeoutError:
                    await self.reconcile()
        finally:
            self.broker.cancel_all()
            self._save_state()
            await self.broker.disconnect()

    def stop(self) -> None:
        self._stop.set()

    async def startup(self) -> None:
        """Bring local state in line with the broker and with today's saved state."""
        # Orders left working by a previous (crashed) run would otherwise fill untracked.
        self.broker.cancel_all()

        held = await self.broker.positions()
        for sym in self.strategy.symbols:
            qty, avg = held.get(sym, (0, 0.0))
            self.risk.set_position(sym, qty, avg)
            if qty and self.journal:
                self.journal.seed_position(sym, qty, avg)
            if qty:
                log.warning("starting with existing position %s %+d @ %.4f", sym, qty, avg)
        others = sorted(set(held) - set(self.strategy.symbols))
        if others:
            log.info("ignoring positions in symbols not traded by this agent: %s", others)

        self._day = self._trading_date(self._clock())
        if self.state_store:
            st = self.state_store.load(self._day)
            self.risk.restore(st.realized_pnl, st.halted, st.halt_reason)
            log.info(
                "restored state for %s: realized=%.2f halted=%s",
                st.date,
                st.realized_pnl,
                st.halted,
            )
            self._save_state()

    async def reconcile(self) -> None:
        """Compare our positions with the broker's. A mismatch that persists across two
        checks with no working orders means our view is wrong: halt rather than trade on it."""
        try:
            held = await self.broker.positions()
        except Exception:
            log.exception("position reconciliation failed")
            return
        for sym in self.strategy.symbols:
            if self._has_working(sym):
                self._mismatch_strikes.pop(sym, None)
                continue
            ours, theirs = self.risk.position(sym), held.get(sym, (0, 0.0))[0]
            if ours == theirs:
                self._mismatch_strikes.pop(sym, None)
                continue
            strikes = self._mismatch_strikes.get(sym, 0) + 1
            self._mismatch_strikes[sym] = strikes
            log.warning(
                "position mismatch %s: ours=%d broker=%d (strike %d)", sym, ours, theirs, strikes
            )
            if strikes >= 2:
                self.risk.halt(f"position mismatch on {sym}: ours={ours} broker={theirs}")
                self._enforce_halt()

    # ---- per tick --------------------------------------------------------------------

    def on_tick(self, tick: Tick) -> None:
        if self.recorder:
            self.recorder.tick(tick)
        if self.journal:
            self.journal.on_tick(tick)
        if not tick.valid:
            return
        self._roll_day()
        self.risk.update_mark(tick.symbol, tick.mid)
        phase = self.session.phase(self._clock()) if self.session else Phase.TRADING
        if self._enforce_halt():
            self._cancel_stale()
            can_flatten = self.risk.allow_flatten and phase is not Phase.CLOSED
            if can_flatten and not self._has_working(tick.symbol):
                self._flatten(tick)
            return
        self._cancel_stale()

        if phase is Phase.CLOSED:
            return
        # One working order per symbol keeps behaviour simple and avoids self-crossing.
        if self._has_working(tick.symbol):
            return
        if phase is Phase.FLATTEN:
            self._flatten(tick)
            return
        for intent in self.strategy.on_tick(tick, self.risk.position(tick.symbol)):
            self._submit(intent, tick.mid, "strategy")

    def _flatten(self, tick: Tick) -> None:
        qty = self.risk.position(tick.symbol)
        if qty == 0:
            return
        # Marketable limit at the touch: crosses the spread but can't run through the book.
        if qty > 0:
            intent = OrderIntent(tick.symbol, Side.SELL, qty, tick.bid)
        else:
            intent = OrderIntent(tick.symbol, Side.BUY, -qty, tick.ask)
        log.info("flattening %s: %s %d", tick.symbol, intent.side.value, intent.qty)
        self._submit(intent, tick.mid, "halt_flatten" if self.risk.halted else "eod_flatten")

    # ---- orders ----------------------------------------------------------------------

    def _submit(self, intent: OrderIntent, mid: float, reason: str) -> None:
        ok, why = self.risk.check(intent, mid)
        if not ok:
            log.debug("rejected %s: %s", intent, why)
            return
        self.risk.on_submit(intent)
        closed = False
        decision_id = None
        if self.journal:
            context = self.strategy.explain(intent.symbol) if reason == "strategy" else {}
            decision_id = self.journal.record_decision(intent, mid, reason, context)

        def on_fill(fill: Fill) -> None:
            if self.journal:
                self.journal.on_fill(fill, decision_id)
            self._on_fill(fill)

        def on_done(order_id: str, unfilled_qty: int) -> None:
            nonlocal closed
            closed = True
            self.working.pop(order_id, None)
            self.risk.on_order_closed(intent.symbol, intent.side, unfilled_qty)

        try:
            oid = self.broker.place_limit(intent, on_fill, on_done)
        except Exception:
            log.exception("order placement failed: %s", intent)
            self.risk.on_order_closed(intent.symbol, intent.side, intent.qty)
            return
        # The simulator may complete an order synchronously inside place_limit.
        if not closed:
            self.working[oid] = WorkingOrder(intent, self._clock())

    def _on_fill(self, fill: Fill) -> None:
        if fill.qty:
            log.info("fill %s %s %d @ %.4f", fill.symbol, fill.side.value, fill.qty, fill.price)
        self.risk.on_fill(fill)
        self.strategy.on_fill(fill)
        self._save_state()
        self._enforce_halt()

    def _has_working(self, symbol: str) -> bool:
        return any(w.intent.symbol == symbol for w in self.working.values())

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
            action = "flattening" if self.risk.allow_flatten else "no further orders"
            log.critical(
                "risk halt -> cancelling all orders, %s: %s", action, self.risk.halt_reason
            )
            self.broker.cancel_all()
            self._save_state()
        return True

    # ---- day / state -----------------------------------------------------------------

    def _trading_date(self, ts: float) -> date:
        if self.session:
            return self.session.trading_date(ts)
        return datetime.fromtimestamp(ts, UTC).date()

    def _roll_day(self) -> None:
        today = self._trading_date(self._clock())
        if self._day is not None and today != self._day:
            log.info("new trading day %s (previous realized=%.2f)", today, self.risk.realized_pnl)
            self.risk.reset_day()
            self._day = today
            self._save_state()
        self._day = today

    def _save_state(self) -> None:
        if not self.state_store or self._day is None:
            return
        try:
            self.state_store.save(
                DailyState(
                    self._day.isoformat(),
                    self.risk.realized_pnl,
                    self.risk.halted,
                    self.risk.halt_reason,
                )
            )
        except OSError:
            log.exception("failed to save state")
