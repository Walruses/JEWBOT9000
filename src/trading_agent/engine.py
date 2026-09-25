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

from .account import AccountGuard, AccountInfo
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
    opening: bool = False  # opens (or flips) a position: will become a day trade


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
        account: AccountGuard | None = None,
    ):
        self.broker = broker
        self.strategy = strategy
        self.risk = risk
        self.order_ttl = order_ttl
        self.session = session
        self.state_store = state_store
        self.recorder = recorder
        self.journal = journal
        self.account = account
        risk.account = account
        self._stop_cooldown_until: dict[str, float] = {}
        self._equity_at_open = account.config.starting_equity if account else 0.0
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
        info = await self._refresh_account()
        if self.account:
            self.account.start_day(self._day, info.settled_cash if info else None)
        if self.state_store:
            st = self.state_store.load(self._day)
            self.risk.restore(st.realized_pnl, st.halted, st.halt_reason)
            if self.account:
                self.account.day_trade_dates = [date.fromisoformat(d) for d in st.day_trade_dates]
            log.info(
                "restored state for %s: realized=%.2f halted=%s",
                st.date,
                st.realized_pnl,
                st.halted,
            )
            self._save_state()

    async def _refresh_account(self) -> AccountInfo | None:
        if not self.account:
            return None
        try:
            info = await self.broker.account()
        except Exception:
            log.exception("account refresh failed")
            info = None
        if info and info.equity:
            self.account.update(info)
        else:
            # Simulator / no report: equity at the start of the day plus today's PnL.
            self.account.equity = self._equity_at_open + self.risk.total_pnl()
        return info

    async def reconcile(self) -> None:
        """Compare our positions with the broker's. A mismatch that persists across two
        checks with no working orders means our view is wrong: halt rather than trade on it."""
        await self._refresh_account()
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
        if phase is not Phase.CLOSED and self._check_stop(tick):
            return

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

    def _check_stop(self, tick: Tick) -> bool:
        """Close a position whose price has moved STOP_LOSS_PCT against its average entry.
        Returns True if the stop is active (nothing else should trade this symbol now)."""
        if not self.account:
            return False
        qty = self.risk.position(tick.symbol)
        if qty == 0:
            return False
        avg = self.risk.avg_price(tick.symbol)
        stop_frac = self.account.config.stop_loss_pct / 100
        hit = tick.mid <= avg * (1 - stop_frac) if qty > 0 else tick.mid >= avg * (1 + stop_frac)
        if not hit:
            return False
        cooldown = self.account.config.stop_cooldown_minutes * 60
        if tick.symbol not in self._stop_cooldown_until or not self._has_working(tick.symbol):
            log.warning("stop loss %s: %+d @ avg %.4f, mid %.4f", tick.symbol, qty, avg, tick.mid)
        self._stop_cooldown_until[tick.symbol] = self._clock() + cooldown
        for oid, w in list(self.working.items()):
            if w.intent.symbol == tick.symbol and not w.cancel_requested:
                w.cancel_requested = True
                self.broker.cancel(oid)
        if not self._has_working(tick.symbol):
            self._flatten(tick, reason="stop_loss")
        return True

    def _flatten(self, tick: Tick, reason: str | None = None) -> None:
        qty = self.risk.position(tick.symbol)
        if qty == 0:
            return
        # Marketable limit at the touch: crosses the spread but can't run through the book.
        if qty > 0:
            intent = OrderIntent(tick.symbol, Side.SELL, qty, tick.bid)
        else:
            intent = OrderIntent(tick.symbol, Side.BUY, -qty, tick.ask)
        log.info("flattening %s: %s %d", tick.symbol, intent.side.value, intent.qty)
        if reason is None:
            reason = "halt_flatten" if self.risk.halted else "eod_flatten"
        self._submit(intent, tick.mid, reason)

    # ---- orders ----------------------------------------------------------------------

    def _opens(self, intent: OrderIntent) -> bool:
        """Does this order start a new position (from flat, or by flipping)?"""
        pos = self.risk.position(intent.symbol)
        after = pos + intent.side.sign * intent.qty
        return after != 0 and (pos == 0 or (pos > 0) != (after > 0))

    def _account_check(self, intent: OrderIntent, opening: bool) -> tuple[bool, str]:
        if opening and self._clock() < self._stop_cooldown_until.get(intent.symbol, 0.0):
            return False, "stop-loss cooldown"
        if not self.account or self._day is None:
            return True, ""
        if opening:
            pending = sum(1 for w in self.working.values() if w.opening)
            remaining = self.account.day_trades_remaining(self._day)
            if remaining is not None and remaining - pending <= 0:
                ok, why = self.account.can_open(self._day)
                return False, why or "pattern day trader limit (orders in flight)"
        if intent.side is Side.BUY:
            pending_buys = sum(
                w.intent.qty * w.intent.limit_price
                for w in self.working.values()
                if w.intent.side is Side.BUY
            )
            return self.account.can_buy(intent.qty * intent.limit_price + pending_buys)
        return True, ""

    def _submit(self, intent: OrderIntent, mid: float, reason: str) -> None:
        ok, why = self.risk.check(intent, mid)
        opening = self._opens(intent)
        if ok and reason == "strategy":
            ok, why = self._account_check(intent, opening)
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
            self._track_account(fill)
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
            self.working[oid] = WorkingOrder(intent, self._clock(), opening=opening)

    def _track_account(self, fill: Fill) -> None:
        """Record day trades (positions opened) and purchases against settled cash."""
        if not self.account or not fill.qty or self._day is None:
            return
        before = self.risk.position(fill.symbol)
        after = before + fill.side.sign * fill.qty
        if after != 0 and (before == 0 or (before > 0) != (after > 0)):
            self.account.record_open(self._day)
            remaining = self.account.day_trades_remaining(self._day)
            if remaining is not None:
                log.info("opened %s: %d day trade(s) left this window", fill.symbol, remaining)
        if fill.side is Side.BUY:
            self.account.record_buy(fill.qty * fill.price)

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
            if self.account:
                # Without a broker report, carry the day's result into equity.
                self._equity_at_open += self.risk.realized_pnl
                self.account.equity = self._equity_at_open
            self.risk.reset_day()
            if self.account:
                self.account.start_day(today)
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
                    [d.isoformat() for d in self.account.day_trade_dates] if self.account else [],
                )
            )
        except OSError:
            log.exception("failed to save state")
