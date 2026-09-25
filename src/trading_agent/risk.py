"""Pre-trade risk checks, position/PnL tracking and the kill switch.

Every order passes through RiskManager.check() before reaching the broker. Worst-case
exposure counts working orders as if they will fully fill, so a burst of orders cannot
stack past the position limit.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .account import AccountGuard
from .config import RiskLimits
from .models import Fill, OrderIntent, Side

log = logging.getLogger(__name__)


@dataclass
class Position:
    qty: int = 0
    avg_price: float = 0.0
    pending_buy: int = 0
    pending_sell: int = 0


class RiskManager:
    def __init__(self, limits: RiskLimits, clock: Callable[[], float] = time.monotonic):
        self.limits = limits
        self._clock = clock
        self._positions: dict[str, Position] = {}
        self._marks: dict[str, float] = {}
        self._order_times: deque[float] = deque()
        self.realized_pnl = 0.0
        self.halted = False
        self.halt_reason = ""
        # Account-level caps (sizing, exposure, cash-account rules); optional.
        self.account: AccountGuard | None = None
        self.allow_flatten = False

    def _pos(self, symbol: str) -> Position:
        return self._positions.setdefault(symbol, Position())

    def position(self, symbol: str) -> int:
        return self._pos(symbol).qty

    def avg_price(self, symbol: str) -> float:
        return self._pos(symbol).avg_price

    def positions(self) -> dict[str, int]:
        return {sym: p.qty for sym, p in self._positions.items() if p.qty}

    def set_position(self, symbol: str, qty: int, avg_price: float) -> None:
        """Overwrite a position from the broker (startup reconciliation)."""
        p = self._pos(symbol)
        p.qty = qty
        p.avg_price = avg_price if qty else 0.0

    def restore(self, realized_pnl: float, halted: bool, halt_reason: str) -> None:
        """Restore today's state after a restart so the daily-loss limit can't be reset
        by restarting the process."""
        self.realized_pnl = realized_pnl
        if halted:
            # Flattening is allowed after a restart: positions were just loaded from the broker.
            self.halt(halt_reason or "halted before restart", allow_flatten=True)
        self._check_loss()

    def reset_day(self) -> None:
        """New trading day: realized PnL starts from zero. A halt stays in force."""
        self.realized_pnl = 0.0

    def halt(self, reason: str, allow_flatten: bool = False) -> None:
        """Stop trading. With allow_flatten, orders that only reduce positions toward flat
        are still accepted (used for the loss limit; a halt caused by an untrustworthy
        position view must not trade at all)."""
        if not self.halted:
            log.critical("TRADING HALTED: %s", reason)
            self.allow_flatten = allow_flatten
        self.halted = True
        self.halt_reason = reason

    def check(
        self,
        intent: OrderIntent,
        mid: float,
        bid: float | None = None,
        ask: float | None = None,
    ) -> tuple[bool, str]:
        """Pre-trade checks. With the current bid/ask, the fat-finger check measures the
        limit price against the quote rather than the mid: an exit at the bid (or ask) of
        a wide-spread stock is legitimate, and must never be refused for being "far from
        mid"."""
        lim = self.limits
        if self.halted and not (self.allow_flatten and self._reduces(intent)):
            return False, f"halted: {self.halt_reason}"
        if intent.qty <= 0:
            return False, "non-positive quantity"
        if intent.qty > lim.max_order_qty:
            return False, f"qty {intent.qty} > max_order_qty {lim.max_order_qty}"
        if intent.limit_price <= 0 or mid <= 0:
            return False, "invalid price"
        if intent.qty * intent.limit_price > lim.max_order_notional:
            return False, "order notional exceeds limit"
        band = lim.max_price_deviation_bps / 10_000
        if bid and ask and 0 < bid <= ask:
            if not bid * (1 - band) <= intent.limit_price <= ask * (1 + band):
                return False, (
                    f"limit price {intent.limit_price} outside quote {bid}/{ask} "
                    f"+/-{lim.max_price_deviation_bps:.0f}bps"
                )
        else:
            deviation_bps = abs(intent.limit_price - mid) / mid * 10_000
            if deviation_bps > lim.max_price_deviation_bps:
                return False, f"limit price {deviation_bps:.1f}bps from mid"

        p = self._pos(intent.symbol)
        if intent.side is Side.BUY:
            worst = p.qty + p.pending_buy + intent.qty
        else:
            worst = p.qty - p.pending_sell - intent.qty
        # Orders that shrink an oversized position (e.g. one inherited at startup) are allowed.
        if abs(worst) > lim.max_position and abs(worst) >= abs(p.qty):
            return False, f"worst-case position {worst} exceeds {lim.max_position}"
        ok, why = self._check_account(intent, p, worst, mid)
        if not ok:
            return False, why

        now = self._clock()
        while self._order_times and now - self._order_times[0] >= 1.0:
            self._order_times.popleft()
        if len(self._order_times) >= lim.max_orders_per_sec:
            return False, "order rate limit"
        return True, ""

    def _check_account(
        self, intent: OrderIntent, p: Position, worst: int, mid: float
    ) -> tuple[bool, str]:
        """Account-level caps: risk per trade, concentration, gross exposure, no shorts
        in a cash account. Only orders that increase exposure are capped."""
        acct = self.account
        if acct is None:
            return True, ""
        if acct.config.is_cash and worst < 0:
            return False, "cash account: short selling not allowed"
        if abs(worst) <= abs(p.qty):
            return True, ""
        cap = acct.max_position_value()
        if abs(worst) * mid > cap * 1.02:  # small tolerance for price moves since sizing
            return False, (
                f"position value ${abs(worst) * mid:,.0f} exceeds ${cap:,.0f} "
                f"(risk per trade / concentration cap)"
            )
        gross = abs(worst) * mid
        for sym, other in self._positions.items():
            if sym == intent.symbol:
                continue
            exposure = max(abs(other.qty + other.pending_buy), abs(other.qty - other.pending_sell))
            gross += exposure * self._marks.get(sym, other.avg_price)
        if gross > acct.max_gross_exposure * 1.001:
            return False, f"gross exposure ${gross:,.0f} exceeds ${acct.max_gross_exposure:,.0f}"
        return True, ""

    def _reduces(self, intent: OrderIntent) -> bool:
        p = self._pos(intent.symbol)
        if intent.side is Side.SELL:
            return p.qty > 0 and p.qty - p.pending_sell - intent.qty >= 0
        return p.qty < 0 and p.qty + p.pending_buy + intent.qty <= 0

    def on_submit(self, intent: OrderIntent) -> None:
        self._order_times.append(self._clock())
        p = self._pos(intent.symbol)
        if intent.side is Side.BUY:
            p.pending_buy += intent.qty
        else:
            p.pending_sell += intent.qty

    def on_order_closed(self, symbol: str, side: Side, unfilled_qty: int) -> None:
        """Release exposure reserved for the unfilled remainder of a cancelled/done order."""
        self._release_pending(self._pos(symbol), side, unfilled_qty)

    def on_fill(self, fill: Fill) -> None:
        self.realized_pnl -= fill.commission
        if fill.qty == 0:  # commission-only report
            self._check_loss()
            return
        p = self._pos(fill.symbol)
        self._release_pending(p, fill.side, fill.qty)

        signed = fill.side.sign * fill.qty
        if p.qty == 0 or (p.qty > 0) == (signed > 0):
            total = abs(p.qty) + fill.qty
            p.avg_price = (p.avg_price * abs(p.qty) + fill.price * fill.qty) / total
            p.qty += signed
        else:
            closing = min(abs(p.qty), fill.qty)
            direction = 1 if p.qty > 0 else -1
            self.realized_pnl += closing * (fill.price - p.avg_price) * direction
            p.qty += signed
            if p.qty == 0:
                p.avg_price = 0.0
            elif fill.qty > closing:  # flipped through flat
                p.avg_price = fill.price
        self._check_loss()

    def update_mark(self, symbol: str, mid: float) -> None:
        self._marks[symbol] = mid
        self._check_loss()

    def unrealized_pnl(self) -> float:
        total = 0.0
        for sym, p in self._positions.items():
            mark = self._marks.get(sym)
            if p.qty and mark is not None:
                total += p.qty * (mark - p.avg_price)
        return total

    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl()

    def _check_loss(self) -> None:
        if self.total_pnl() <= -self.limits.max_daily_loss:
            self.halt(f"daily loss limit hit (pnl={self.total_pnl():.2f})", allow_flatten=True)

    @staticmethod
    def _release_pending(p: Position, side: Side, qty: int) -> None:
        if side is Side.BUY:
            p.pending_buy = max(0, p.pending_buy - qty)
        else:
            p.pending_sell = max(0, p.pending_sell - qty)
