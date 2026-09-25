"""Account-level rules: risk-based position sizing, the pattern day trader rule, and
cash-account settlement.

Sizing: every position gets a protective stop STOP_LOSS_PCT from its entry. A position
is sized so that hitting that stop loses at most RISK_PER_TRADE_PCT of equity, and is
further capped at MAX_POSITION_PCT of equity (and all positions together at
MAX_GROSS_EXPOSURE_PCT). With $16,000, 1% risk and a 2% stop: $160 at risk, up to an
$8,000 position.

Pattern day trader rule (FINRA): a margin account under $25,000 may make at most 3 day
trades in any 5 business days. This agent closes every position the same day, so each
position it opens becomes a day trade; new positions are refused once the limit would be
exceeded. IBKR's own count is used when it reports one.

Cash accounts: no day-trade limit, but no short selling, and only settled cash can be
used. Sale proceeds settle the next business day, so the day's total purchases are
capped at the settled cash available at the start of the day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

PDT_EQUITY_THRESHOLD = 25_000.0
PDT_MAX_DAY_TRADES = 3
PDT_WINDOW_BUSINESS_DAYS = 5


@dataclass(frozen=True)
class AccountConfig:
    account_type: str = "margin"  # "margin" or "cash"
    starting_equity: float = 16_000.0  # used when the broker doesn't report equity
    risk_per_trade_pct: float = 1.0
    stop_loss_pct: float = 2.0
    max_position_pct: float = 50.0
    max_gross_exposure_pct: float = 100.0
    stop_cooldown_minutes: float = 60.0

    @property
    def is_cash(self) -> bool:
        return self.account_type == "cash"


@dataclass(frozen=True)
class AccountInfo:
    """What the broker reports; any field may be unknown."""

    equity: float | None = None
    settled_cash: float | None = None
    day_trades_remaining: int | None = None  # None = not reported; -1 = unlimited


@dataclass
class AccountGuard:
    config: AccountConfig
    equity: float = 0.0
    settled_cash_at_open: float = 0.0
    broker_day_trades_remaining: int | None = None
    # Dates on which this agent opened a position (each becomes a day trade).
    day_trade_dates: list[date] = field(default_factory=list)
    bought_today: float = 0.0
    _today: date | None = None

    def __post_init__(self):
        if not self.equity:
            self.equity = self.config.starting_equity
        if not self.settled_cash_at_open:
            self.settled_cash_at_open = self.equity

    # ---- sizing ----------------------------------------------------------------------

    @property
    def risk_budget(self) -> float:
        return self.equity * self.config.risk_per_trade_pct / 100

    def max_position_value(self) -> float:
        by_risk = self.risk_budget / (self.config.stop_loss_pct / 100)
        by_concentration = self.equity * self.config.max_position_pct / 100
        return min(by_risk, by_concentration)

    def max_shares(self, price: float) -> int:
        if price <= 0:
            return 0
        return math.floor(self.max_position_value() / price)

    @property
    def max_gross_exposure(self) -> float:
        return self.equity * self.config.max_gross_exposure_pct / 100

    # ---- day rollover / broker updates -----------------------------------------------

    def start_day(self, today: date, settled_cash: float | None = None) -> None:
        if today == self._today:
            return
        self._today = today
        self.bought_today = 0.0
        self.settled_cash_at_open = settled_cash if settled_cash is not None else self.equity

    def update(self, info: AccountInfo) -> None:
        if info.equity:
            self.equity = info.equity
        if info.day_trades_remaining is not None:
            self.broker_day_trades_remaining = info.day_trades_remaining

    # ---- pattern day trader ----------------------------------------------------------

    @property
    def pdt_applies(self) -> bool:
        return not self.config.is_cash and self.equity < PDT_EQUITY_THRESHOLD

    def day_trades_used(self, today: date) -> int:
        window = business_days_back(today, PDT_WINDOW_BUSINESS_DAYS)
        return sum(1 for d in self.day_trade_dates if window <= d <= today)

    def day_trades_remaining(self, today: date) -> int | None:
        """None = unlimited."""
        if not self.pdt_applies:
            return None
        ours = PDT_MAX_DAY_TRADES - self.day_trades_used(today)
        broker = self.broker_day_trades_remaining
        if broker is not None and broker >= 0:
            ours = min(ours, broker)
        return max(0, ours)

    def record_open(self, today: date) -> None:
        self.day_trade_dates.append(today)
        cutoff = business_days_back(today, PDT_WINDOW_BUSINESS_DAYS)
        self.day_trade_dates = [d for d in self.day_trade_dates if d >= cutoff]
        if self.broker_day_trades_remaining is not None and self.broker_day_trades_remaining > 0:
            self.broker_day_trades_remaining -= 1  # until the broker's next update

    # ---- checks ----------------------------------------------------------------------

    def can_open(self, today: date) -> tuple[bool, str]:
        remaining = self.day_trades_remaining(today)
        if remaining is not None and remaining <= 0:
            return False, (
                f"pattern day trader limit: {PDT_MAX_DAY_TRADES} day trades per "
                f"{PDT_WINDOW_BUSINESS_DAYS} business days used (equity < $25k, margin)"
            )
        return True, ""

    def can_buy(self, notional: float) -> tuple[bool, str]:
        if self.config.is_cash and self.bought_today + notional > self.settled_cash_at_open:
            return False, "cash account: purchase would exceed today's settled cash"
        return True, ""

    def record_buy(self, notional: float) -> None:
        self.bought_today += notional


def business_days_back(today: date, n: int) -> date:
    """The earliest date of the n-business-day window ending today. Weekends are skipped
    but exchange holidays aren't, so across a holiday this window is a day short of
    FINRA's; IBKR's own DayTradesRemaining, used when reported, covers that case."""
    d, counted = today, 1
    while counted < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            counted += 1
    return d
