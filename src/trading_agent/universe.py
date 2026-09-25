"""Which stocks may be traded, and how: rules per price tier plus volatility-scaled stops.

Two tiers, by price:
- standard ($5 and up, including smaller exchange-listed companies): normal rules; entries
  need a spread of at most MAX_SPREAD_PCT.
- penny (MIN_PRICE to $5, exchange-listed): entries only when the analyst is very
  confident (sentiment and confidence both >= 0.8 on a fresh view), half the risk per
  trade, a smaller position cap, a wider spread allowance, long only, and no
  microstructure adjustments (thin order books are easy to spoof).
Below MIN_PRICE nothing new is opened.

Stops scale with each stock's recent volatility: STOP_VOL_MULTIPLE x the standard
deviation of 1-minute returns over a 30-minute horizon, bounded per tier. Until enough
history has been seen, the tier's default stop applies. Position size then follows from
the stop so that a stop-out loses the tier's risk per trade.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .account import AccountGuard
from .models import Signal, Tick


@dataclass(frozen=True)
class TierRules:
    name: str
    risk_per_trade_pct: float
    max_position_pct: float
    max_spread_pct: float
    stop_default_pct: float
    stop_min_pct: float
    stop_max_pct: float
    # Minimum analyst conviction to open: |sentiment| and confidence of the LLM signal,
    # and how fresh it must be (1.0 = just published, 0 = about to expire).
    min_score: float = 0.0
    min_confidence: float = 0.0
    min_freshness: float = 0.0
    allow_short: bool = True
    use_microstructure: bool = True


STANDARD = TierRules(
    name="standard",
    risk_per_trade_pct=1.0,
    max_position_pct=50.0,
    max_spread_pct=0.5,
    stop_default_pct=2.0,
    stop_min_pct=1.0,
    stop_max_pct=5.0,
)

PENNY = TierRules(
    name="penny",
    risk_per_trade_pct=0.5,
    max_position_pct=10.0,
    max_spread_pct=2.0,
    stop_default_pct=8.0,
    stop_min_pct=3.0,
    stop_max_pct=15.0,
    min_score=0.8,
    min_confidence=0.8,
    min_freshness=0.5,
    allow_short=False,
    use_microstructure=False,
)


class VolatilityTracker:
    """Standard deviation of 1-minute log returns per symbol, from sampled mids."""

    def __init__(self, sample_seconds: float = 60.0, window: int = 60, min_samples: int = 15):
        self.sample_seconds = sample_seconds
        self.min_samples = min_samples
        self._samples: dict[str, deque[float]] = {}
        self._last_ts: dict[str, float] = {}
        self._window = window

    def update(self, tick: Tick) -> None:
        if not tick.valid:
            return
        last = self._last_ts.get(tick.symbol)
        if last is not None and tick.ts - last < self.sample_seconds:
            return
        self._last_ts[tick.symbol] = tick.ts
        self._samples.setdefault(tick.symbol, deque(maxlen=self._window + 1)).append(tick.mid)

    def per_minute(self, symbol: str) -> float | None:
        mids = self._samples.get(symbol)
        if not mids or len(mids) <= self.min_samples:
            return None
        rets = [math.log(b / a) for a, b in zip(mids, list(mids)[1:], strict=False) if a > 0]
        mean = sum(rets) / len(rets)
        return math.sqrt(sum((r - mean) ** 2 for r in rets) / len(rets))


@dataclass
class Universe:
    account: AccountGuard
    standard: TierRules = STANDARD
    penny: TierRules = PENNY
    penny_below: float = 5.0
    min_price: float = 1.0
    stop_vol_multiple: float = 2.0
    stop_horizon_minutes: float = 30.0
    vol: VolatilityTracker | None = None

    def __post_init__(self):
        self.vol = self.vol or VolatilityTracker()

    def tier(self, price: float) -> TierRules | None:
        if price < self.min_price:
            return None
        return self.penny if price < self.penny_below else self.standard

    def update(self, tick: Tick) -> None:
        self.vol.update(tick)

    def stop_pct(self, symbol: str, price: float) -> float:
        tier = self.tier(price) or self.penny
        sigma = self.vol.per_minute(symbol)
        if sigma is None:
            return tier.stop_default_pct
        pct = self.stop_vol_multiple * sigma * math.sqrt(self.stop_horizon_minutes) * 100
        return max(tier.stop_min_pct, min(tier.stop_max_pct, pct))

    def max_shares(self, symbol: str, price: float) -> int:
        tier = self.tier(price)
        if tier is None or price <= 0:
            return 0
        equity = self.account.equity
        by_risk = equity * tier.risk_per_trade_pct / 100 / (self.stop_pct(symbol, price) / 100)
        by_concentration = equity * tier.max_position_pct / 100
        # Never above the account-wide cap the risk manager enforces.
        cap = min(by_risk, by_concentration, self.account.max_position_value())
        return math.floor(cap / price)

    def entry_allowed(
        self, tick: Tick, short: bool, opener: tuple[Signal, float] | None
    ) -> tuple[bool, str]:
        """May exposure be added in this symbol now? `opener` is the strongest active
        LLM signal and its decay weight."""
        tier = self.tier(tick.mid)
        if tier is None:
            return False, f"price {tick.mid:.2f} below minimum {self.min_price:.2f}"
        spread_pct = (tick.ask - tick.bid) / tick.mid * 100
        if spread_pct > tier.max_spread_pct:
            return False, f"spread {spread_pct:.2f}% > {tier.max_spread_pct}% ({tier.name})"
        if short and not tier.allow_short:
            return False, f"no short selling in {tier.name} tier"
        if tier.min_score or tier.min_confidence:
            if opener is None:
                return False, f"{tier.name} tier needs a confident analyst view"
            sig, freshness = opener
            if (
                abs(sig.score) < tier.min_score
                or sig.confidence < tier.min_confidence
                or freshness < tier.min_freshness
            ):
                return False, (
                    f"{tier.name} tier needs |score| >= {tier.min_score}, confidence >= "
                    f"{tier.min_confidence}, freshness >= {tier.min_freshness}; have "
                    f"{abs(sig.score):.2f}/{sig.confidence:.2f}/{freshness:.2f}"
                )
        return True, ""
