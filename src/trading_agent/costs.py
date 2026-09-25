"""Trading cost model, used to skip trades whose expected gain doesn't cover their cost.

Defaults are IBKR Pro "Tiered" US stock commissions: $0.0035/share, $0.35 minimum, capped
at 1% of trade value. Exchange, clearing and regulatory fees (and liquidity rebates for
passive orders) come on top; `extra_per_share` approximates them. Check IBKR's current
schedule for your account.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    per_share: float = 0.0035
    minimum: float = 0.35
    max_pct_of_value: float = 0.01
    extra_per_share: float = 0.0

    def commission(self, qty: int, price: float) -> float:
        if qty <= 0:
            return 0.0
        fee = max(self.minimum, self.per_share * qty)
        fee = min(fee, self.max_pct_of_value * qty * price)
        return fee + self.extra_per_share * qty

    def round_trip(self, qty: int, price: float) -> float:
        """Commission to open qty now and close it later."""
        return 2 * self.commission(qty, price)
