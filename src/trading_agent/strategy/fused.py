"""Trades each symbol toward the target position implied by all active signals.

Cost-aware: an order that adds exposure is only sent when its expected gain covers the
round-trip commission by a safety multiple. Expected gain is modelled as
    shares x price x |conviction| x edge_bps_at_full_conviction / 10,000
where edge_bps_at_full_conviction is the average move a full-conviction view has been
worth (the journal report estimates it from real signal outcomes). Orders that reduce
exposure are never blocked: those costs are already committed.

Anti-churn: a position needs an LLM view strong enough on its own; microstructure
signals may then enlarge or speed up the entry, but reductions
follow the LLM view alone, and only once it has fallen to half the position or less
(going straight to flat when the remainder would be small). Otherwise every wiggle in
the order book, and every step of a decaying view, would cost a commission.
"""

from __future__ import annotations

from collections.abc import Callable

from ..costs import CostModel
from ..models import OrderIntent, Side, Tick
from ..signals import MicrostructureSignals, SignalFusion, SignalHub
from ..universe import Universe
from .base import Strategy


class FusedSignalStrategy(Strategy):
    def __init__(
        self,
        symbols: list[str],
        hub: SignalHub,
        fusion: SignalFusion,
        micro: MicrostructureSignals | None = None,
        costs: CostModel | None = None,
        edge_bps_at_full_conviction: float = 50.0,
        cost_safety_multiple: float = 2.0,
        min_rebalance_fraction: float = 0.2,
        sizer: Callable[[float], int] | None = None,
        universe: Universe | None = None,
    ):
        super().__init__(symbols)
        self.hub = hub
        self.fusion = fusion
        self.micro = micro or MicrostructureSignals()
        self.costs = costs or CostModel()
        self.edge_bps = edge_bps_at_full_conviction
        self.safety = cost_safety_multiple
        # Ignore target changes smaller than this share of max position (anti-churn),
        # except when going flat.
        self.min_rebalance_fraction = min_rebalance_fraction
        # price -> largest position in shares (the account's risk-per-trade sizing);
        # without one, fusion.max_position shares.
        self.sizer = sizer
        # Per-tier rules (penny stocks etc.) and volatility-scaled sizing; overrides sizer.
        self.universe = universe
        self._max_qty: dict[str, int] = {}
        self._last_eval: dict[str, dict] = {}

    def max_qty(self, symbol: str, price: float) -> int:
        if self.universe:
            qty = self.universe.max_shares(symbol, price)
        elif self.sizer:
            qty = self.sizer(price)
        else:
            qty = self.fusion.max_position
        self._max_qty[symbol] = qty
        return qty

    def min_rebalance_qty(self, max_qty: int) -> int:
        return max(1, round(self.min_rebalance_fraction * max_qty))

    def _reduction_target(self, symbol: str, position: int, max_qty: int) -> int:
        primary = self.fusion.target_for(self.fusion.primary_conviction(symbol), max_qty)
        if primary != 0 and (primary > 0) != (position > 0):
            return 0  # view reversed: flat first; the next tick decides on a new entry
        if abs(primary) > abs(position) / 2:
            return position  # hold
        return primary if abs(primary) >= self.min_rebalance_qty(max_qty) else 0

    def expected_gain(self, qty: int, price: float, conviction: float) -> float:
        return qty * price * abs(conviction) * self.edge_bps / 10_000

    def explain(self, symbol: str) -> dict:
        return {
            "conviction": round(self.fusion.conviction(symbol), 6),
            "target": self.fusion.target_position(symbol, self._max_qty.get(symbol)),
            "max_qty": self._max_qty.get(symbol),
            "signals": self.fusion.breakdown(symbol),
            **self._last_eval.get(symbol, {}),
        }

    def on_tick(self, tick: Tick, position: int) -> list[OrderIntent]:
        if tick.symbol not in self.symbols or not tick.valid:
            return []
        for sig in self.micro.update(tick):
            self.hub.publish(sig)
        tier = None
        if self.universe:
            self.universe.update(tick)
            tier = self.universe.tier(tick.mid)

        # The LLM view alone must justify holding a position; microstructure only
        # resizes it. (Entering on a combined view but exiting on the LLM view alone
        # would churn.)
        max_qty = self.max_qty(tick.symbol, tick.mid)
        primary = self.fusion.target_for(self.fusion.primary_conviction(tick.symbol), max_qty)
        use_micro = tier is None or tier.use_microstructure
        if not primary:
            target = 0
        elif use_micro:
            target = self.fusion.target_position(tick.symbol, max_qty)
        else:
            target = primary
        same_side = position == 0 or target == 0 or (position > 0) == (target > 0)
        if position and same_side and abs(target) < abs(position):
            target = self._reduction_target(tick.symbol, position, max_qty)
        delta = target - position
        if delta == 0 or (target != 0 and abs(delta) < self.min_rebalance_qty(max_qty)):
            return []

        # Shares that would add exposure: beyond the current position on the same side,
        # or everything past zero when flipping.
        same_side = position == 0 or target == 0 or (position > 0) == (target > 0)
        adding = max(0, abs(target) - abs(position)) if same_side else abs(target)
        if adding:
            conviction = (
                self.fusion.conviction(tick.symbol)
                if use_micro
                else self.fusion.primary_conviction(tick.symbol)
            )
            gain = self.expected_gain(adding, tick.mid, conviction)
            cost = self.costs.round_trip(adding, tick.mid)
            allowed, why = True, ""
            if self.universe:
                opener = self.fusion.strongest_opener(tick.symbol)
                allowed, why = self.universe.entry_allowed(tick, target < 0, opener)
            self._last_eval[tick.symbol] = {
                "expected_gain": round(gain, 4),
                "expected_cost": round(cost, 4),
                "tier": tier.name if tier else None,
                "stop_pct": round(self.universe.stop_pct(tick.symbol, tick.mid), 3)
                if self.universe
                else None,
                "blocked": why or None,
            }
            if gain < self.safety * cost or not allowed:
                if not allowed:
                    self._skip(tick.symbol, "tier_rules", why)
                else:
                    self._skip(
                        tick.symbol,
                        "cost_check",
                        f"expected gain ${gain:.2f} < {self.safety:g}x round-trip cost "
                        f"${cost:.2f} for {adding} shares",
                    )
                if same_side:
                    return []
                target = 0  # flip not worth it: just close
                delta = -position
                if delta == 0:
                    return []

        # Rest passively at the touch to earn the spread instead of paying it.
        if delta > 0:
            return [OrderIntent(tick.symbol, Side.BUY, delta, tick.bid)]
        return [OrderIntent(tick.symbol, Side.SELL, -delta, tick.ask)]
