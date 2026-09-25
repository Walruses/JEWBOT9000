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

from ..costs import CostModel
from ..models import OrderIntent, Side, Tick
from ..signals import MicrostructureSignals, SignalFusion, SignalHub
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
        self.min_rebalance_qty = max(1, round(min_rebalance_fraction * fusion.max_position))
        self._last_eval: dict[str, dict] = {}

    def _reduction_target(self, symbol: str, position: int) -> int:
        primary = self.fusion.target_for(self.fusion.primary_conviction(symbol))
        if primary != 0 and (primary > 0) != (position > 0):
            return 0  # view reversed: flat first; the next tick decides on a new entry
        if abs(primary) > abs(position) / 2:
            return position  # hold
        return primary if abs(primary) >= self.min_rebalance_qty else 0

    def expected_gain(self, qty: int, price: float, conviction: float) -> float:
        return qty * price * abs(conviction) * self.edge_bps / 10_000

    def explain(self, symbol: str) -> dict:
        return {
            "conviction": round(self.fusion.conviction(symbol), 6),
            "target": self.fusion.target_position(symbol),
            "signals": self.fusion.breakdown(symbol),
            **self._last_eval.get(symbol, {}),
        }

    def on_tick(self, tick: Tick, position: int) -> list[OrderIntent]:
        if tick.symbol not in self.symbols or not tick.valid:
            return []
        for sig in self.micro.update(tick):
            self.hub.publish(sig)

        # The LLM view alone must justify holding a position; microstructure only
        # resizes it. (Entering on a combined view but exiting on the LLM view alone
        # would churn.)
        primary = self.fusion.target_for(self.fusion.primary_conviction(tick.symbol))
        target = self.fusion.target_position(tick.symbol) if primary else 0
        same_side = position == 0 or target == 0 or (position > 0) == (target > 0)
        if position and same_side and abs(target) < abs(position):
            target = self._reduction_target(tick.symbol, position)
        delta = target - position
        if delta == 0 or (target != 0 and abs(delta) < self.min_rebalance_qty):
            return []

        # Shares that would add exposure: beyond the current position on the same side,
        # or everything past zero when flipping.
        same_side = position == 0 or target == 0 or (position > 0) == (target > 0)
        adding = max(0, abs(target) - abs(position)) if same_side else abs(target)
        if adding:
            conviction = self.fusion.conviction(tick.symbol)
            gain = self.expected_gain(adding, tick.mid, conviction)
            cost = self.costs.round_trip(adding, tick.mid)
            self._last_eval[tick.symbol] = {
                "expected_gain": round(gain, 4),
                "expected_cost": round(cost, 4),
            }
            if gain < self.safety * cost:
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
