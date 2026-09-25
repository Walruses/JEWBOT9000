"""Trades each symbol toward the target position implied by all active signals."""

from __future__ import annotations

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
        min_trade_qty: int = 1,
    ):
        super().__init__(symbols)
        self.hub = hub
        self.fusion = fusion
        self.micro = micro or MicrostructureSignals()
        self.min_trade_qty = min_trade_qty

    def on_tick(self, tick: Tick, position: int) -> list[OrderIntent]:
        if tick.symbol not in self.symbols or not tick.valid:
            return []
        for sig in self.micro.update(tick):
            self.hub.publish(sig)

        delta = self.fusion.target_position(tick.symbol) - position
        if abs(delta) < self.min_trade_qty:
            return []
        # Rest passively at the touch to earn the spread instead of paying it.
        if delta > 0:
            return [OrderIntent(tick.symbol, Side.BUY, delta, tick.bid)]
        return [OrderIntent(tick.symbol, Side.SELL, -delta, tick.ask)]
