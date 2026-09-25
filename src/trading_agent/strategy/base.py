"""Strategy interface.

A strategy turns market data into OrderIntents. It never talks to the broker directly:
the engine routes every intent through the risk manager first. This is also where an
ML/AI signal model plugs in -- wrap the model in a Strategy subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from ..models import Fill, OrderIntent, Tick


class Strategy(ABC):
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        # (symbol, stage, reason, context) -> None; set by the engine to journal trades
        # the strategy wanted but decided against (cost check, tier rules, ...).
        self.skip_listener: Callable[[str, str, str, dict], None] | None = None

    def _skip(self, symbol: str, stage: str, reason: str) -> None:
        if self.skip_listener:
            self.skip_listener(symbol, stage, reason, self.explain(symbol))

    @abstractmethod
    def on_tick(self, tick: Tick, position: int) -> list[OrderIntent]:
        """Return orders to place given the latest tick and current position."""

    def explain(self, symbol: str) -> dict:
        """What the strategy based its latest decision on; stored with each order."""
        return {}

    def on_fill(self, fill: Fill) -> None:  # noqa: B027 - optional hook
        """Called for every execution of an order this strategy requested."""
