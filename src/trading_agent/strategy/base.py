"""Strategy interface.

A strategy turns market data into OrderIntents. It never talks to the broker directly:
the engine routes every intent through the risk manager first. This is also where an
ML/AI signal model plugs in -- wrap the model in a Strategy subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Fill, OrderIntent, Tick


class Strategy(ABC):
    def __init__(self, symbols: list[str]):
        self.symbols = symbols

    @abstractmethod
    def on_tick(self, tick: Tick, position: int) -> list[OrderIntent]:
        """Return orders to place given the latest tick and current position."""

    def on_fill(self, fill: Fill) -> None:  # noqa: B027 - optional hook
        """Called for every execution of an order this strategy requested."""
