"""Broker abstraction so the engine runs identically against IBKR or the simulator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ..models import Fill, OrderIntent, Tick

TickCallback = Callable[[Tick], None]
FillCallback = Callable[[Fill], None]
# (order_id, unfilled_qty) -- called exactly once when an order is filled, cancelled or rejected.
DoneCallback = Callable[[str, int], None]


class Broker(Protocol):
    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def positions(self) -> dict[str, tuple[int, float]]:
        """Current stock positions as {symbol: (qty, avg_price)}."""
        ...

    async def subscribe(self, symbols: list[str], on_tick: TickCallback) -> None: ...

    def place_limit(
        self, intent: OrderIntent, on_fill: FillCallback, on_done: DoneCallback
    ) -> str: ...

    def cancel(self, order_id: str) -> None: ...

    def cancel_all(self) -> None:
        """Cancel every working order placed by this agent (not orders placed elsewhere)."""
        ...
