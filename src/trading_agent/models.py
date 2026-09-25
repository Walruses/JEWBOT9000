"""Core value types shared across the agent."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    bid: float
    ask: float
    last: float
    ts: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def valid(self) -> bool:
        return self.bid > 0 and self.ask > 0 and self.ask >= self.bid


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """A strategy's request to trade. Only becomes an order after passing risk checks."""

    symbol: str
    side: Side
    qty: int
    limit_price: float


@dataclass(frozen=True, slots=True)
class Fill:
    """An execution. Brokers that report commissions separately (IBKR) send a follow-up
    Fill with qty=0 carrying only the commission."""

    order_id: str
    symbol: str
    side: Side
    qty: int
    price: float
    ts: float
    commission: float = 0.0


@dataclass(frozen=True, slots=True)
class Signal:
    """A directional view on a symbol from one source.

    score is -1 (strong sell) .. +1 (strong buy); confidence is 0..1. A signal is only
    considered for ttl seconds after ts, with its weight decaying linearly to zero.
    """

    symbol: str
    source: str
    score: float
    confidence: float
    ts: float
    ttl: float
    rationale: str = ""
    # For journal attribution: a unique id, the news items analysed, and the ones the
    # analyst said drove its view.
    id: str = ""
    inputs: tuple[str, ...] = ()
    drivers: tuple[str, ...] = ()
    model: str = ""  # which LLM produced it


@dataclass(frozen=True, slots=True)
class NewsItem:
    """A piece of text-based information about a symbol from any source."""

    id: str
    source: str
    symbol: str
    headline: str
    published_at: float
    body: str = ""
    url: str = ""
