"""Combines active signals into a target position per symbol."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Signal
from .hub import SignalHub

DEFAULT_WEIGHTS = {"llm": 0.6, "micro:imbalance": 0.25, "micro:reversion": 0.15}
# Sources allowed to originate a position. Others (the microstructure signals) only
# adjust the timing and size of a view an opener already holds.
DEFAULT_OPENERS = frozenset({"llm"})


@dataclass
class SignalFusion:
    hub: SignalHub
    max_position: int
    # Keyed by exact source name, or by the prefix before ":" (e.g. "llm" covers "llm:news").
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    # Ignore combined conviction below this to avoid churning on noise.
    entry_threshold: float = 0.15
    openers: frozenset[str] = DEFAULT_OPENERS

    def weight(self, source: str) -> float:
        if source in self.weights:
            return self.weights[source]
        return self.weights.get(source.split(":", 1)[0], 0.0)

    def is_opener(self, source: str) -> bool:
        return source in self.openers or source.split(":", 1)[0] in self.openers

    def primary_conviction(self, symbol: str) -> float:
        """Conviction from openers (the LLM) alone, without microstructure adjustments."""
        total = sum(
            self.weight(sig.source) * sig.confidence * decay * sig.score
            for sig, decay in self.hub.active(symbol)
            if self.is_opener(sig.source)
        )
        return max(-1.0, min(1.0, total))

    def target_for(self, conviction: float, max_qty: int | None = None) -> int:
        if abs(conviction) < self.entry_threshold:
            return 0
        return round(conviction * (self.max_position if max_qty is None else max_qty))

    def conviction(self, symbol: str) -> float:
        """Weighted sum of signals in [-1, 1]; sources agreeing reinforce each other.

        Zero unless an opener (the LLM) holds a view. Other sources can strengthen or
        weaken that view but never create one or flip its direction."""
        primary = modifiers = 0.0
        for sig, decay in self.hub.active(symbol):
            contribution = self.weight(sig.source) * sig.confidence * decay * sig.score
            if self.is_opener(sig.source):
                primary += contribution
            else:
                modifiers += contribution
        if primary == 0:
            return 0.0
        total = primary + modifiers
        if (total > 0) != (primary > 0):
            return 0.0
        return max(-1.0, min(1.0, total))

    def strongest_opener(self, symbol: str) -> tuple[Signal, float] | None:
        """The active opener (LLM) signal with the most weight right now, and its decay."""
        best = None
        for sig, decay in self.hub.active(symbol):
            if self.is_opener(sig.source) and sig.score:
                strength = abs(sig.score) * sig.confidence * decay
                if best is None or strength > best[0]:
                    best = (strength, sig, decay)
        return (best[1], best[2]) if best else None

    def breakdown(self, symbol: str) -> list[dict]:
        """Each active signal's contribution to conviction, for the trade journal."""
        rows = []
        for sig, decay in self.hub.active(symbol):
            weight = self.weight(sig.source)
            rows.append(
                {
                    "source": sig.source,
                    "signal_id": sig.id,
                    "score": round(sig.score, 4),
                    "confidence": round(sig.confidence, 4),
                    "decay": round(decay, 4),
                    "weight": weight,
                    "contribution": round(weight * sig.confidence * decay * sig.score, 6),
                }
            )
        return rows

    def target_position(self, symbol: str, max_qty: int | None = None) -> int:
        return self.target_for(self.conviction(symbol), max_qty)
