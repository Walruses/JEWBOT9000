"""Combines active signals into a target position per symbol."""

from __future__ import annotations

from dataclasses import dataclass, field

from .hub import SignalHub

DEFAULT_WEIGHTS = {"llm": 0.6, "micro:imbalance": 0.25, "micro:reversion": 0.15}


@dataclass
class SignalFusion:
    hub: SignalHub
    max_position: int
    # Keyed by exact source name, or by the prefix before ":" (e.g. "llm" covers "llm:news").
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    # Ignore combined conviction below this to avoid churning on noise.
    entry_threshold: float = 0.15

    def weight(self, source: str) -> float:
        if source in self.weights:
            return self.weights[source]
        return self.weights.get(source.split(":", 1)[0], 0.0)

    def conviction(self, symbol: str) -> float:
        """Weighted sum of signals in [-1, 1]; sources agreeing reinforce each other."""
        total = 0.0
        for sig, decay in self.hub.active(symbol):
            total += self.weight(sig.source) * sig.confidence * decay * sig.score
        return max(-1.0, min(1.0, total))

    def target_position(self, symbol: str) -> int:
        c = self.conviction(symbol)
        if abs(c) < self.entry_threshold:
            return 0
        return round(c * self.max_position)
