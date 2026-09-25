"""Persists the day's realized PnL and halt state across restarts.

Positions are not stored here: the broker is the source of truth and the engine loads
them on startup. This file only guarantees that a restart can't silently reset the
daily-loss limit or clear a halt.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class DailyState:
    date: str
    realized_pnl: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    # Dates this agent opened positions (= day trades), for the pattern day trader rule.
    # Kept across days: the rule looks back 5 business days.
    day_trade_dates: list[str] = field(default_factory=list)


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self, today: date) -> DailyState:
        try:
            raw = json.loads(self.path.read_text())
        except FileNotFoundError:
            return DailyState(date=today.isoformat())
        state = DailyState(**raw)
        if state.date != today.isoformat():
            # A halt carries over to the next day: someone should look before trading resumes.
            return DailyState(
                date=today.isoformat(),
                halted=state.halted,
                halt_reason=state.halt_reason,
                day_trade_dates=state.day_trade_dates,
            )
        return state

    def save(self, state: DailyState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(state), indent=2))
        os.replace(tmp, self.path)  # atomic: a crash never leaves a half-written file
