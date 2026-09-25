"""US equity trading session: when to trade and when to flatten before the close.

Exchange holidays and half days are not modelled: on those days IBKR sends no (or
early-ending) data. Set the times to match if you trade a half day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")


class Phase(Enum):
    CLOSED = "closed"
    TRADING = "trading"
    FLATTEN = "flatten"


@dataclass(frozen=True)
class TradingSession:
    # Skip the first minutes after the 09:30 open, when spreads and volatility are extreme.
    start: time = time(9, 35)
    # Stop opening positions and work every position back to flat before the 16:00 close.
    flatten: time = time(15, 50)
    end: time = time(16, 0)
    tz: ZoneInfo = NEW_YORK

    def phase(self, ts: float) -> Phase:
        dt = datetime.fromtimestamp(ts, self.tz)
        if dt.weekday() >= 5:
            return Phase.CLOSED
        t = dt.time()
        if self.start <= t < self.flatten:
            return Phase.TRADING
        if self.flatten <= t < self.end:
            return Phase.FLATTEN
        return Phase.CLOSED

    def trading_date(self, ts: float) -> date:
        return datetime.fromtimestamp(ts, self.tz).date()


def parse_hhmm(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))
