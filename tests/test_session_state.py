from datetime import date, datetime

from trading_agent.events import decode, encode
from trading_agent.models import NewsItem, Signal, Tick
from trading_agent.session import NEW_YORK, Phase, TradingSession
from trading_agent.state import DailyState, StateStore


def et(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=NEW_YORK).timestamp()


def test_session_phases():
    s = TradingSession()
    assert s.phase(et(2026, 9, 24, 9, 31)) is Phase.CLOSED  # opening minutes skipped
    assert s.phase(et(2026, 9, 24, 12, 0)) is Phase.TRADING
    assert s.phase(et(2026, 9, 24, 15, 55)) is Phase.FLATTEN
    assert s.phase(et(2026, 9, 24, 16, 5)) is Phase.CLOSED
    assert s.phase(et(2026, 9, 26, 12, 0)) is Phase.CLOSED  # Saturday


def test_state_roundtrip_and_day_rollover(tmp_path):
    store = StateStore(tmp_path / "state.json")
    today = date(2026, 9, 24)
    assert store.load(today) == DailyState("2026-09-24")
    store.save(DailyState("2026-09-24", -120.5, True, "daily loss"))
    assert store.load(today).realized_pnl == -120.5
    tomorrow = store.load(date(2026, 9, 25))
    assert tomorrow.realized_pnl == 0.0
    assert tomorrow.halted  # halts need a human to clear them


def test_event_roundtrip():
    for event in (
        Tick("AAPL", 1.0, 1.1, 1.05, 5.0, 100, 200),
        NewsItem("n1", "edgar", "AAPL", "h", 7.0, "body", "url"),
        Signal("AAPL", "llm:news", 0.5, 0.8, 9.0, 600, "why"),
    ):
        assert decode(encode(event)) == event
