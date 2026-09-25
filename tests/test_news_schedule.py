import asyncio
from datetime import datetime

from trading_agent.models import NewsItem, Signal
from trading_agent.pipeline import BRIEFING_NOTE, NewsMode, NewsPipeline, NewsSchedule
from trading_agent.session import NEW_YORK
from trading_agent.signals import SignalHub


def et(d, h, m):
    return datetime(2026, 9, d, h, m, tzinfo=NEW_YORK).timestamp()  # 24 = Thursday


def test_modes_and_sleep_until_briefing():
    s = NewsSchedule()
    assert s.mode(et(24, 10, 0)) is NewsMode.LIVE
    assert s.mode(et(24, 9, 20)) is NewsMode.LIVE
    assert s.mode(et(24, 17, 0)) is NewsMode.EXTENDED
    assert s.mode(et(24, 23, 0)) is NewsMode.QUIET
    assert s.mode(et(26, 10, 0)) is NewsMode.QUIET  # Saturday
    assert s.sleep_seconds(et(24, 10, 0)) == 60
    assert s.sleep_seconds(et(24, 17, 0)) == 300
    assert s.sleep_seconds(et(24, 23, 0)) == 1800
    assert s.sleep_seconds(et(25, 9, 17)) == 180  # capped: wake for the 09:20 briefing
    assert s.next_briefing(et(25, 17, 0)) == et(28, 9, 20)  # Friday evening -> Monday


class Source:
    name = "s"

    def __init__(self):
        self.items = []

    async def fetch(self, symbols):
        return self.items


class Analyst:
    def __init__(self):
        self.calls = []

    async def analyze(self, symbol, items, now=None, note=""):
        self.calls.append((symbol, sorted(i.id for i in items), note))
        return Signal(symbol, "llm:news", 0.5, 0.5, now, 600, id=f"s{len(self.calls)}")


def test_off_hours_items_are_buffered_then_briefed_once():
    clock = [et(24, 18, 0)]
    source, analyst = Source(), Analyst()
    pipe = NewsPipeline(
        [source],
        analyst,
        SignalHub(clock=lambda: clock[0]),
        ["AAPL"],
        clock=lambda: clock[0],
        schedule=NewsSchedule(),
    )
    # After-hours earnings release, and a Reddit post overnight.
    source.items = [NewsItem("a", "edgar", "AAPL", "8-K", et(24, 16, 5))]
    asyncio.run(pipe.poll_once(analyze=False))
    clock[0] = et(25, 2, 0)
    source.items.append(NewsItem("b", "reddit", "AAPL", "post", et(25, 1, 0)))
    asyncio.run(pipe.poll_once(analyze=False))
    assert analyst.calls == []

    clock[0] = et(25, 9, 20)
    [signal] = asyncio.run(pipe.briefing())
    assert analyst.calls == [("AAPL", ["a", "b"], BRIEFING_NOTE)]
    assert signal.symbol == "AAPL"
    assert asyncio.run(pipe.briefing()) == []  # buffer cleared
