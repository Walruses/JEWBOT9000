import asyncio

from trading_agent.models import NewsItem, Signal
from trading_agent.pipeline import NewsPipeline
from trading_agent.signals import SignalHub

NOW = 10_000.0


class StaticSource:
    name = "static"

    def __init__(self, items):
        self.items = items

    async def fetch(self, symbols):
        return self.items


class BrokenSource:
    name = "broken"

    async def fetch(self, symbols):
        raise RuntimeError("down")


class RecordingAnalyst:
    def __init__(self):
        self.batches = []

    async def analyze(self, symbol, items, now=None):
        self.batches.append((symbol, [i.id for i in items]))
        return Signal(symbol, "llm:news", 0.5, 0.5, NOW, 600)


def item(id_, published_at=NOW - 10, symbol="AAPL"):
    return NewsItem(id_, "static", symbol, "headline", published_at)


def test_dedupes_filters_stale_and_survives_broken_source():
    analyst = RecordingAnalyst()
    hub = SignalHub(clock=lambda: NOW)
    source = StaticSource(
        [item("a"), item("old", published_at=NOW - 7200), item("m", symbol="MSFT")]
    )
    pipeline = NewsPipeline(
        [source, BrokenSource()], analyst, hub, ["AAPL", "MSFT"], clock=lambda: NOW
    )

    asyncio.run(pipeline.poll_once())
    assert sorted(analyst.batches) == [("AAPL", ["a"]), ("MSFT", ["m"])]
    assert len(hub.active("AAPL")) == 1

    asyncio.run(pipeline.poll_once())  # same items again -> nothing new to analyse
    assert len(analyst.batches) == 2
