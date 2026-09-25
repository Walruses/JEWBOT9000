import asyncio
from datetime import datetime

from trading_agent.backtest import CachedAnalyst, run_backtest
from trading_agent.config import RiskLimits
from trading_agent.models import NewsItem, Signal, Tick
from trading_agent.session import NEW_YORK

T0 = datetime(2026, 9, 24, 10, 0, tzinfo=NEW_YORK).timestamp()
LLM_ONLY = {"llm": 1.0}


def ticks(start, n, bid=100.00, step=0.0):
    return [
        Tick(
            "AAPL",
            round(bid + i * step, 2),
            round(bid + i * step + 0.02, 2),
            bid + i * step + 0.01,
            start + i,
        )
        for i in range(n)
    ]


def test_recorded_signal_drives_a_profitable_round_trip():
    # Bullish LLM signal, then price rises; the position is flattened at 15:50.
    events = [
        *ticks(T0, 5),
        Signal("AAPL", "llm:news", 1.0, 1.0, T0 + 5, 4 * 3600),
        *ticks(T0 + 6, 1, bid=100.00),  # strategy bids 100.00
        *ticks(T0 + 7, 3, bid=99.98),  # ask drops to 100.00: our bid fills
        *ticks(T0 + 100, 5, bid=101.00),
        *ticks(datetime(2026, 9, 24, 15, 51, tzinfo=NEW_YORK).timestamp(), 5, bid=101.00),
    ]
    result = asyncio.run(
        run_backtest(
            events,
            ["AAPL"],
            RiskLimits(max_position=10, max_daily_loss=1e9),
            commission_per_share=0.01,
            min_commission=0.0,
            fusion_weights=LLM_ONLY,
        )
    )
    assert result.fills == 2
    assert result.unrealized_pnl == 0.0
    assert result.fees == 0.2
    assert result.realized_pnl == 10.0 - 0.2  # bought 10 @ 100.00, sold 10 @ 101.00


class FakeAnalyst:
    def __init__(self):
        self.calls = []

    async def analyze(self, symbol, items, now=None):
        self.calls.append((now, [i.id for i in items]))
        return Signal(symbol, "llm:news", 1.0, 1.0, now, 3600)


def test_reanalyze_respects_latency_and_caches(tmp_path):
    news = NewsItem("n1", "edgar", "AAPL", "guidance raised", T0 + 30)
    # We bid 99.99 once the signal is usable; the ask reaches 99.99 at T0+85.
    events = [*ticks(T0, 85, bid=99.99), *ticks(T0 + 85, 5, bid=99.97), news]
    events.sort(key=lambda e: e.ts if isinstance(e, Tick) else e.published_at)

    fake = FakeAnalyst()
    analyst = CachedAnalyst(fake, tmp_path / "cache.jsonl", latency=20.0, model="m")
    result = asyncio.run(
        run_backtest(
            events,
            ["AAPL"],
            RiskLimits(max_position=10),
            analyst=analyst,
            poll_interval=60.0,
            commission_per_share=0.0,
            min_commission=0.0,
            fusion_weights=LLM_ONLY,
        )
    )
    # Item published at T0+30 is first seen at the T0+60 poll and usable from T0+80.
    assert fake.calls == [(T0 + 60, ["n1"])]
    assert result.signals_used == 1 and result.fills == 1

    fake2 = FakeAnalyst()
    cached = CachedAnalyst(fake2, tmp_path / "cache.jsonl", latency=20.0, model="m")
    asyncio.run(
        run_backtest(
            events,
            ["AAPL"],
            RiskLimits(max_position=10),
            analyst=cached,
            poll_interval=60.0,
            fusion_weights=LLM_ONLY,
        )
    )
    assert fake2.calls == [] and cached.calls == 0
