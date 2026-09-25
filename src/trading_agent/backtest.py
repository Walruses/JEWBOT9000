"""Replay recorded or downloaded market data and news through the live code path.

    python -m trading_agent.backtest data/recordings/2026-09-24.jsonl
    python -m trading_agent.backtest data/history/*.jsonl --reanalyze   # re-run Claude

The engine, strategy, signal fusion and risk manager are the same objects used live;
only the broker (SimBroker) and the clock are simulated.

News signals come from one of two places:
- recorded (default): the LLM signals captured live with --record. Free and exactly
  what the agent saw, but you can't change the analyst.
- --reanalyze: news items are fed through the real NewsPipeline and Claude on a
  simulated poll schedule. Results are cached on disk, so re-runs cost nothing.
  CAUTION: the model may know how events after its training data turned out. For
  dates it could know about, this is lookahead bias and inflates results.

Fill model: resting limit orders fill only once the opposite side trades through the
limit price (no queue position, no partial fills), and a commission is charged per
share. That is conservative for entries but ignores market impact.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .broker.sim import SimBroker
from .config import RiskLimits, risk_limits_from_env
from .engine import Engine
from .events import Event, event_ts, read_events
from .models import NewsItem, Signal, Tick
from .pipeline import Analyst, NewsPipeline
from .risk import RiskManager
from .session import TradingSession
from .signals import SignalFusion, SignalHub
from .strategy import FusedSignalStrategy

log = logging.getLogger(__name__)

DEFAULT_SESSION = TradingSession()


class SimClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class ReplaySource:
    """Serves recorded news items once the simulated clock reaches their publish time."""

    name = "replay"

    def __init__(self, items: list[NewsItem], clock: SimClock):
        self._items = sorted(items, key=lambda i: i.published_at)
        self._clock = clock
        self._next = 0

    async def fetch(self, symbols: list[str]) -> list[NewsItem]:
        out = []
        while (
            self._next < len(self._items) and self._items[self._next].published_at <= self._clock()
        ):
            item = self._items[self._next]
            self._next += 1
            if item.symbol in symbols:
                out.append(item)
        return out


class CachedAnalyst:
    """Wraps an analyst with a disk cache and simulated analysis latency: a signal is
    stamped `latency` seconds after the poll, so the strategy can't act on it sooner
    than it could live."""

    def __init__(self, inner: Analyst, cache_path: Path, latency: float, model: str):
        self.inner = inner
        self.cache_path = cache_path
        self.latency = latency
        self.model = model
        self.calls = 0
        self._cache: dict[str, dict | None] = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                row = json.loads(line)
                self._cache[row["key"]] = row["signal"]

    async def analyze(
        self, symbol: str, items: list[NewsItem], now: float | None = None
    ) -> Signal | None:
        assert now is not None
        ids = ",".join(sorted(i.id for i in items))
        key = hashlib.sha256(f"{self.model}|{symbol}|{now:.0f}|{ids}".encode()).hexdigest()
        if key not in self._cache:
            self.calls += 1
            sig = await self.inner.analyze(symbol, items, now=now)
            self._cache[key] = asdict(sig) if sig else None
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "signal": self._cache[key]}) + "\n")
        cached = self._cache[key]
        if cached is None:
            return None
        sig = Signal(**cached)
        return replace(sig, ts=sig.ts + self.latency)


@dataclass
class BacktestResult:
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    fills: int
    shares_traded: int
    max_drawdown: float
    signals_used: int
    analyst_calls: int = 0
    halted: str = ""
    equity_curve: list[tuple[float, float]] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    def summary(self) -> str:
        lines = [
            f"total PnL       {self.total_pnl:12.2f}  (net of fees)",
            f"  realized      {self.realized_pnl:12.2f}",
            f"  unrealized    {self.unrealized_pnl:12.2f}",
            f"fees            {self.fees:12.2f}",
            f"max drawdown    {self.max_drawdown:12.2f}",
            f"fills           {self.fills:12d}",
            f"shares traded   {self.shares_traded:12d}",
            f"LLM signals     {self.signals_used:12d}",
        ]
        if self.analyst_calls:
            lines.append(f"new LLM calls   {self.analyst_calls:12d}")
        if self.halted:
            lines.append(f"HALTED: {self.halted}")
        return "\n".join(lines)


async def run_backtest(
    events: list[Event],
    symbols: list[str],
    limits: RiskLimits,
    session: TradingSession | None = DEFAULT_SESSION,
    analyst: CachedAnalyst | None = None,
    poll_interval: float = 60.0,
    commission_per_share: float = 0.0035,
    min_commission: float = 0.35,
    order_ttl: float = 2.0,
    fusion_weights: dict[str, float] | None = None,
) -> BacktestResult:
    if not events:
        raise ValueError("no events to replay")
    clock = SimClock(event_ts(events[0]))
    hub = SignalHub(clock=clock)
    fusion = SignalFusion(hub, max_position=limits.max_position)
    if fusion_weights:
        fusion.weights = fusion_weights
    risk = RiskManager(limits, clock=clock)
    broker = SimBroker(commission_per_share=commission_per_share, min_commission=min_commission)
    engine = Engine(
        broker,
        FusedSignalStrategy(symbols, hub, fusion),
        risk,
        order_ttl=order_ttl,
        clock=clock,
        session=session,
    )
    await broker.subscribe(symbols, engine.on_tick)

    pipeline = None
    if analyst:
        news = [e for e in events if isinstance(e, NewsItem)]
        pipeline = NewsPipeline(
            [ReplaySource(news, clock)],
            analyst,
            hub,
            symbols,
            poll_interval=poll_interval,
            clock=clock,
        )
    next_poll = clock.t

    signals_used = 0
    peak = equity = 0.0
    max_dd = 0.0
    curve: list[tuple[float, float]] = []
    wanted = set(symbols)

    for event in events:
        ts = event_ts(event)
        while pipeline and next_poll <= ts:
            clock.t = next_poll
            signals_used += len(await pipeline.poll_once())
            next_poll += poll_interval
        clock.t = max(clock.t, ts)

        if isinstance(event, Tick) and event.symbol in wanted:
            broker.push_tick(event)
            equity = risk.total_pnl()
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
            if not curve or ts - curve[-1][0] >= 60:
                curve.append((ts, equity))
        elif isinstance(event, Signal) and pipeline is None and event.symbol in wanted:
            hub.publish(event)
            signals_used += 1

    return BacktestResult(
        realized_pnl=risk.realized_pnl,
        unrealized_pnl=risk.unrealized_pnl(),
        fees=broker.fees_paid,
        fills=broker.fill_count,
        shares_traded=broker.shares_traded,
        max_drawdown=max_dd,
        signals_used=signals_used,
        analyst_calls=analyst.calls if analyst else 0,
        halted=risk.halt_reason,
        equity_curve=curve,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="trading_agent.backtest")
    parser.add_argument("files", nargs="+", type=Path, help="JSONL event files")
    parser.add_argument("--symbols", nargs="+", help="default: every symbol with ticks")
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="run Claude over recorded news instead of using recorded signals",
    )
    parser.add_argument("--cache", type=Path, default=Path("data/analysis_cache.jsonl"))
    parser.add_argument(
        "--latency",
        type=float,
        default=10.0,
        help="simulated seconds from news poll to usable LLM signal",
    )
    parser.add_argument("--poll", type=float, default=60.0, help="news poll interval (s)")
    parser.add_argument("--commission", type=float, default=0.0035, help="$ per share")
    parser.add_argument("--min-commission", type=float, default=0.35)
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="trade around the clock instead of the 09:35-15:50 ET session",
    )
    parser.add_argument("--equity-csv", type=Path)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    events = read_events(args.files)
    symbols = args.symbols or sorted({e.symbol for e in events if isinstance(e, Tick)})
    analyst = None
    if args.reanalyze:
        from .analyst import ClaudeAnalyst
        from .config import data_config_from_env

        cfg = data_config_from_env()
        inner = ClaudeAnalyst(model=cfg.analyst_model, effort=cfg.analyst_effort)
        analyst = CachedAnalyst(inner, args.cache, args.latency, cfg.analyst_model)
        print(
            "NOTE: --reanalyze asks the model about past events; if they predate its "
            "training data, results carry lookahead bias.\n"
        )

    result = asyncio.run(
        run_backtest(
            events,
            symbols,
            risk_limits_from_env(),
            session=None if args.no_session else TradingSession(),
            analyst=analyst,
            poll_interval=args.poll,
            commission_per_share=args.commission,
            min_commission=args.min_commission,
        )
    )
    print(f"symbols: {' '.join(symbols)}   events: {len(events):,}")
    print(result.summary())
    if args.equity_csv:
        with open(args.equity_csv, "w") as f:
            f.write("ts,equity\n")
            f.writelines(f"{ts:.0f},{eq:.2f}\n" for ts, eq in result.equity_curve)


if __name__ == "__main__":
    main()
