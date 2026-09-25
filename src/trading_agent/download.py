"""Download a day of historical quotes and news into the backtest event format.

    python -m trading_agent.download --symbols AAPL MSFT --date 2026-09-24

Quotes come from IBKR (needs TWS/Gateway running; paper login is fine):
- default: 1-second BID_ASK bars for the regular session (13 requests per symbol).
  IBKR defines these bars as open = time-averaged bid, close = time-averaged ask, so each
  bar becomes one synthetic quote. Book sizes aren't available, so the order-book
  imbalance signal is inactive in these backtests.
- --ticks: every individual quote with sizes. IBKR returns 1000 per request and paces
  historical requests, so this is only practical for short windows (--start/--end).

Your own recordings (--record when trading) remain the best backtest data: exact quotes
with sizes, plus the LLM signals the agent actually produced.

News: IBKR headlines, SEC EDGAR filings and Finnhub (when configured) from 04:00 ET.
Reddit is not included (its search API has no reliable date range).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from ib_async import IB, Stock

from .config import data_config_from_env, ib_config_from_env
from .events import encode
from .models import NewsItem, Tick
from .session import NEW_YORK, parse_hhmm

log = logging.getLogger(__name__)

BAR_CHUNK = timedelta(minutes=30)  # IBKR's maximum duration for 1-second bars
PACING_DELAY = 2.0  # IBKR: at most 6 historical requests per contract in 2 seconds


def _et(day: date, t: time) -> datetime:
    return datetime.combine(day, t, NEW_YORK)


async def download_bars(ib: IB, contract: Stock, start: datetime, end: datetime) -> list[Tick]:
    ticks: list[Tick] = []
    chunk_end = end
    while chunk_end > start:
        duration = int(min(BAR_CHUNK, chunk_end - start).total_seconds())
        bars = await ib.reqHistoricalDataAsync(
            contract,
            chunk_end.astimezone(UTC),
            f"{duration} S",
            "1 secs",
            "BID_ASK",
            useRTH=True,
            formatDate=2,
        )
        for bar in bars:
            if not isinstance(bar.date, datetime):
                continue
            # A bar's averages are only known once it closes: stamp it at the bar's end.
            ts = bar.date.timestamp() + 1.0
            if bar.open > 0 and bar.close >= bar.open:
                ticks.append(
                    Tick(contract.symbol, bar.open, bar.close, (bar.open + bar.close) / 2, ts)
                )
        log.info("%s: %d bars up to %s", contract.symbol, len(bars), chunk_end.time())
        chunk_end -= BAR_CHUNK
        await asyncio.sleep(PACING_DELAY)
    return sorted(ticks, key=lambda t: t.ts)


async def download_ticks(ib: IB, contract: Stock, start: datetime, end: datetime) -> list[Tick]:
    ticks: list[Tick] = []
    seen: set[tuple] = set()
    cursor = start
    while cursor < end:
        rows = await ib.reqHistoricalTicksAsync(
            contract, cursor.astimezone(UTC), "", 1000, "BID_ASK", useRth=True
        )
        if not rows:
            break
        for r in rows:
            key = (r.time, r.priceBid, r.priceAsk, r.sizeBid, r.sizeAsk)
            if r.time >= end or key in seen:
                continue
            seen.add(key)
            ticks.append(
                Tick(
                    contract.symbol,
                    r.priceBid,
                    r.priceAsk,
                    (r.priceBid + r.priceAsk) / 2,
                    r.time.timestamp(),
                    float(r.sizeBid),
                    float(r.sizeAsk),
                )
            )
        last = rows[-1].time
        # Pages are keyed by second; if a whole page fell in one second, step past it.
        cursor = last if last > cursor else cursor + timedelta(seconds=1)
        await asyncio.sleep(PACING_DELAY)
    return ticks


async def download_news(
    ib: IB, symbols: list[str], contracts: dict, start: float, end: float
) -> list[NewsItem]:
    from .sources import EdgarSource, FinnhubSource
    from .sources.ibkr_news import IBKRNewsSource

    cfg = data_config_from_env()

    class _BrokerShim:  # IBKRNewsSource only needs .ib and .contract()
        def __init__(self):
            self.ib = ib

        def contract(self, symbol):
            return contracts.get(symbol)

    sources: list = [IBKRNewsSource(_BrokerShim())]  # type: ignore[arg-type]
    if cfg.sec_user_agent:
        sources.append(EdgarSource(cfg.sec_user_agent))
    if cfg.finnhub_api_key:
        sources.append(FinnhubSource(cfg.finnhub_api_key))
    items: list[NewsItem] = []
    for source in sources:
        try:
            got = await source.fetch_between(symbols, start, end)
            log.info("%s: %d news items", source.name, len(got))
            items.extend(got)
        except Exception:
            log.exception("news source %s failed", source.name)
    return items


async def run(args: argparse.Namespace) -> Path:
    day = date.fromisoformat(args.date)
    start, end = _et(day, parse_hhmm(args.start)), _et(day, parse_hhmm(args.end))
    cfg = ib_config_from_env()
    ib = IB()
    await ib.connectAsync(cfg.host, cfg.port, clientId=cfg.client_id + 100)
    try:
        contracts = {s: Stock(s, "SMART", "USD") for s in args.symbols}
        await ib.qualifyContractsAsync(*contracts.values())
        ticks: list[Tick] = []
        for contract in contracts.values():
            fetch = download_ticks if args.ticks else download_bars
            ticks.extend(await fetch(ib, contract, start, end))
        news: list[NewsItem] = []
        if not args.no_news:
            news_start = _et(day, time(4, 0)).timestamp()
            news = await download_news(ib, args.symbols, contracts, news_start, end.timestamp())
    finally:
        ib.disconnect()

    out = args.out or Path("data/history") / f"{day.isoformat()}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    events = sorted([*ticks, *news], key=lambda e: e.ts if isinstance(e, Tick) else e.published_at)
    with open(out, "w") as f:
        f.writelines(encode(e) + "\n" for e in events)
    print(f"wrote {len(ticks):,} quotes and {len(news):,} news items to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(prog="trading_agent.download")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--date", required=True, help="trading day, YYYY-MM-DD")
    parser.add_argument("--start", default="09:30", help="ET, HH:MM")
    parser.add_argument("--end", default="16:00", help="ET, HH:MM")
    parser.add_argument("--ticks", action="store_true", help="every quote with sizes (slow)")
    parser.add_argument("--no-news", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
