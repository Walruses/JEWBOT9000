"""Build a watchlist from IBKR market scanners.

    python -m trading_agent.scan --preset smallcap penny          # print + save watchlist
    python -m trading_agent --mode paper --scan smallcap penny    # scan at startup

Presets look for stocks trading on unusually high volume (IBKR's HOT_BY_VOLUME scan),
which is where news tends to be. Only exchange-listed US stocks are scanned
(STK.US.MAJOR: no OTC/pink sheets), operating companies only (no ETFs).

- largecap: market cap above $10B, price above $5
- smallcap: market cap $300M-$2B, price above $5, 500k+ shares a day
- penny:    price $1-$5, 1M+ shares a day (traded only on very confident views)

Scanner filters are applied by IBKR; check the output on a paper account before relying
on it. The list is fixed for the session: the agent subscribes to these symbols at start.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from ib_async import IB, ScannerSubscription, TagValue

from .config import ib_config_from_env

log = logging.getLogger(__name__)

DEFAULT_WATCHLIST = Path("data/watchlist.txt")

PRESETS: dict[str, tuple[dict, dict[str, str]]] = {
    "largecap": ({"abovePrice": 5.0}, {"marketCapAbove1e6": "10000"}),
    "smallcap": (
        {"abovePrice": 5.0, "aboveVolume": 500_000},
        {"marketCapAbove1e6": "300", "marketCapBelow1e6": "2000"},
    ),
    "penny": ({"abovePrice": 1.0, "belowPrice": 5.0, "aboveVolume": 1_000_000}, {}),
}


async def scan(ib: IB, preset: str, rows: int = 20) -> list[str]:
    fields, filters = PRESETS[preset]
    sub = ScannerSubscription(
        instrument="STK",
        locationCode="STK.US.MAJOR",
        scanCode="HOT_BY_VOLUME",
        numberOfRows=rows,
        stockTypeFilter="CORP",
        **fields,
    )
    tags = [TagValue(k, v) for k, v in filters.items()]
    data = await ib.reqScannerDataAsync(sub, scannerSubscriptionFilterOptions=tags)
    return [d.contractDetails.contract.symbol for d in data]


async def build_watchlist(presets: list[str], rows: int, max_symbols: int) -> list[str]:
    cfg = ib_config_from_env()
    ib = IB()
    await ib.connectAsync(cfg.host, cfg.port, clientId=cfg.client_id + 200)
    try:
        symbols: list[str] = []
        for preset in presets:
            found = await scan(ib, preset, rows)
            log.info("scan %s: %s", preset, " ".join(found))
            symbols.extend(found)
    finally:
        ib.disconnect()
    unique = list(dict.fromkeys(symbols))
    if len(unique) > max_symbols:
        log.warning("keeping the first %d of %d scanned symbols", max_symbols, len(unique))
    return unique[:max_symbols]


def read_watchlist(path: Path) -> list[str]:
    lines = (line.split("#", 1)[0].strip().upper() for line in path.read_text().splitlines())
    return [s for s in lines if s]


def main() -> None:
    parser = argparse.ArgumentParser(prog="trading_agent.scan")
    parser.add_argument("--preset", nargs="+", choices=sorted(PRESETS), required=True)
    parser.add_argument("--rows", type=int, default=20, help="results per preset")
    parser.add_argument("--max-symbols", type=int, default=30)
    parser.add_argument("--out", type=Path, default=DEFAULT_WATCHLIST)
    args = parser.parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    symbols = asyncio.run(build_watchlist(args.preset, args.rows, args.max_symbols))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(symbols) + "\n")
    print(f"{len(symbols)} symbols -> {args.out}: {' '.join(symbols)}")


if __name__ == "__main__":
    main()
