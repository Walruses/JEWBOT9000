"""Pre-flight checks before paper (or live) trading.

    python -m trading_agent.preflight --symbols AAPL MSFT
    python -m trading_agent.preflight --symbols-file data/watchlist.txt

Checks the IBKR connection and that it's a paper account, the account size against
ACCOUNT_EQUITY (paper accounts often start at $1,000,000, which would scale every
position up), live market data, IBKR news, open positions and orders, every API key the
data sources and analyst need, and that the data directory is writable. Nothing is
traded. Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
from dataclasses import dataclass
from pathlib import Path

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Result:
    status: str
    check: str
    detail: str


def evaluate_equity(net_liq: float | None, expected: float) -> Result:
    if net_liq is None:
        return Result(FAIL, "account size", "IBKR did not report NetLiquidation")
    if net_liq < 25_000:
        return Result(
            WARN,
            "account size",
            f"${net_liq:,.0f} is under $25,000: the pattern day trader limit "
            "(3 day trades per 5 business days) will apply",
        )
    if net_liq > expected * 1.5:
        return Result(
            FAIL,
            "account size",
            f"${net_liq:,.0f}, but ACCOUNT_EQUITY is ${expected:,.0f}. Positions are "
            f"sized from the real balance, so they'd be ~{net_liq / expected:.0f}x "
            "the plan. Reset the paper account balance (Client Portal > Settings > "
            "Paper Trading Account > Reset) to your planned amount.",
        )
    return Result(PASS, "account size", f"${net_liq:,.0f}")


async def check_ibkr(symbols: list[str], expected_equity: float) -> list[Result]:
    from ib_async import IB, Stock

    from .broker.ibkr import OTC_EXCHANGES, is_paper_account
    from .config import PAPER_PORTS, ib_config_from_env

    out: list[Result] = []
    cfg = ib_config_from_env()
    port_kind = "paper" if cfg.port in PAPER_PORTS else "LIVE" if cfg.is_live_port else "custom"
    out.append(
        Result(
            PASS if port_kind == "paper" else FAIL,
            "IBKR port",
            f"{cfg.host}:{cfg.port} ({port_kind})",
        )
    )
    ib = IB()
    try:
        await ib.connectAsync(cfg.host, cfg.port, clientId=cfg.client_id + 300, timeout=10)
    except Exception as e:
        out.append(
            Result(
                FAIL,
                "IBKR connection",
                f"{e!r}. Is IB Gateway/TWS running and logged in, with "
                "Configure > API > Settings > 'Enable ActiveX and Socket Clients' "
                f"on, port {cfg.port}, and 'Read-Only API' off?",
            )
        )
        return out
    try:
        accounts = list(ib.managedAccounts())
        paper = accounts and all(is_paper_account(a) for a in accounts)
        out.append(
            Result(
                PASS if paper else FAIL,
                "paper account",
                f"accounts {accounts}"
                + (
                    ""
                    if paper
                    else ": paper account IDs start with 'D'; log in to the paper account"
                ),
            )
        )

        values = {v.tag: v.value for v in ib.accountValues() if v.currency in ("USD", "BASE")}
        net_liq = float(values["NetLiquidation"]) if "NetLiquidation" in values else None
        out.append(evaluate_equity(net_liq, expected_equity))

        positions = [(p.contract.symbol, p.position) for p in ib.positions()]
        orders = [t for t in ib.openTrades() if not t.isDone()]
        out.append(
            Result(
                WARN if positions or orders else PASS,
                "existing positions/orders",
                f"positions {positions or 'none'}, {len(orders)} open orders"
                + (
                    " (the agent adopts positions in its symbols and cancels its "
                    "own leftover orders)"
                    if positions or orders
                    else ""
                ),
            )
        )

        contracts = [Stock(s, "SMART", "USD") for s in symbols]
        await ib.qualifyContractsAsync(*contracts)
        bad = [c.symbol for c in contracts if not c.conId]
        otc = [c.symbol for c in contracts if c.conId and c.primaryExchange in OTC_EXCHANGES]
        out.append(
            Result(
                FAIL if bad else WARN if otc else PASS,
                "symbols",
                f"{len(contracts)} checked"
                + (f"; not found: {bad}" if bad else "")
                + (f"; OTC, will be skipped: {otc}" if otc else ""),
            )
        )

        good = [c for c in contracts if c.conId and c.symbol not in otc]
        if good:
            ticker = ib.reqMktData(good[0], "", True, False)  # one snapshot
            await asyncio.sleep(4)
            bid, ask = ticker.bid, ticker.ask
            live = bid and ask and not (math.isnan(bid) or math.isnan(ask)) and bid > 0
            dtype = getattr(ticker, "marketDataType", 1)
            if live and dtype == 1:
                out.append(Result(PASS, "market data", f"{good[0].symbol} {bid} x {ask}"))
            elif live:
                out.append(
                    Result(
                        WARN,
                        "market data",
                        f"{good[0].symbol} quotes are delayed (type {dtype}); "
                        "real-time data needs a subscription",
                    )
                )
            else:
                out.append(
                    Result(
                        FAIL,
                        "market data",
                        f"no quote for {good[0].symbol}. Outside market hours this "
                        "can be normal; otherwise subscribe to US real-time data and, "
                        "for a paper account, share the live account's market data "
                        "(Client Portal > Settings > Paper Trading Account)",
                    )
                )

        providers = await ib.reqNewsProvidersAsync()
        codes = [p.code for p in providers]
        out.append(
            Result(
                PASS if codes else WARN,
                "IBKR news",
                f"providers {codes}"
                if codes
                else "no news providers enabled; IBKR news will be empty",
            )
        )
    finally:
        ib.disconnect()
    return out


async def check_services() -> list[Result]:
    import httpx

    from .config import data_config_from_env

    cfg = data_config_from_env()
    out: list[Result] = []

    # Claude: confirm credentials and that each configured model is available.
    try:
        import anthropic

        client = anthropic.AsyncAnthropic()
        models = {
            cfg.analyst_model,
            cfg.analyst_routine_model,
            os.environ.get("REVIEW_MODEL", "claude-fable-5-1"),
        } - {""}
        for model in sorted(models):
            await client.models.retrieve(model)
        out.append(Result(PASS, "Anthropic API", f"models available: {sorted(models)}"))
    except Exception as e:
        out.append(
            Result(FAIL, "Anthropic API", f"{e!r}. Set ANTHROPIC_API_KEY (or `ant auth login`).")
        )

    async with httpx.AsyncClient(timeout=10) as http:
        if cfg.sec_user_agent:
            try:
                r = await http.get(
                    "https://www.sec.gov/files/company_tickers.json",
                    headers={"User-Agent": cfg.sec_user_agent},
                )
                r.raise_for_status()
                out.append(Result(PASS, "SEC EDGAR", "reachable"))
            except Exception as e:
                out.append(Result(FAIL, "SEC EDGAR", repr(e)))
        else:
            out.append(Result(WARN, "SEC EDGAR", "SEC_USER_AGENT not set: filings disabled"))

        if cfg.finnhub_api_key:
            try:
                r = await http.get(
                    "https://finnhub.io/api/v1/quote",
                    params={"symbol": "AAPL", "token": cfg.finnhub_api_key},
                )
                r.raise_for_status()
                out.append(Result(PASS, "Finnhub", "key accepted"))
            except Exception as e:
                out.append(Result(FAIL, "Finnhub", repr(e)))
        else:
            out.append(Result(WARN, "Finnhub", "FINNHUB_API_KEY not set: news API disabled"))

        if cfg.reddit_client_id and cfg.reddit_client_secret:
            try:
                r = await http.post(
                    "https://www.reddit.com/api/v1/access_token",
                    auth=(cfg.reddit_client_id, cfg.reddit_client_secret),
                    data={"grant_type": "client_credentials"},
                    headers={"User-Agent": cfg.reddit_user_agent},
                )
                r.raise_for_status()
                out.append(Result(PASS, "Reddit", "credentials accepted"))
            except Exception as e:
                out.append(Result(FAIL, "Reddit", repr(e)))
        else:
            out.append(Result(WARN, "Reddit", "REDDIT_CLIENT_ID/SECRET not set: disabled"))
    return out


def check_local() -> list[Result]:
    from .config import live_trading_allowed, runtime_config_from_env

    out = []
    rt = runtime_config_from_env()
    try:
        probe = Path(rt.journal_file).parent / ".write_test"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok")
        probe.unlink()
        out.append(Result(PASS, "data directory", f"{probe.parent} writable"))
    except OSError as e:
        out.append(Result(FAIL, "data directory", repr(e)))
    out.append(
        Result(
            WARN if live_trading_allowed() else PASS,
            "live trading switch",
            "TRADING_ALLOW_LIVE=yes: set it back to no for the training period"
            if live_trading_allowed()
            else "off",
        )
    )
    return out


def render(results: list[Result]) -> str:
    width = max(len(r.check) for r in results)
    return "\n".join(f"[{r.status}] {r.check:<{width}}  {r.detail}" for r in results)


async def run(symbols: list[str]) -> list[Result]:
    from .config import account_config_from_env
    from .tuning import apply_to_environment

    apply_to_environment()
    expected = account_config_from_env().starting_equity
    return check_local() + await check_ibkr(symbols, expected) + await check_services()


def main() -> None:
    parser = argparse.ArgumentParser(prog="trading_agent.preflight")
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--symbols-file", type=Path)
    args = parser.parse_args()
    symbols = [s.upper() for s in args.symbols]
    if args.symbols_file:
        from .scan import read_watchlist

        symbols += read_watchlist(args.symbols_file)
    results = asyncio.run(run(symbols or ["AAPL"]))
    print(render(results))
    failed = [r for r in results if r.status == FAIL]
    print(f"\n{len(failed)} failed, {sum(r.status == WARN for r in results)} warnings")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
