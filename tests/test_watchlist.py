import argparse
import asyncio

from trading_agent.__main__ import resolve_symbols
from trading_agent.broker.ibkr import IBKRBroker
from trading_agent.config import IBConfig
from trading_agent.scan import read_watchlist
from trading_agent.sources import EdgarSource


def test_watchlist_file_and_symbol_resolution(tmp_path):
    f = tmp_path / "w.txt"
    f.write_text("sofi\n# comment\nPLTR  # growth\n\nAAPL\n")
    assert read_watchlist(f) == ["SOFI", "PLTR", "AAPL"]
    args = argparse.Namespace(
        symbols=["aapl", "MSFT"], symbols_file=f, scan=None, mode="paper", max_symbols=3
    )
    assert resolve_symbols(args) == ["AAPL", "MSFT", "SOFI"]  # de-duplicated, capped
    empty = argparse.Namespace(symbols=[], symbols_file=None, scan=None, mode="sim", max_symbols=30)
    assert resolve_symbols(empty) == ["AAPL"]


class _Event(list):
    def __iadd__(self, handler):
        self.append(handler)
        return self


class FakeIB:
    exchanges = {"AAPL": "NASDAQ", "SCAM": "PINK"}

    def __init__(self):
        self.pendingTickersEvent = _Event()
        self.subscribed = []

    async def qualifyContractsAsync(self, *contracts):
        for i, c in enumerate(contracts, 1):
            if c.symbol in self.exchanges:
                c.conId, c.primaryExchange = i, self.exchanges[c.symbol]

    def reqMktData(self, contract, *args):
        self.subscribed.append(contract.symbol)


def test_otc_and_unknown_symbols_are_skipped():
    broker = IBKRBroker(IBConfig())
    broker.ib = FakeIB()
    asyncio.run(broker.subscribe(["AAPL", "SCAM", "NOPE"], lambda t: None))
    assert broker.ib.subscribed == ["AAPL"]
    assert broker.contract("SCAM") is None


def test_dilution_filings_are_tracked():
    subs = {
        "name": "Tiny Co",
        "filings": {
            "recent": {
                "accessionNumber": ["0001-24-000001"],
                "form": ["424B5"],
                "acceptanceDateTime": ["2026-09-24T08:00:00.000Z"],
                "primaryDocument": ["p.htm"],
                "primaryDocDescription": ["424B5"],
                "items": [""],
            }
        },
    }
    [filing] = EdgarSource("ua").parse_submissions("TINY", 1, subs)
    assert "dilution" in filing.item.body
