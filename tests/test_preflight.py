import asyncio

import pytest

from trading_agent.broker.ibkr import IBKRBroker, is_paper_account
from trading_agent.config import IBConfig
from trading_agent.preflight import FAIL, PASS, WARN, check_local, evaluate_equity, render


def test_account_size_check():
    assert evaluate_equity(27_000, 25_000).status == PASS
    big = evaluate_equity(1_000_000, 25_000)
    assert big.status == FAIL and "40x" in big.detail
    assert evaluate_equity(24_000, 25_000).status == WARN  # pattern day trader limit
    assert evaluate_equity(None, 25_000).status == FAIL


def test_paper_account_ids():
    assert is_paper_account("DU1234567")
    assert not is_paper_account("U1234567")


class _Event(list):
    def __iadd__(self, h):
        self.append(h)
        return self


class FakeIB:
    def __init__(self, accounts):
        self.accounts = accounts
        self.errorEvent = _Event()
        self.pendingTickersEvent = _Event()
        self.disconnected = False

    async def connectAsync(self, *a, **kw):
        pass

    def managedAccounts(self):
        return self.accounts

    def disconnect(self):
        self.disconnected = True


def test_paper_mode_refuses_a_live_account():
    broker = IBKRBroker(IBConfig(), expect_paper=True)
    broker.ib = FakeIB(["U7654321"])
    with pytest.raises(RuntimeError, match="paper account"):
        asyncio.run(broker.connect())
    assert broker.ib.disconnected
    ok = IBKRBroker(IBConfig(), expect_paper=True)
    ok.ib = FakeIB(["DU7654321"])
    asyncio.run(ok.connect())


def test_local_checks_render(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_FILE", str(tmp_path / "j.db"))
    monkeypatch.setenv("TRADING_ALLOW_LIVE", "no")
    results = check_local()
    assert all(r.status == PASS for r in results)
    assert "[PASS] data directory" in render(results)
