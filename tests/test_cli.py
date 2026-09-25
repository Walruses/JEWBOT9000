import pytest

from trading_agent.__main__ import build_broker


def test_paper_mode_refuses_live_port(monkeypatch):
    monkeypatch.setenv("IB_PORT", "7496")
    with pytest.raises(SystemExit):
        build_broker("paper")


def test_live_mode_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("IB_PORT", "4001")
    monkeypatch.setenv("TRADING_ALLOW_LIVE", "no")
    with pytest.raises(SystemExit):
        build_broker("live")
