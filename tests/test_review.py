import asyncio
import json
import logging
import sqlite3
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from trading_agent import review as review_mod
from trading_agent import tuning
from trading_agent.__main__ import run_config_snapshot
from trading_agent.analyst import ClaudeAnalyst
from trading_agent.journal import Journal, JournalLogHandler
from trading_agent.models import Fill, NewsItem, OrderIntent, Side, Signal, Tick
from trading_agent.session import NEW_YORK

DAY = date(2026, 9, 24)


def et(h, m, s=0):
    return datetime(2026, 9, 24, h, m, s, tzinfo=NEW_YORK).timestamp()


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def build_day(path):
    """A small but complete trading day in a journal."""
    clock = Clock(et(9, 25))
    j = Journal(path, clock=clock)
    j.start_run("paper", ["AAPL"], {"limits": {"max_position": 5000}}, "abc123")
    item = NewsItem("n1", "edgar", "AAPL", "8-K: results", et(9, 20), "Revenue rose " * 500)
    sig = Signal(
        "AAPL",
        "llm:news",
        0.8,
        0.9,
        et(9, 30),
        3600,
        "strong quarter",
        id="s1",
        inputs=("n1",),
        drivers=("n1",),
        model="claude-opus-5",
    )
    j.record_signal(sig, [item])
    j.record_llm_call(
        ts=et(9, 30),
        symbol="AAPL",
        model="claude-opus-5",
        prompt="p",
        response='{"material": true}',
        stop_reason="end_turn",
        input_tokens=2000,
        output_tokens=500,
        latency_s=3.2,
        signal_id="s1",
    )
    for minute in range(0, 70):  # prices rising after the signal
        ts = et(9, 30) + minute * 60
        mid = 200 + minute * 0.05
        j.on_tick(Tick("AAPL", mid - 0.01, mid + 0.01, mid, ts, 300, 100))
    clock.t = et(9, 31)
    ctx = {
        "conviction": 0.43,
        "target": 40,
        "tier": "standard",
        "stop_pct": 1.5,
        "signals": [
            {
                "source": "llm:news",
                "signal_id": "s1",
                "score": 0.8,
                "confidence": 0.9,
                "decay": 1,
                "weight": 0.6,
                "contribution": 0.43,
            }
        ],
    }
    d1 = j.record_decision(OrderIntent("AAPL", Side.BUY, 40, 200.05), 200.06, "strategy", ctx)
    j.on_fill(Fill("o1", "AAPL", Side.BUY, 40, 200.05, et(9, 31), 0.35), d1)
    clock.t = et(10, 0)
    j.skipped("TINY", "tier_rules", "penny tier needs |score| >= 0.8; have 0.60")
    j.skipped("TINY", "tier_rules", "penny tier needs |score| >= 0.8; have 0.61")  # deduped
    j.event("stop_loss", "MSFT", position=10, stop_pct=2.0)
    clock.t = et(15, 51)
    d2 = j.record_decision(OrderIntent("AAPL", Side.SELL, 40, 203.4), 203.41, "eod_flatten", {})
    j.on_fill(Fill("o2", "AAPL", Side.SELL, 40, 203.40, et(15, 51), 0.35), d2)
    j.close()


def test_journal_records_review_data(tmp_path):
    path = tmp_path / "j.db"
    build_day(path)
    db = sqlite3.connect(path)
    [(exit_reason, tier, stop, net)] = db.execute(
        "SELECT exit_reason, tier, stop_pct, net_pnl FROM trades"
    ).fetchall()
    assert (exit_reason, tier, stop) == ("eod_flatten", "standard", 1.5)
    assert net == pytest.approx(40 * 3.35 - 0.70)
    assert db.execute("SELECT repeats FROM skipped").fetchall() == [(2,)]
    assert db.execute("SELECT length(body) FROM signal_inputs").fetchone()[0] > 4000
    assert db.execute("SELECT COUNT(*) FROM bars").fetchone()[0] == 70
    [(config, ended)] = db.execute("SELECT config, ended_at FROM runs").fetchall()
    assert json.loads(config)["limits"]["max_position"] == 5000 and ended


def test_bundle_contains_the_whole_day(tmp_path, monkeypatch):
    path = tmp_path / "j.db"
    build_day(path)
    monkeypatch.setattr(review_mod, "REVIEW_DIR", tmp_path / "reviews")
    out = review_mod.write_bundle(path, DAY, tmp_path / "reviews")
    b = json.loads((out / "bundle.json").read_text())

    [trade] = b["trades"]
    assert trade["exit_reason"] == "eod_flatten"
    assert [d["role"] for d in trade["decisions"]] == ["entry", "exit"]
    assert trade["decisions"][0]["context"]["signals"][0]["signal_id"] == "s1"
    [sig] = b["signals"]
    assert sig["used_in_trade"] and sig["outcomes"]["30m"]["directional_bps"] > 0
    assert sig["inputs"][0]["body"].endswith("chars in journal]")  # capped in the bundle
    assert b["skipped_summary"] == [{"stage": "tier_rules", "symbol": "TINY", "count": 2}]
    assert b["day_summary"]["llm_usage"]["claude-opus-5"]["cost_usd"] == pytest.approx(0.0225)
    assert b["price_bars_5m"]["AAPL"][0][0] == "09:30"
    assert "RISK_PER_TRADE_PCT" in b["tunable_parameters"]
    assert [e["kind"] for e in b["events"]] == ["stop_loss"]
    assert "1 trades" in (out / "summary.md").read_text()


class FakeStream:
    def __init__(self, message):
        self.message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self.message


REVIEW = {
    "summary": "One profitable trade on a strong 8-K.",
    "what_worked": ["edgar-driven long"],
    "what_failed": [],
    "anomalies": [],
    "source_assessment": [{"source": "edgar", "verdict": "keep", "evidence": "s1 +"}],
    "parameter_changes": [
        {
            "key": "STOP_VOL_MULTIPLE",
            "proposed": "2.5",
            "rationale": "r",
            "evidence": "e",
            "confidence": "medium",
        },
        {
            "key": "RISK_PER_TRADE_PCT",
            "proposed": "2.0",
            "rationale": "r",
            "evidence": "e",
            "confidence": "low",
        },
        {
            "key": "TRADING_ALLOW_LIVE",
            "proposed": "yes",
            "rationale": "r",
            "evidence": "e",
            "confidence": "high",
        },
    ],
    "strategy_suggestions": [],
    "data_gaps": [],
}


def test_reviewer_proposals_are_validated_and_applied_only_on_approval(tmp_path, monkeypatch):
    path = tmp_path / "j.db"
    build_day(path)
    monkeypatch.setattr(review_mod, "REVIEW_DIR", tmp_path / "reviews")
    monkeypatch.setattr(tuning, "DEFAULT_TUNED_FILE", tmp_path / "tuned.json")
    monkeypatch.setattr(tuning, "HISTORY_FILE", tmp_path / "history.jsonl")

    calls = []
    message = SimpleNamespace(
        stop_reason="end_turn",
        model="claude-fable-5-1",
        content=[SimpleNamespace(type="text", text=json.dumps(REVIEW))],
        usage=SimpleNamespace(input_tokens=90_000, output_tokens=4_000),
    )

    class Client:
        beta = SimpleNamespace(
            messages=SimpleNamespace(stream=lambda **kw: calls.append(kw) or FakeStream(message))
        )

    async def reviewer(bundle_json, model, effort="high", client=None):
        return await review_mod.__dict__["_orig_run_reviewer"](bundle_json, model, effort, Client())

    monkeypatch.setitem(review_mod.__dict__, "_orig_run_reviewer", review_mod.run_reviewer)
    monkeypatch.setattr(review_mod, "run_reviewer", reviewer)
    out = asyncio.run(review_mod.analyze_day(DAY, path, "claude-fable-5-1"))

    [call] = calls
    assert call["model"] == "claude-fable-5-1" and call["fallbacks"] == "default"
    assert "thinking" not in call  # always on for this model; must be omitted
    saved = json.loads((out / "review.json").read_text())
    valid = {p["key"]: p["valid"] for p in saved["validated_changes"]}
    assert valid == {
        "STOP_VOL_MULTIPLE": True,
        "RISK_PER_TRADE_PCT": False,
        "TRADING_ALLOW_LIVE": False,
    }
    assert "REJECTED" in (out / "review.md").read_text()

    assert not (tmp_path / "tuned.json").exists()  # nothing applied yet
    applied = tuning.approve(
        {"STOP_VOL_MULTIPLE": 2.5}, "test", tmp_path / "tuned.json", tmp_path / "history.jsonl"
    )
    assert applied == {"STOP_VOL_MULTIPLE": 2.5}
    assert tuning.load_tuned(tmp_path / "tuned.json") == {"STOP_VOL_MULTIPLE": 2.5}


def test_tunable_bounds():
    with pytest.raises(ValueError):
        tuning.TUNABLES["RISK_PER_TRADE_PCT"].validate(1.5)  # never above 1%
    with pytest.raises(ValueError):
        tuning.TUNABLES["ANALYST_EFFORT"].validate("max")
    with pytest.raises(ValueError):
        tuning.approve({"ACCOUNT_TYPE": "cash"}, "test")
    assert tuning.TUNABLES["PENNY_MIN_SCORE"].validate("0.9") == 0.9


def test_config_snapshot_redacts_secrets():
    from trading_agent.config import DataConfig

    snap = run_config_snapshot(data=DataConfig(finnhub_api_key="abc", reddit_client_secret="x"))
    assert snap["data"]["finnhub_api_key"] == "<redacted>"
    assert snap["data"]["reddit_client_secret"] == "<redacted>"
    assert snap["data"]["analyst_model"] == "claude-opus-5"


def test_warnings_become_journal_events(tmp_path):
    j = Journal(tmp_path / "j.db")
    handler = JournalLogHandler(j)
    logger = logging.getLogger("trading_agent.test")
    logger.addHandler(handler)
    logger.info("routine")
    logger.warning("IBKR error code=%s", 1100)
    logger.removeHandler(handler)
    j.db.commit()
    [(kind, detail)] = j.db.execute("SELECT kind, detail FROM events").fetchall()
    assert kind == "log" and "1100" in json.loads(detail)["message"]


def test_analyst_calls_are_logged():
    logged = []
    payload = {
        "material": True,
        "sentiment": 0.5,
        "confidence": 0.5,
        "horizon_minutes": 30,
        "drivers": [1],
        "rationale": "r",
    }
    msg = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
    )

    class Messages:
        async def create(self, **kw):
            return msg

    client = SimpleNamespace(messages=Messages(), beta=SimpleNamespace(messages=Messages()))
    analyst = ClaudeAnalyst(client=client)
    analyst.call_log = lambda **c: logged.append(c)
    item = NewsItem("r1", "reddit/r/stocks", "AAPL", "chatter", 1.0)
    sig = asyncio.run(analyst.analyze("AAPL", [item], now=100.0))
    [call] = logged
    assert call["signal_id"] == sig.id and call["model"] == "claude-sonnet-5"
    assert call["input_tokens"] == 1200 and "chatter" in call["prompt"]
    assert call["latency_s"] >= 0


def test_engine_and_strategy_skips_reach_the_journal(tmp_path):
    from trading_agent.costs import CostModel
    from trading_agent.signals import SignalFusion, SignalHub
    from trading_agent.strategy import FusedSignalStrategy

    j = Journal(tmp_path / "j.db", clock=lambda: 1000.0)
    hub = SignalHub(clock=lambda: 1000.0)
    strat = FusedSignalStrategy(
        ["AAPL"], hub, SignalFusion(hub, max_position=10), costs=CostModel(1.0, 1.0)
    )  # $1/share: nothing is worth it
    strat.skip_listener = lambda s, st, r, c: j.skipped(s, st, r, context=c)
    hub.publish(Signal("AAPL", "llm:news", 1.0, 1.0, 1000.0, 600))
    assert strat.on_tick(Tick("AAPL", 99.99, 100.01, 100.0, 1000.0), 0) == []
    j.db.commit()
    [(stage, reason, ctx)] = j.db.execute("SELECT stage, reason, context FROM skipped").fetchall()
    assert stage == "cost_check" and "round-trip cost" in reason
    assert json.loads(ctx)["signals"][0]["source"] == "llm:news"


def test_old_journal_is_upgraded_in_place(tmp_path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE signals (id TEXT PRIMARY KEY, ts REAL, symbol TEXT, source TEXT,
            score REAL, confidence REAL, ttl REAL, rationale TEXT);
        CREATE TABLE signal_inputs (signal_id TEXT, news_id TEXT, news_source TEXT,
            headline TEXT, url TEXT, published_at REAL, driver INTEGER);
        CREATE TABLE trades (id INTEGER PRIMARY KEY, symbol TEXT, direction INTEGER,
            opened_at REAL, closed_at REAL, max_qty INTEGER, avg_entry REAL, avg_exit REAL,
            gross_pnl REAL, fees REAL, net_pnl REAL, inherited INTEGER);
        INSERT INTO trades (symbol, net_pnl) VALUES ('AAPL', 5.0);
    """)
    db.close()
    j = Journal(path)
    cols = {r[1] for r in j.db.execute("PRAGMA table_info(trades)")}
    assert {"exit_reason", "tier", "stop_pct"} <= cols
    assert j.db.execute("SELECT net_pnl FROM trades").fetchone() == (5.0,)
