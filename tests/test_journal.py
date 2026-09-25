import sqlite3

import pytest

from trading_agent.journal import Journal
from trading_agent.models import Fill, NewsItem, OrderIntent, Side, Signal, Tick
from trading_agent.report import (
    MIN_SAMPLES,
    build_report,
    load_quality,
    render,
    render_trade,
    suggest_fusion_weights,
    track_records,
    write_quality,
)
from trading_agent.signals.fusion import DEFAULT_WEIGHTS


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def ctx(llm=0.4, imbalance=0.1, reversion=-0.05, signal_id="sig1"):
    return {
        "conviction": llm + imbalance + reversion,
        "target": 10,
        "signals": [
            {
                "source": "llm:news",
                "signal_id": signal_id,
                "score": 1,
                "confidence": 1,
                "decay": 1,
                "weight": 0.6,
                "contribution": llm,
            },
            {
                "source": "micro:imbalance",
                "signal_id": "",
                "score": 1,
                "confidence": 1,
                "decay": 1,
                "weight": 0.25,
                "contribution": imbalance,
            },
            {
                "source": "micro:reversion",
                "signal_id": "",
                "score": -1,
                "confidence": 1,
                "decay": 1,
                "weight": 0.15,
                "contribution": reversion,
            },
        ],
    }


def fill(side, qty, px, fee=0.0, sym="AAPL"):
    return Fill("o", sym, side, qty, px, 0.0, fee)


def rows(j, sql):
    j.db.commit()
    return j.db.execute(sql).fetchall()


def open_and_close(j, entry=100.0, exit_=101.0, direction=Side.BUY, context=None):
    d1 = j.record_decision(
        OrderIntent("AAPL", direction, 10, entry), entry, "strategy", context or ctx()
    )
    j.on_fill(fill(direction, 10, entry, fee=0.35), d1)
    closing = Side.SELL if direction is Side.BUY else Side.BUY
    d2 = j.record_decision(OrderIntent("AAPL", closing, 10, exit_), exit_, "strategy", {})
    j.on_fill(fill(closing, 10, exit_, fee=0.35), d2)
    return d1, d2


def test_round_trip_trade_pnl_and_attribution():
    j = Journal(":memory:", clock=Clock())
    d1, d2 = open_and_close(j)
    [(direction, gross, fees, net)] = rows(
        j, "SELECT direction, gross_pnl, fees, net_pnl FROM trades"
    )
    assert (direction, gross, fees) == (1, pytest.approx(10.0), pytest.approx(0.70))
    assert net == pytest.approx(9.30)
    attr = dict(rows(j, "SELECT source, share FROM trade_attribution"))
    # Only sources pointing the way of the trade get credit: 0.4 vs 0.1.
    assert attr == {"llm:news": pytest.approx(0.8), "micro:imbalance": pytest.approx(0.2)}
    pnl = sum(r[0] for r in rows(j, "SELECT pnl FROM trade_attribution"))
    assert pnl == pytest.approx(net)
    links = rows(j, "SELECT decision_id, role, qty FROM trade_decisions ORDER BY role")
    assert links == [(d1, "entry", 10), (d2, "exit", 10)]


def test_flip_splits_fill_into_close_and_new_trade():
    j = Journal(":memory:", clock=Clock())
    d1 = j.record_decision(OrderIntent("AAPL", Side.BUY, 10, 100), 100, "strategy", ctx())
    j.on_fill(fill(Side.BUY, 10, 100.0), d1)
    d2 = j.record_decision(
        OrderIntent("AAPL", Side.SELL, 15, 102),
        102,
        "strategy",
        ctx(llm=-0.5, imbalance=-0.1, reversion=0.0),
    )
    j.on_fill(fill(Side.SELL, 15, 102.0, fee=1.5), d2)  # 1.0 of fee to the close
    [(gross, fees)] = rows(j, "SELECT gross_pnl, fees FROM trades")
    assert (gross, fees) == (pytest.approx(20.0), pytest.approx(1.0))
    j.on_fill(fill(Side.BUY, 5, 101.0), None)  # close the 5-share short
    trades = rows(j, "SELECT direction, gross_pnl, fees FROM trades ORDER BY id")
    assert trades[1] == (-1, pytest.approx(5.0), pytest.approx(0.5))


def test_late_commission_updates_closed_trade():
    j = Journal(":memory:", clock=Clock())
    open_and_close(j)
    j.on_fill(Fill("o", "AAPL", Side.SELL, 0, 0.0, 0.0, 0.30), None)
    [(fees, net)] = rows(j, "SELECT fees, net_pnl FROM trades")
    assert (fees, net) == (pytest.approx(1.0), pytest.approx(9.0))
    assert sum(r[0] for r in rows(j, "SELECT pnl FROM trade_attribution")) == pytest.approx(9.0)


def test_inherited_position_is_unattributed():
    j = Journal(":memory:", clock=Clock())
    j.seed_position("AAPL", 10, 99.0)
    j.on_fill(fill(Side.SELL, 10, 100.0), None)
    assert rows(j, "SELECT inherited, gross_pnl FROM trades") == [(1, pytest.approx(10.0))]
    assert rows(j, "SELECT source, share FROM trade_attribution") == [("unattributed", 1.0)]


def news(i, source):
    return NewsItem(i, source, "AAPL", f"headline {i}", 900.0)


def test_signal_outcomes_at_horizons():
    j = Journal(":memory:", clock=Clock())
    sig = Signal(
        "AAPL",
        "llm:news",
        -0.8,
        0.9,
        1000.0,
        1800,
        "bad news",
        id="s1",
        inputs=("a", "b"),
        drivers=("a",),
    )
    j.record_signal(sig, [news("a", "edgar"), news("b", "reddit/r/stocks")])
    for dt, mid in [(0, 100.0), (61, 99.9), (301, 99.5), (1801, 99.0), (3601, 101.0)]:
        j.on_tick(Tick("AAPL", mid - 0.01, mid + 0.01, mid, 1000.0 + dt))
    out = dict(rows(j, "SELECT horizon, directional_bps FROM signal_outcomes"))
    assert out[1800.0] == pytest.approx(100.0)  # fell 1% as predicted
    assert out[3600.0] == pytest.approx(-100.0)
    assert rows(j, "SELECT news_source, driver FROM signal_inputs ORDER BY driver") == [
        ("reddit/r/stocks", 0),
        ("edgar", 1),
    ]


def make_history(path, n_good, n_bad):
    """n_good winning trades driven by an edgar signal that predicted correctly, n_bad
    losing trades driven by a reddit signal that predicted wrongly."""
    clock = Clock()
    j = Journal(path, clock=clock)
    for k in range(n_good + n_bad):
        good = k < n_good
        sid = f"s{k}"
        src = "edgar" if good else "reddit/r/wsb"
        sig = Signal(
            "AAPL",
            "llm:news",
            1.0,
            1.0,
            clock.t,
            3600,
            "",
            id=sid,
            inputs=(f"n{k}",),
            drivers=(f"n{k}",),
        )
        j.record_signal(sig, [news(f"n{k}", src)])
        j.on_tick(Tick("AAPL", 99.99, 100.01, 100.0, clock.t))
        end = 100.5 if good else 99.5
        j.on_tick(Tick("AAPL", end - 0.01, end + 0.01, end, clock.t + 1801))
        open_and_close(j, 100.0, end, context=ctx(signal_id=sid))
        clock.t += 4000
    j.close()


def test_report_scores_sources_and_suggests_weights(tmp_path):
    db_path = tmp_path / "journal.db"
    make_history(db_path, n_good=MIN_SAMPLES + 5, n_bad=3)
    db = sqlite3.connect(db_path)
    report = build_report(db)

    assert report.trades["count"] == MIN_SAMPLES + 8
    edgar, wsb = report.news_sources["edgar"], report.news_sources["reddit/r/wsb"]
    assert (edgar.scored, edgar.hits) == (MIN_SAMPLES + 5, MIN_SAMPLES + 5)
    assert (wsb.scored, wsb.hits) == (3, 0)
    assert edgar.pnl > 0 > wsb.pnl
    assert edgar.avg_bps(1800.0) == pytest.approx(50.0)

    weights = suggest_fusion_weights(report)
    assert weights["llm"] > DEFAULT_WEIGHTS["llm"]  # mostly winning trades
    assert weights["micro:reversion"] == DEFAULT_WEIGHTS["micro:reversion"]  # no support

    records = track_records(report)
    assert records["edgar"].startswith("right 15 of 15")
    assert "reddit/r/wsb" not in records  # too few samples to show

    quality_file = tmp_path / "quality.json"
    write_quality(report, quality_file)
    assert load_quality(quality_file)["fusion_weights"] == weights
    write_quality(report, quality_file)  # idempotent: no compounding
    assert load_quality(quality_file)["fusion_weights"] == weights

    text = render(report, DEFAULT_WEIGHTS)
    assert "edgar" in text and "FUSION SOURCES" in text
    detail = render_trade(db, 1)
    assert "ENTRY decision" in detail and "headline n0" in detail
