import asyncio
import math

from trading_agent.account import AccountConfig, AccountGuard
from trading_agent.broker.sim import SimBroker
from trading_agent.config import RiskLimits
from trading_agent.costs import CostModel
from trading_agent.engine import Engine
from trading_agent.models import Side, Signal, Tick
from trading_agent.risk import RiskManager
from trading_agent.signals import SignalFusion, SignalHub
from trading_agent.strategy import FusedSignalStrategy
from trading_agent.strategy.base import Strategy
from trading_agent.universe import Universe, VolatilityTracker

NOW = 1000.0


def universe(equity=25_000):
    return Universe(AccountGuard(AccountConfig(starting_equity=equity)))


def quote(sym, bid, ask, ts=NOW, bs=0, as_=0):
    return Tick(sym, bid, ask, (bid + ask) / 2, ts, bs, as_)


def test_tiers_and_sizing():
    u = universe()
    assert u.tier(50.0).name == "standard"
    assert u.tier(3.0).name == "penny"
    assert u.tier(0.50) is None
    # Standard: $250 risk / 2% default stop = $12,500 -> 250 shares at $50.
    assert u.max_shares("BIG", 50.0) == 250
    # Penny: 0.5% risk = $125 / 8% default stop = $1,562 -> 520 shares at $3.
    assert u.max_shares("TINY", 3.0) == 520
    assert u.max_shares("SUB", 0.50) == 0


def test_stops_scale_with_volatility():
    u = universe()
    calm, wild = "CALM", "WILD"
    for i in range(31):
        u.update(quote(calm, 50.0 + 0.01 * (i % 2), 50.02 + 0.01 * (i % 2), NOW + 60 * i))
        wild_mid = 50.0 * (1.01 if i % 2 else 0.99)
        u.update(quote(wild, wild_mid - 0.01, wild_mid + 0.01, NOW + 60 * i))
    assert u.stop_pct(calm, 50.0) == 1.0  # floor for the standard tier
    assert u.stop_pct(wild, 50.0) == 5.0  # capped
    assert u.max_shares(wild, 50.0) == 100  # wider stop -> smaller position ($5,000)


def test_penny_entries_need_a_very_confident_fresh_view():
    u = universe()
    q = quote("TINY", 2.99, 3.01)
    weak = Signal("TINY", "llm:news", 0.7, 0.9, NOW, 600)
    strong = Signal("TINY", "llm:news", 0.9, 0.85, NOW, 600)
    assert not u.entry_allowed(q, False, None)[0]
    assert not u.entry_allowed(q, False, (weak, 1.0))[0]
    assert not u.entry_allowed(q, False, (strong, 0.3))[0]  # stale
    assert u.entry_allowed(q, False, (strong, 0.9))[0]
    assert not u.entry_allowed(q, True, (strong, 0.9))[0]  # no shorting penny stocks
    ok, why = u.entry_allowed(quote("TINY", 2.90, 3.10), False, (strong, 0.9))
    assert not ok and "spread" in why


def test_standard_spread_filter():
    u = universe()
    assert u.entry_allowed(quote("SMALL", 20.00, 20.04), False, None)[0]  # 0.2%
    assert not u.entry_allowed(quote("SMALL", 20.00, 20.20), False, None)[0]  # 1%


def make_strategy():
    hub = SignalHub(clock=lambda: NOW)
    fusion = SignalFusion(hub, max_position=5_000)
    u = universe()
    strat = FusedSignalStrategy(["TINY", "BIG"], hub, fusion, universe=u, costs=CostModel(0, 0))
    return hub, strat, u


def test_strategy_penny_rules_end_to_end():
    hub, strat, _ = make_strategy()
    hub.publish(Signal("TINY", "llm:news", 0.7, 0.9, NOW, 600))
    assert strat.on_tick(quote("TINY", 2.99, 3.01, bs=5000, as_=10), 0) == []
    assert "penny tier needs" in strat.explain("TINY")["blocked"]
    hub.publish(Signal("TINY", "llm:news", 0.9, 0.9, NOW, 600))
    [order] = strat.on_tick(quote("TINY", 2.99, 3.01, bs=5000, as_=10), 0)
    # Primary view only (order-book imbalance ignored for penny stocks):
    # 0.6 x 0.9 x 0.9 = 0.486 of 520 max shares.
    assert (order.side, order.qty) == (Side.BUY, round(0.486 * 520))


class Idle(Strategy):
    def __init__(self):
        super().__init__(["BIG"])

    def on_tick(self, tick, position):
        return []


def test_engine_uses_the_stop_fixed_at_entry():
    u = universe()
    for i in range(31):  # volatile history: 5% stop
        mid = 50.0 * (1.01 if i % 2 else 0.99)
        u.update(quote("BIG", mid - 0.01, mid + 0.01, NOW - 3600 + 60 * i))
    broker = SimBroker()
    clock = lambda: NOW  # noqa: E731
    risk = RiskManager(RiskLimits(max_order_notional=1e9), clock=clock)
    engine = Engine(
        broker,
        Idle(),
        risk,
        clock=clock,
        account=u.account,
        stop_pct=u.stop_pct,
    )
    asyncio.run(engine.startup())
    broker._on_tick = engine.on_tick
    from trading_agent.models import Fill

    engine._track_account(Fill("x", "BIG", Side.BUY, 100, 50.0, NOW))
    risk.on_fill(Fill("x", "BIG", Side.BUY, 100, 50.0, NOW))
    assert math.isclose(engine._entry_stop["BIG"], 5.0)
    broker.push_tick(quote("BIG", 48.50, 48.52))  # -3%: inside a 5% stop, no exit
    assert broker.open_orders == {}
    broker.push_tick(quote("BIG", 47.40, 47.42))  # -5.2%: stop
    [order] = broker.open_orders.values()
    assert (order.side, order.qty) == (Side.SELL, 100)


def test_volatility_tracker_needs_history():
    v = VolatilityTracker()
    v.update(quote("X", 10, 10.01))
    assert v.per_minute("X") is None
