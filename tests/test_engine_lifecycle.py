import asyncio
from datetime import datetime

from trading_agent.broker.sim import SimBroker
from trading_agent.config import RiskLimits
from trading_agent.engine import Engine
from trading_agent.models import OrderIntent, Side, Tick
from trading_agent.risk import RiskManager
from trading_agent.session import NEW_YORK, TradingSession
from trading_agent.state import StateStore
from trading_agent.strategy.base import Strategy


class Idle(Strategy):
    def __init__(self):
        super().__init__(["AAPL"])

    def on_tick(self, tick, position):
        return []


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def et(h, mi):
    return datetime(2026, 9, 24, h, mi, tzinfo=NEW_YORK).timestamp()


def tick(ts, bid=100.00, ask=100.02):
    return Tick("AAPL", bid, ask, (bid + ask) / 2, ts)


class HeldBroker(SimBroker):
    """Simulator that reports a pre-existing position and one leftover order."""

    def __init__(self, held):
        super().__init__()
        self.held = held
        self.cancel_all_calls = 0

    async def positions(self):
        return self.held

    def cancel_all(self):
        self.cancel_all_calls += 1
        super().cancel_all()


def make(broker, clock, tmp_path=None, **limits):
    risk = RiskManager(RiskLimits(**limits), clock=clock)
    store = StateStore(tmp_path / "state.json") if tmp_path else None
    engine = Engine(broker, Idle(), risk, clock=clock, session=TradingSession(), state_store=store)
    broker._on_tick = engine.on_tick
    return engine, risk, store


def test_startup_loads_broker_positions_and_cancels_leftovers(tmp_path):
    broker = HeldBroker({"AAPL": (30, 99.5), "TSLA": (5, 200.0)})
    engine, risk, store = make(broker, Clock(et(10, 0)), tmp_path)
    asyncio.run(engine.startup())
    assert broker.cancel_all_calls == 1
    assert risk.position("AAPL") == 30
    assert risk.position("TSLA") == 0  # not ours to manage
    assert store.path.exists()


def test_startup_restores_todays_loss_and_halt(tmp_path):
    store = StateStore(tmp_path / "state.json")
    from trading_agent.state import DailyState

    store.save(DailyState("2026-09-24", -40.0, True, "daily loss limit hit"))
    engine, risk, _ = make(HeldBroker({}), Clock(et(10, 0)), tmp_path)
    asyncio.run(engine.startup())
    assert risk.realized_pnl == -40.0 and risk.halted


def test_flatten_phase_closes_positions_and_opens_nothing():
    broker = HeldBroker({"AAPL": (30, 99.5)})
    clock = Clock(et(15, 51))
    engine, risk, _ = make(broker, clock)
    asyncio.run(engine.startup())
    broker.push_tick(tick(clock.t))
    [order] = broker.open_orders.values()
    assert (order.side, order.qty, order.limit_price) == (Side.SELL, 30, 100.00)
    broker.push_tick(tick(clock.t, bid=100.00, ask=100.01))
    assert risk.position("AAPL") == 0


def test_no_orders_outside_session():
    broker = HeldBroker({"AAPL": (30, 99.5)})
    engine, risk, _ = make(broker, Clock(et(8, 0)))
    asyncio.run(engine.startup())
    broker.push_tick(tick(et(8, 0)))
    assert broker.open_orders == {}


def test_loss_halt_flattens():
    broker = HeldBroker({"AAPL": (100, 100.0)})
    clock = Clock(et(11, 0))
    engine, risk, _ = make(broker, clock, max_daily_loss=50)
    asyncio.run(engine.startup())
    broker.push_tick(tick(clock.t, bid=99.0, ask=99.02))  # -99 unrealized -> halt
    assert risk.halted
    [order] = broker.open_orders.values()
    assert order == OrderIntent("AAPL", Side.SELL, 100, 99.0)


def test_persistent_position_mismatch_halts_without_trading():
    broker = HeldBroker({"AAPL": (30, 99.5)})
    engine, risk, _ = make(broker, Clock(et(11, 0)))
    asyncio.run(engine.startup())
    broker.held = {"AAPL": (10, 99.5)}  # e.g. someone traded manually
    asyncio.run(engine.reconcile())
    assert not risk.halted  # one strike can be a race with a fill in flight
    asyncio.run(engine.reconcile())
    assert risk.halted and not risk.allow_flatten
    broker.push_tick(tick(et(11, 0)))
    assert broker.open_orders == {}
