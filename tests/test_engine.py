from trading_agent.broker.sim import SimBroker
from trading_agent.config import RiskLimits
from trading_agent.engine import Engine
from trading_agent.models import OrderIntent, Side, Tick
from trading_agent.risk import RiskManager
from trading_agent.strategy.base import Strategy


class ScriptedStrategy(Strategy):
    """Buys `qty` whenever flat."""

    def __init__(self, qty=10):
        super().__init__(["AAPL"])
        self.qty = qty

    def on_tick(self, tick, position):
        return [] if position else [OrderIntent("AAPL", Side.BUY, self.qty, tick.bid)]


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def tick(bid, ask):
    return Tick("AAPL", bid, ask, (bid + ask) / 2, 0.0)


def setup(**limits):
    broker = SimBroker()
    clock = Clock()
    risk = RiskManager(RiskLimits(**limits), clock=clock)
    engine = Engine(broker, ScriptedStrategy(), risk, order_ttl=2.0, clock=clock)
    broker._on_tick = engine.on_tick
    return broker, engine, risk, clock


def test_order_placed_then_filled_updates_position():
    broker, engine, risk, _ = setup()
    broker.push_tick(tick(100.00, 100.02))
    assert len(broker.open_orders) == 1
    broker.push_tick(tick(99.98, 100.00))  # ask trades down through our bid
    assert risk.position("AAPL") == 10
    assert engine.working == {}


def test_stale_order_cancelled_and_exposure_released():
    broker, engine, risk, clock = setup()
    broker.push_tick(tick(100.00, 100.02))
    clock.t = 2.5
    broker.push_tick(tick(100.01, 100.03))  # cancels stale order, then re-quotes
    assert len(broker.open_orders) == 1
    assert list(broker.open_orders.values())[0].limit_price == 100.01
    assert risk._pos("AAPL").pending_buy == 10


def test_halt_cancels_everything_and_stops_quoting():
    broker, engine, risk, _ = setup()
    broker.push_tick(tick(100.00, 100.02))
    risk.halt("test")
    broker.push_tick(tick(100.00, 100.02))
    assert broker.open_orders == {}
    assert engine.working == {}
