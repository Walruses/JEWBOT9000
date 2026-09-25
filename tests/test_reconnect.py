import asyncio
import sqlite3

from trading_agent.broker.sim import SimBroker
from trading_agent.config import RiskLimits
from trading_agent.engine import Engine
from trading_agent.journal import Journal
from trading_agent.models import Fill, OrderIntent, Side
from trading_agent.risk import RiskManager
from trading_agent.strategy.base import Strategy


class Idle(Strategy):
    def __init__(self):
        super().__init__(["AAPL"])

    def on_tick(self, tick, position):
        return []


class FlakyBroker(SimBroker):
    """Drops the connection once; the first reconnect attempt fails."""

    def __init__(self):
        super().__init__()
        self.connected = True
        self.held = {"AAPL": (50, 100.0)}
        self.reconnect_attempts = 0
        self.resubscribed = 0

    def is_connected(self):
        return self.connected

    async def reconnect(self):
        self.reconnect_attempts += 1
        if self.reconnect_attempts == 1:
            raise ConnectionError("gateway still restarting")
        self.connected = True

    async def resubscribe(self):
        self.resubscribed += 1

    async def positions(self):
        return self.held


def test_engine_reconnects_and_resyncs_from_broker(tmp_path):
    broker = FlakyBroker()
    risk = RiskManager(RiskLimits())
    journal = Journal(tmp_path / "j.db")
    engine = Engine(broker, Idle(), risk, reconcile_interval=0.01, journal=journal)
    engine.reconnect_backoff = 0.01

    async def scenario():
        task = asyncio.create_task(engine.run())
        await engine.ready.wait()
        assert risk.position("AAPL") == 50
        # An order was working, then the gateway restarted and the position was sold
        # while we couldn't see it.
        risk.update_mark("AAPL", 101.0)
        risk.on_submit(OrderIntent("AAPL", Side.SELL, 50, 101.0))
        broker.connected = False
        broker.held = {}
        for _ in range(200):
            await asyncio.sleep(0.01)
            if broker.resubscribed:
                break
        engine.stop()
        await task

    asyncio.run(scenario())
    assert broker.reconnect_attempts == 2 and broker.resubscribed == 1
    assert risk.position("AAPL") == 0
    assert risk._pos("AAPL").pending_sell == 0  # stale reservation cleared
    journal.close()
    db = sqlite3.connect(tmp_path / "j.db")
    kinds = [k for (k,) in db.execute("SELECT kind FROM events ORDER BY id")]
    assert kinds[:3] == ["inherited_position", "broker_disconnected", "position_resync"]
    assert "broker_reconnected" in kinds
    [(reason, pnl)] = db.execute("SELECT exit_reason, gross_pnl FROM trades").fetchall()
    assert reason == "resync_after_disconnect" and pnl == 50.0  # closed at last mark 101


def test_journal_resync_keeps_matching_positions():
    j = Journal(":memory:")
    j.on_fill(Fill("o", "AAPL", Side.BUY, 10, 100.0, 0.0), None)
    j.resync_position("AAPL", 10, 100.0, 101.0)
    assert j.db.execute("SELECT COUNT(*) FROM trades").fetchone() == (0,)
    j.resync_position("AAPL", 4, 100.0, 101.0)  # partially sold while disconnected
    assert j.db.execute("SELECT exit_reason FROM trades").fetchone() == ("resync_after_disconnect",)
    assert j._open["AAPL"].qty == 4 and j._open["AAPL"].inherited
