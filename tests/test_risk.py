from trading_agent.config import RiskLimits
from trading_agent.models import Fill, OrderIntent, Side
from trading_agent.risk import RiskManager


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(**kw):
    clock = Clock()
    return RiskManager(RiskLimits(**kw), clock=clock), clock


def buy(qty, px=100.0, sym="AAPL"):
    return OrderIntent(sym, Side.BUY, qty, px)


def sell(qty, px=100.0, sym="AAPL"):
    return OrderIntent(sym, Side.SELL, qty, px)


def fill(side, qty, px, sym="AAPL"):
    return Fill("1", sym, side, qty, px, 0.0)


def test_rejects_oversized_order_and_fat_finger():
    risk, _ = make(max_order_qty=10, max_price_deviation_bps=50)
    assert not risk.check(buy(11), 100.0)[0]
    assert not risk.check(buy(5, px=101.0), 100.0)[0]  # 100bps from mid
    assert risk.check(buy(5, px=100.2), 100.0)[0]


def test_pending_orders_count_toward_position_limit():
    risk, _ = make(max_position=10, max_order_qty=10)
    risk.on_submit(buy(8))
    assert not risk.check(buy(3), 100.0)[0]
    risk.on_order_closed("AAPL", Side.BUY, 8)  # cancelled unfilled
    assert risk.check(buy(3), 100.0)[0]


def test_order_rate_limit_window():
    risk, clock = make(max_orders_per_sec=2, max_position=1000, max_order_qty=1000)
    for _ in range(2):
        assert risk.check(buy(1), 100.0)[0]
        risk.on_submit(buy(1))
    assert risk.check(buy(1), 100.0) == (False, "order rate limit")
    clock.t = 1.0
    assert risk.check(buy(1), 100.0)[0]


def test_pnl_through_flip():
    risk, _ = make(max_daily_loss=1e9)
    risk.on_fill(fill(Side.BUY, 10, 100.0))
    risk.on_fill(fill(Side.SELL, 15, 101.0))  # close 10 (+10), open short 5 @ 101
    assert risk.realized_pnl == 10.0
    assert risk.position("AAPL") == -5
    risk.update_mark("AAPL", 100.0)
    assert risk.unrealized_pnl() == 5.0


def test_daily_loss_halt_allows_only_reducing_orders():
    risk, _ = make(max_daily_loss=50)
    risk.on_fill(fill(Side.BUY, 10, 100.0))
    risk.update_mark("AAPL", 94.0)
    assert risk.halted and risk.allow_flatten
    assert not risk.check(buy(1, px=94.0), 94.0)[0]
    assert not risk.check(sell(11, px=94.0), 94.0)[0]  # would flip short
    assert risk.check(sell(10, px=94.0), 94.0)[0]


def test_hard_halt_blocks_everything():
    risk, _ = make()
    risk.on_fill(fill(Side.BUY, 10, 100.0))
    risk.halt("position mismatch")
    assert not risk.check(sell(10), 100.0)[0]


def test_commissions_reduce_realized_pnl():
    risk, _ = make()
    risk.on_fill(Fill("1", "AAPL", Side.BUY, 10, 100.0, 0.0, commission=0.35))
    risk.on_fill(Fill("1", "AAPL", Side.BUY, 0, 0.0, 0.0, commission=0.15))  # report only
    assert risk.position("AAPL") == 10
    assert risk.realized_pnl == -0.5


def test_can_reduce_an_oversized_inherited_position():
    risk, _ = make(max_position=100, max_order_qty=100)
    risk.set_position("AAPL", 150, 100.0)
    assert not risk.check(buy(1), 100.0)[0]
    assert risk.check(sell(10), 100.0)[0]


def test_restore_reapplies_loss_limit():
    risk, _ = make(max_daily_loss=50)
    risk.restore(-60.0, halted=False, halt_reason="")
    assert risk.halted
