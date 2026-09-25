import asyncio
from datetime import date, datetime

from trading_agent.account import AccountConfig, AccountGuard, AccountInfo, business_days_back
from trading_agent.broker.sim import SimBroker
from trading_agent.config import RiskLimits
from trading_agent.engine import Engine
from trading_agent.models import OrderIntent, Side, Signal, Tick
from trading_agent.risk import RiskManager
from trading_agent.session import NEW_YORK, TradingSession
from trading_agent.signals import SignalFusion, SignalHub
from trading_agent.state import StateStore
from trading_agent.strategy import FusedSignalStrategy
from trading_agent.strategy.base import Strategy

THU = date(2026, 9, 24)


def guard(**kw):
    return AccountGuard(AccountConfig(**kw))


def test_sizing_from_one_percent_risk():
    g = guard(starting_equity=16_000, risk_per_trade_pct=1, stop_loss_pct=2)
    assert g.risk_budget == 160
    assert g.max_position_value() == 8_000  # $160 / 2%
    assert g.max_shares(200.0) == 40
    tight = guard(starting_equity=16_000, stop_loss_pct=0.5)  # $32,000 by risk...
    assert tight.max_position_value() == 8_000  # ...but capped at 50% of equity


def test_pattern_day_trader_window():
    g = guard(starting_equity=16_000)
    assert g.pdt_applies
    for d in (date(2026, 9, 18), date(2026, 9, 21), date(2026, 9, 23)):  # Fri, Mon, Wed
        g.record_open(d)
    assert g.day_trades_remaining(THU) == 0
    assert not g.can_open(THU)[0]
    # Five business days ending Fri 9/25 start Mon 9/21: Friday 9/18 drops out.
    assert business_days_back(date(2026, 9, 25), 5) == date(2026, 9, 21)
    assert g.day_trades_remaining(date(2026, 9, 25)) == 1
    g.update(AccountInfo(day_trades_remaining=0))  # IBKR's count wins if lower
    assert not g.can_open(date(2026, 9, 25))[0]


def test_no_day_trade_limit_for_cash_or_large_margin_accounts():
    assert guard(account_type="cash").day_trades_remaining(THU) is None
    assert guard(starting_equity=30_000).day_trades_remaining(THU) is None


def test_cash_account_settled_cash_limit():
    g = guard(account_type="cash", starting_equity=16_000)
    g.start_day(THU, settled_cash=10_000)
    assert g.can_buy(8_000)[0]
    g.record_buy(8_000)
    assert not g.can_buy(3_000)[0]


def test_risk_manager_enforces_account_caps():
    risk = RiskManager(RiskLimits(max_order_notional=1e9))
    risk.account = guard(starting_equity=16_000)
    risk.update_mark("AAPL", 200.0)
    assert risk.check(OrderIntent("AAPL", Side.BUY, 40, 200.0), 200.0)[0]  # $8,000
    ok, why = risk.check(OrderIntent("AAPL", Side.BUY, 45, 200.0), 200.0)
    assert not ok and "risk per trade" in why
    risk.set_position("MSFT", 40, 200.0)
    risk.set_position("NVDA", 40, 200.0)
    risk.update_mark("MSFT", 200.0)
    risk.update_mark("NVDA", 200.0)
    ok, why = risk.check(OrderIntent("AAPL", Side.BUY, 10, 200.0), 200.0)
    assert not ok and "gross exposure" in why  # already $16,000 invested

    cash = RiskManager(RiskLimits())
    cash.account = guard(account_type="cash")
    ok, why = cash.check(OrderIntent("AAPL", Side.SELL, 1, 200.0), 200.0)
    assert not ok and "short" in why


def test_strategy_sizes_positions_from_the_risk_budget():
    hub = SignalHub(clock=lambda: 1000.0)
    fusion = SignalFusion(hub, max_position=5_000, weights={"llm": 1.0})
    g = guard(starting_equity=16_000)
    strat = FusedSignalStrategy(["AAPL"], hub, fusion, sizer=g.max_shares)
    hub.publish(Signal("AAPL", "llm:news", 0.5, 1.0, 1000.0, 600))
    [order] = strat.on_tick(Tick("AAPL", 199.99, 200.01, 200.0, 1000.0), position=0)
    assert order.qty == 20  # half conviction x 40 shares ($8,000 / $200)


# ---- engine: stops and day-trade limit ---------------------------------------------


class BuyWhenFlat(Strategy):
    def __init__(self, symbols=("AAPL",), qty=40):
        super().__init__(list(symbols))
        self.qty = qty

    def on_tick(self, tick, position):
        return [] if position else [OrderIntent(tick.symbol, Side.BUY, self.qty, tick.bid)]


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def et(h, m):
    return datetime(2026, 9, 24, h, m, tzinfo=NEW_YORK).timestamp()


def make_engine(tmp_path, strategy, clock, **acct):
    broker = SimBroker()
    risk = RiskManager(RiskLimits(max_order_notional=1e9), clock=clock)
    account = guard(starting_equity=16_000, **acct)
    engine = Engine(
        broker,
        strategy,
        risk,
        clock=clock,
        session=TradingSession(),
        state_store=StateStore(tmp_path / "state.json"),
        account=account,
    )
    asyncio.run(engine.startup())
    broker._on_tick = engine.on_tick
    return broker, engine, risk, account


def q(sym, bid, ts):
    return Tick(sym, bid, round(bid + 0.02, 2), bid + 0.01, ts)


def test_stop_loss_closes_and_blocks_reentry(tmp_path):
    clock = Clock(et(10, 0))
    broker, engine, risk, _ = make_engine(tmp_path, BuyWhenFlat(), clock)
    broker.push_tick(q("AAPL", 200.00, clock.t))  # bid 200.00
    broker.push_tick(q("AAPL", 199.97, clock.t))  # ask 199.99 <= 200: filled
    assert risk.position("AAPL") == 40
    clock.t += 5
    broker.push_tick(q("AAPL", 195.90, clock.t))  # mid 195.91: -2.05% -> stop
    [order] = broker.open_orders.values()
    assert (order.side, order.qty) == (Side.SELL, 40)
    broker.push_tick(q("AAPL", 195.90, clock.t))  # bid 195.90 >= limit: stopped out
    assert risk.position("AAPL") == 0
    assert risk.realized_pnl == (195.90 - 200.00) * 40  # -$164: 1% plus slippage
    clock.t += 60
    broker.push_tick(q("AAPL", 196.00, clock.t))
    assert broker.open_orders == {}  # cooldown: no re-entry


def test_fourth_day_trade_is_refused_and_count_persists(tmp_path):
    clock = Clock(et(10, 0))
    strat = BuyWhenFlat(symbols=("A", "B", "C", "D"), qty=10)
    broker, engine, risk, account = make_engine(tmp_path, strat, clock)
    for sym in "ABCD":
        broker.push_tick(q(sym, 100.00, clock.t))
        broker.push_tick(q(sym, 99.97, clock.t))
    assert [risk.position(s) for s in "ABCD"] == [10, 10, 10, 0]
    assert account.day_trades_remaining(THU) == 0

    # A restart the same week still knows three day trades were used.
    _, _, _, account2 = make_engine(tmp_path, strat, clock)
    assert account2.day_trades_remaining(THU) == 0
