from trading_agent.costs import CostModel
from trading_agent.models import Side, Signal, Tick
from trading_agent.signals import SignalFusion, SignalHub
from trading_agent.strategy import FusedSignalStrategy

NOW = 1000.0
WEIGHTS = {"llm": 0.6, "micro:imbalance": 0.25, "micro:reversion": 0.15}


def setup(max_position=100, **kw):
    hub = SignalHub(clock=lambda: NOW)
    fusion = SignalFusion(hub, max_position=max_position, weights=dict(WEIGHTS))
    strat = FusedSignalStrategy(["AAPL"], hub, fusion, **kw)
    return hub, fusion, strat


def sig(source, score, conf=1.0):
    return Signal("AAPL", source, score, conf, NOW, 600)


def tick(bid_size=0, ask_size=0):
    return Tick("AAPL", 100.00, 100.02, 100.01, NOW, bid_size, ask_size)


def test_microstructure_alone_never_opens_a_position():
    hub, fusion, strat = setup()
    hub.publish(sig("micro:imbalance", 1.0))
    hub.publish(sig("micro:reversion", 1.0))
    assert fusion.conviction("AAPL") == 0.0
    assert strat.on_tick(tick(bid_size=1000, ask_size=10), position=0) == []


def test_microstructure_adjusts_but_cannot_flip_an_llm_view():
    hub, fusion, _ = setup()
    hub.publish(sig("llm:news", 0.5))  # +0.30
    hub.publish(sig("micro:imbalance", 1.0))  # +0.25
    assert round(fusion.conviction("AAPL"), 6) == 0.55
    hub.publish(sig("micro:imbalance", -1.0))  # -0.25 -> weaker but same side
    assert round(fusion.conviction("AAPL"), 6) == 0.05
    hub.publish(sig("llm:news", 0.1))  # +0.06 - 0.25 would flip -> no view
    assert fusion.conviction("AAPL") == 0.0


def test_cost_check_blocks_small_edge_but_allows_worthwhile_trades():
    costs = CostModel(per_share=0.0035, minimum=0.35)
    hub, fusion, strat = setup(max_position=20, costs=costs, edge_bps_at_full_conviction=5.0)
    hub.publish(sig("llm:news", 0.5))  # conviction 0.3 -> target 6 shares
    # gain = 6 x 100 x 0.3 x 5bps = $0.09 < 2 x $0.70 round trip
    assert strat.on_tick(tick(), position=0) == []
    assert strat.explain("AAPL")["expected_cost"] == 0.7

    hub, fusion, strat = setup(max_position=200, costs=costs, edge_bps_at_full_conviction=50.0)
    hub.publish(sig("llm:news", 1.0))  # conviction 0.6 -> 120 shares
    [order] = strat.on_tick(tick(), position=0)
    assert (order.side, order.qty) == (Side.BUY, 120)  # $36 expected vs $0.84 cost


def test_reducing_is_never_blocked_and_unprofitable_flip_only_closes():
    costs = CostModel(per_share=0.0035, minimum=0.35)
    hub, fusion, strat = setup(max_position=20, costs=costs, edge_bps_at_full_conviction=1.0)
    [order] = strat.on_tick(tick(), position=15)  # no view -> target 0 -> close
    assert (order.side, order.qty) == (Side.SELL, 15)
    hub.publish(sig("llm:news", -1.0))  # wants -12, flip not worth it
    [order] = strat.on_tick(tick(), position=15)
    assert (order.side, order.qty) == (Side.SELL, 15)


def test_small_rebalances_are_ignored():
    hub, fusion, strat = setup(max_position=100, costs=CostModel(0, 0))
    hub.publish(sig("llm:news", 1.0))  # target 60
    assert strat.on_tick(tick(), position=55) == []  # 5 < 20% of max position
    [order] = strat.on_tick(tick(), position=30)
    assert order.qty == 30


def test_commission_model():
    c = CostModel()
    assert c.commission(10, 100.0) == 0.35  # minimum
    assert c.commission(1000, 100.0) == 3.5  # per share
    assert c.commission(1, 1.0) == 0.01  # capped at 1% of value


def test_microstructure_cannot_trim_a_held_position():
    hub, fusion, strat = setup(max_position=100, costs=CostModel(0, 0))
    hub.publish(sig("llm:news", 1.0))  # primary target 60
    hub.publish(sig("micro:imbalance", -1.0))  # combined target 35
    assert fusion.target_position("AAPL") == 35
    assert strat.on_tick(tick(), position=60) == []  # hold: the LLM view hasn't changed


def test_decaying_view_steps_down_in_large_chunks():
    hub, fusion, strat = setup(max_position=100, costs=CostModel(0, 0))
    hub.publish(sig("llm:news", 0.6))  # target 36: more than half of 60 -> hold
    assert strat.on_tick(tick(), position=60) == []
    hub.publish(sig("llm:news", 0.5))  # target 30: half of 60 -> reduce to 30
    [order] = strat.on_tick(tick(), position=60)
    assert (order.side, order.qty) == (Side.SELL, 30)
    hub.publish(sig("llm:news", 0.3))  # target 18: more than half of 30 -> hold
    assert strat.on_tick(tick(), position=30) == []
    hub.publish(sig("llm:news", 0.2))  # conviction 0.12: below entry threshold -> flat
    [order] = strat.on_tick(tick(), position=30)
    assert (order.side, order.qty) == (Side.SELL, 30)


def test_microstructure_cannot_lift_a_weak_llm_view_into_a_trade():
    hub, fusion, strat = setup(max_position=100, costs=CostModel(0, 0))
    hub.publish(sig("llm:news", 0.2))  # 0.12 alone: below the 0.15 entry threshold
    hub.publish(sig("micro:imbalance", 1.0))  # combined 0.37
    assert strat.on_tick(tick(), position=0) == []
    hub.publish(sig("llm:news", 0.5))  # 0.30 alone: enough; micro enlarges to 55
    [order] = strat.on_tick(tick(), position=0)
    assert order.qty == 55
