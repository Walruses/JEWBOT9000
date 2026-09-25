from trading_agent.models import Signal, Tick
from trading_agent.signals import MicrostructureSignals, SignalFusion, SignalHub


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def test_hub_decays_and_expires_signals():
    clock = Clock()
    hub = SignalHub(clock=clock)
    hub.publish(Signal("AAPL", "llm:news", 1.0, 1.0, ts=1000.0, ttl=100.0))
    clock.t = 1050.0
    [(sig, decay)] = hub.active("AAPL")
    assert decay == 0.5
    clock.t = 1100.0
    assert hub.active("AAPL") == []


def test_fusion_combines_weighted_sources():
    clock = Clock()
    hub = SignalHub(clock=clock)
    fusion = SignalFusion(hub, max_position=100, weights={"llm": 0.6, "micro:imbalance": 0.4})
    hub.publish(Signal("AAPL", "llm:news", 1.0, 1.0, 1000.0, 60))
    hub.publish(Signal("AAPL", "micro:imbalance", 0.5, 1.0, 1000.0, 60))
    assert fusion.conviction("AAPL") == 0.8
    assert fusion.target_position("AAPL") == 80


def test_fusion_deadband_and_unknown_sources():
    clock = Clock()
    hub = SignalHub(clock=clock)
    fusion = SignalFusion(hub, max_position=100, weights={"llm": 0.6}, entry_threshold=0.15)
    hub.publish(Signal("AAPL", "llm:news", 0.2, 1.0, 1000.0, 60))  # 0.12 < threshold
    hub.publish(Signal("AAPL", "unknown", 1.0, 1.0, 1000.0, 60))  # weight 0
    assert fusion.target_position("AAPL") == 0


def test_microstructure_imbalance():
    micro = MicrostructureSignals(window=5)
    [sig] = micro.update(Tick("AAPL", 99.99, 100.01, 100.0, 0.0, bid_size=300, ask_size=100))
    assert sig.source == "micro:imbalance"
    assert sig.score == 0.5
