"""Command-line entry point.

python -m trading_agent --mode sim   --symbols AAPL MSFT   # synthetic prices, no broker
python -m trading_agent --mode paper --symbols AAPL        # IBKR paper account
python -m trading_agent --mode live  --symbols AAPL        # real money; gated
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys

from .config import (
    DataConfig,
    cost_config_from_env,
    data_config_from_env,
    ib_config_from_env,
    live_trading_allowed,
    risk_limits_from_env,
    runtime_config_from_env,
)
from .costs import CostModel
from .engine import Engine
from .events import Recorder
from .journal import Journal
from .pipeline import NewsPipeline, NewsSchedule
from .risk import RiskManager
from .session import TradingSession, parse_hhmm
from .signals import SignalFusion, SignalHub
from .state import StateStore
from .strategy import FusedSignalStrategy

log = logging.getLogger("trading_agent")


def build_broker(mode: str, costs: CostModel | None = None):
    if mode == "sim":
        from .broker.sim import SimBroker

        costs = costs or CostModel()
        return SimBroker(
            synthetic_feed=True,
            commission_per_share=costs.per_share,
            min_commission=costs.minimum,
        )

    from .broker.ibkr import IBKRBroker

    cfg = ib_config_from_env()
    if mode == "paper" and cfg.is_live_port:
        sys.exit(f"refusing to run paper mode against live port {cfg.port}")
    if mode == "live":
        if not live_trading_allowed():
            sys.exit("live mode requires TRADING_ALLOW_LIVE=yes")
        if not cfg.is_live_port:
            sys.exit(f"live mode requested but port {cfg.port} is not a live TWS/Gateway port")
    return IBKRBroker(cfg)


def build_sources(cfg: DataConfig, broker) -> list:
    from .sources import EdgarSource, FinnhubSource, RedditSource

    sources: list = []
    if cfg.ibkr_news and hasattr(broker, "ib"):
        from .sources.ibkr_news import IBKRNewsSource

        sources.append(IBKRNewsSource(broker))
    if cfg.sec_user_agent:
        sources.append(EdgarSource(cfg.sec_user_agent))
    if cfg.finnhub_api_key:
        sources.append(FinnhubSource(cfg.finnhub_api_key))
    if cfg.reddit_client_id and cfg.reddit_client_secret:
        sources.append(
            RedditSource(cfg.reddit_client_id, cfg.reddit_client_secret, cfg.reddit_user_agent)
        )
    return sources


async def run(args: argparse.Namespace) -> None:
    limits = risk_limits_from_env()
    data_cfg = data_config_from_env()
    risk = RiskManager(limits)
    hub = SignalHub()
    fusion = SignalFusion(hub, max_position=limits.max_position)
    cost_cfg = cost_config_from_env()
    costs = CostModel(
        per_share=cost_cfg.commission_per_share,
        minimum=cost_cfg.commission_min,
        extra_per_share=cost_cfg.extra_fees_per_share,
    )
    broker = build_broker(args.mode, costs)

    rt = runtime_config_from_env()
    state_store = StateStore(rt.state_file if args.mode != "sim" else "data/state-sim.json")
    if args.clear_halt:
        _clear_halt(state_store)
    # The simulator's synthetic prices run around the clock; real markets don't.
    session = None
    if args.mode != "sim":
        session = TradingSession(parse_hhmm(rt.session_start), parse_hhmm(rt.session_flatten))
    recorder = Recorder(rt.record_dir) if args.record else None
    journal = Journal(rt.journal_file if args.mode != "sim" else "data/journal-sim.db")

    # Source quality from `python -m trading_agent.report --write`, if it exists.
    from pathlib import Path

    from .report import load_quality

    quality = load_quality(Path(rt.quality_file)) or {}
    if quality.get("fusion_weights"):
        fusion.weights = dict(quality["fusion_weights"])
        log.info("fusion weights from %s: %s", rt.quality_file, fusion.weights)

    edge_bps = quality.get("edge_bps_at_full_conviction") or cost_cfg.edge_bps
    strategy = FusedSignalStrategy(
        args.symbols,
        hub,
        fusion,
        costs=costs,
        edge_bps_at_full_conviction=edge_bps,
        cost_safety_multiple=cost_cfg.safety_multiple,
    )
    log.info(
        "cost check: edge %.1f bps at full conviction, %.1fx safety",
        edge_bps,
        cost_cfg.safety_multiple,
    )
    engine = Engine(
        broker,
        strategy,
        risk,
        session=session,
        state_store=state_store,
        recorder=recorder,
        journal=journal,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, engine.stop)

    pipeline_task = None
    sources = [] if args.no_news else build_sources(data_cfg, broker)
    if sources:
        from .analyst import ClaudeAnalyst

        analyst = ClaudeAnalyst(
            model=data_cfg.analyst_model,
            effort=data_cfg.analyst_effort,
            track_records=quality.get("track_records"),
            routine_model=data_cfg.analyst_routine_model or None,
        )
        pipeline = NewsPipeline(
            sources,
            analyst,
            hub,
            args.symbols,
            poll_interval=data_cfg.news_poll_seconds,
            recorder=recorder,
            journal=journal,
            # Poll less often outside market hours and brief once before the open.
            schedule=(
                NewsSchedule(live_end=session.flatten, live_interval=data_cfg.news_poll_seconds)
                if session
                else None
            ),
        )
        log.info("news sources: %s", ", ".join(s.name for s in sources))

        async def start_pipeline() -> None:
            await engine.ready.wait()
            await pipeline.run()

        pipeline_task = asyncio.create_task(start_pipeline())
    elif args.mode == "sim":
        pipeline_task = asyncio.create_task(_synthetic_views(hub, args.symbols))
        log.info("sim mode: publishing random synthetic LLM views to exercise the pipeline")
    else:
        log.warning("no news sources configured: without LLM views no positions are opened")

    try:
        await engine.run()
    finally:
        if pipeline_task:
            pipeline_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pipeline_task
        if recorder:
            recorder.close()
        journal.close()
        log.info("stopped; realized=%.2f total=%.2f", risk.realized_pnl, risk.total_pnl())


async def _synthetic_views(hub: SignalHub, symbols: list[str]) -> None:
    """Sim mode only: stand-in LLM views, since only LLM views may open positions."""
    import random
    import time

    from .models import Signal

    rng = random.Random(1)
    while True:
        for sym in symbols:
            score = rng.choice([-1, 1]) * rng.uniform(0.3, 1.0)
            hub.publish(Signal(sym, "llm:sim", score, rng.uniform(0.5, 1.0), time.time(), 120.0))
        await asyncio.sleep(60)


def _clear_halt(store: StateStore) -> None:
    from datetime import date

    state = store.load(date.today())
    if state.halted:
        log.warning("clearing halt: %s", state.halt_reason)
        state.halted, state.halt_reason = False, ""
        store.save(state)


def main() -> None:
    parser = argparse.ArgumentParser(prog="trading-agent")
    parser.add_argument("--mode", choices=["sim", "paper", "live"], default="sim")
    parser.add_argument("--symbols", nargs="+", default=["AAPL"])
    parser.add_argument("--no-news", action="store_true", help="disable the LLM news pipeline")
    parser.add_argument(
        "--record", action="store_true", help="record ticks, news and signals for backtesting"
    )
    parser.add_argument(
        "--clear-halt", action="store_true", help="resume after a halt (after investigating it)"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
