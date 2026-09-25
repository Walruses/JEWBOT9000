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
    data_config_from_env,
    ib_config_from_env,
    live_trading_allowed,
    risk_limits_from_env,
)
from .engine import Engine
from .pipeline import NewsPipeline
from .risk import RiskManager
from .signals import SignalFusion, SignalHub
from .strategy import FusedSignalStrategy

log = logging.getLogger("trading_agent")


def build_broker(mode: str):
    if mode == "sim":
        from .broker.sim import SimBroker

        return SimBroker(synthetic_feed=True)

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
    broker = build_broker(args.mode)
    engine = Engine(broker, FusedSignalStrategy(args.symbols, hub, fusion), risk)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, engine.stop)

    pipeline_task = None
    sources = [] if args.no_news else build_sources(data_cfg, broker)
    if sources:
        from .analyst import ClaudeAnalyst

        analyst = ClaudeAnalyst(model=data_cfg.analyst_model, effort=data_cfg.analyst_effort)
        pipeline = NewsPipeline(
            sources, analyst, hub, args.symbols, poll_interval=data_cfg.news_poll_seconds
        )
        log.info("news sources: %s", ", ".join(s.name for s in sources))

        async def start_pipeline() -> None:
            await engine.ready.wait()
            await pipeline.run()

        pipeline_task = asyncio.create_task(start_pipeline())
    else:
        log.info("no news sources configured; trading on microstructure signals only")

    try:
        await engine.run()
    finally:
        if pipeline_task:
            pipeline_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pipeline_task
        log.info("stopped; realized=%.2f total=%.2f", risk.realized_pnl, risk.total_pnl())


def main() -> None:
    parser = argparse.ArgumentParser(prog="trading-agent")
    parser.add_argument("--mode", choices=["sim", "paper", "live"], default="sim")
    parser.add_argument("--symbols", nargs="+", default=["AAPL"])
    parser.add_argument("--no-news", action="store_true", help="disable the LLM news pipeline")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
