"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

LIVE_PORTS = frozenset({7496, 4001})
PAPER_PORTS = frozenset({7497, 4002})


@dataclass(frozen=True)
class IBConfig:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 1

    @property
    def is_live_port(self) -> bool:
        return self.port in LIVE_PORTS


@dataclass(frozen=True)
class RiskLimits:
    # Share-count backstops; real sizing comes from the account's risk-per-trade rules.
    max_position: int = 5_000
    max_order_qty: int = 5_000
    max_order_notional: float = 25_000.0
    max_daily_loss: float = 500.0
    # IBKR rejects/disconnects clients above ~50 messages/sec; stay well below it.
    max_orders_per_sec: int = 40
    # Reject limit prices further than this from the current mid (fat-finger guard).
    max_price_deviation_bps: float = 50.0


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def ib_config_from_env() -> IBConfig:
    return IBConfig(
        host=_env("IB_HOST", "127.0.0.1"),
        port=int(_env("IB_PORT", "4002")),
        client_id=int(_env("IB_CLIENT_ID", "1")),
    )


def risk_limits_from_env() -> RiskLimits:
    d = RiskLimits()
    return RiskLimits(
        max_position=int(_env("RISK_MAX_POSITION", str(d.max_position))),
        max_order_qty=int(_env("RISK_MAX_ORDER_QTY", str(d.max_order_qty))),
        max_order_notional=float(_env("RISK_MAX_ORDER_NOTIONAL", str(d.max_order_notional))),
        max_daily_loss=float(_env("RISK_MAX_DAILY_LOSS", str(d.max_daily_loss))),
        max_orders_per_sec=int(_env("RISK_MAX_ORDERS_PER_SEC", str(d.max_orders_per_sec))),
        max_price_deviation_bps=float(
            _env("RISK_MAX_PRICE_DEVIATION_BPS", str(d.max_price_deviation_bps))
        ),
    )


def live_trading_allowed() -> bool:
    return _env("TRADING_ALLOW_LIVE", "no").strip().lower() == "yes"


@dataclass(frozen=True)
class DataConfig:
    analyst_model: str = "claude-opus-5"
    # Cheaper model for routine news/social batches; empty = use analyst_model for all.
    analyst_routine_model: str = "claude-sonnet-5"
    analyst_effort: str = "medium"
    news_poll_seconds: float = 60.0
    ibkr_news: bool = True
    sec_user_agent: str = ""
    finnhub_api_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = ""


def data_config_from_env() -> DataConfig:
    d = DataConfig()
    return DataConfig(
        analyst_model=_env("ANALYST_MODEL", d.analyst_model),
        analyst_routine_model=_env("ANALYST_ROUTINE_MODEL", d.analyst_routine_model),
        analyst_effort=_env("ANALYST_EFFORT", d.analyst_effort),
        news_poll_seconds=float(_env("NEWS_POLL_SECONDS", str(d.news_poll_seconds))),
        ibkr_news=_env("IBKR_NEWS", "yes").strip().lower() == "yes",
        sec_user_agent=_env("SEC_USER_AGENT", ""),
        finnhub_api_key=_env("FINNHUB_API_KEY", ""),
        reddit_client_id=_env("REDDIT_CLIENT_ID", ""),
        reddit_client_secret=_env("REDDIT_CLIENT_SECRET", ""),
        reddit_user_agent=_env("REDDIT_USER_AGENT", "trading-agent/0.1"),
    )


@dataclass(frozen=True)
class RuntimeConfig:
    state_file: str = "data/state.json"
    record_dir: str = "data/recordings"
    session_start: str = "09:35"
    session_flatten: str = "15:50"
    journal_file: str = "data/journal.db"
    quality_file: str = "data/source_quality.json"


def runtime_config_from_env() -> RuntimeConfig:
    d = RuntimeConfig()
    return RuntimeConfig(
        state_file=_env("STATE_FILE", d.state_file),
        record_dir=_env("RECORD_DIR", d.record_dir),
        session_start=_env("SESSION_START", d.session_start),
        session_flatten=_env("SESSION_FLATTEN", d.session_flatten),
        journal_file=_env("JOURNAL_FILE", d.journal_file),
        quality_file=_env("QUALITY_FILE", d.quality_file),
    )


@dataclass(frozen=True)
class CostConfig:
    commission_per_share: float = 0.0035
    commission_min: float = 0.35
    extra_fees_per_share: float = 0.0
    # Expected move (bps) of a full-conviction view; replaced by the journal report's
    # estimate once source_quality.json has one.
    edge_bps: float = 50.0
    safety_multiple: float = 2.0


def cost_config_from_env() -> CostConfig:
    d = CostConfig()
    return CostConfig(
        commission_per_share=float(_env("COMMISSION_PER_SHARE", str(d.commission_per_share))),
        commission_min=float(_env("COMMISSION_MIN", str(d.commission_min))),
        extra_fees_per_share=float(_env("EXTRA_FEES_PER_SHARE", str(d.extra_fees_per_share))),
        edge_bps=float(_env("EDGE_BPS_FULL_CONVICTION", str(d.edge_bps))),
        safety_multiple=float(_env("COST_SAFETY_MULTIPLE", str(d.safety_multiple))),
    )


def account_config_from_env():
    from .account import AccountConfig

    d = AccountConfig()
    account_type = _env("ACCOUNT_TYPE", d.account_type).strip().lower()
    if account_type not in ("cash", "margin"):
        raise ValueError(f"ACCOUNT_TYPE must be cash or margin, not {account_type!r}")
    return AccountConfig(
        account_type=account_type,
        starting_equity=float(_env("ACCOUNT_EQUITY", str(d.starting_equity))),
        risk_per_trade_pct=float(_env("RISK_PER_TRADE_PCT", str(d.risk_per_trade_pct))),
        stop_loss_pct=float(_env("STOP_LOSS_PCT", str(d.stop_loss_pct))),
        max_position_pct=float(_env("MAX_POSITION_PCT", str(d.max_position_pct))),
        max_gross_exposure_pct=float(_env("MAX_GROSS_EXPOSURE_PCT", str(d.max_gross_exposure_pct))),
        stop_cooldown_minutes=float(_env("STOP_COOLDOWN_MINUTES", str(d.stop_cooldown_minutes))),
    )
