"""Parameters the post-close reviewer may propose changing, with hard bounds.

Proposals are never applied automatically: `python -m trading_agent.review apply` writes
the approved ones to data/tuned_params.json, which the agent loads at startup (values
there override the environment for these keys only). Anything not listed here -- account
type, live-trading switches, daily loss limit, gross exposure -- can't be tuned by the
reviewer at all. The bounds encode the account's ground rules, e.g. risk per trade can
never exceed 1%.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_TUNED_FILE = Path("data/tuned_params.json")
HISTORY_FILE = Path("data/tuning_history.jsonl")


@dataclass(frozen=True)
class Tunable:
    key: str
    description: str
    default: float | str
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()

    def validate(self, value) -> float | str:
        if self.choices:
            if str(value) not in self.choices:
                raise ValueError(f"{self.key} must be one of {self.choices}, not {value!r}")
            return str(value)
        v = float(value)
        if not self.low <= v <= self.high:
            raise ValueError(f"{self.key}={v} outside allowed range [{self.low}, {self.high}]")
        return v


TUNABLES = {
    t.key: t
    for t in (
        # Sizing and stops (standard tier)
        Tunable(
            "RISK_PER_TRADE_PCT", "max loss per trade at the stop, % of equity", 1.0, 0.25, 1.0
        ),
        Tunable("STOP_LOSS_PCT", "default stop until volatility is known, %", 2.0, 1.0, 5.0),
        Tunable("STOP_VOL_MULTIPLE", "stop = this x 30-minute volatility", 2.0, 1.0, 4.0),
        Tunable("MAX_POSITION_PCT", "max position per stock, % of equity", 50.0, 10.0, 50.0),
        Tunable("MAX_SPREAD_PCT", "max bid-ask spread to enter (standard), %", 0.5, 0.1, 1.0),
        Tunable(
            "STOP_COOLDOWN_MINUTES", "no re-entry after a stop for this long", 60.0, 15.0, 390.0
        ),
        # Penny tier
        Tunable("PENNY_RISK_PER_TRADE_PCT", "penny: max loss per trade, %", 0.5, 0.1, 0.5),
        Tunable("PENNY_MAX_POSITION_PCT", "penny: max position, % of equity", 10.0, 2.0, 10.0),
        Tunable("PENNY_MAX_SPREAD_PCT", "penny: max spread to enter, %", 2.0, 0.5, 3.0),
        Tunable("PENNY_MIN_SCORE", "penny: min |sentiment| to enter", 0.8, 0.7, 1.0),
        Tunable("PENNY_MIN_CONFIDENCE", "penny: min analyst confidence to enter", 0.8, 0.7, 1.0),
        # Signal fusion and the cost check
        Tunable("FUSION_WEIGHT_LLM", "weight of the analyst's view", 0.6, 0.3, 1.0),
        Tunable("FUSION_WEIGHT_IMBALANCE", "weight of order-book imbalance", 0.25, 0.0, 0.5),
        Tunable("FUSION_WEIGHT_REVERSION", "weight of short-term mean reversion", 0.15, 0.0, 0.5),
        Tunable("ENTRY_THRESHOLD", "min |conviction| to hold a position", 0.15, 0.1, 0.6),
        Tunable(
            "EDGE_BPS_FULL_CONVICTION",
            "expected move of a full-conviction view, bps",
            50.0,
            0.0,
            300.0,
        ),
        Tunable(
            "COST_SAFETY_MULTIPLE", "expected gain must beat cost by this factor", 2.0, 1.5, 5.0
        ),
        # News analysis
        Tunable(
            "ANALYST_EFFORT",
            "analyst reasoning effort",
            "medium",
            choices=("low", "medium", "high"),
        ),
        Tunable("NEWS_POLL_SECONDS", "news poll interval in market hours, s", 60.0, 30.0, 300.0),
    )
}


def load_tuned(path: Path = DEFAULT_TUNED_FILE) -> dict[str, float | str]:
    try:
        raw = json.loads(path.read_text()).get("params", {})
    except FileNotFoundError:
        return {}
    params = {}
    for key, value in raw.items():
        if key not in TUNABLES:
            log.warning("ignoring unknown tuned parameter %s", key)
            continue
        try:
            params[key] = TUNABLES[key].validate(value)
        except ValueError as e:
            log.error("ignoring tuned parameter: %s", e)
    return params


def apply_to_environment(path: Path = DEFAULT_TUNED_FILE) -> dict[str, float | str]:
    """Load approved tuned parameters into os.environ (overriding .env for these keys)."""
    params = load_tuned(path)
    for key, value in params.items():
        os.environ[key] = str(value)
    if params:
        log.info("tuned parameters from %s: %s", path, params)
    return params


def current_values() -> dict[str, float | str]:
    out = {}
    for key, t in TUNABLES.items():
        raw = os.environ.get(key)
        out[key] = t.default if raw is None else (raw if t.choices else float(raw))
    return out


def approve(
    changes: dict[str, float | str],
    source: str,
    path: Path = DEFAULT_TUNED_FILE,
    history: Path = HISTORY_FILE,
) -> dict[str, float | str]:
    """Validate and persist approved changes; every approval is appended to the history."""
    validated = {
        key: TUNABLES[key].validate(value) for key, value in changes.items() if _known(key)
    }
    params = load_tuned(path) | validated
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": params, "updated_at": time.time()}, indent=2))
    with open(history, "a") as f:
        f.write(json.dumps({"ts": time.time(), "source": source, "changes": validated}) + "\n")
    return params


def _known(key: str) -> bool:
    if key not in TUNABLES:
        raise ValueError(f"{key} is not a tunable parameter")
    return True
