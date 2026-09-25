"""Trade journal: what was traded, why, and how it turned out.

Everything goes into one SQLite database so it can be queried directly
(`sqlite3 data/journal.db`) as well as through `python -m trading_agent.report`.

- decisions:         every order sent, with the full signal breakdown behind it
- fills:             every execution, linked to its decision
- trades:            round trips (flat -> position -> flat) with gross/net PnL and fees
- trade_decisions:   which decisions opened/added to and closed each trade
- trade_attribution: each trade's net PnL split across the signal sources that
                     supported the entry, in proportion to their contribution
- signals:           every LLM signal with its rationale
- signal_inputs:     the news items behind each LLM signal (and which were drivers)
- signal_outcomes:   the price move after each LLM signal at fixed horizons, so a source
                     can be judged on its predictions, not only on trades that happened

Attribution is correlational: a source that often agrees with a profitable one looks
good too. Judge sources on enough samples, and read the signal outcomes alongside PnL.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .models import Fill, NewsItem, OrderIntent, Signal, Tick

log = logging.getLogger(__name__)

HORIZONS = (60.0, 300.0, 1800.0, 3600.0)

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY, ts REAL, symbol TEXT, side TEXT, qty INTEGER,
    limit_price REAL, mid REAL, reason TEXT, conviction REAL, target INTEGER, context TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY, decision_id INTEGER, order_id TEXT, ts REAL, symbol TEXT,
    side TEXT, qty INTEGER, price REAL, commission REAL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY, symbol TEXT, direction INTEGER, opened_at REAL,
    closed_at REAL, max_qty INTEGER, avg_entry REAL, avg_exit REAL, gross_pnl REAL,
    fees REAL, net_pnl REAL, inherited INTEGER
);
CREATE TABLE IF NOT EXISTS trade_decisions (
    trade_id INTEGER, decision_id INTEGER, role TEXT, qty INTEGER
);
CREATE TABLE IF NOT EXISTS trade_attribution (
    trade_id INTEGER, source TEXT, signal_id TEXT, share REAL, pnl REAL
);
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY, ts REAL, symbol TEXT, source TEXT, score REAL,
    confidence REAL, ttl REAL, rationale TEXT, model TEXT
);
CREATE TABLE IF NOT EXISTS signal_inputs (
    signal_id TEXT, news_id TEXT, news_source TEXT, headline TEXT, url TEXT,
    published_at REAL, driver INTEGER
);
CREATE TABLE IF NOT EXISTS signal_outcomes (
    signal_id TEXT, horizon REAL, base_mid REAL, mid REAL, ret_bps REAL,
    directional_bps REAL, PRIMARY KEY (signal_id, horizon)
);
CREATE INDEX IF NOT EXISTS idx_attr_trade ON trade_attribution(trade_id);
CREATE INDEX IF NOT EXISTS idx_inputs_signal ON signal_inputs(signal_id);
"""


@dataclass
class _Entry:
    decision_id: int | None
    qty: int
    context: dict


@dataclass
class _OpenTrade:
    symbol: str
    direction: int
    opened_at: float
    inherited: bool = False
    qty: int = 0  # signed
    avg: float = 0.0
    max_qty: int = 0
    entry_qty: int = 0
    entry_value: float = 0.0
    exit_qty: int = 0
    exit_value: float = 0.0
    gross: float = 0.0
    fees: float = 0.0
    entries: list[_Entry] = field(default_factory=list)
    exits: list[tuple[int | None, int]] = field(default_factory=list)


@dataclass
class _PendingOutcome:
    signal_id: str
    symbol: str
    ts: float
    score: float
    base_mid: float | None = None
    base_ts: float = 0.0
    remaining: list[float] = field(default_factory=lambda: list(HORIZONS))


class Journal:
    def __init__(
        self,
        path: str | Path,
        clock: Callable[[], float] = time.time,
        commit_interval: float = 1.0,
    ):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(signals)")}
        if "model" not in columns:  # journals created before model routing
            self.db.execute("ALTER TABLE signals ADD COLUMN model TEXT")
        self._clock = clock
        self._commit_interval = commit_interval
        self._last_commit = 0.0
        # decision_id -> context, needed only until the order's fills arrive (orders live
        # seconds); bounded so cancelled orders' contexts don't accumulate.
        self._decisions: OrderedDict[int, dict] = OrderedDict()
        self._open: dict[str, _OpenTrade] = {}
        self._last_closed: dict[str, int] = {}
        self._pending: dict[str, list[_PendingOutcome]] = {}

    # ---- decisions & fills -----------------------------------------------------------

    def record_decision(self, intent: OrderIntent, mid: float, reason: str, context: dict) -> int:
        cur = self.db.execute(
            "INSERT INTO decisions (ts, symbol, side, qty, limit_price, mid, reason, conviction,"
            " target, context) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                self._clock(),
                intent.symbol,
                intent.side.value,
                intent.qty,
                intent.limit_price,
                mid,
                reason,
                context.get("conviction"),
                context.get("target"),
                json.dumps(context),
            ),
        )
        decision_id = int(cur.lastrowid)
        self._decisions[decision_id] = context
        if len(self._decisions) > 10_000:
            self._decisions.popitem(last=False)
        self._maybe_commit()
        return decision_id

    def seed_position(self, symbol: str, qty: int, avg_price: float) -> None:
        """A position that existed before this run (no decision data behind it)."""
        if qty == 0 or symbol in self._open:
            return
        t = _OpenTrade(symbol, 1 if qty > 0 else -1, self._clock(), inherited=True)
        t.qty, t.avg, t.max_qty = qty, avg_price, abs(qty)
        t.entry_qty, t.entry_value = abs(qty), abs(qty) * avg_price
        self._open[symbol] = t

    def on_fill(self, fill: Fill, decision_id: int | None) -> None:
        self.db.execute(
            "INSERT INTO fills (decision_id, order_id, ts, symbol, side, qty, price, commission)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                decision_id,
                fill.order_id,
                fill.ts,
                fill.symbol,
                fill.side.value,
                fill.qty,
                fill.price,
                fill.commission,
            ),
        )
        if fill.qty == 0:  # a commission reported after the fill (IBKR)
            self._add_late_fee(fill.symbol, fill.commission)
        else:
            self._apply_fill(fill, decision_id)
        self._maybe_commit()

    def _apply_fill(self, fill: Fill, decision_id: int | None) -> None:
        signed = fill.side.sign * fill.qty
        remaining, fee_per_share = fill.qty, fill.commission / fill.qty
        t = self._open.get(fill.symbol)
        context = self._decisions.get(decision_id, {}) if decision_id else {}

        if t is not None and (t.qty > 0) != (signed > 0):
            closing = min(abs(t.qty), remaining)
            t.gross += closing * (fill.price - t.avg) * t.direction
            t.exit_qty += closing
            t.exit_value += closing * fill.price
            t.fees += closing * fee_per_share
            t.qty += t.direction * -closing
            t.exits.append((decision_id, closing))
            remaining -= closing
            if t.qty == 0:
                self._close(t, fill.ts)
                t = None
        if remaining:
            if t is None:
                t = _OpenTrade(fill.symbol, 1 if signed > 0 else -1, fill.ts)
                self._open[fill.symbol] = t
            t.avg = (t.avg * abs(t.qty) + fill.price * remaining) / (abs(t.qty) + remaining)
            t.qty += t.direction * remaining
            t.max_qty = max(t.max_qty, abs(t.qty))
            t.entry_qty += remaining
            t.entry_value += remaining * fill.price
            t.fees += remaining * fee_per_share
            t.entries.append(_Entry(decision_id, remaining, context))

    def _close(self, t: _OpenTrade, ts: float) -> None:
        del self._open[t.symbol]
        net = t.gross - t.fees
        cur = self.db.execute(
            "INSERT INTO trades (symbol, direction, opened_at, closed_at, max_qty, avg_entry,"
            " avg_exit, gross_pnl, fees, net_pnl, inherited) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                t.symbol,
                t.direction,
                t.opened_at,
                ts,
                t.max_qty,
                t.entry_value / t.entry_qty if t.entry_qty else 0.0,
                t.exit_value / t.exit_qty if t.exit_qty else 0.0,
                t.gross,
                t.fees,
                net,
                int(t.inherited),
            ),
        )
        trade_id = int(cur.lastrowid)
        self._last_closed[t.symbol] = trade_id
        self.db.executemany(
            "INSERT INTO trade_decisions (trade_id, decision_id, role, qty) VALUES (?,?,?,?)",
            [(trade_id, e.decision_id, "entry", e.qty) for e in t.entries]
            + [(trade_id, d, "exit", q) for d, q in t.exits],
        )
        self.db.executemany(
            "INSERT INTO trade_attribution (trade_id, source, signal_id, share, pnl)"
            " VALUES (?,?,?,?,?)",
            [
                (trade_id, src, sid, share, net * share)
                for (src, sid), share in attribute(t.entries, t.direction).items()
            ],
        )
        self.db.commit()
        log.info(
            "trade closed %s %s net=%.2f", t.symbol, "long" if t.direction > 0 else "short", net
        )

    def _add_late_fee(self, symbol: str, fee: float) -> None:
        t = self._open.get(symbol)
        if t is not None:
            t.fees += fee
        elif symbol in self._last_closed:
            trade_id = self._last_closed[symbol]
            self.db.execute(
                "UPDATE trades SET fees = fees + ?, net_pnl = net_pnl - ? WHERE id = ?",
                (fee, fee, trade_id),
            )
            # Keep attributed PnL summing to net PnL.
            self.db.execute(
                "UPDATE trade_attribution SET pnl = pnl - ? * share WHERE trade_id = ?",
                (fee, trade_id),
            )

    # ---- LLM signals and their outcomes ----------------------------------------------

    def record_signal(self, signal: Signal, items: list[NewsItem]) -> None:
        if not signal.id:
            return
        self.db.execute(
            "INSERT OR REPLACE INTO signals (id, ts, symbol, source, score, confidence, ttl,"
            " rationale, model) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                signal.id,
                signal.ts,
                signal.symbol,
                signal.source,
                signal.score,
                signal.confidence,
                signal.ttl,
                signal.rationale,
                signal.model,
            ),
        )
        drivers = set(signal.drivers)
        self.db.executemany(
            "INSERT INTO signal_inputs VALUES (?,?,?,?,?,?,?)",
            [
                (signal.id, i.id, i.source, i.headline, i.url, i.published_at, int(i.id in drivers))
                for i in items
            ],
        )
        self._pending.setdefault(signal.symbol, []).append(
            _PendingOutcome(signal.id, signal.symbol, signal.ts, signal.score)
        )
        self._maybe_commit()

    def on_tick(self, tick: Tick) -> None:
        pending = self._pending.get(tick.symbol)
        if not pending or not tick.valid:
            return
        done = []
        for p in pending:
            if tick.ts < p.ts:
                continue
            if p.base_mid is None:
                # The first quote after the signal is usable is the reference price.
                p.base_mid, p.base_ts = tick.mid, tick.ts
                continue
            while p.remaining and tick.ts - p.base_ts >= p.remaining[0]:
                horizon = p.remaining.pop(0)
                ret = (tick.mid / p.base_mid - 1) * 10_000
                direction = (p.score > 0) - (p.score < 0)
                self.db.execute(
                    "INSERT OR REPLACE INTO signal_outcomes VALUES (?,?,?,?,?,?)",
                    (
                        p.signal_id,
                        horizon,
                        p.base_mid,
                        tick.mid,
                        ret,
                        ret * direction if direction else None,
                    ),
                )
            if not p.remaining:
                done.append(p)
        for p in done:
            pending.remove(p)
        self._maybe_commit()

    # ---- housekeeping ----------------------------------------------------------------

    def _maybe_commit(self) -> None:
        now = time.monotonic()
        if now - self._last_commit >= self._commit_interval:
            self.db.commit()
            self._last_commit = now

    def close(self) -> None:
        self.db.commit()
        self.db.close()


def attribute(entries: list[_Entry], direction: int) -> dict[tuple[str, str], float]:
    """Split credit for a trade across the signals that supported its entries.

    Each entry's signal contributions are weighted by the shares it added. Only
    contributions pointing the way the trade went count; shares sum to 1. Trades with no
    decision data (inherited positions, flatten-only entries) get a single "unattributed"
    row so every trade's PnL is accounted for.
    """
    support: dict[tuple[str, str], float] = {}
    for e in entries:
        for sig in e.context.get("signals", []):
            aligned = sig["contribution"] * direction
            if aligned > 0:
                key = (sig["source"], sig.get("signal_id", ""))
                support[key] = support.get(key, 0.0) + aligned * e.qty
    total = sum(support.values())
    if total <= 0:
        return {("unattributed", ""): 1.0}
    return {k: v / total for k, v in support.items()}
