"""Performance and source-quality report from the trade journal.

    python -m trading_agent.report                      # summary + source scorecards
    python -m trading_agent.report --since 2026-09-01   # limit the period
    python -m trading_agent.report --trade 42           # everything behind one trade
    python -m trading_agent.report --write              # save updated weights/track records

Two views of each source, because trades alone are too few and too noisy:
- Fusion sources (llm, micro:*): net PnL of the trades they supported, attributed by
  their share of the entry conviction, and the win rate of those trades.
- News sources (edgar, finnhub/Reuters, reddit/r/..., ibkr/...): how the price moved
  after each LLM signal they drove -- hit rate and average move in the predicted
  direction -- plus the trade PnL attributed to them.

--write stores suggested fusion weights and news-source track records in
data/source_quality.json. The agent loads that file on startup: the weights replace the
defaults and the track records are shown to the analyst next to each item. Suggestions
are shrunk toward neutral and only move off the defaults once a source has
MIN_SAMPLES observations; review them before writing.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .session import NEW_YORK
from .signals.fusion import DEFAULT_WEIGHTS

DEFAULT_QUALITY_FILE = Path("data/source_quality.json")
HIT_HORIZON = 1800.0
MIN_SAMPLES = 10
PRIOR_N = 20  # pseudo-observations at 50%: a source must earn its way off neutral
DEFAULT_EDGE_BPS = 50.0  # matches FusedSignalStrategy's default
EDGE_PRIOR_N = 30


def shrunk_rate(hits: int, n: int) -> float:
    return (hits + PRIOR_N / 2) / (n + PRIOR_N)


@dataclass
class SourceStats:
    name: str
    signals: int = 0
    directional: int = 0
    scored: int = 0
    hits: int = 0
    bps: dict[float, list[float]] = field(default_factory=lambda: defaultdict(list))
    trades: set[int] = field(default_factory=set)
    winning_trades: set[int] = field(default_factory=set)
    pnl: float = 0.0

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.scored if self.scored else None

    def avg_bps(self, horizon: float) -> float | None:
        values = self.bps.get(horizon)
        return sum(values) / len(values) if values else None

    @property
    def trade_win_rate(self) -> float | None:
        return len(self.winning_trades) / len(self.trades) if self.trades else None


@dataclass
class Report:
    trades: dict
    by_symbol: list[tuple]
    fusion_sources: dict[str, SourceStats]
    news_sources: dict[str, SourceStats]
    models: dict[str, SourceStats] = field(default_factory=dict)
    edge_moves: list[tuple[float, float]] = field(default_factory=list)


def _since_clause(since: float | None, column: str) -> tuple[str, tuple]:
    return (f" WHERE {column} >= ?", (since,)) if since else ("", ())


def build_report(db: sqlite3.Connection, since: float | None = None) -> Report:
    where, args = _since_clause(since, "closed_at")
    row = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(net_pnl > 0), 0), COALESCE(SUM(net_pnl), 0),"
        " COALESCE(SUM(fees), 0),"
        " AVG(CASE WHEN net_pnl > 0 THEN net_pnl END),"
        " AVG(CASE WHEN net_pnl <= 0 THEN net_pnl END),"
        " COALESCE(SUM(CASE WHEN net_pnl > 0 THEN net_pnl END), 0),"
        " COALESCE(-SUM(CASE WHEN net_pnl < 0 THEN net_pnl END), 0)"
        f" FROM trades{where}",
        args,
    ).fetchone()
    n, wins, net, fees, avg_win, avg_loss, gross_win, gross_loss = row
    trades = {
        "count": n,
        "wins": wins,
        "net_pnl": net,
        "fees": fees,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": gross_win / gross_loss if gross_loss else None,
    }
    by_symbol = db.execute(
        "SELECT symbol, COUNT(*), SUM(net_pnl > 0), SUM(net_pnl) FROM trades"
        f"{where} GROUP BY symbol ORDER BY SUM(net_pnl) DESC",
        args,
    ).fetchall()

    # Fusion-source attribution of trade PnL.
    fusion: dict[str, SourceStats] = {}
    attr = db.execute(
        "SELECT a.trade_id, a.source, a.signal_id, a.pnl, t.net_pnl FROM trade_attribution a"
        f" JOIN trades t ON t.id = a.trade_id{where.replace('closed_at', 't.closed_at')}",
        args,
    ).fetchall()
    for trade_id, source, _sid, pnl, trade_net in attr:
        s = fusion.setdefault(source, SourceStats(source))
        s.trades.add(trade_id)
        if trade_net > 0:
            s.winning_trades.add(trade_id)
        s.pnl += pnl

    # News sources: which sources drove each LLM signal.
    swhere, sargs = _since_clause(since, "ts")
    rows = db.execute(f"SELECT id, score, confidence, model FROM signals{swhere}", sargs)
    signals: dict[str, float] = {}
    meta: dict[str, tuple[float, str]] = {}
    for sid, score, confidence, model in rows:
        signals[sid] = score
        meta[sid] = (confidence or 0.0, model or "unknown")
    inputs: dict[str, dict[str, bool]] = defaultdict(dict)
    for sid, source, driver in db.execute(
        "SELECT signal_id, news_source, driver FROM signal_inputs"
    ):
        if sid in signals:
            inputs[sid][source] = inputs[sid].get(source, False) or bool(driver)

    def sources_of(sid: str) -> list[str]:
        srcs = inputs.get(sid, {})
        drivers = [s for s, d in srcs.items() if d]
        return drivers or list(srcs)

    outcomes: dict[str, dict[float, float | None]] = defaultdict(dict)
    for sid, horizon, directional in db.execute(
        "SELECT signal_id, horizon, directional_bps FROM signal_outcomes"
    ):
        if sid in signals:
            outcomes[sid][horizon] = directional

    news: dict[str, SourceStats] = {}
    models: dict[str, SourceStats] = {}
    edge_moves: list[tuple[float, float]] = []  # (|score| x confidence, directional bps)

    def score_signal(s: SourceStats, sid: str, score: float) -> None:
        s.signals += 1
        if score == 0:
            return
        s.directional += 1
        for horizon, bps in outcomes.get(sid, {}).items():
            if bps is not None:
                s.bps[horizon].append(bps)
        hit = outcomes.get(sid, {}).get(HIT_HORIZON)
        if hit is not None:
            s.scored += 1
            s.hits += hit > 0

    for sid, score in signals.items():
        for src in sources_of(sid):
            score_signal(news.setdefault(src, SourceStats(src)), sid, score)
        confidence, model = meta[sid]
        score_signal(models.setdefault(model, SourceStats(model)), sid, score)
        hit = outcomes.get(sid, {}).get(HIT_HORIZON)
        if score and hit is not None:
            edge_moves.append((abs(score) * confidence, hit))

    for trade_id, _source, sid, pnl, trade_net in attr:
        if not sid or sid not in signals:
            continue
        srcs = sources_of(sid)
        for src in srcs:
            s = news.setdefault(src, SourceStats(src))
            s.trades.add(trade_id)
            if trade_net > 0:
                s.winning_trades.add(trade_id)
            s.pnl += pnl / len(srcs)

    return Report(trades, by_symbol, fusion, news, models, edge_moves)


def suggest_fusion_weights(
    report: Report, base: dict[str, float] = DEFAULT_WEIGHTS
) -> dict[str, float]:
    """Scale each fusion source's default weight by its shrunk win rate on the trades it
    supported (0.5 -> unchanged), bounded to 0.25x..2x. Always relative to the defaults,
    so re-running on the same data gives the same answer instead of compounding."""
    out = dict(base)
    groups = fusion_groups(report, list(base))
    for name, weight in base.items():
        n, wins = len(groups[name].trades), len(groups[name].winning_trades)
        if n >= MIN_SAMPLES:
            factor = max(0.25, min(2.0, shrunk_rate(wins, n) / 0.5))
            out[name] = round(weight * factor, 4)
    return out


def estimate_edge_bps(report: Report, llm_weight: float = DEFAULT_WEIGHTS["llm"]) -> float:
    """Average 30-minute move a full-conviction view has been worth, for the strategy's
    cost check. Fused conviction from one LLM signal is weight x |score| x confidence, so
    the edge per unit conviction is mean(directional move) / mean(that). Shrunk toward
    the default with EDGE_PRIOR_N pseudo-observations and floored at zero: if signals
    haven't predicted anything, the cost check should stop opening positions."""
    moves = report.edge_moves
    strength = sum(llm_weight * m for m, _ in moves)
    if not moves or strength <= 0:
        return DEFAULT_EDGE_BPS
    estimate = sum(bps for _, bps in moves) / strength
    n = len(moves)
    shrunk = (n * estimate + EDGE_PRIOR_N * DEFAULT_EDGE_BPS) / (n + EDGE_PRIOR_N)
    return round(max(0.0, min(500.0, shrunk)), 1)


def track_records(report: Report) -> dict[str, str]:
    out = {}
    for name, s in report.news_sources.items():
        if s.scored >= MIN_SAMPLES // 2:
            out[name] = f"right {s.hits} of {s.scored} past calls ({s.hits / s.scored:.0%})"
    return out


def load_quality(path: Path = DEFAULT_QUALITY_FILE) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def write_quality(report: Report, path: Path) -> dict:
    data = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "fusion_weights": suggest_fusion_weights(report),
        "track_records": track_records(report),
        # Only replaces the default once there's enough evidence.
        "edge_bps_at_full_conviction": (
            estimate_edge_bps(report) if len(report.edge_moves) >= MIN_SAMPLES else DEFAULT_EDGE_BPS
        ),
        "news_sources": {
            n: {
                "signals": s.signals,
                "scored": s.scored,
                "hit_rate": s.hit_rate,
                "shrunk_hit_rate": shrunk_rate(s.hits, s.scored),
                "avg_bps_30m": s.avg_bps(HIT_HORIZON),
                "attributed_pnl": s.pnl,
            }
            for n, s in report.news_sources.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return data


def _fmt(v, spec: str) -> str:
    if v is None:
        width = re.match(r"\D*(\d*)", spec).group(1)
        return format("-", f">{width}") if width else "-"
    return format(v, spec)


def _matches(source: str, key: str) -> bool:
    return source == key or source.split(":", 1)[0] == key


def fusion_groups(report: Report, keys: list[str]) -> dict[str, SourceStats]:
    """Attribution stats grouped by fusion-weight key ("llm" covers "llm:news"); sources
    without a weight (e.g. "unattributed") keep their own row."""
    groups: dict[str, SourceStats] = {}
    for src, s in report.fusion_sources.items():
        key = next((k for k in keys if _matches(src, k)), src)
        g = groups.setdefault(key, SourceStats(key))
        g.trades |= s.trades
        g.winning_trades |= s.winning_trades
        g.pnl += s.pnl
    for k in keys:
        groups.setdefault(k, SourceStats(k))
    return groups


def render(report: Report, current: dict[str, float]) -> str:
    t = report.trades
    lines = ["TRADES", "------"]
    if not t["count"]:
        lines.append("no closed trades yet")
    else:
        lines += [
            f"closed trades   {t['count']:>10}   win rate {t['wins'] / t['count']:.0%}",
            f"net PnL         {t['net_pnl']:>10.2f}   fees {t['fees']:.2f}",
            f"avg win / loss  {_fmt(t['avg_win'], '>10.2f')} / {_fmt(t['avg_loss'], '.2f')}"
            f"   profit factor {_fmt(t['profit_factor'], '.2f')}",
            "",
            f"{'symbol':<8}{'trades':>8}{'wins':>6}{'net PnL':>12}",
        ]
        lines += [f"{s:<8}{c:>8}{w:>6}{p:>12.2f}" for s, c, w, p in report.by_symbol]

    suggested = suggest_fusion_weights(report)
    lines += [
        "",
        "FUSION SOURCES (trade attribution)",
        "----------------------------------",
        f"{'source':<18}{'trades':>7}{'win%':>6}{'attrib PnL':>12}{'weight':>8}{'suggested':>11}",
    ]
    for name, s in sorted(fusion_groups(report, list(current)).items()):
        w = current.get(name)
        sug = suggested.get(name)
        lines.append(
            f"{name:<18}{len(s.trades):>7}{_fmt(s.trade_win_rate, '>6.0%')}{s.pnl:>12.2f}"
            f"{_fmt(w, '>8.3f')}{_fmt(sug, '>11.3f')}"
        )

    lines += [
        "",
        "NEWS SOURCES (LLM signal outcomes; hit = moved the predicted way after 30m)",
        "--------------------------------------------------------------------------",
        f"{'source':<28}{'signals':>8}{'dir':>5}{'scored':>7}{'hit%':>6}{'adj%':>6}"
        f"{'bps 5m':>8}{'30m':>7}{'60m':>7}{'trades':>7}{'attrib PnL':>12}",
    ]
    for name, s in sorted(report.news_sources.items(), key=lambda kv: -kv[1].signals):
        lines.append(
            f"{name[:27]:<28}{s.signals:>8}{s.directional:>5}{s.scored:>7}"
            f"{_fmt(s.hit_rate, '>6.0%')}{shrunk_rate(s.hits, s.scored):>6.0%}"
            f"{_fmt(s.avg_bps(300.0), '>8.1f')}{_fmt(s.avg_bps(1800.0), '>7.1f')}"
            f"{_fmt(s.avg_bps(3600.0), '>7.1f')}{len(s.trades):>7}{s.pnl:>12.2f}"
        )
    if not report.news_sources:
        lines.append("no LLM signals recorded yet")
    if report.models:
        lines += [
            "",
            "ANALYST MODELS (same scoring as news sources)",
            "---------------------------------------------",
            f"{'model':<28}{'signals':>8}{'dir':>5}{'scored':>7}{'hit%':>6}{'adj%':>6}"
            f"{'bps 30m':>9}",
        ]
        for name, m in sorted(report.models.items()):
            lines.append(
                f"{name[:27]:<28}{m.signals:>8}{m.directional:>5}{m.scored:>7}"
                f"{_fmt(m.hit_rate, '>6.0%')}{shrunk_rate(m.hits, m.scored):>6.0%}"
                f"{_fmt(m.avg_bps(HIT_HORIZON), '>9.1f')}"
            )
        lines.append(
            f"estimated edge at full conviction: {estimate_edge_bps(report):.1f} bps "
            f"from {len(report.edge_moves)} scored signals (default {DEFAULT_EDGE_BPS:.0f})"
        )
    lines += [
        "",
        f"adj% = hit rate shrunk toward 50% ({PRIOR_N} pseudo-observations). "
        f"Weights change only after {MIN_SAMPLES} supported trades.",
    ]
    return "\n".join(lines)


def render_trade(db: sqlite3.Connection, trade_id: int) -> str:
    t = db.execute(
        "SELECT symbol, direction, opened_at, closed_at, max_qty, avg_entry, avg_exit,"
        " gross_pnl, fees, net_pnl, inherited FROM trades WHERE id = ?",
        (trade_id,),
    ).fetchone()
    if not t:
        return f"no trade {trade_id}"
    sym, direction, opened, closed, qty, entry, exit_, gross, fees, net, inherited = t

    def ts(v: float) -> str:
        return datetime.fromtimestamp(v, NEW_YORK).strftime("%Y-%m-%d %H:%M:%S ET")

    lines = [
        f"TRADE {trade_id}: {sym} {'LONG' if direction > 0 else 'SHORT'} up to {qty} shares"
        + (" (position inherited from before startup)" if inherited else ""),
        f"  {ts(opened)} -> {ts(closed)}",
        f"  entry {entry:.4f}  exit {exit_:.4f}  gross {gross:.2f}  fees {fees:.2f}  net {net:.2f}",
        "",
        "ATTRIBUTION",
    ]
    for source, sid, share, pnl in db.execute(
        "SELECT source, signal_id, share, pnl FROM trade_attribution WHERE trade_id = ?"
        " ORDER BY share DESC",
        (trade_id,),
    ):
        lines.append(f"  {source:<18}{share:>6.0%}{pnl:>10.2f}  {sid or ''}")

    for decision_id, role, dqty in db.execute(
        "SELECT decision_id, role, qty FROM trade_decisions WHERE trade_id = ?", (trade_id,)
    ):
        d = db.execute(
            "SELECT ts, side, limit_price, mid, reason, conviction, target, context"
            " FROM decisions WHERE id = ?",
            (decision_id,),
        ).fetchone()
        if not d:
            continue
        dts, side, px, mid, reason, conv, target, context = d
        lines += [
            "",
            f"{role.upper()} decision {decision_id} ({reason}): {side} {dqty} @ {px:.4f}"
            f" (mid {mid:.4f}) at {ts(dts)}  conviction {_fmt(conv, '+.3f')}"
            f"  target {target}",
        ]
        for sig in json.loads(context).get("signals", []):
            lines.append(
                f"    {sig['source']:<18} score {sig['score']:+.2f} conf {sig['confidence']:.2f}"
                f" decay {sig['decay']:.2f} x weight {sig['weight']:.2f}"
                f" = {sig['contribution']:+.3f}"
            )
            if sig.get("signal_id"):
                lines += _render_signal(db, sig["signal_id"])
    return "\n".join(lines)


def _render_signal(db: sqlite3.Connection, signal_id: str) -> list[str]:
    row = db.execute("SELECT rationale FROM signals WHERE id = ?", (signal_id,)).fetchone()
    lines = [f"      rationale: {row[0]}"] if row else []
    for source, headline, driver in db.execute(
        "SELECT news_source, headline, driver FROM signal_inputs WHERE signal_id = ?",
        (signal_id,),
    ):
        lines.append(f"      {'*' if driver else ' '} [{source}] {headline[:100]}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(prog="trading_agent.report")
    parser.add_argument("--db", type=Path, default=Path("data/journal.db"))
    parser.add_argument("--since", help="YYYY-MM-DD (Eastern)")
    parser.add_argument("--trade", type=int, help="show everything behind one trade")
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"save suggested weights and track records to {DEFAULT_QUALITY_FILE}",
    )
    parser.add_argument("--quality-file", type=Path, default=DEFAULT_QUALITY_FILE)
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"no journal at {args.db}")
    db = sqlite3.connect(args.db)
    if args.trade is not None:
        print(render_trade(db, args.trade))
        return
    since = None
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=NEW_YORK).timestamp()
    report = build_report(db, since)
    existing = load_quality(args.quality_file)
    current = dict(DEFAULT_WEIGHTS) | ((existing or {}).get("fusion_weights") or {})
    print(render(report, current))
    if args.write:
        data = write_quality(report, args.quality_file)
        print(
            f"\nwrote {args.quality_file}: weights {data['fusion_weights']}, "
            f"{len(data['track_records'])} track records"
        )


if __name__ == "__main__":
    main()
