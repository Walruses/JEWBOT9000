# AI Intraday Trading Agent for Interactive Brokers

An automated trading agent for US equities. It combines an LLM analyst (Claude) that reads
news, SEC filings and social media with fast quantitative signals computed on every tick,
and trades through Interactive Brokers. Every order passes pre-trade risk checks first.

> **Speed target.** IBKR's API is limited to about 50 messages/sec, has tens of milliseconds
> of latency and offers no colocation, so true sub-millisecond HFT isn't possible through it.
> This agent is built for automated **intraday trading on horizons of seconds to minutes**.

## Architecture

```
 SLOW LOOP (every ~60s)                          FAST LOOP (every tick)
 ┌─────────────────────────────┐                 ┌───────────────────────────┐
 │ Sources                     │                 │ IBKR market data (L1)     │
 │  IBKR news · SEC EDGAR      │                 └────────────┬──────────────┘
 │  Finnhub · Reddit           │                              │
 └────────────┬────────────────┘                              ▼
              ▼ de-dupe, drop stale                 MicrostructureSignals
      ClaudeAnalyst (LLM)                          (book imbalance, mean reversion)
   structured score + confidence + horizon                    │
              │                                               │
              └──────────────►  SignalHub  ◄──────────────────┘
                          (latest per source, TTL + decay)
                                    │
                                    ▼
                    SignalFusion → target position
                                    │
                                    ▼
                 FusedSignalStrategy → OrderIntent (passive limit)
                                    │
                                    ▼
          RiskManager  (position incl. working orders, size, notional,
                        fat-finger, rate limit, daily-loss kill switch)
                                    │
                                    ▼
                 Engine → Broker (IBKRBroker | SimBroker)
```

| Module | Role |
|---|---|
| `sources/` | `IBKRNewsSource`, `EdgarSource`, `FinnhubSource`, `RedditSource`, all sharing the `NewsSource` interface |
| `analyst.py` | Claude reads each symbol's new items and returns a schema-constrained, clamped `Signal` |
| `pipeline.py` | Slow loop: polls sources, de-duplicates, drops items older than 1h, runs the analyst |
| `signals/` | `SignalHub` (TTL and decay), `MicrostructureSignals`, `SignalFusion` (weighted to a target position) |
| `strategy/` | `FusedSignalStrategy` works each symbol toward its target with passive limit orders |
| `session.py`, `state.py` | Trading hours / end-of-day flattening; daily PnL and halt persisted across restarts |
| `events.py`, `backtest.py`, `download.py` | Recording format, replay backtester, IBKR history downloader |
| `journal.py`, `report.py` | Trade journal (SQLite) and per-source performance report / weight updates |
| `risk.py` | Pre-trade checks, position and PnL tracking, kill switch |
| `engine.py` | Order lifecycle: one working order per symbol, stale-order cancels, halt handling |
| `broker/` | `IBKRBroker` (ib_async) and `SimBroker` (offline, fills conservatively) |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env   # fill in keys; each data source turns on only when configured
set -a; source .env; set +a
```

Run TWS or IB Gateway with the API enabled (Configure → API → Settings).

```bash
python -m trading_agent --mode sim   --symbols AAPL MSFT   # synthetic prices, no broker
python -m trading_agent --mode paper --symbols AAPL MSFT   # IBKR paper account
python -m trading_agent --mode live  --symbols AAPL        # real money (see below)
```

`--no-news` turns off the LLM pipeline and `--record` saves everything for backtesting.
Ctrl-C cancels the agent's working orders and disconnects.

### Trading day and restarts

- **Session** (paper/live): new positions only between `SESSION_START` (09:35 ET) and
  `SESSION_FLATTEN` (15:50 ET). After that the agent works every position back to flat
  with marketable limit orders before the 16:00 close, and places nothing outside hours.
  Holidays and half days aren't modelled.
- **Startup:** cancels orders left over from a previous run (this client ID's orders
  only, never ones you placed by hand) and loads current IBKR positions for the traded
  symbols. Holdings in other symbols are ignored.
- **Reconciliation:** every 15s the agent compares its positions with IBKR's. A mismatch
  that persists across two checks with no orders in flight halts trading completely.
- **Saved state** (`STATE_FILE`, default `data/state.json`): today's realized PnL and any
  halt survive a restart, so restarting can't reset the daily-loss limit. A halt carries
  over to the next day. Resume with `--clear-halt` once you've investigated it.
- **Daily-loss halt:** cancels everything, then only accepts orders that reduce positions
  toward flat, so a losing position isn't carried overnight.

### Data sources

| Source | Needs | Notes |
|---|---|---|
| IBKR news | Paper/live mode, IBKR news subscriptions | Headlines from whatever providers your account has enabled |
| SEC EDGAR | `SEC_USER_AGENT` (name + email, required by the SEC) | 8-K/6-K: main document and EX-99 exhibits (press releases), capped at 20k characters with a visible truncation marker. Form 4: insider, role, each buy/sell with size and price, 10b5-1 flag. 10-Q/10-K/13D/G: metadata only |
| Finnhub | `FINNHUB_API_KEY` | Free tier: 60 calls/min, enough for ~60 symbols at the default 60s poll |
| Reddit | `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Official OAuth API; r/wallstreetbets, r/stocks, r/investing |
| Claude | `ANTHROPIC_API_KEY` | `ANALYST_MODEL` (default `claude-opus-5`), `ANALYST_EFFORT` (default `medium`) |

The analyst makes one Claude call per symbol per poll, and only when that symbol has new items.

## Safety

- **Paper by default.** Paper mode refuses to connect to a live port (7496/4001). Live mode
  needs both `--mode live` and `TRADING_ALLOW_LIVE=yes`.
- **Risk limits** (`RISK_*` in `.env`): max position (counting working orders as filled),
  max order size and notional, max distance from mid, orders/sec, and a daily-loss limit
  that halts trading, cancels working orders and flattens.
- **Untrusted text.** News and social posts are untrusted input to the LLM, and its output
  moves money. So the output is schema-constrained, numbers are clamped, and the prompt
  treats embedded instructions as data. Fusion weights cap the LLM's influence (0.6 by
  default), and risk limits bound the book whatever any signal says.

## Trade journal and source quality

Every run writes to `data/journal.db` (SQLite; `JOURNAL_FILE`). It records:

- **every order decision** with the full reasoning behind it: each active signal's
  score, confidence, decay, weight and contribution, plus the resulting conviction and
  target
- **every fill** with its commission, linked to its decision
- **every round-trip trade** (flat -> position -> flat) with entry/exit prices, gross and
  net PnL and fees, and the decisions that opened and closed it
- **attribution**: each trade's net PnL split across the signal sources that supported
  its entry, in proportion to their contribution. Sources that pointed the other way get
  no credit.
- **every Claude signal** with its rationale and the news items it read. Claude now
  names which items drove its view, so credit goes to the right source.
- **signal outcomes**: the price move 1, 5, 30 and 60 minutes after each signal, so news
  sources are judged on every prediction, not only the few that became trades

```bash
python -m trading_agent.report                   # performance + source scorecards
python -m trading_agent.report --trade 42        # everything behind one trade
python -m trading_agent.report --since 2026-09-01
python -m trading_agent.report --write           # update data/source_quality.json
```

The report scores the **fusion sources** (`llm`, `micro:imbalance`, `micro:reversion`) on
the trades they supported. It scores the **news sources** (e.g. `edgar`,
`finnhub/Reuters`, `reddit/r/wallstreetbets`, `ibkr/BRFG`) on the hit rate and average
move of the signals they drove, plus their attributed PnL.

**Closing the loop.** `--write` saves suggested fusion weights and each news source's track
record. The agent loads that file on its next start:
- **weights:** the suggested weights replace the defaults.
- **track records:** Claude sees each item's record next to it, e.g.
  `track_record="right 31 of 44 past calls (70%)"`.

Suggestions are shrunk toward neutral and move off the defaults only after 10 supported
trades per source. They're capped at 0.25x-2x and always computed from the defaults, so
re-running doesn't compound. Attribution is correlational: a source that often agrees
with a good one looks good too. Review the report before writing.

Backtests can journal too (`--journal data/backtest.db`), which gives you source
scorecards over historical data.

## Backtesting

The backtester replays events through the same engine, strategy, fusion and risk code
used live. Only the broker and the clock are simulated.

```bash
# 1. Data: record your own sessions (best: exact quotes with sizes, plus the LLM signals
#    the agent actually produced) ...
python -m trading_agent --mode paper --symbols AAPL MSFT --record   # -> data/recordings/
# ... or download history from IBKR (1-second quotes) plus news from EDGAR/Finnhub/IBKR
python -m trading_agent.download --symbols AAPL MSFT --date 2026-09-24   # -> data/history/

# 2. Replay
python -m trading_agent.backtest data/recordings/2026-09-24.jsonl        # recorded signals
python -m trading_agent.backtest data/history/2026-09-24.jsonl --reanalyze --equity-csv eq.csv
```

- `--reanalyze` runs Claude over the recorded news on a simulated 60s poll. Each signal
  becomes usable only after `--latency` seconds (default 10). Results are cached in
  `data/analysis_cache.jsonl`, so re-runs are free and deterministic.
- **Lookahead bias:** a model asked about past news may already know how the story ended.
  Treat `--reanalyze` results for dates the model could know about as optimistic. Signals
  recorded live don't have this problem.
- **Fill model:** resting orders fill only when the opposite side trades through the
  limit (conservative for entries). There's no queue position, partial fills or market
  impact, and commissions default to IBKR's fixed $0.0035/share, $0.35 minimum.
- Downloaded 1-second bars have no book sizes, so the imbalance signal is inactive in
  those backtests (`download --ticks` gets sizes but is only practical for short windows).

## Development

```bash
pytest          # offline; no broker, network or API key needed
ruff check src tests && ruff format src tests
```

## Known gaps / next steps

- **Validate on paper.** The IBKR adapter, downloader and live Claude calls are tested
  only against fakes. Run on a paper account and watch the reconciliation logs.
- **Tune with backtests.** Fusion weights, thresholds and position sizes are placeholders.
- Short-sale locate/borrow checks, exchange holiday calendar, IBKR L2 depth for better
  microstructure signals, X/Twitter (paid API).

This software can lose money. Nothing here is investment advice.
