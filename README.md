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
| `review.py`, `tuning.py` | Daily review bundle, reviewer model, bounded parameter tuning |
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
| Claude | `ANTHROPIC_API_KEY` | `ANALYST_MODEL` (default `claude-opus-5`) for SEC filings and pre-open briefings; `ANALYST_ROUTINE_MODEL` (default `claude-sonnet-5`, empty to disable) for everything else; `ANALYST_EFFORT` (default `medium`) |

**News schedule** (paper/live):

| When (ET, weekdays) | Poll every | What happens |
|---|---|---|
| 09:20 – 15:50 | 60 s | New items are analysed as they arrive (one call per symbol with new items) |
| 04:00 – 09:20, 15:50 – 20:00 | 5 min | Items are collected, not analysed |
| Overnight, weekends | 30 min | Items are collected, not analysed |
| 09:20 | once | **Pre-open briefing**: one call per symbol over everything collected since the close (up to 100 items), on the main model |

Off-hours news can't be traded until the open, and by then the opening price partly
reflects it. So it's analysed once, in context, rather than item by item overnight. This
also keeps Claude costs to trading hours plus one call per symbol per day.

**Model routing:** routine newswire, IBKR and Reddit batches go to Sonnet 5 ($2/$10 per
million tokens vs $5/$25 for Opus 5). Batches containing an SEC filing, and the pre-open
briefings, go to Opus 5. If Sonnet declines to answer, the batch is retried on Opus. The
journal records which model produced each signal. The report's "ANALYST MODELS" table
compares their hit rates, so you can check the cheaper model holds up.

### What gets traded: watchlists and tiers

Choose symbols with `--symbols`, a watchlist file (`--symbols-file`, one per line), or IBKR
scanners (`--scan smallcap penny`, or `python -m trading_agent.scan --preset ...` to save
`data/watchlist.txt`). At most `--max-symbols` (30) are traded. Scanner presets look for
unusually heavy volume, which is where news tends to be, among exchange-listed US
operating companies:

| Preset | Filter |
|---|---|
| `largecap` | market cap > $10B, price > $5 |
| `smallcap` | market cap $300M–$2B, price > $5, 500k+ shares/day |
| `penny` | price $1–$5, 1M+ shares/day |

OTC and pink-sheet stocks are always skipped, even if listed by hand: IBKR restricts
buying many of them, and their data and news are poor.

Each stock is handled by its price tier:

| | Standard ($5+, including small caps) | Penny ($1–$5) |
|---|---|---|
| Opens a position when | the Claude view clears the entry threshold | Claude is **very confident**: sentiment and confidence both >= 0.8 on a fresh view (`PENNY_MIN_SCORE`, `PENNY_MIN_CONFIDENCE`) |
| Risk per trade | `RISK_PER_TRADE_PCT` (1%) | `PENNY_RISK_PER_TRADE_PCT` (0.5%) |
| Max position | `MAX_POSITION_PCT` (50%) | `PENNY_MAX_POSITION_PCT` (10%) |
| Max spread to enter | `MAX_SPREAD_PCT` (0.5%) | `PENNY_MAX_SPREAD_PCT` (2%) |
| Stop | 1–5%, from volatility | 3–15%, from volatility |
| Short selling | yes | no |
| Order-book signals | adjust size and timing | ignored (thin books are easy to spoof) |

Below `MIN_PRICE` ($1) nothing new is opened.

**Stops follow each stock's volatility:** `STOP_VOL_MULTIPLE` (2) x the standard deviation
of 1-minute returns over 30 minutes, within the tier's range. The tier default (2% / 8%)
applies until 15 minutes of history exist. Each position's stop is fixed when it opens,
and the position is sized from it: a wider stop means fewer shares, same dollar risk.

For small companies, EDGAR also tracks offering filings (S-1, S-3, 424B prospectuses),
which usually mean dilution. Claude is told to be especially skeptical of promotional
releases and social-media hype for these stocks.

The fat-finger check measures limit prices against the current **bid/ask**, not the mid.
So stop-loss, end-of-day and halt exits at the touch are never refused on wide-spread
stocks, while a genuinely mispriced order still is.

### Account rules and position sizing

Positions are sized by **risk per trade**, not by share count:

- Every position gets a protective stop `STOP_LOSS_PCT` (default 2%) from its average
  entry. Size is chosen so hitting the stop loses at most `RISK_PER_TRADE_PCT` (default 1%)
  of equity. With $25,000 that's $250 at risk, so positions go up to $12,500.
- Also capped at `MAX_POSITION_PCT` of equity per stock (50%) and
  `MAX_GROSS_EXPOSURE_PCT` across all positions (100%). A weaker view gets a
  proportionally smaller position.
- When the stop is hit, the position is closed with a marketable limit order and the
  stock can't be re-entered for `STOP_COOLDOWN_MINUTES` (60).
- Equity comes from IBKR (NetLiquidation) and is refreshed every 15 seconds.
  `ACCOUNT_EQUITY` is used in sim and backtests.

**Pattern day trader rule** (`ACCOUNT_TYPE=margin`, the default): a margin account under
$25,000 is limited to 3 day trades per 5 business days. The test uses equity at the start
of the day. With $25,000 or more there's no limit, but a day that closes below $25,000
brings the limit into force the next morning, and the agent switches automatically.
Funding a buffer above $25,000 avoids that. The agent closes every position
the same day, so each position it opens is a day trade. It opens at most 3 per rolling
window and uses IBKR's own `DayTradesRemaining` when that's lower. The count is saved in
the state file, so restarts don't reset it.

**Cash account** (`ACCOUNT_TYPE=cash`): there's no day-trade limit, but also no short
selling. Only settled cash can be used, and sale proceeds settle the next business day,
so each day's purchases are capped at the settled cash available at the open.

Limits of the stop:
- **It's enforced by the agent, not placed at IBKR.** If the agent or its connection goes
  down, positions have no stop until it's back.
- **The exit is a limit order at the bid/ask,** so a fast move or a gap can lose more than
  1% (the tests show $164 on a $160 budget).
- **Commissions are extra.**

### How positions are opened, sized and closed

- **Only a Claude view can open a position.** The microstructure signals (book imbalance,
  mean reversion) can enlarge an entry or change its timing. They can't create a
  position, flip one, or lift a view that's too weak on its own over the entry threshold.
- **Cost check:** an order that adds exposure is sent only if its expected gain covers
  the round-trip commission `COST_SAFETY_MULTIPLE` times (default 2). Expected gain is
  shares x price x |conviction| x `EDGE_BPS_FULL_CONVICTION` (default 50 bps). The report
  replaces that default with an estimate from real signal outcomes once there's enough
  data. Commissions default to IBKR Tiered ($0.0035/share, $0.35 minimum; see
  `COMMISSION_*`). Orders that reduce exposure are never blocked.
- **Anti-churn:** exits follow the Claude view alone, not the order book. A position is
  reduced only once that view has fallen to half the position or less, going straight to
  flat when the rest would be under 20% of the maximum position. Target changes smaller
  than that are ignored.

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

It also compares the analyst models and estimates the edge a full-conviction view has
been worth. `--write` saves that estimate for the cost check.

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

## Daily review and the training period

The plan: paper-trade first, have a more capable model review every day after the close,
tune parameters from its proposals, and only then use real money. Everything the review
needs is recorded automatically in `data/journal.db`:

| Recorded | Where |
|---|---|
| Configuration of every run (all limits, tiers, weights, models, tuned values; secrets redacted) and the code version | `runs` |
| Every order decision with the full signal breakdown, conviction, target, tier, stop and cost estimate | `decisions` |
| Fills, round-trip trades (with exit reason, tier and stop), attribution to sources | `fills`, `trades`, `trade_decisions`, `trade_attribution` |
| **Trades not taken** and why: cost check, tier rules, spread, day-trade limit, stop cooldown, risk checks (repeats within a minute are counted, not duplicated) | `skipped` |
| Every analyst signal, the full text of every news item it read, which items drove it, and the price move 1/5/30/60 minutes later | `signals`, `signal_inputs`, `signal_outcomes` |
| Every Claude API call: full prompt, response, model, tokens, latency, errors | `llm_calls` |
| 1-minute price bars per stock (mid OHLC, average spread and book sizes) | `bars` |
| Stops, halts, position mismatches, end-of-day flatten, account snapshots every 5 min, and every warning or error logged | `events` |

**After each close** (16:10 ET, paper/live) the agent writes `data/reviews/<date>/`:
- `bundle.json`: the whole day, self-contained, with the cumulative scorecards and the list
  of tunable parameters with their allowed ranges.
- `summary.md`: a readable version.

```bash
python -m trading_agent.review bundle  --date 2026-09-24   # rebuild a day's bundle
python -m trading_agent.review analyze --date 2026-09-24   # have the reviewer model assess it
python -m trading_agent.review apply --date 2026-09-24 --keys STOP_VOL_MULTIPLE
python -m trading_agent.review apply --set PENNY_MIN_SCORE=0.85   # or set values by hand
```

**The reviewer:**
- **Model:** `analyze` sends the bundle to `REVIEW_MODEL` (default `claude-fable-5-1`, the
  most capable model), roughly $1–5 a day depending on activity. Your organization needs
  30-day data retention for that model; otherwise set `REVIEW_MODEL=claude-opus-5-5`.
- **What it returns:** an assessment, anomalies, a verdict on each source, proposed
  parameter changes with rationale, evidence and confidence, strategy suggestions, and data
  gaps. It's saved as `review.json` and `review.md`.
- **Running it automatically:** set `REVIEW_AUTO_ANALYZE=yes` to analyse after every close.

**Guardrails on tuning:**
- **Proposals are never applied automatically.** `apply` validates each change against the
  bounds in `tuning.py` and writes approved values to `data/tuned_params.json`.
- **Every approval is logged** in `data/tuning_history.jsonl`.
- **The agent loads tuned values on its next start;** they override `.env` for those keys.
- **Hard bounds:** only listed parameters can be tuned, within fixed ranges. Risk per trade
  can never exceed 1%, and penny confidence can't go below 0.7.
- **Off limits:** account type, live-trading switches, the daily loss limit and gross
  exposure can't be changed by the reviewer at all.

Backtests can journal too (`--journal data/backtest.db`), so the same review works on
historical days: `review bundle --db data/backtest.db --date ...`.

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
