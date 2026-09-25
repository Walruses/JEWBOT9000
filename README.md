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

`--no-news` turns off the LLM pipeline. Ctrl-C cancels all working orders and disconnects.

### Data sources

| Source | Needs | Notes |
|---|---|---|
| IBKR news | Paper/live mode, IBKR news subscriptions | Headlines from whatever providers your account has enabled |
| SEC EDGAR | `SEC_USER_AGENT` (name + email, required by the SEC) | 8-K / 10-Q / 10-K / Form 4 / 13D/G metadata and 8-K item codes |
| Finnhub | `FINNHUB_API_KEY` | Free tier: 60 calls/min, enough for ~60 symbols at the default 60s poll |
| Reddit | `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Official OAuth API; r/wallstreetbets, r/stocks, r/investing |
| Claude | `ANTHROPIC_API_KEY` | `ANALYST_MODEL` (default `claude-opus-5`), `ANALYST_EFFORT` (default `medium`) |

The analyst makes one Claude call per symbol per poll, and only when that symbol has new items.

## Safety

- **Paper by default.** Paper mode refuses to connect to a live port (7496/4001). Live mode
  needs both `--mode live` and `TRADING_ALLOW_LIVE=yes`.
- **Risk limits** (`RISK_*` in `.env`): max position (counting working orders as filled),
  max order size and notional, max distance from mid, orders/sec, and a daily-loss limit
  that halts trading and cancels every working order.
- **Untrusted text.** News and social posts are untrusted input to the LLM, and its output
  moves money. So the output is schema-constrained, numbers are clamped, and the prompt
  treats embedded instructions as data. Fusion weights cap the LLM's influence (0.6 by
  default), and risk limits bound the book whatever any signal says.

## Development

```bash
pytest          # offline; no broker, network or API key needed
ruff check src tests && ruff format src tests
```

## Known gaps / next steps

- **Backtesting.** The strategy and sources run live only. Recording ticks and news and
  replaying them through `SimBroker` is the next step before tuning weights.
- **Filing contents.** EDGAR items carry metadata only. Fetching the 8-K exhibits (e.g.
  EX-99.1 press releases) would give the analyst much more to work with.
- **End-of-day flattening**, short-sale locate checks, and persistence of positions and PnL
  across restarts. On startup, reconcile with IBKR positions.
- **Order-book depth** (IBKR L2) for better microstructure signals, and X/Twitter (paid API).
- The IBKR adapter and live Claude calls are covered by type-level wiring only; test them
  on a paper account before relying on them.

This software can lose money. Nothing here is investment advice.
