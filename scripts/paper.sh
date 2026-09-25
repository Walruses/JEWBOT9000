#!/usr/bin/env bash
# Start a paper-trading session: pre-flight checks, watchlist, then the agent.
#
#   scripts/paper.sh                 # uses data/watchlist.txt (built by a scan if missing)
#   WATCHLIST=my.txt scripts/paper.sh
#   SCAN_PRESETS="largecap smallcap" scripts/paper.sh
#
# Run it in tmux/screen or as a service so it survives logging out; stop with Ctrl-C.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "No .env: cp .env.example .env and fill it in"; exit 1; }
source .venv/bin/activate
set -a; source .env; set +a

WATCHLIST="${WATCHLIST:-data/watchlist.txt}"
if [ ! -s "$WATCHLIST" ]; then
  echo "Building $WATCHLIST from IBKR scanners: ${SCAN_PRESETS:-largecap smallcap penny}"
  python -m trading_agent.scan --preset ${SCAN_PRESETS:-largecap smallcap penny} --out "$WATCHLIST"
fi

python -m trading_agent.preflight --symbols-file "$WATCHLIST" || {
  echo "Pre-flight failed: fix the [FAIL] items above first."; exit 1; }

mkdir -p logs
python -m trading_agent --mode paper --symbols-file "$WATCHLIST" --record \
  2>&1 | tee -a "logs/agent-$(date +%F).log"
