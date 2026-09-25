#!/usr/bin/env bash
# Interactive paper-trading session on this machine (for a VPS service, see deploy/).
#   scripts/paper.sh        Ctrl-C to stop; run inside tmux/screen to survive logout
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "No .env: cp .env.example .env and fill it in"; exit 1; }
set -a; source .env; set +a
export MODE=paper REFRESH_WATCHLIST="${REFRESH_WATCHLIST:-no}"
mkdir -p logs
scripts/run.sh 2>&1 | tee -a "logs/agent-$(date +%F).log"
