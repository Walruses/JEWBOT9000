#!/usr/bin/env bash
# Start the agent unattended (used by Docker, systemd and scripts/paper.sh).
#
# Environment (all optional):
#   MODE=paper|live            trading mode (default paper)
#   WATCHLIST=path             symbols file (default data/watchlist.txt)
#   SCAN_PRESETS="..."         scanner presets for the watchlist (default: largecap smallcap penny)
#   REFRESH_WATCHLIST=yes      rebuild the watchlist from scanners on every start (default yes)
#   RESTART_DAILY_AT=HH:MM     exit at this ET time daily so the service restarts fresh
#   GATEWAY_WAIT_SECONDS=600   how long to wait for IB Gateway to come up
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
[ -x .venv/bin/python ] && PY=.venv/bin/python
MODE="${MODE:-paper}"
WATCHLIST="${WATCHLIST:-data/watchlist.txt}"
mkdir -p data logs "$(dirname "$WATCHLIST")"

# 1. Wait until IB Gateway has logged in and answers API requests. (Its port -- and the
#    Docker image's forwarder -- accepts TCP connections before the login finishes.)
"$PY" - <<'PYEOF'
import asyncio, os, sys, time
from ib_async import IB

host, port = os.environ.get("IB_HOST", "127.0.0.1"), int(os.environ.get("IB_PORT", "4002"))
client_id = int(os.environ.get("IB_CLIENT_ID", "1")) + 400
deadline = time.time() + float(os.environ.get("GATEWAY_WAIT_SECONDS", "600"))


async def ready() -> bool:
    ib = IB()
    try:
        await ib.connectAsync(host, port, clientId=client_id, timeout=10)
        return bool(ib.managedAccounts())
    except Exception:
        return False
    finally:
        ib.disconnect()


while not asyncio.run(ready()):
    if time.time() > deadline:
        sys.exit(f"IB Gateway at {host}:{port} did not become ready")
    print(f"waiting for IB Gateway at {host}:{port} to finish logging in ...", flush=True)
    time.sleep(15)
print("IB Gateway is ready", flush=True)
PYEOF

# 2. Watchlist from today's scanners (falls back to the previous one if the scan fails).
if [ "${REFRESH_WATCHLIST:-yes}" = "yes" ] || [ ! -s "$WATCHLIST" ]; then
  # shellcheck disable=SC2086
  "$PY" -m trading_agent.scan --preset ${SCAN_PRESETS:-largecap smallcap penny} \
    --out "$WATCHLIST.new" && mv "$WATCHLIST.new" "$WATCHLIST" \
    || echo "scan failed; using the existing $WATCHLIST"
fi
[ -s "$WATCHLIST" ] || { echo "no symbols in $WATCHLIST"; exit 1; }

# 3. Pre-flight checks; Gateway may still be finishing its login, so retry a few times.
for attempt in 1 2 3 4 5; do
  if "$PY" -m trading_agent.preflight --symbols-file "$WATCHLIST"; then break; fi
  [ "$attempt" = 5 ] && { echo "pre-flight failed; not starting"; exit 1; }
  echo "pre-flight failed (attempt $attempt); retrying in 60s"; sleep 60
done

# 4. Run. Exits cleanly at RESTART_DAILY_AT; the service manager starts it again.
ARGS=(--mode "$MODE" --symbols-file "$WATCHLIST" --record)
[ -n "${RESTART_DAILY_AT:-}" ] && ARGS+=(--restart-daily-at "$RESTART_DAILY_AT")
exec "$PY" -m trading_agent "${ARGS[@]}"
