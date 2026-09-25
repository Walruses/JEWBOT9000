"""News headlines through the IBKR connection (feeds depend on your IBKR subscriptions)."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from ..broker.ibkr import IBKRBroker
from ..models import NewsItem

log = logging.getLogger(__name__)

# IBKR prefixes headlines with metadata like "{A:800015:L:en:K:0.97:C:0.99}".
_META = re.compile(r"^\{[^}]*\}\s*")


class IBKRNewsSource:
    name = "ibkr"

    def __init__(self, broker: IBKRBroker, per_symbol: int = 10):
        self.broker = broker
        self.per_symbol = per_symbol
        self._providers = ""

    async def fetch(self, symbols: list[str]) -> list[NewsItem]:
        return await self._fetch(symbols, "", "", self.per_symbol)

    async def fetch_between(self, symbols: list[str], start: float, end: float) -> list[NewsItem]:
        return await self._fetch(
            symbols,
            datetime.fromtimestamp(start, UTC),
            datetime.fromtimestamp(end, UTC),
            300,  # IBKR's maximum per request
        )

    async def _fetch(self, symbols: list[str], start, end, limit: int) -> list[NewsItem]:
        ib = self.broker.ib
        if not self._providers:
            providers = await ib.reqNewsProvidersAsync()
            self._providers = "+".join(p.code for p in providers)
            if not self._providers:
                log.warning("IBKR: no news providers enabled on this account")
                return []
        items: list[NewsItem] = []
        for sym in symbols:
            contract = self.broker.contract(sym)
            if contract is None:
                continue
            result = await ib.reqHistoricalNewsAsync(
                contract.conId, self._providers, start, end, limit
            )
            rows = result if isinstance(result, list) else [result] if result else []
            for row in rows:
                ts = row.time.timestamp() if isinstance(row.time, datetime) else 0.0
                items.append(
                    NewsItem(
                        id=f"ibkr:{row.providerCode}:{row.articleId}",
                        source=f"{self.name}/{row.providerCode}",
                        symbol=sym,
                        headline=_META.sub("", row.headline),
                        published_at=ts,
                    )
                )
        return items
