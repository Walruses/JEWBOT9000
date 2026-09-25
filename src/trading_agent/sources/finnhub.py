"""Company news from Finnhub (https://finnhub.io; free tier allows 60 calls/minute)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx

from ..models import NewsItem

NEWS_URL = "https://finnhub.io/api/v1/company-news"


class FinnhubSource:
    name = "finnhub"

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None):
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=10.0)

    async def fetch(self, symbols: list[str]) -> list[NewsItem]:
        today = datetime.now(UTC).date()
        params = {"from": str(today - timedelta(days=1)), "to": str(today), "token": self._api_key}
        items: list[NewsItem] = []
        for sym in symbols:
            resp = await self._client.get(NEWS_URL, params={**params, "symbol": sym})
            resp.raise_for_status()
            items.extend(self.parse(sym, resp.json()))
        return items

    def parse(self, symbol: str, rows: list[dict]) -> list[NewsItem]:
        return [
            NewsItem(
                id=f"finnhub:{row['id']}",
                source=f"{self.name}/{row.get('source', '')}",
                symbol=symbol,
                headline=row.get("headline", ""),
                body=row.get("summary", ""),
                url=row.get("url", ""),
                published_at=float(row.get("datetime", 0)),
            )
            for row in rows
            if row.get("headline")
        ]
