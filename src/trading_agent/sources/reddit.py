"""Social chatter from Reddit via the official OAuth API (application-only auth).

Create a "script" app at https://www.reddit.com/prefs/apps for the client id/secret.
Very short tickers (e.g. "A", "IT", "ON") produce noisy matches.
"""

from __future__ import annotations

import time

import httpx

from ..models import NewsItem

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
SEARCH_URL = "https://oauth.reddit.com/r/{subs}/search"


class RedditSource:
    name = "reddit"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        subreddits: tuple[str, ...] = ("wallstreetbets", "stocks", "investing"),
        min_score: int = 5,
        client: httpx.AsyncClient | None = None,
    ):
        self._auth = (client_id, client_secret)
        self._client = client or httpx.AsyncClient(headers={"User-Agent": user_agent}, timeout=10.0)
        self.subreddits = subreddits
        self.min_score = min_score
        self._token = ""
        self._token_expiry = 0.0

    async def _ensure_token(self) -> None:
        if self._token and time.time() < self._token_expiry - 60:
            return
        resp = await self._client.post(
            TOKEN_URL, auth=self._auth, data={"grant_type": "client_credentials"}
        )
        resp.raise_for_status()
        body = resp.json()
        self._token = body["access_token"]
        self._token_expiry = time.time() + float(body.get("expires_in", 3600))

    async def fetch(self, symbols: list[str]) -> list[NewsItem]:
        await self._ensure_token()
        headers = {"Authorization": f"bearer {self._token}"}
        url = SEARCH_URL.format(subs="+".join(self.subreddits))
        items: list[NewsItem] = []
        for sym in symbols:
            params = {
                "q": f'"{sym}" OR "${sym}"',
                "restrict_sr": "1",
                "sort": "new",
                "t": "day",
                "limit": "25",
            }
            resp = await self._client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            items.extend(self.parse(sym, resp.json()))
        return items

    def parse(self, symbol: str, listing: dict) -> list[NewsItem]:
        out = []
        for child in listing.get("data", {}).get("children", []):
            post = child.get("data", {})
            if post.get("score", 0) < self.min_score:
                continue
            out.append(
                NewsItem(
                    id=f"reddit:{post['id']}",
                    source=f"{self.name}/r/{post.get('subreddit', '')}",
                    symbol=symbol,
                    headline=post.get("title", ""),
                    body=(post.get("selftext") or "")[:2000],
                    url="https://www.reddit.com" + post.get("permalink", ""),
                    published_at=float(post.get("created_utc", 0)),
                )
            )
        return out
