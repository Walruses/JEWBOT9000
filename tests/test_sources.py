import asyncio

import httpx

from trading_agent.sources import EdgarSource, FinnhubSource, RedditSource

SUBMISSIONS = {
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-24-000001", "0000320193-24-000002"],
            "form": ["8-K", "S-8"],
            "acceptanceDateTime": ["2024-02-01T16:30:12.000Z", "2024-02-01T17:00:00.000Z"],
            "primaryDocument": ["aapl-8k.htm", "s8.htm"],
            "primaryDocDescription": ["8-K", "S-8"],
            "items": ["2.02,9.01", ""],
        }
    },
}


def test_edgar_parses_relevant_filings():
    items = EdgarSource("test agent test@example.com").parse_submissions(
        "AAPL", 320193, SUBMISSIONS
    )
    [filing] = items  # S-8 is filtered out
    assert filing.id == "edgar:0000320193-24-000001"
    assert "Results of Operations" in filing.body
    assert filing.url.endswith("/320193/000032019324000001/aapl-8k.htm")
    assert filing.published_at == 1706805012.0


def test_finnhub_parse():
    rows = [
        {
            "id": 7,
            "headline": "Apple news",
            "summary": "s",
            "source": "Reuters",
            "url": "u",
            "datetime": 1706805012,
        },
        {"id": 8, "headline": ""},
    ]
    [item] = FinnhubSource("k").parse("AAPL", rows)
    assert (item.id, item.source, item.published_at) == (
        "finnhub:7",
        "finnhub/Reuters",
        1706805012.0,
    )


def test_reddit_fetch_authenticates_and_filters_low_score():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/access_token":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        assert request.headers["Authorization"] == "bearer tok"
        posts = [
            {
                "id": "a",
                "title": "AAPL to the moon",
                "score": 50,
                "created_utc": 1.0,
                "permalink": "/r/stocks/a",
                "subreddit": "stocks",
                "selftext": "",
            },
            {
                "id": "b",
                "title": "AAPL?",
                "score": 1,
                "created_utc": 2.0,
                "permalink": "/r/stocks/b",
                "subreddit": "stocks",
            },
        ]
        return httpx.Response(200, json={"data": {"children": [{"data": p} for p in posts]}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = RedditSource("id", "secret", "ua", client=client)
    [item] = asyncio.run(source.fetch(["AAPL"]))
    assert item.id == "reddit:a"
    assert item.source == "reddit/r/stocks"
