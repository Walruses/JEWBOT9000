import asyncio
import time
from datetime import datetime

import httpx

from trading_agent.session import NEW_YORK
from trading_agent.sources import EdgarSource, FinnhubSource, RedditSource
from trading_agent.sources.edgar import parse_form4, truncate

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
    source = EdgarSource("test agent test@example.com")
    [filing] = source.parse_submissions("AAPL", 320193, SUBMISSIONS)  # S-8 filtered out
    item = filing.item
    assert item.id == "edgar:0000320193-24-000001"
    assert "Results of Operations" in item.body
    assert item.url.endswith("/320193/000032019324000001/aapl-8k.htm")
    # Read as Eastern wall-clock time: 16:30:12 EST == 21:30:12 UTC
    assert item.published_at == 1706823012.0


FORM4 = """<?xml version="1.0"?>
<ownershipDocument>
  <aff10b5One>1</aff10b5One>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>DOE JANE</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector><isOfficer>1</isOfficer><officerTitle>CEO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable><nonDerivativeTransaction>
    <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
    <transactionAmounts>
      <transactionShares><value>1000</value></transactionShares>
      <transactionPricePerShare><value>150.50</value></transactionPricePerShare>
    </transactionAmounts>
    <postTransactionAmounts>
      <sharesOwnedFollowingTransaction><value>50000</value></sharesOwnedFollowingTransaction>
    </postTransactionAmounts>
  </nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>"""


def test_parse_form4():
    text = parse_form4(FORM4)
    assert "DOE JANE (CEO)" in text
    assert "10b5-1" in text
    assert "open-market SALE: 1,000 shares @ $150.50 ($150,500); owns 50,000" in text


def test_edgar_fetch_enriches_8k_with_exhibit_text():
    now = time.time()
    accepted = datetime.fromtimestamp(now - 60, NEW_YORK).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    subs = {
        "name": "Apple Inc.",
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-24-000009"],
                "form": ["8-K"],
                "acceptanceDateTime": [accepted],
                "primaryDocument": ["main.htm"],
                "primaryDocDescription": ["8-K"],
                "items": ["2.02"],
            }
        },
    }
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        path = request.url.path
        if path.endswith("company_tickers.json"):
            return httpx.Response(200, json={"0": {"cik_str": 320193, "ticker": "AAPL"}})
        if "submissions" in path:
            return httpx.Response(200, json=subs)
        if path.endswith("index.json"):
            names = ["main.htm", "ex99-1.htm", "logo.jpg"]
            return httpx.Response(200, json={"directory": {"item": [{"name": n} for n in names]}})
        if path.endswith("main.htm"):
            return httpx.Response(
                200, text="<html><ix:header>hidden</ix:header><p>Cover</p></html>"
            )
        if path.endswith("ex99-1.htm"):
            return httpx.Response(200, text="<p>Revenue rose <b>12%</b></p><script>x</script>")
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = EdgarSource("ua", client=client, min_request_interval=0)
    [item] = asyncio.run(source.fetch(["AAPL"]))
    assert "Revenue rose 12%" in item.body
    assert "Cover" in item.body and "hidden" not in item.body and "x" not in item.body.split()
    assert "Item 2.02" in item.body

    n = len(requests)
    asyncio.run(source.fetch(["AAPL"]))  # contents cached: only the submissions call repeats
    assert len(requests) == n + 1


def test_truncate_marks_cut_text():
    assert truncate("abcdef", 3) == "abc\n[truncated: first 3 of 6 characters shown]"


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
