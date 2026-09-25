"""SEC EDGAR filings (free; SEC requires a descriptive User-Agent with contact info and
at most 10 requests/second).

For each new filing the analyst gets real content, not just metadata:
- 8-K / 6-K: text of the main document plus EX-99 exhibits (usually the press release),
  capped at MAX_FILING_CHARS with an explicit truncation marker.
- Form 4: the insider's name/role and each transaction (buy/sell, shares, price,
  whether it was a pre-planned 10b5-1 trade).
- 10-Q / 10-K / 13D / 13G: metadata only; they are long and rarely move prices intraday
  beyond what the accompanying 8-K press release already says.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime
from html.parser import HTMLParser

import httpx

from ..models import NewsItem
from ..session import NEW_YORK

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FILING_DIR = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}"

DEFAULT_FORMS = frozenset({"8-K", "8-K/A", "4", "10-Q", "10-K", "SC 13D", "SC 13G", "6-K"})
TEXT_FORMS = frozenset({"8-K", "8-K/A", "6-K"})
MAX_FILING_CHARS = 20_000
_EXHIBIT_99 = re.compile(r"ex[-_]?99", re.IGNORECASE)

# Most price-relevant 8-K items, so the analyst sees what a filing is about.
EIGHT_K_ITEMS = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy Listing Rule",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.02": "Departure/Appointment of Directors or Officers",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
}

FORM4_CODES = {
    "P": "open-market PURCHASE",
    "S": "open-market SALE",
    "A": "grant/award",
    "M": "option exercise",
    "F": "shares withheld for tax",
    "G": "gift",
    "C": "conversion",
    "X": "option exercise",
}


class EdgarSource:
    name = "edgar"

    def __init__(
        self,
        user_agent: str,
        client: httpx.AsyncClient | None = None,
        forms: frozenset[str] = DEFAULT_FORMS,
        max_age: float = 3600.0,
        min_request_interval: float = 0.12,
    ):
        self._client = client or httpx.AsyncClient(headers={"User-Agent": user_agent}, timeout=15.0)
        self.forms = forms
        self.max_age = max_age
        self._ciks: dict[str, int] = {}
        # accession -> enriched body, so repeated polls don't refetch documents
        self._bodies: OrderedDict[str, str] = OrderedDict()
        self._min_interval = min_request_interval
        self._throttle = asyncio.Lock()
        self._last_request = 0.0

    async def fetch(self, symbols: list[str]) -> list[NewsItem]:
        now = time.time()
        return await self.fetch_between(symbols, now - self.max_age, now)

    async def fetch_between(self, symbols: list[str], start: float, end: float) -> list[NewsItem]:
        if not self._ciks:
            resp = await self._get(TICKERS_URL)
            self._ciks = {r["ticker"].upper(): int(r["cik_str"]) for r in resp.json().values()}
        items: list[NewsItem] = []
        for sym in symbols:
            cik = self._ciks.get(sym.upper())
            if cik is None:
                log.warning("EDGAR: no CIK for %s", sym)
                continue
            resp = await self._get(SUBMISSIONS_URL.format(cik=cik))
            now = time.time()
            for filing in self.parse_submissions(sym, cik, resp.json()):
                published = min(filing.item.published_at, now)
                if start <= published <= end:
                    filing.item = replace(filing.item, published_at=published)
                    items.append(await self._enrich(filing))
        return items

    def parse_submissions(self, symbol: str, cik: int, data: dict) -> list[_Filing]:
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])

        def col(name: str, i: int) -> str:
            values = recent.get(name) or [""] * len(forms)
            return values[i] or ""

        out: list[_Filing] = []
        for i, form in enumerate(forms):
            if form not in self.forms:
                continue
            acc = col("accessionNumber", i)
            desc = col("primaryDocDescription", i)
            codes = [c.strip() for c in col("items", i).split(",") if c.strip()]
            summary = "; ".join(f"Item {c}: {EIGHT_K_ITEMS.get(c, 'see filing')}" for c in codes)
            directory = FILING_DIR.format(cik=cik, acc=acc.replace("-", ""))
            primary = col("primaryDocument", i)
            item = NewsItem(
                id=f"edgar:{acc}",
                source=self.name,
                symbol=symbol,
                headline=f"{data.get('name', symbol)} filed {form}"
                + (f" ({desc})" if desc else ""),
                body=summary,
                url=f"{directory}/{primary}",
                published_at=_parse_ts(col("acceptanceDateTime", i)),
            )
            out.append(_Filing(item, form, directory, primary))
        return out

    async def _enrich(self, filing: _Filing) -> NewsItem:
        item = filing.item
        cached = self._bodies.get(item.id)
        if cached is None:
            try:
                if filing.form in TEXT_FORMS:
                    cached = await self._filing_text(filing)
                elif filing.form == "4":
                    # primaryDocument points at an XSL-rendered view; the raw XML sits beside it.
                    raw = filing.primary.rsplit("/", 1)[-1]
                    cached = parse_form4((await self._get(f"{filing.directory}/{raw}")).text)
                else:
                    cached = ""
            except (httpx.HTTPError, ET.ParseError) as e:
                # Not cached: try again on the next poll.
                log.warning("EDGAR: could not fetch contents of %s: %r", item.id, e)
                return item
            self._bodies[item.id] = cached
            if len(self._bodies) > 5000:
                self._bodies.popitem(last=False)
        return replace(item, body="\n\n".join(part for part in (item.body, cached) if part))

    async def _filing_text(self, filing: _Filing) -> str:
        index = (await self._get(f"{filing.directory}/index.json")).json()
        names = [f["name"] for f in index.get("directory", {}).get("item", [])]
        docs = [filing.primary] + [
            n
            for n in names
            if _EXHIBIT_99.search(n) and n.lower().endswith((".htm", ".html", ".txt"))
        ]
        parts = []
        for name in dict.fromkeys(docs):  # de-dupe, keep order
            resp = await self._get(f"{filing.directory}/{name}")
            text = resp.text if name.lower().endswith(".txt") else html_to_text(resp.text)
            parts.append(f"--- {name} ---\n{text}")
        return truncate("\n\n".join(parts), MAX_FILING_CHARS)

    async def _get(self, url: str) -> httpx.Response:
        async with self._throttle:
            wait = self._last_request + self._min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()
        resp = await self._client.get(url)
        resp.raise_for_status()
        return resp


class _Filing:
    __slots__ = ("item", "form", "directory", "primary")

    def __init__(self, item: NewsItem, form: str, directory: str, primary: str):
        self.item = item
        self.form = form
        self.directory = directory
        self.primary = primary


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated: first {limit:,} of {len(text):,} characters shown]"


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "ix:header", "head"}
    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line.strip("| "))


def parse_form4(xml_text: str) -> str:
    root = ET.fromstring(xml_text.strip())

    def text(node, path: str) -> str:
        found = node.find(path) if node is not None else None
        return (found.text or "").strip() if found is not None else ""

    owners = []
    for owner in root.findall("reportingOwner"):
        name = text(owner, "reportingOwnerId/rptOwnerName")
        rel = owner.find("reportingOwnerRelationship")
        roles = [text(rel, "officerTitle")] if text(rel, "isOfficer") in ("1", "true") else []
        if text(rel, "isDirector") in ("1", "true"):
            roles.append("Director")
        if text(rel, "isTenPercentOwner") in ("1", "true"):
            roles.append("10% owner")
        owners.append(f"{name} ({', '.join(r for r in roles if r) or 'insider'})")

    planned = text(root, "aff10b5One") in ("1", "true")
    lines = [f"Insider: {'; '.join(owners) or 'unknown'}"]
    if planned:
        lines.append("Reported as a pre-arranged Rule 10b5-1 plan trade.")
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = text(tx, "transactionCoding/transactionCode")
        shares = _num(text(tx, "transactionAmounts/transactionShares/value"))
        price = _num(text(tx, "transactionAmounts/transactionPricePerShare/value"))
        after = _num(text(tx, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"))
        what = FORM4_CODES.get(code, f"code {code}")
        value = f" (${shares * price:,.0f})" if shares and price else ""
        lines.append(
            f"- {what}: {shares:,.0f} shares @ ${price:,.2f}{value}; owns {after:,.0f} afterwards"
        )
    return "\n".join(lines)


def _num(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _parse_ts(value: str) -> float:
    """acceptanceDateTime carries a "Z" suffix but is reported to be Eastern wall-clock
    time. Reading it as Eastern is the safe choice either way: if it really were UTC,
    filings would only look a few hours *late* (never early, which would leak future
    information into backtests), and live fetches clamp times to "now"."""
    try:
        naive = datetime.fromisoformat(value.removesuffix("Z")).replace(tzinfo=None)
    except ValueError:
        return 0.0
    return naive.replace(tzinfo=NEW_YORK).timestamp()
