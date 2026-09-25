"""SEC EDGAR filings (free; SEC requires a descriptive User-Agent with contact info).

Uses the submissions JSON API. Only filing metadata is sent to the analyst for now --
fetching and summarising the filing documents themselves is a natural next step.
"""

from __future__ import annotations

import logging
from datetime import datetime

import httpx

from ..models import NewsItem

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

DEFAULT_FORMS = frozenset({"8-K", "8-K/A", "4", "10-Q", "10-K", "SC 13D", "SC 13G", "6-K"})

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


class EdgarSource:
    name = "edgar"

    def __init__(
        self,
        user_agent: str,
        client: httpx.AsyncClient | None = None,
        forms: frozenset[str] = DEFAULT_FORMS,
        per_symbol: int = 10,
    ):
        self._client = client or httpx.AsyncClient(headers={"User-Agent": user_agent}, timeout=10.0)
        self.forms = forms
        self.per_symbol = per_symbol
        self._ciks: dict[str, int] = {}

    async def _load_ciks(self) -> None:
        resp = await self._client.get(TICKERS_URL)
        resp.raise_for_status()
        self._ciks = {row["ticker"].upper(): int(row["cik_str"]) for row in resp.json().values()}

    async def fetch(self, symbols: list[str]) -> list[NewsItem]:
        if not self._ciks:
            await self._load_ciks()
        items: list[NewsItem] = []
        for sym in symbols:
            cik = self._ciks.get(sym.upper())
            if cik is None:
                log.warning("EDGAR: no CIK for %s", sym)
                continue
            resp = await self._client.get(SUBMISSIONS_URL.format(cik=cik))
            resp.raise_for_status()
            items.extend(self.parse_submissions(sym, cik, resp.json()))
        return items

    def parse_submissions(self, symbol: str, cik: int, data: dict) -> list[NewsItem]:
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        out: list[NewsItem] = []
        for i, form in enumerate(forms):
            if form not in self.forms:
                continue
            acc = recent["accessionNumber"][i]
            accepted = recent.get("acceptanceDateTime", [""] * len(forms))[i]
            desc = recent.get("primaryDocDescription", [""] * len(forms))[i]
            doc = recent.get("primaryDocument", [""] * len(forms))[i]
            item_codes = [c.strip() for c in recent.get("items", [""] * len(forms))[i].split(",")]
            item_text = "; ".join(
                f"Item {c}: {EIGHT_K_ITEMS.get(c, 'see filing')}" for c in item_codes if c
            )
            out.append(
                NewsItem(
                    id=f"edgar:{acc}",
                    source=self.name,
                    symbol=symbol,
                    headline=f"{data.get('name', symbol)} filed {form}"
                    + (f" ({desc})" if desc else ""),
                    body=item_text,
                    url=ARCHIVE_URL.format(cik=cik, acc=acc.replace("-", ""), doc=doc),
                    published_at=_parse_ts(accepted),
                )
            )
            if len(out) >= self.per_symbol:
                break
        return out


def _parse_ts(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0
