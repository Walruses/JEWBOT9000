"""Interface for text/event data sources feeding the LLM analyst."""

from __future__ import annotations

from typing import Protocol

from ..models import NewsItem


class NewsSource(Protocol):
    name: str

    async def fetch(self, symbols: list[str]) -> list[NewsItem]:
        """Return recent items for the given symbols. Duplicates across calls are fine;
        the pipeline de-duplicates on NewsItem.id."""
        ...
