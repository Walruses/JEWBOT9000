"""JSONL event format shared by the live recorder, the historical downloader and the
backtester. One JSON object per line: {"type": "tick"|"news"|"signal", "ts": ..., ...}."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from .models import NewsItem, Signal, Tick

Event = Tick | NewsItem | Signal


def encode(event: Event) -> str:
    if isinstance(event, Tick):
        return json.dumps({"type": "tick", **asdict(event)})
    if isinstance(event, NewsItem):
        return json.dumps({"type": "news", "ts": event.published_at, **asdict(event)})
    return json.dumps({"type": "signal", **asdict(event)})


def decode(line: str) -> Event:
    raw = json.loads(line)
    kind = raw.pop("type")
    if kind == "tick":
        return Tick(**raw)
    if kind == "news":
        raw.pop("ts", None)
        return NewsItem(**raw)
    if kind == "signal":
        raw["inputs"] = tuple(raw.get("inputs", ()))
        raw["drivers"] = tuple(raw.get("drivers", ()))
        return Signal(**raw)
    raise ValueError(f"unknown event type {kind!r}")


def event_ts(event: Event) -> float:
    return event.published_at if isinstance(event, NewsItem) else event.ts


def read_events(paths: list[Path]) -> list[Event]:
    """Load events from one or more files, sorted by time (stable within equal times)."""
    events: list[Event] = []
    for path in paths:
        with open(path) as f:
            events.extend(decode(line) for line in f if line.strip())
    events.sort(key=event_ts)
    return events


class Recorder:
    """Appends live events to <dir>/<UTC date>.jsonl, rotating daily."""

    def __init__(self, directory: str | Path, flush_interval: float = 1.0):
        self.directory = Path(directory)
        self.flush_interval = flush_interval
        self._file: IO[str] | None = None
        self._date = ""
        self._last_flush = 0.0

    def tick(self, tick: Tick) -> None:
        self._write(encode(tick))

    def news(self, item: NewsItem) -> None:
        self._write(encode(item))

    def signal(self, signal: Signal) -> None:
        self._write(encode(signal))

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def _write(self, line: str) -> None:
        now = time.time()
        today = datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d")
        if today != self._date:
            self.close()
            self.directory.mkdir(parents=True, exist_ok=True)
            self._file = open(self.directory / f"{today}.jsonl", "a")  # noqa: SIM115
            self._date = today
        assert self._file is not None
        self._file.write(line + "\n")
        if now - self._last_flush >= self.flush_interval:
            self._file.flush()
            self._last_flush = now
