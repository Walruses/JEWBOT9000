"""Logging setup shared by the command-line entry points.

HTTP client libraries log every request URL at INFO, and some APIs (Finnhub) take the key
as a query parameter, so request logging is turned down to WARNING. As a second line of
defence, every handler redacts credential-looking query parameters and bearer tokens from
messages and tracebacks (an HTTP error's repr also contains the URL).
"""

from __future__ import annotations

import logging
import re

_SECRET_PARAM = re.compile(
    r"((?:token|api_?key|apikey|key|secret|password|client_secret)=)[^&\s'\"<>)]+",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE)

QUIET_LOGGERS = ("httpx", "httpx2", "httpcore", "anthropic._base_client")


def redact(text: str) -> str:
    return _BEARER.sub(r"\1***", _SECRET_PARAM.sub(r"\1***", text))


class RedactSecrets(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # a malformed log call: leave it to the formatter to report
            return True
        clean = redact(message)
        if clean != message:
            record.msg, record.args = clean, None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


def configure_logging(level: str = "INFO", fmt: str = "%(levelname)s %(name)s: %(message)s"):
    logging.basicConfig(level=level.upper(), format=fmt)
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for handler in logging.getLogger().handlers:
        protect(handler)


def protect(handler: logging.Handler) -> None:
    if not any(isinstance(f, RedactSecrets) for f in handler.filters):
        handler.addFilter(RedactSecrets())
