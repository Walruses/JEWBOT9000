import logging

import httpx

from trading_agent.logsetup import RedactSecrets, redact


def test_redacts_query_tokens_and_bearer():
    url = "https://finnhub.io/api/v1/company-news?symbol=NVDA&token=abc123&from=2026-09-28"
    assert (
        redact(url)
        == "https://finnhub.io/api/v1/company-news?symbol=NVDA&token=***&from=2026-09-28"
    )
    assert redact("Authorization: Bearer sk-ant-xyz") == "Authorization: Bearer ***"


def test_filter_redacts_args_and_tracebacks():
    request = httpx.Request("GET", "https://finnhub.io/api/v1/quote?token=SECRET")
    err = httpx.HTTPStatusError(
        f"Client error '401 Unauthorized' for url '{request.url}'",
        request=request,
        response=httpx.Response(401, request=request),
    )
    try:
        raise err
    except httpx.HTTPStatusError:
        record = logging.LogRecord(
            "x", logging.WARNING, __file__, 1, "source failed: %r", (err,), None
        )
        import sys

        record.exc_info = sys.exc_info()
    RedactSecrets().filter(record)
    text = logging.Formatter().format(record)
    assert "SECRET" not in text
    assert "token=***" in text
