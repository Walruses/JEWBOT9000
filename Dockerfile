# The trading agent. IB Gateway runs in its own container (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 trader
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
COPY scripts ./scripts

# Recorded in every run's configuration snapshot (containers have no .git).
ARG CODE_VERSION=unknown
ENV CODE_VERSION=${CODE_VERSION}

RUN mkdir -p data logs && chown -R trader:trader /app
USER trader
VOLUME ["/app/data"]
ENTRYPOINT ["scripts/run.sh"]
