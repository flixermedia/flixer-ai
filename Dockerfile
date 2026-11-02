FROM python:3.12-slim

ARG APP_VERSION=0.1.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    PORT=8094

WORKDIR /app

LABEL org.opencontainers.image.version="${APP_VERSION}"

COPY pyproject.toml .
COPY VERSION .
COPY src ./src

RUN pip install --no-cache-dir .

RUN groupadd --system --gid 10001 flixer-ai \
    && useradd --system --uid 10001 --gid flixer-ai --home-dir /nonexistent --shell /usr/sbin/nologin flixer-ai \
    && mkdir -p /data/ai/models \
    && chown -R 10001:10001 /data/ai

USER 10001:10001

EXPOSE 8094

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8094/healthz', timeout=2)"

CMD ["sh", "-c", "uvicorn flixer_ai.main:app --host 0.0.0.0 --port ${PORT:-8094}"]
