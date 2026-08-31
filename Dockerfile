# Vantage: self-correcting multi-agent data analyst.
#
# The warehouse is generated at build time, so the image runs offline and the
# container's data is byte-identical to the one the benchmark was scored on.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VANTAGE_DB=/app/data/warehouse.db \
    VANTAGE_LOG_DIR=/app/logs \
    VANTAGE_MODEL=mock

WORKDIR /app

# Dependencies first: they change far less often than the source.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY bench ./bench
COPY tests ./tests
RUN pip install --no-deps -e .

# Deterministic 258,000-row warehouse, baked into the image.
RUN python -m vantage.warehouse.generate --out /app/data/warehouse.db

# Run as a non-root user; the database is mounted read-only at runtime anyway.
RUN useradd --create-home --uid 10001 vantage \
    && mkdir -p /app/logs \
    && chown -R vantage:vantage /app
USER vantage

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health', timeout=4).status_code == 200 else 1)"

CMD ["uvicorn", "vantage.api:app", "--host", "0.0.0.0", "--port", "8000"]
