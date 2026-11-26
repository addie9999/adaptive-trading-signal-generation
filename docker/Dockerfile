# syntax=docker/dockerfile:1
# ---- Multi-stage build for the AQAF prediction service ----
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    AQAF_ENV=production

WORKDIR /app

# System deps for scientific Python wheels (xgboost/scipy need libgomp).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install ".[backtest]"

# Copy source.
COPY src ./src
COPY configs ./configs
COPY main.py ./

EXPOSE 8000

# Default: serve the API. Override CMD for training jobs.
CMD ["uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
