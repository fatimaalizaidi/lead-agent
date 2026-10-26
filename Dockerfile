FROM python:3.13-slim

WORKDIR /app

# PORT is overridden by hosts (Render). Secrets are NOT baked in —
# pass OPENAI_API_KEY at runtime (-e / env_file / platform secret).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser requirements.txt .

# chroma-hnswlib may compile from source (needs g++); purge toolchain afterward.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=appuser:appuser main.py rag.py config.py errors.py guardrails.py observability.py ./
COPY --chown=appuser:appuser knowledge/ ./knowledge/

# Writable dirs for Chroma persistence + rotating logs (app runs as non-root).
RUN mkdir -p /app/data/chroma /app/logs \
    && chown -R appuser:appuser /app/data /app/logs

USER appuser

EXPOSE 8000

# Shell form so $PORT expands (Render injects PORT).
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
