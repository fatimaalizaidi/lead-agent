FROM python:3.13-slim

WORKDIR /app

# PORT is overridden by hosts (Render/Railway). Secrets are NOT baked in —
# pass OPENAI_API_KEY at runtime (-e / env_file / platform secret).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser main.py rag.py config.py errors.py guardrails.py observability.py ./
COPY --chown=appuser:appuser knowledge/ ./knowledge/

# Writable dirs for Chroma persistence + rotating logs (app runs as non-root).
RUN mkdir -p /app/data/chroma /app/logs \
    && chown -R appuser:appuser /app/data /app/logs

USER appuser

EXPOSE 8000

# Shell form so $PORT expands (Render/Railway inject PORT).
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
