# ═══════════════════════════════════════════════════════════════
# Dockerfile — FastAPI Backend + Core Engine
# ═══════════════════════════════════════════════════════════════
# Build context: project root
# Runs: uvicorn backend.main:app on port 8000
# ═══════════════════════════════════════════════════════════════

FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (cached layer — only re-runs if requirements.txt changes) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ────────────────────────────────────────────────
COPY backend/      ./backend/
COPY core_engine.py .
COPY database.py    .

# ── Runtime directories (SQLite DB + ML models live here) ──────
# These are overridden by Docker volumes in docker-compose.yml
# so data persists across container restarts.
RUN mkdir -p db models

# ── Non-root user for security ─────────────────────────────────
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# ── Health check — used by docker-compose depends_on ───────────
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# ── Start FastAPI ───────────────────────────────────────────────
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
