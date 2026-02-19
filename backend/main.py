"""
FastAPI application entry point.

Start with:
    uvicorn backend.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.analysis import router
from backend.services.analysis_service import ensure_db_ready


# ---------------------------------------------------------------------------
# Lifespan — runs once at startup and once at shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the SQLite database before the first request is served."""
    ensure_db_ready()
    print("✓ Database ready")
    yield
    # Nothing special needed on shutdown


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="URL Risk Analyzer API",
    description=(
        "AI-powered URL safety checker. Detects phishing, malware, scams, "
        "piracy, financial fraud, and gambling sites using Scikit-learn models."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the Streamlit frontend (default port 8501) to call this API
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit default
        "http://127.0.0.1:8501",
        "*",                       # remove in production and restrict to your domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------
app.include_router(router, prefix="/api/v1", tags=["Analysis"])


# ---------------------------------------------------------------------------
# Root redirect — helpful when opening http://localhost:8000 in a browser
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "URL Risk Analyzer API v3.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
