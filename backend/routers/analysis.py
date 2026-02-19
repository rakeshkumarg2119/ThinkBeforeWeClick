"""
API routers — defines all HTTP endpoints.

Endpoints:
    POST /analyze   →  run URL risk analysis via core_engine
    GET  /health    →  liveness probe
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from backend.models.schemas import (
    URLRequest,
    AnalysisResult,
    HealthResponse,
    ErrorResponse,
)
from backend.services.analysis_service import run_analysis

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------
@router.post(
    "/analyze",
    response_model=AnalysisResult,
    summary="Analyse a URL for security risks",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or un-parseable URL"},
        422: {"description": "Request body validation failed"},
        500: {"model": ErrorResponse, "description": "Internal analysis engine error"},
    },
)
async def analyze_url_endpoint(request: URLRequest):
    """
    **Accepts** a JSON body: `{"url": "https://example.com"}`

    Passes the URL through `core_engine.analyze_url()` and returns the full
    risk report including per-component scores, risk level, confidence,
    anomaly flag, and (if applicable) a gambling financial warning.

    Results are automatically cached in SQLite — submitting the same URL a
    second time returns the cached result in milliseconds.
    """
    try:
        result = run_analysis(request.url)
    except ValueError as exc:
        # Bad URL or core_engine returned an error dict
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except RuntimeError as exc:
        # Unexpected engine failure
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return AnalysisResult(**result)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / health check",
)
async def health_check():
    """
    Returns a simple status payload. Use this to verify the backend is running
    before wiring up the Streamlit frontend.
    """
    return HealthResponse(
        status="ok",
        message="URL Risk Analyzer API is running",
        version="3.0.0",
    )
