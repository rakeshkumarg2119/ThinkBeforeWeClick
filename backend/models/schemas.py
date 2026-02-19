"""
Pydantic schemas for request/response validation.
All fields map directly to what core_engine.analyze_url() returns.
"""
from pydantic import BaseModel, HttpUrl, field_validator
from typing import Optional


class URLRequest(BaseModel):
    """Request body for POST /analyze"""
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        if len(v) < 10:
            raise ValueError("URL is too short to be valid")
        return v


class AnalysisResult(BaseModel):
    """
    Full analysis result returned by the API.
    Mirrors the dict returned by core_engine.analyze_url().
    """
    url: str
    domain: str

    # Individual component scores
    domain_score: int
    url_score: int
    keyword_score: int
    security_score: int
    redirect_score: int
    total_score: int

    # Risk assessment
    risk_level: str                  # "Low" | "Medium" | "High"
    risk_level_numeric: int          # 0 | 1 | 2
    risk_type: str                   # "Phishing" | "Malware" | "Scam" | etc.
    confidence_percent: float        # 0.0 – 100.0
    risk_severity_index: int         # 0 – 100

    # Extra flags
    anomaly_detected: bool
    why_risk: str
    gambling_warning: Optional[str] = None  # None unless it's a gambling site
    cached: bool = False                    # True if result was served from DB cache


class HealthResponse(BaseModel):
    """Response for GET /health"""
    status: str
    message: str
    version: str


class ErrorResponse(BaseModel):
    """Standardised error envelope"""
    error: str
    detail: Optional[str] = None