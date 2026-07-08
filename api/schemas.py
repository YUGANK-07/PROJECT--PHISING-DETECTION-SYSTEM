"""
api/schemas.py
───────────────
Pydantic v2 request / response models for the phishing detection API.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, HttpUrl
import re


# ── Request models ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Single URL or HTML prediction request."""

    url: str = Field(
        ...,
        min_length=4,
        max_length=2048,
        description="URL to analyse for phishing",
        examples=["https://example.com"],
    )
    html: Optional[str] = Field(
        default=None,
        max_length=2_000_000,
        description="Pre-fetched HTML (optional — avoids server-side fetch)",
    )
    include_explanation: bool = Field(
        default=True,
        description="Include SHAP feature explanations in response",
    )
    reference_url: Optional[str] = Field(
        default=None,
        description="Legitimate counterpart URL for visual similarity verification",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        # Basic injection prevention
        if any(c in v for c in ["\x00", "\r", "\n"]):
            raise ValueError("URL contains invalid characters")
        # Must look like a URL
        if not re.match(r"^https?://|^[a-zA-Z0-9]", v):
            raise ValueError("URL must begin with http(s):// or a hostname")
        return v


class BatchPredictRequest(BaseModel):
    """Batch URL prediction request."""

    urls: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of URLs (max 100 per batch)",
    )
    include_explanation: bool = Field(default=False)

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, v: list[str]) -> list[str]:
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 URLs")
        return [u.strip() for u in v if u.strip()]


class AuthRequest(BaseModel):
    """API key / JWT login request."""
    api_key: str = Field(..., min_length=8, description="Your API key")


class PreviewRequest(BaseModel):
    """Secure website preview request."""
    url: str = Field(
        ...,
        min_length=4,
        max_length=2048,
        description="URL to preview securely",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if any(c in v for c in ["\x00", "\r", "\n"]):
            raise ValueError("URL contains invalid characters")
        if not re.match(r"^https?://|^[a-zA-Z0-9]", v):
            raise ValueError("URL must begin with http(s):// or a hostname")
        if not v.startswith("http"):
            v = "http://" + v
        return v


# ── Response models ───────────────────────────────────────────────────────────

class FeatureExplanationOut(BaseModel):
    """Single feature contribution in a prediction explanation."""
    feature: str
    value: float
    shap_value: float
    impact: Literal["increases_risk", "decreases_risk", "neutral"]
    human_label: str


class PredictResponse(BaseModel):
    """Phishing detection result for a single URL."""

    model_config = {"protected_namespaces": ()}

    url: str
    phishing_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Phishing probability score [0-1]",
    )
    risk_level: Literal["Low", "Medium", "High"] = Field(
        ...,
        description="Risk classification based on probability",
    )
    explanation: Optional[List[FeatureExplanationOut]] = Field(
        default=None,
        description="Top SHAP feature contributions",
    )
    text_summary: Optional[str] = Field(
        default=None,
        description="Human-readable explanation",
    )
    model_scores: Optional[dict[str, float]] = Field(
        default=None,
        description="Individual base model probabilities",
    )
    processing_time_ms: float = Field(
        ..., description="End-to-end processing time in milliseconds"
    )
    cached: bool = Field(default=False, description="Result served from cache")


class BatchPredictResponse(BaseModel):
    """Batch prediction results."""
    results: List[PredictResponse]
    total_urls: int
    processing_time_ms: float


class PreviewResponse(BaseModel):
    """Response for secure preview."""
    image_b64: str = Field(..., description="Base64 encoded PNG image data of the secure preview")
    processing_time_ms: float = Field(..., description="Time taken to render preview")


class HealthResponse(BaseModel):
    """Service health check response."""
    status: Literal["ok", "degraded", "down"]
    model_loaded: bool
    redis_connected: bool
    version: str = "1.0.0"
    uptime_seconds: float


class TokenResponse(BaseModel):
    """JWT authentication response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int   # seconds


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    status_code: int
