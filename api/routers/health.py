"""
api/routers/health.py
──────────────────────
GET /health  — Liveness + readiness probe
GET /metrics — Prometheus-style counters (text format)
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from api.cache import redis_ping
from api.schemas import HealthResponse
from api.security import optional_auth
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])

_start_time = time.time()

# Simple in-memory counters (replace with Prometheus client in production)
_counters: dict[str, int] = {
    "predictions_total": 0,
    "predictions_cached": 0,
    "predictions_high_risk": 0,
    "batch_requests_total": 0,
    "errors_total": 0,
}


def increment(key: str, n: int = 1) -> None:
    """Increment an in-memory metric counter."""
    _counters[key] = _counters.get(key, 0) + n


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["Health"],
)
async def health(user: Optional[str] = Depends(optional_auth)) -> HealthResponse:
    """Returns service status, model availability, and Redis connectivity."""
    from api.routers.predict import _model
    redis_ok = await redis_ping()

    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        redis_connected=redis_ok,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus-style metrics",
    tags=["Health"],
)
async def metrics() -> str:
    """Returns metrics in Prometheus text format."""
    lines = [
        "# HELP phishguard_predictions_total Total prediction requests",
        "# TYPE phishguard_predictions_total counter",
        f"phishguard_predictions_total {_counters['predictions_total']}",
        "",
        "# HELP phishguard_predictions_cached Cache hit count",
        "# TYPE phishguard_predictions_cached counter",
        f"phishguard_predictions_cached {_counters['predictions_cached']}",
        "",
        "# HELP phishguard_high_risk_total High-risk URL detections",
        "# TYPE phishguard_high_risk_total counter",
        f"phishguard_high_risk_total {_counters['predictions_high_risk']}",
        "",
        "# HELP phishguard_uptime_seconds Service uptime",
        "# TYPE phishguard_uptime_seconds gauge",
        f"phishguard_uptime_seconds {round(time.time() - _start_time, 1)}",
    ]
    return "\n".join(lines)
