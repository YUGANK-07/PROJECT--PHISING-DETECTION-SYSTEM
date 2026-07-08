"""
api/main.py
────────────
FastAPI application entrypoint.

Startup sequence
----------------
1. Load environment config
2. Initialise Redis connection
3. Load trained model + feature pipeline + SHAP explainer
4. Register routers with rate limiting + CORS middleware
5. Serve on configured host:port

Run locally
-----------
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

JWT token for testing
---------------------
    POST /auth/token  {"api_key": "demo-key-phishguard-2024"}
"""

from __future__ import annotations

import json
import time
import sys
import asyncio
from contextlib import asynccontextmanager

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from pathlib import Path
from typing import Optional

import joblib
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.cache import get_redis
from api.routers import predict as predict_router_module
from api.routers.predict import router as predict_router, set_model
from api.routers.health import router as health_router
from api.routers.preview import router as preview_router
from api.schemas import AuthRequest, ErrorResponse, TokenResponse
from api.security import create_access_token, require_auth
from features.feature_pipeline import FeaturePipeline
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Rate limiting ─────────────────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    RATE_LIMIT_ENABLED = True
except ImportError:
    logger.warning("slowapi not installed. Rate limiting disabled.")
    limiter = None
    RATE_LIMIT_ENABLED = False


# ── Application lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("=" * 55)
    logger.info("  PhishGuard API — Starting up")
    logger.info("=" * 55)

    # ── 1. Redis ──────────────────────────────────────────────────────────────
    await get_redis()

    # ── 2. Load model artifacts ───────────────────────────────────────────────
    artifact_dir = settings.MODEL_ARTIFACT_DIR

    rf_path  = artifact_dir / "random_forest.joblib"
    xgb_path = artifact_dir / "xgboost.joblib"
    meta_path = artifact_dir / "meta_learner.joblib"
    fn_path   = artifact_dir / "feature_names.json"

    model     = None
    pipeline  = FeaturePipeline(mode="url_only")
    explainer = None
    feature_names: list[str] = pipeline.feature_names

    if fn_path.exists():
        with open(fn_path) as f:
            feature_names = json.load(f)
        logger.info(f"Feature names loaded: {len(feature_names)} features")

    # Load ensemble or best single model
    if meta_path.exists() and rf_path.exists() and xgb_path.exists():
        try:
            from models.ensemble import PhishingEnsemble
            nn_path = artifact_dir / "neural_network.pt"  # optional
            model = PhishingEnsemble.load_from_artifacts(rf_path, xgb_path, meta_path, nn_path)
            logger.info("Ensemble model loaded (RF + XGBoost + meta-learner)")

            # SHAP explainer (use XGBoost — fastest TreeExplainer)
            try:
                from models.explainer import PhishingExplainer
                explainer = PhishingExplainer(model=model.xgb, model_type="xgb")
                logger.info("SHAP explainer initialised")
            except Exception as exc:
                logger.warning(f"SHAP explainer unavailable: {exc}")

        except Exception as exc:
            logger.error(f"Model load failed: {exc}")
    elif xgb_path.exists():
        try:
            from models.xgboost_model import PhishingXGBoost
            xgb = PhishingXGBoost.load(xgb_path)
            model = xgb
            logger.info("XGBoost model loaded (standalone)")
        except Exception as exc:
            logger.error(f"XGBoost load failed: {exc}")
    else:
        logger.warning(
            "No trained models found in models/artifacts/. "
            "Run: python -m models.trainer"
        )

    # Inject into predict router
    set_model(model, pipeline, explainer, feature_names)

    logger.info("PhishGuard API ready.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("PhishGuard API shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="PhishGuard — Phishing Detection API",
        description=(
            "Production-grade phishing detection powered by an ensemble of "
            "Random Forest, XGBoost, and Neural Network models with "
            "SHAP-based explanations."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # open for local dev; restrict in prod
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # Serve frontend static files
    from fastapi.staticfiles import StaticFiles
    from pathlib import Path as _Path
    _fe = _Path("frontend")
    if _fe.exists():
        app.mount("/ui", StaticFiles(directory=str(_fe), html=True), name="frontend")

    # ── Rate limiting ─────────────────────────────────────────────────────────
    if RATE_LIMIT_ENABLED and limiter:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Global error handler ──────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error on {request.url}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc), "status_code": 500},
        )

    # ── Auth endpoint ─────────────────────────────────────────────────────────
    @app.post(
        "/auth/token",
        response_model=TokenResponse,
        tags=["Auth"],
        summary="Get a JWT token using your API key",
    )
    async def get_token(req: AuthRequest) -> TokenResponse:
        """Exchange an API key for a short-lived JWT token.

        Use the returned ``access_token`` as:
        ```
        Authorization: Bearer <access_token>
        ```
        """
        from api.security import _API_KEYS
        if req.api_key not in _API_KEYS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        username = _API_KEYS[req.api_key]
        token = create_access_token(subject=username)
        return TokenResponse(
            access_token=token,
            expires_in=settings.JWT_EXPIRE_MINUTES * 60,
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(preview_router)

    # ── Root redirect ─────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": "PhishGuard", "docs": "/docs", "health": "/health"}

    return app


app = create_app()
