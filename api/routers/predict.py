"""
api/routers/predict.py
───────────────────────
POST /predict   — Single URL phishing analysis
POST /batch     — Batch URL analysis (up to 100 URLs)
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from api.cache import cache_get, cache_set
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    FeatureExplanationOut,
    PredictRequest,
    PredictResponse,
)
from api.security import require_auth
from utils.helpers import extract_domain
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])

SAFE_DOMAINS = frozenset([
    "microsoft.com", "paypal.com", "apple.com", "google.com", 
    "amazon.com", "netflix.com", "facebook.com", "instagram.com", 
    "twitter.com", "ebay.com", "chase.com", "wellsfargo.com", 
    "bankofamerica.com", "citibank.com", "hsbc.com", "linkedin.com", 
    "dropbox.com", "icloud.com", "outlook.com", "yahoo.com", 
    "dhl.com", "fedex.com", "usps.com", "ups.com", 
    "steamcommunity.com", "roblox.com", "coinbase.com", 
    "binance.com", "kraken.com"
])

# ── Model singleton (lazy-loaded from main.py lifespan) ───────────────────────
_model = None
_pipeline = None
_explainer = None
_feature_names: list[str] = []


def set_model(model, pipeline, explainer, feature_names):
    """Called from app lifespan to inject model dependencies."""
    global _model, _pipeline, _explainer, _feature_names
    _model         = model
    _pipeline      = pipeline
    _explainer     = explainer
    _feature_names = feature_names


def _get_model():
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please wait for startup to complete.",
        )
    return _model


# ── Core prediction logic ─────────────────────────────────────────────────────

async def _predict_single(
    url: str,
    html: str = "",
    include_explanation: bool = True,
    reference_url: Optional[str] = None,
) -> dict:
    """Run the full prediction pipeline for one URL.

    Returns
    -------
    dict
        Data compatible with PredictResponse.
    """
    # ── Safe Domain Whitelist Check ───────────────────────────────────────────
    domain = extract_domain(url)
    if domain in SAFE_DOMAINS:
        return {
            "url": url,
            "phishing_probability": 0.0,
            "risk_level": "Low",
            "explanation": [],
            "text_summary": f"✅ Risk Level: Low (0.0% phishing probability)\nURL: {url}\n\nSafe domain matched.",
            "model_scores": None,
            "processing_time_ms": 0.0,
            "cached": False,
        }

    model    = _get_model()
    pipeline = _pipeline
    t0 = time.perf_counter()

    # ── Feature extraction ────────────────────────────────────────────────────
    result = await pipeline.extract_async(url, html=html)
    X = result.vector.reshape(1, -1)

    # ── Inference ─────────────────────────────────────────────────────────────
    base_proba = float(model.predict_proba(X)[0])
    proba = base_proba
    
    if reference_url and hasattr(model, "predict_with_visual_verification"):
        final_preds = model.predict_with_visual_verification(X, [url], [reference_url])
        is_phishing = bool(final_preds[0])
        
        # Override probability if visual similarity flagged it
        if is_phishing and base_proba < 0.5:
            proba = 0.99

    # Individual base model scores (if ensemble)
    model_scores = None
    if hasattr(model, "predict_proba_with_base"):
        _, base = model.predict_proba_with_base(X)
        model_scores = {k: round(float(v[0]), 4) for k, v in base.items()}

    # ── Risk level ────────────────────────────────────────────────────────────
    from models.ensemble import PhishingEnsemble
    risk_level = PhishingEnsemble.score_to_risk(proba)

    # ── Explanation ───────────────────────────────────────────────────────────
    explanation_out  = None
    text_summary_out = None

    if include_explanation and _explainer is not None:
        try:
            expl = _explainer.explain(
                X,
                feature_names=_feature_names,
                url=url,
                phishing_proba=proba,
                risk_level=risk_level,
                top_n=8,
            )
            explanation_out = [
                FeatureExplanationOut(
                    feature=f.feature,
                    value=f.value,
                    shap_value=f.shap_value,
                    impact=f.impact,
                    human_label=f.human_label,
                )
                for f in expl.top_features
            ]
            text_summary_out = expl.text_summary
        except Exception as exc:
            logger.warning(f"Explanation failed for {url}: {exc}")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "url": url,
        "phishing_probability": round(proba, 4),
        "risk_level": risk_level,
        "explanation": explanation_out,
        "text_summary": text_summary_out,
        "model_scores": model_scores,
        "processing_time_ms": round(elapsed_ms, 2),
        "cached": False,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=PredictResponse,
    summary="Analyse a single URL for phishing",
    response_description="Phishing probability, risk level, and SHAP explanation",
)
async def predict(
    req: PredictRequest,
    user: str = Depends(require_auth),
) -> PredictResponse:
    """Analyse a URL for phishing indicators.

    - **url**: The URL to check (required)
    - **html**: Pre-fetched HTML content (optional)
    - **include_explanation**: Include SHAP feature explanations

    Returns a probability score, risk level (Low/Medium/High), and
    a list of the most influential features.
    """
    # ── Cache lookup ──────────────────────────────────────────────────────────
    if not req.html:   # Don't cache if HTML was provided (dynamic content)
        cached = await cache_get("prediction", req.url)
        if cached:
            cached["cached"] = True
            return PredictResponse(**cached)

    # ── Predict ───────────────────────────────────────────────────────────────
    data = await _predict_single(
        url=req.url,
        html=req.html or "",
        include_explanation=req.include_explanation,
        reference_url=req.reference_url,
    )

    # ── Cache result ──────────────────────────────────────────────────────────
    if not req.html:
        cache_data = {k: v for k, v in data.items() if k != "explanation"}
        if data.get("explanation"):
            cache_data["explanation"] = [e.model_dump() for e in data["explanation"]]
        await cache_set("prediction", req.url, cache_data)

    logger.info(
        f"[{user}] {req.url[:60]} -> {data['risk_level']} "
        f"({data['phishing_probability']:.3f}) {data['processing_time_ms']:.0f}ms"
    )

    return PredictResponse(**data)


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Analyse multiple URLs in one request",
)
async def batch_predict(
    req: BatchPredictRequest,
    user: str = Depends(require_auth),
) -> BatchPredictResponse:
    """Analyse up to 100 URLs in a single request.

    Results are returned in the same order as the input URLs.
    Explanations are disabled by default for batch requests
    (set ``include_explanation=true`` to enable — slower).
    """
    t0 = time.perf_counter()
    results = []

    for url in req.urls:
        try:
            # Check cache first
            cached = await cache_get("prediction", url)
            if cached:
                cached["cached"] = True
                results.append(PredictResponse(**cached))
                continue

            data = await _predict_single(
                url=url,
                include_explanation=req.include_explanation,
            )
            results.append(PredictResponse(**data))

            # Cache
            cache_data = {k: v for k, v in data.items() if k != "explanation"}
            if data.get("explanation"):
                cache_data["explanation"] = [e.model_dump() for e in data["explanation"]]
            await cache_set("prediction", url, cache_data)

        except Exception as exc:
            logger.error(f"Batch prediction failed for {url}: {exc}")
            results.append(PredictResponse(
                url=url,
                phishing_probability=0.5,
                risk_level="Medium",
                processing_time_ms=0,
                cached=False,
            ))

    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[{user}] Batch: {len(req.urls)} URLs in {total_ms:.0f}ms")

    return BatchPredictResponse(
        results=results,
        total_urls=len(results),
        processing_time_ms=round(total_ms, 2),
    )
