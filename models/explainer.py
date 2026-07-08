"""
models/explainer.py
────────────────────
SHAP-based explainability for all three model types.

Provides:
  1. Per-prediction feature explanations (force plots)
  2. Global feature importance (summary plots)
  3. Human-readable explanation text for the API response

Usage
-----
    from models.explainer import PhishingExplainer

    explainer = PhishingExplainer(model=rf_model, model_type="rf")
    explanation = explainer.explain(X_sample, feature_names=names)
    print(explanation.text_summary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

_SHAP_AVAILABLE = True
try:
    import shap
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("shap not installed. Explainability disabled. pip install shap")


# ── Explanation result ────────────────────────────────────────────────────────

@dataclass
class FeatureExplanation:
    """Single feature's contribution to a prediction."""
    feature: str
    value: float
    shap_value: float
    impact: str        # "increases_risk" | "decreases_risk" | "neutral"
    human_label: str   # Human-readable description


@dataclass
class PredictionExplanation:
    """Full explanation for one prediction."""
    url: str
    phishing_probability: float
    risk_level: str
    top_features: list[FeatureExplanation]
    text_summary: str
    base_value: float = 0.5   # SHAP expected value


# ── Human-readable feature descriptions ──────────────────────────────────────

_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "url_length":             "URL is unusually long",
    "url_entropy":            "URL contains random-looking characters (high entropy)",
    "hostname_entropy":       "Hostname contains random characters",
    "num_dots":               "URL has many dots (deep subdomain nesting)",
    "num_hyphens":            "URL has many hyphens (common in phishing domains)",
    "num_at":                 "URL contains @ symbol (redirect trick)",
    "has_at_symbol":          "URL uses @ redirect trick",
    "has_ip":                 "URL uses raw IP address instead of domain",
    "subdomain_depth":        "Unusually deep subdomain nesting",
    "suspicious_kw_count":    "URL contains suspicious phishing keywords",
    "tld_risk_score":         "High-risk top-level domain (e.g. .xyz, .tk)",
    "tld_is_free":            "Uses a free TLD commonly abused for phishing",
    "brand_in_domain":        "Legitimate brand name found in suspicious domain",
    "brand_in_subdomain":     "Brand name used in subdomain (impersonation)",
    "is_shortened_url":       "URL is from a known shortener service",
    "url_entropy_high":       "Very high URL entropy suggests obfuscation",
    "has_double_slash":       "Double slash in path (redirect indicator)",
    "digit_ratio":            "Unusually high proportion of digits in URL",
    "domain_age_days":        "Domain was registered very recently",
    "is_recently_registered": "Domain registered less than 30 days ago",
    "whois_available":        "WHOIS information is hidden/unavailable",
    "has_a_record":           "Domain has no DNS A record",
    "has_ssl":                "Page served without HTTPS",
    "ssl_valid":              "SSL certificate is invalid or expired",
    "ssl_self_signed":        "SSL certificate is self-signed",
    "ssl_issuer_trusted":     "SSL issued by untrusted authority",
    "ssl_days_to_expiry":     "SSL certificate expiring very soon",
    "num_forms":              "Page contains multiple forms",
    "num_password_fields":    "Page contains password input fields",
    "form_action_external":   "Form submits data to external domain",
    "has_login_form":         "Page has a login/credential form",
    "js_obfuscation_score":   "JavaScript appears obfuscated",
    "has_eval":               "JavaScript uses eval() (code injection risk)",
    "has_window_location":    "JavaScript redirects via window.location",
    "num_hidden_iframes":     "Page contains hidden iframes",
    "external_link_ratio":    "Most links point to external domains",
    "suspicious_kw_density":  "High density of phishing-related keywords",
    "brand_impersonation_score": "Content impersonates a legitimate brand",
    "phishing_phrase_count":  "Contains known phishing phrases",
    "urgency_word_count":     "Contains urgency language (act now, expire, etc.)",
    "num_meta_refresh":       "Page auto-redirects via meta refresh",
    "has_punycode":           "Uses Punycode (international homograph attack)",
}


class PhishingExplainer:
    """SHAP-based model explainer.

    Parameters
    ----------
    model:
        A fitted model with ``predict_proba(X)`` returning phishing probabilities.
    model_type:
        One of ``"rf"``, ``"xgb"``, ``"nn"``, ``"ensemble"``.
        Determines which SHAP explainer backend is used.
    background_data:
        Background dataset for KernelExplainer (required for NN/ensemble).
        Use a representative sample of ~100-500 rows.
    """

    def __init__(
        self,
        model,
        model_type: str = "xgb",
        background_data: Optional[np.ndarray] = None,
    ):
        if not _SHAP_AVAILABLE:
            raise ImportError("shap package required. pip install shap")

        self.model      = model
        self.model_type = model_type
        self._explainer = None
        self._background = background_data

    def _get_explainer(self, X_sample: np.ndarray):
        """Lazily instantiate the SHAP explainer on first call."""
        if self._explainer is not None:
            return self._explainer

        if self.model_type in ("rf",):
            self._explainer = shap.TreeExplainer(self.model.model)
        elif self.model_type in ("xgb",):
            self._explainer = shap.TreeExplainer(self.model.model)
        elif self.model_type in ("nn", "ensemble"):
            # KernelExplainer for black-box models
            if self._background is None:
                raise ValueError(
                    "background_data required for NN/ensemble SHAP explanations"
                )
            predict_fn = lambda X: self.model.predict_proba(X).reshape(-1, 1)
            self._explainer = shap.KernelExplainer(
                predict_fn,
                shap.kmeans(self._background, 50),
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return self._explainer

    # ── Per-sample explanation ────────────────────────────────────────────────

    def explain(
        self,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
        url: str = "",
        phishing_proba: Optional[float] = None,
        risk_level: str = "Unknown",
        top_n: int = 8,
    ) -> PredictionExplanation:
        """Generate a SHAP explanation for a single sample.

        Parameters
        ----------
        X:
            Feature vector, shape (1, n_features) or (n_features,).
        feature_names:
            Ordered feature names.
        url:
            URL being analysed (for the report).
        phishing_proba:
            Pre-computed probability (skip recompute if available).
        risk_level:
            ``"Low"``, ``"Medium"``, or ``"High"``.
        top_n:
            Number of top features to include in explanation.

        Returns
        -------
        PredictionExplanation
        """
        X = np.atleast_2d(X).astype(np.float32)
        names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        if phishing_proba is None:
            phishing_proba = float(self.model.predict_proba(X)[0])

        explainer = self._get_explainer(X)

        try:
            shap_values = explainer.shap_values(X)
            # TreeExplainer returns list [legit_vals, phish_vals] for binary
            if isinstance(shap_values, list) and len(shap_values) == 2:
                sv = shap_values[1][0]
            else:
                sv = np.array(shap_values).squeeze()

            base_value = float(
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )
        except Exception as exc:
            logger.warning(f"SHAP computation failed: {exc}. Using feature values.")
            sv = X[0]
            base_value = 0.5

        # Build feature explanations
        pairs = list(zip(names, X[0].tolist(), sv.tolist()))
        pairs_sorted = sorted(pairs, key=lambda p: abs(p[2]), reverse=True)[:top_n]

        feature_explanations = []
        for fname, fval, sval in pairs_sorted:
            if abs(sval) < 1e-6:
                continue
            impact = "increases_risk" if sval > 0 else "decreases_risk"
            human  = _FEATURE_DESCRIPTIONS.get(fname, fname.replace("_", " "))
            feature_explanations.append(FeatureExplanation(
                feature=fname,
                value=round(fval, 4),
                shap_value=round(sval, 4),
                impact=impact,
                human_label=human,
            ))

        text_summary = _build_text_summary(
            url, phishing_proba, risk_level, feature_explanations
        )

        return PredictionExplanation(
            url=url,
            phishing_probability=round(phishing_proba, 4),
            risk_level=risk_level,
            top_features=feature_explanations,
            text_summary=text_summary,
            base_value=round(base_value, 4),
        )

    # ── Global summary ────────────────────────────────────────────────────────

    def global_summary(
        self,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        """Generate a SHAP summary plot (beeswarm) for the dataset."""
        X = X.astype(np.float32)
        explainer = self._get_explainer(X)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        import matplotlib
        matplotlib.use("Agg")
        shap.summary_plot(
            shap_values, X,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        if save_path:
            import matplotlib.pyplot as plt
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP summary plot saved → {save_path}")
            plt.close()


# ── Text builder ──────────────────────────────────────────────────────────────

def _build_text_summary(
    url: str,
    probability: float,
    risk_level: str,
    features: list[FeatureExplanation],
) -> str:
    """Build a human-readable explanation paragraph."""
    pct = round(probability * 100, 1)
    risk_emojis = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}
    emoji = risk_emojis.get(risk_level, "")

    risk_factors = [f.human_label for f in features if f.impact == "increases_risk"]
    safe_factors = [f.human_label for f in features if f.impact == "decreases_risk"]

    lines = [
        f"{emoji} Risk Level: {risk_level} ({pct}% phishing probability)",
        f"URL: {url[:80]}{'...' if len(url) > 80 else ''}",
        "",
    ]

    if risk_factors:
        lines.append("Suspicious signals detected:")
        for r in risk_factors[:5]:
            lines.append(f"  • {r}")

    if safe_factors:
        lines.append("Legitimacy signals:")
        for s in safe_factors[:3]:
            lines.append(f"  • {s}")

    return "\n".join(lines)
