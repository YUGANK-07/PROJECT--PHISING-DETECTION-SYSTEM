"""
models/random_forest.py
────────────────────────
Random Forest classifier wrapper with:
  - Sklearn-compatible API
  - Class-weight balancing
  - Feature importance extraction
  - Save / load via joblib
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
    "oob_score": True,
}


class PhishingRandomForest:
    """Random Forest phishing classifier.

    Wraps sklearn's RandomForestClassifier with convenience methods
    for training, evaluation, saving, and feature importance reporting.

    Parameters
    ----------
    params:
        Hyperparameter overrides.  Merged with ``_DEFAULT_PARAMS``.
    """

    MODEL_FILE = "random_forest.joblib"

    def __init__(self, params: Optional[dict[str, Any]] = None):
        merged = {**_DEFAULT_PARAMS, **(params or {})}
        self.model = RandomForestClassifier(**merged)
        self.is_fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PhishingRandomForest":
        """Fit the model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)  — 0=legit, 1=phishing

        Returns
        -------
        self
        """
        logger.info(f"Training Random Forest on {X.shape[0]:,} samples, {X.shape[1]} features")
        self.model.fit(X, y)
        self.is_fitted = True
        if hasattr(self.model, "oob_score_"):
            logger.info(f"OOB accuracy: {self.model.oob_score_:.4f}")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return phishing probability scores.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Phishing probability for each sample (column 1 of proba matrix).
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importances(
        self,
        feature_names: Optional[list[str]] = None,
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """Return top-n feature importances.

        Parameters
        ----------
        feature_names:
            Names matching model input columns.
        top_n:
            Number of top features to return.

        Returns
        -------
        list of (name, importance) sorted descending.
        """
        importances = self.model.feature_importances_
        names = feature_names or [f"f{i}" for i in range(len(importances))]
        pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        return pairs[:top_n]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Serialise model to disk with joblib."""
        if path is None:
            path = settings.MODEL_ARTIFACT_DIR / self.MODEL_FILE
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Random Forest saved → {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PhishingRandomForest":
        """Load a previously saved model."""
        if path is None:
            path = settings.MODEL_ARTIFACT_DIR / cls.MODEL_FILE
        obj = joblib.load(path)
        logger.info(f"Random Forest loaded from {path}")
        return obj
