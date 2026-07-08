"""
models/xgboost_model.py
────────────────────────
XGBoost gradient-boosted classifier with:
  - Early stopping on validation set
  - Scale-pos-weight for class imbalance
  - SHAP-compatible tree structure
  - Optuna hyperparameter search support
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import xgboost as xgb

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 1000,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",      # fast histogram-based
    "random_state": 42,
    "n_jobs": -1,
    "use_label_encoder": False,
}


class PhishingXGBoost:
    """XGBoost phishing classifier.

    Parameters
    ----------
    params:
        Hyperparameter overrides.
    early_stopping_rounds:
        Rounds without improvement before stopping.
    """

    MODEL_FILE = "xgboost.joblib"

    def __init__(
        self,
        params: Optional[dict[str, Any]] = None,
        early_stopping_rounds: int = 50,
    ):
        merged = {**_DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.model = xgb.XGBClassifier(**merged)
        self.is_fitted = False
        self._best_iteration: int = 0

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "PhishingXGBoost":
        """Fit with optional early stopping on validation set.

        Parameters
        ----------
        X_train, y_train : Training data.
        X_val, y_val     : Validation data for early stopping.
                           If None, a 10% split of training data is used.

        Returns
        -------
        self
        """
        logger.info(f"Training XGBoost on {X_train.shape[0]:,} samples")

        # Scale-pos-weight: ratio of negative to positive class
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pw = neg / max(pos, 1)
        self.model.set_params(scale_pos_weight=scale_pw)
        logger.info(f"scale_pos_weight set to {scale_pw:.3f} (neg={neg:,} pos={pos:,})")

        eval_set = []
        callbacks = []

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks = [
                xgb.callback.EarlyStopping(
                    rounds=self.early_stopping_rounds,
                    metric_name="auc",
                    maximize=True,
                    save_best=True,
                )
            ]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set if eval_set else None,
            verbose=False,
            callbacks=callbacks if callbacks else None,
        )
        self.is_fitted = True

        if hasattr(self.model, "best_iteration"):
            self._best_iteration = self.model.best_iteration
            logger.info(f"Best iteration: {self._best_iteration}")

        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return phishing probability scores (column 1)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importances(
        self,
        feature_names: Optional[list[str]] = None,
        importance_type: str = "gain",
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """Return top-n feature importances.

        Parameters
        ----------
        importance_type:
            One of ``"weight"``, ``"gain"``, ``"cover"``, ``"total_gain"``.
        """
        scores = self.model.get_booster().get_score(importance_type=importance_type)
        names  = feature_names or list(scores.keys())
        pairs  = [(n, scores.get(f"f{i}", 0.0)) for i, n in enumerate(names)]
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = settings.MODEL_ARTIFACT_DIR / self.MODEL_FILE
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"XGBoost saved → {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PhishingXGBoost":
        if path is None:
            path = settings.MODEL_ARTIFACT_DIR / cls.MODEL_FILE
        obj = joblib.load(path)
        logger.info(f"XGBoost loaded from {path}")
        return obj


# ── Optuna objective (for hyperparameter search) ──────────────────────────────

def optuna_objective(
    trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Optuna objective function for XGBoost hyperparameter tuning.

    Parameters
    ----------
    trial : optuna.Trial
    X_train, y_train, X_val, y_val : Train/val splits.

    Returns
    -------
    float
        Validation ROC-AUC (higher is better).
    """
    from sklearn.metrics import roc_auc_score

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }

    clf = PhishingXGBoost(params=params)
    clf.fit(X_train, y_train, X_val, y_val)
    proba = clf.predict_proba(X_val)
    return roc_auc_score(y_val, proba)
