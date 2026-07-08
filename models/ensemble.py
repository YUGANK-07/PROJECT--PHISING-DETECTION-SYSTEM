"""
models/ensemble.py
───────────────────
Stacking ensemble that combines Random Forest, XGBoost, and Neural Network
predictions via a Logistic Regression meta-learner.

Architecture
────────────
                  ┌─────────────────┐
     X ──────────►│  Random Forest  │──► P_rf
     X ──────────►│    XGBoost      │──► P_xgb    ──► [P_rf, P_xgb, P_nn]
     X ──────────►│  Neural Network │──► P_nn          │
                  └─────────────────┘                  ▼
                                               LogisticRegression
                                               (meta-learner)
                                                       │
                                                       ▼
                                               phishing_probability

The meta-learner is trained on out-of-fold (OOF) predictions to prevent
data leakage — each base model predicts on held-out folds.

Usage
-----
    ensemble = PhishingEnsemble()
    ensemble.fit(X_train, y_train, X_val, y_val)
    proba = ensemble.predict_proba(X_test)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from models.random_forest import PhishingRandomForest
from models.xgboost_model import PhishingXGBoost
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_NN_AVAILABLE = True
try:
    from models.neural_network import PhishingMLP
except ImportError:
    _NN_AVAILABLE = False
    logger.warning("PyTorch not available — ensemble will use RF + XGB only")


class PhishingEnsemble:
    """Stacking ensemble phishing classifier.

    Parameters
    ----------
    use_nn:
        Include the neural network base model.  Requires PyTorch.
    n_folds:
        Number of cross-validation folds for OOF meta-features.
    rf_params, xgb_params, nn_params:
        Hyperparameter dicts forwarded to each base model.
    """

    MODEL_FILE = "ensemble.joblib"

    def __init__(
        self,
        use_nn: bool = True,
        n_folds: int = 5,
        rf_params: Optional[dict] = None,
        xgb_params: Optional[dict] = None,
        nn_params: Optional[dict] = None,
    ):
        self.use_nn  = use_nn and _NN_AVAILABLE
        self.n_folds = n_folds
        self.is_fitted = False

        # Base models (trained on full train set after OOF)
        self.rf  = PhishingRandomForest(params=rf_params)
        self.xgb = PhishingXGBoost(params=xgb_params)
        self.nn  = PhishingMLP(**(nn_params or {})) if self.use_nn else None

        # Meta-learner
        self.meta = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )

        # Store base model names for SHAP / explanation
        self.base_names = ["random_forest", "xgboost"] + (["neural_net"] if self.use_nn else [])

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        input_dim: Optional[int] = None,
    ) -> "PhishingEnsemble":
        """Train the full stacking ensemble.

        Steps
        -----
        1. Generate out-of-fold predictions for each base model.
        2. Train meta-learner on OOF predictions.
        3. Retrain each base model on the full training set.

        Parameters
        ----------
        X_train, y_train : Full training data.
        X_val, y_val     : Validation data (forwarded to base models for early stopping).
        input_dim        : Override MLP input dimension if different from X_train.shape[1].

        Returns
        -------
        self
        """
        n = len(X_train)
        n_base = 2 + int(self.use_nn)

        logger.info(f"Training ensemble: {self.base_names}")
        logger.info(f"Generating OOF predictions with {self.n_folds}-fold CV ...")

        oof_preds = np.zeros((n, n_base), dtype=np.float32)
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            Xf_tr, yf_tr = X_train[trn_idx], y_train[trn_idx]
            Xf_vl, yf_vl = X_train[val_idx],  y_train[val_idx]

            logger.info(f"  Fold {fold + 1}/{self.n_folds}")

            # Random Forest
            rf_fold = PhishingRandomForest()
            rf_fold.fit(Xf_tr, yf_tr)
            oof_preds[val_idx, 0] = rf_fold.predict_proba(Xf_vl)

            # XGBoost
            xgb_fold = PhishingXGBoost()
            xgb_fold.fit(Xf_tr, yf_tr, Xf_vl, yf_vl)
            oof_preds[val_idx, 1] = xgb_fold.predict_proba(Xf_vl)

            # Neural Network
            if self.use_nn:
                dim = input_dim or X_train.shape[1]
                nn_fold = PhishingMLP(input_dim=dim, epochs=20)
                nn_fold.fit(Xf_tr, yf_tr, Xf_vl, yf_vl)
                oof_preds[val_idx, 2] = nn_fold.predict_proba(Xf_vl)

        # Train meta-learner on OOF predictions
        logger.info("Training meta-learner on OOF predictions ...")
        self.meta.fit(oof_preds, y_train)

        # Retrain base models on full training set
        logger.info("Retraining base models on full training data ...")
        self.rf.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train, X_val, y_val)
        if self.use_nn:
            dim = input_dim or X_train.shape[1]
            self.nn = PhishingMLP(input_dim=dim, epochs=50)
            self.nn.fit(X_train, y_train, X_val, y_val)

        self.is_fitted = True
        logger.info("Ensemble training complete.")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def _base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get base model probability predictions.

        Returns
        -------
        np.ndarray, shape (n_samples, n_base_models)
        """
        preds = [
            self.rf.predict_proba(X),
            self.xgb.predict_proba(X),
        ]
        if self.use_nn and self.nn is not None:
            preds.append(self.nn.predict_proba(X))
        return np.column_stack(preds).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return final ensemble phishing probability scores.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        base_preds = self._base_predictions(X)
        return self.meta.predict_proba(base_preds)[:, 1]

    def predict_proba_with_base(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return ensemble score AND individual base model scores.

        Useful for the explanation layer.

        Returns
        -------
        (ensemble_proba, {"random_forest": ..., "xgboost": ..., "neural_net": ...})
        """
        base = self._base_predictions(X)
        ensemble = self.meta.predict_proba(base)[:, 1]
        individual = {name: base[:, i] for i, name in enumerate(self.base_names)}
        return ensemble, individual

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def predict_with_visual_verification(
        self, 
        X: np.ndarray, 
        input_urls: list[str], 
        reference_urls: list[str],
        visual_threshold: float = 0.85
    ) -> list[int]:
        """
        Predict class using the ensemble model, enhanced by visual similarity.
        If the model predicts benign (0), but visual similarity to a reference domain is high,
        flag as phishing (1).
        
        Args:
            X: Feature array, shape (n_samples, n_features)
            input_urls: List of input URLs being tested
            reference_urls: List of reference URLs (legitimate counterparts) for comparison
            visual_threshold: Threshold for visual similarity to flag as phishing
            
        Returns:
            list of predicted classes (0 or 1)
        """
        from features.visual_similarity import detect_visual_similarity_sync
        
        base_preds = self.predict(X)
        final_preds = []
        
        for i, (pred, input_url, ref_url) in enumerate(zip(base_preds, input_urls, reference_urls)):
            if pred == 1:
                # Already flagged as phishing by ensemble model
                final_preds.append(1)
                continue
                
            if not ref_url:
                # No reference URL provided, stick with base prediction
                final_preds.append(int(pred))
                continue
                
            # Extract domain to check for mismatch
            from urllib.parse import urlparse
            input_domain = urlparse(input_url).netloc
            ref_domain = urlparse(ref_url).netloc
            
            if input_domain == ref_domain or not input_domain or not ref_domain:
                # Same domain or invalid, stick with base prediction
                final_preds.append(int(pred))
                continue
                
            # Model predicts benign, verify with visual similarity
            sim_score, is_phishing, conf = detect_visual_similarity_sync(input_url, ref_url, visual_threshold)
            
            if is_phishing:
                logger.warning(f"Visual similarity override! {input_url} visually matches {ref_url}. Flagging as phishing.")
                final_preds.append(1)
            else:
                final_preds.append(int(pred))
                
        return final_preds

    # ── Risk level ────────────────────────────────────────────────────────────

    @staticmethod
    def score_to_risk(score: float) -> str:
        """Map a probability score to a human-readable risk level.

        Thresholds
        ----------
        - Low    : score < 0.4
        - Medium : 0.4 ≤ score < 0.7
        - High   : score ≥ 0.7

        Parameters
        ----------
        score:
            Phishing probability in [0, 1].

        Returns
        -------
        str
            ``"Low"``, ``"Medium"``, or ``"High"``.
        """
        if score < 0.4:
            return "Low"
        elif score < 0.7:
            return "Medium"
        else:
            return "High"

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = settings.MODEL_ARTIFACT_DIR / self.MODEL_FILE
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Ensemble saved → {path}")
        return path

    @classmethod
    def load_from_artifacts(
        cls, 
        rf_path: Path, 
        xgb_path: Path, 
        meta_path: Path, 
        nn_path: Optional[Path] = None
    ) -> "PhishingEnsemble":
        """Load an ensemble from independently trained artifact files."""
        instance = cls(use_nn=False)
        
        from models.random_forest import PhishingRandomForest
        from models.xgboost_model import PhishingXGBoost
        
        instance.rf = PhishingRandomForest.load(rf_path)
        instance.xgb = PhishingXGBoost.load(xgb_path)
        instance.meta = joblib.load(meta_path)
        
        # Determine base names based on loaded models
        instance.base_names = ["random_forest", "xgboost"]
        
        if nn_path and nn_path.exists():
            try:
                from models.neural_network import PhishingMLP
                instance.nn = PhishingMLP.load(nn_path)
                instance.use_nn = True
                instance.base_names.append("neural_net")
            except ImportError:
                logger.warning("NN artifact found, but PyTorch not installed.")
                
        instance.is_fitted = True
        logger.info("Ensemble loaded from separated artifacts.")
        return instance

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PhishingEnsemble":
        if path is None:
            path = settings.MODEL_ARTIFACT_DIR / cls.MODEL_FILE
        obj = joblib.load(path)
        logger.info(f"Ensemble loaded from {path}")
        return obj
