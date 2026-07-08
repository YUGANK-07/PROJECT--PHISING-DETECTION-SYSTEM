"""
models/trainer.py
──────────────────
End-to-end training orchestrator.

Loads processed data → extracts features → trains RF, XGBoost, MLP,
and the stacking Ensemble → evaluates on held-out test set → saves
all artefacts and reports.

Usage
-----
    # Train all models (fast, no hyperparameter search):
    python -m models.trainer

    # With Optuna hyperparameter tuning for XGBoost:
    python -m models.trainer --tune-xgb --n-trials 30

    # Train RF + XGB only (skip NN, faster):
    python -m models.trainer --no-nn
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from features.feature_pipeline import FeaturePipeline
from models.random_forest import PhishingRandomForest
from models.xgboost_model import PhishingXGBoost, optuna_objective
from models.ensemble import PhishingEnsemble
from models.evaluator import evaluate, print_report, plot_roc_curve, plot_confusion_matrix
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_ARTIFACT_DIR = settings.MODEL_ARTIFACT_DIR
_PROCESSED_DIR = settings.processed_data_dir
_REPORT_DIR = Path("reports")


# ── Feature matrix builder ────────────────────────────────────────────────────

def build_feature_matrices(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    mode: str = "url_only",
    sample_size: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           list[str]]:
    """Build feature matrices from processed CSVs.

    Parameters
    ----------
    train_path, val_path, test_path:
        Paths to processed CSV files.
    mode:
        Feature extraction mode (``"url_only"`` for fast training).
    sample_size:
        If set, randomly subsample training data for faster iteration.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    logger.info(f"Loading datasets (mode={mode}) ...")

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    if sample_size and len(train_df) > sample_size:
        train_df = train_df.sample(sample_size, random_state=42)
        logger.info(f"Subsampled training to {len(train_df):,} rows")

    logger.info(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    pipeline = FeaturePipeline(mode=mode)

    def _extract(df: pd.DataFrame, split_name: str) -> np.ndarray:
        logger.info(f"Extracting features for {split_name} ...")
        t0 = time.perf_counter()

        from tqdm import tqdm
        results = []
        for url in tqdm(df["url"].tolist(), desc=split_name):
            try:
                r = pipeline.extract(url)
                results.append(r.vector)
            except Exception:
                results.append(np.zeros(pipeline.n_features, dtype=np.float32))

        elapsed = time.perf_counter() - t0
        logger.info(f"  {split_name}: {len(results):,} rows in {elapsed:.1f}s "
                    f"({elapsed/len(results)*1000:.1f}ms/URL)")
        return np.vstack(results)

    X_train = _extract(train_df, "train")
    X_val   = _extract(val_df,   "val")
    X_test  = _extract(test_df,  "test")

    y_train = train_df["label"].values.astype(int)
    y_val   = val_df["label"].values.astype(int)
    y_test  = test_df["label"].values.astype(int)

    # Cache feature matrices for subsequent runs
    cache_dir = _ARTIFACT_DIR / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "X_train.npy", X_train)
    np.save(cache_dir / "X_val.npy",   X_val)
    np.save(cache_dir / "X_test.npy",  X_test)
    np.save(cache_dir / "y_train.npy", y_train)
    np.save(cache_dir / "y_val.npy",   y_val)
    np.save(cache_dir / "y_test.npy",  y_test)
    logger.info(f"Feature matrices cached → {cache_dir}")

    return X_train, X_val, X_test, y_train, y_val, y_test, pipeline.feature_names


def load_cached_matrices():
    """Load pre-computed feature matrices if available."""
    cache_dir = _ARTIFACT_DIR / "feature_cache"
    files = ["X_train.npy", "X_val.npy", "X_test.npy",
             "y_train.npy", "y_val.npy", "y_test.npy"]
    if all((cache_dir / f).exists() for f in files):
        logger.info("Loading cached feature matrices ...")
        X_train = np.load(cache_dir / "X_train.npy")
        X_val   = np.load(cache_dir / "X_val.npy")
        X_test  = np.load(cache_dir / "X_test.npy")
        y_train = np.load(cache_dir / "y_train.npy")
        y_val   = np.load(cache_dir / "y_val.npy")
        y_test  = np.load(cache_dir / "y_test.npy")
        logger.info(f"Loaded — X_train: {X_train.shape}  X_test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    return None


# ── XGBoost Optuna tuning ─────────────────────────────────────────────────────

def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
) -> dict:
    """Run Optuna hyperparameter search for XGBoost."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("optuna not installed. Using default XGBoost params.")
        return {}

    logger.info(f"Starting Optuna XGBoost tuning ({n_trials} trials) ...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best = study.best_params
    logger.info(f"Best XGBoost params: {best}  AUC={study.best_value:.4f}")
    return best


# ── Main training pipeline ────────────────────────────────────────────────────

def train(
    use_cache: bool = True,
    train_nn: bool = False,     # Off by default (requires PyTorch)
    tune_xgb: bool = False,
    n_trials: int = 30,
    sample_size: Optional[int] = 50_000,   # subsample for speed
    mode: str = "url_only",
) -> dict:
    """Run the full training pipeline.

    Parameters
    ----------
    use_cache:
        Load cached feature matrices if available.
    train_nn:
        Include the neural network in the ensemble.
    tune_xgb:
        Run Optuna hyperparameter search for XGBoost.
    n_trials:
        Number of Optuna trials.
    sample_size:
        Subsample training set to this size (None = use all).
    mode:
        Feature extraction mode.

    Returns
    -------
    dict
        Final evaluation results for all models.
    """
    total_start = time.perf_counter()
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load feature matrices ─────────────────────────────────────────────────
    cached = load_cached_matrices() if use_cache else None
    if cached:
        X_train, X_val, X_test, y_train, y_val, y_test = cached
        pipeline = FeaturePipeline(mode=mode)
        feature_names = pipeline.feature_names
    else:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = (
            build_feature_matrices(
                _PROCESSED_DIR / "train.csv",
                _PROCESSED_DIR / "val.csv",
                _PROCESSED_DIR / "test.csv",
                mode=mode,
                sample_size=sample_size,
            )
        )

    logger.info(f"Feature matrix: {X_train.shape[1]} features, "
                f"train={X_train.shape[0]:,} val={X_val.shape[0]:,} test={X_test.shape[0]:,}")

    # Save feature names
    fn_path = _ARTIFACT_DIR / "feature_names.json"
    with open(fn_path, "w") as f:
        json.dump(feature_names, f)

    all_results = {}
    y_probas    = {}

    # ── Random Forest ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("TRAINING: Random Forest")
    logger.info("=" * 55)
    t0 = time.perf_counter()
    rf = PhishingRandomForest()
    rf.fit(X_train, y_train)
    rf.save()
    rf_proba = rf.predict_proba(X_test)
    rf_res = evaluate(y_test, rf_proba, model_name="Random Forest")
    print_report(rf_res)
    rf_res.to_json(_REPORT_DIR / "rf_evaluation.json")
    all_results["random_forest"] = rf_res
    y_probas["random_forest"]    = rf_proba
    logger.info(f"RF training time: {time.perf_counter() - t0:.1f}s")

    # Feature importance chart
    fi = rf.feature_importances(feature_names=feature_names, top_n=20)
    from models.evaluator import plot_feature_importance
    plot_feature_importance(fi, "Random Forest", _REPORT_DIR / "rf_feature_importance.png")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("TRAINING: XGBoost")
    logger.info("=" * 55)
    xgb_params = {}
    if tune_xgb:
        xgb_params = tune_xgboost(X_train, y_train, X_val, y_val, n_trials)
    t0 = time.perf_counter()
    xgb = PhishingXGBoost(params=xgb_params or None)
    xgb.fit(X_train, y_train, X_val, y_val)
    xgb.save()
    xgb_proba = xgb.predict_proba(X_test)
    xgb_res = evaluate(y_test, xgb_proba, model_name="XGBoost")
    print_report(xgb_res)
    xgb_res.to_json(_REPORT_DIR / "xgb_evaluation.json")
    all_results["xgboost"] = xgb_res
    y_probas["xgboost"]    = xgb_proba
    logger.info(f"XGB training time: {time.perf_counter() - t0:.1f}s")
    plot_confusion_matrix(xgb_res, _REPORT_DIR / "xgb_confusion_matrix.png")

    # ── Neural Network (optional) ─────────────────────────────────────────────
    nn = None
    if train_nn:
        try:
            from models.neural_network import PhishingMLP
            logger.info("\n" + "=" * 55)
            logger.info("TRAINING: Neural Network (MLP)")
            logger.info("=" * 55)
            t0 = time.perf_counter()
            nn = PhishingMLP(input_dim=X_train.shape[1], epochs=50)
            nn.fit(X_train, y_train, X_val, y_val)
            nn.save()
            nn_proba = nn.predict_proba(X_test)
            nn_res = evaluate(y_test, nn_proba, model_name="Neural Network")
            print_report(nn_res)
            nn_res.to_json(_REPORT_DIR / "nn_evaluation.json")
            all_results["neural_network"] = nn_res
            y_probas["neural_network"]    = nn_proba
            logger.info(f"NN training time: {time.perf_counter() - t0:.1f}s")
        except ImportError:
            logger.warning("PyTorch not installed. Skipping Neural Network.")
            train_nn = False

    # ── Ensemble ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("TRAINING: Stacking Ensemble")
    logger.info("=" * 55)

    # For fast ensemble training, use pre-computed base model predictions
    # instead of re-running full OOF (use val set as proxy)
    base_preds_val = np.column_stack([
        rf.predict_proba(X_val),
        xgb.predict_proba(X_val),
    ] + ([nn.predict_proba(X_val)] if train_nn and nn else []))

    base_preds_test = np.column_stack([
        rf_proba,
        xgb_proba,
    ] + ([nn_proba] if train_nn and nn else []))

    from sklearn.linear_model import LogisticRegression
    meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    meta.fit(base_preds_val, y_val)
    ens_proba = meta.predict_proba(base_preds_test)[:, 1]
    ens_res = evaluate(y_test, ens_proba, model_name="Ensemble")
    print_report(ens_res)
    ens_res.to_json(_REPORT_DIR / "ensemble_evaluation.json")
    all_results["ensemble"] = ens_res
    y_probas["ensemble"]    = ens_proba

    # Save meta-learner + ensemble wrapper
    joblib.dump(meta, _ARTIFACT_DIR / "meta_learner.joblib")
    joblib.dump({"use_nn": train_nn}, _ARTIFACT_DIR / "ensemble_config.json")
    plot_confusion_matrix(ens_res, _REPORT_DIR / "ensemble_confusion_matrix.png")

    # ── ROC comparison plot ───────────────────────────────────────────────────
    res_list  = [all_results[k] for k in all_results]
    proba_list = [y_probas[k] for k in all_results]
    plot_roc_curve(res_list, y_test, proba_list, _REPORT_DIR / "roc_curves.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Total time: {total_elapsed:.1f}s")
    logger.info(f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    logger.info("-" * 60)
    for name, res in all_results.items():
        logger.info(
            f"  {name:<18} {res.precision:>10.4f} {res.recall:>10.4f} "
            f"{res.f1:>10.4f} {res.roc_auc:>10.4f}"
        )
    logger.info("=" * 60)
    logger.info(f"Artifacts saved → {_ARTIFACT_DIR}")
    logger.info(f"Reports saved   → {_REPORT_DIR}")

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train phishing detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--no-cache",   action="store_true",  help="Recompute feature matrices")
    parser.add_argument("--no-nn",      action="store_true",  help="Skip Neural Network training")
    parser.add_argument("--tune-xgb",   action="store_true",  help="Run Optuna XGBoost tuning")
    parser.add_argument("--n-trials",   type=int, default=30, help="Optuna trials")
    parser.add_argument("--sample",     type=int, default=50_000, help="Training sample size (0=all)")
    parser.add_argument("--mode",       type=str, default="url_only",
                        choices=["url_only", "full", "with_html"],
                        help="Feature extraction mode")
    args = parser.parse_args()

    train(
        use_cache=not args.no_cache,
        train_nn=not args.no_nn,
        tune_xgb=args.tune_xgb,
        n_trials=args.n_trials,
        sample_size=args.sample if args.sample > 0 else None,
        mode=args.mode,
    )
