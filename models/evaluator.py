"""
models/evaluator.py
────────────────────
Model evaluation utilities: metrics, confusion matrix,
ROC-AUC curve, and error analysis reports.

Usage
-----
    from models.evaluator import evaluate, print_report

    results = evaluate(y_true, y_pred_proba)
    print_report(results)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)

from utils.logger import get_logger

logger = get_logger(__name__)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Results dataclass ─────────────────────────────────────────────────────────

@dataclass
class EvaluationResults:
    """Container for all evaluation metrics."""
    precision: float
    recall: float
    f1: float
    roc_auc: float
    avg_precision: float                      # PR-AUC
    threshold: float
    confusion: list[list[int]]
    tp: int
    fp: int
    tn: int
    fn: int
    fpr: float                                # False Positive Rate
    fnr: float                                # False Negative Rate (missed phishing)
    report: str = ""
    model_name: str = "model"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Evaluation results saved → {path}")


# ── Core evaluation ───────────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> float:
    """Find the decision threshold that maximises *metric* on the given data.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_proba:
        Predicted probabilities.
    metric:
        One of ``"f1"``, ``"recall"`` (prioritise catching all phishing),
        ``"precision"``.

    Returns
    -------
    float
        Optimal threshold in [0, 1].
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = -1.0
    best_thresh = 0.5

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, preds, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, preds, zero_division=0)
        else:
            score = f1_score(y_true, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = t

    return round(float(best_thresh), 2)


def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: Optional[float] = None,
    model_name: str = "model",
    optimise_for: str = "f1",
) -> EvaluationResults:
    """Compute comprehensive evaluation metrics.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0=legit, 1=phishing).
    y_proba:
        Predicted phishing probabilities in [0, 1].
    threshold:
        Decision threshold.  If None, auto-selects to maximise ``optimise_for``.
    model_name:
        Label for reports and plots.
    optimise_for:
        Metric to optimise threshold for: ``"f1"`` or ``"recall"``.

    Returns
    -------
    EvaluationResults
    """
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_proba, metric=optimise_for)
        logger.info(f"Optimal {optimise_for} threshold: {threshold:.2f}")

    y_pred = (y_proba >= threshold).astype(int)

    # Core metrics
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    auc    = roc_auc_score(y_true, y_proba)
    ap     = average_precision_score(y_true, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / max(fp + tn, 1)
    fnr_val = fn / max(fn + tp, 1)   # missed phishing rate

    report = classification_report(
        y_true, y_pred,
        target_names=["Legitimate", "Phishing"],
        digits=4,
    )

    return EvaluationResults(
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        roc_auc=round(auc, 4),
        avg_precision=round(ap, 4),
        threshold=threshold,
        confusion=cm.tolist(),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        fpr=round(fpr_val, 4),
        fnr=round(fnr_val, 4),
        report=report,
        model_name=model_name,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(results: EvaluationResults) -> None:
    """Pretty-print evaluation results to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  EVALUATION REPORT — {results.model_name.upper()}")
    print(sep)
    print(f"  Threshold   : {results.threshold:.2f}")
    print(f"  Precision   : {results.precision:.4f}")
    print(f"  Recall      : {results.recall:.4f}   (catch rate — HIGH PRIORITY)")
    print(f"  F1-Score    : {results.f1:.4f}")
    print(f"  ROC-AUC     : {results.roc_auc:.4f}")
    print(f"  PR-AUC      : {results.avg_precision:.4f}")
    print(f"  FPR         : {results.fpr:.4f}   (legit flagged as phishing)")
    print(f"  FNR         : {results.fnr:.4f}   (phishing missed — keep low!)")
    print()
    print("  Confusion Matrix:")
    print(f"    True Legit  → Legit  (TN): {results.tn:>8,}")
    print(f"    True Legit  → Phish  (FP): {results.fp:>8,}")
    print(f"    True Phish  → Legit  (FN): {results.fn:>8,}  <-- missed phishing!")
    print(f"    True Phish  → Phish  (TP): {results.tp:>8,}")
    print()
    print(results.report)
    print(sep)


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_roc_curve(
    results_list: list[EvaluationResults],
    y_true: np.ndarray,
    y_probas: list[np.ndarray],
    save_path: Optional[Path] = None,
) -> None:
    """Plot ROC curves for multiple models on one chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]

    for res, proba, color in zip(results_list, y_probas, colors):
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{res.model_name} (AUC={res.roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Phishing Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve saved → {save_path}")
    plt.close()


def plot_confusion_matrix(
    results: EvaluationResults,
    save_path: Optional[Path] = None,
) -> None:
    """Plot a styled confusion matrix heatmap."""
    cm = np.array(results.confusion)
    labels = ["Legitimate", "Phishing"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(
        f"Confusion Matrix — {results.model_name}\n"
        f"F1={results.f1:.4f}  AUC={results.roc_auc:.4f}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved → {save_path}")
    plt.close()


def plot_feature_importance(
    importances: list[tuple[str, float]],
    model_name: str = "Model",
    save_path: Optional[Path] = None,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of top-n feature importances."""
    importances = importances[:top_n]
    names  = [p[0] for p in importances][::-1]
    values = [p[1] for p in importances][::-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(names, values, color="#6366f1", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Feature importance chart saved → {save_path}")
    plt.close()
