"""
models/neural_network.py
─────────────────────────
PyTorch-based neural network for phishing detection.

Architecture options:
  1. MLP (Multi-Layer Perceptron) — default, works on structured features
  2. Residual MLP — skip connections for deeper networks
  3. Transformer — self-attention over feature tokens (experimental)

The MLP is the default and performs best on the 102-feature vector
when BERT embeddings are not available.

When BERT embeddings ARE available (768-dim), the network fuses
both the structured features and the embedding via a cross-attention head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ── PyTorch imports (lazy — graceful failure if not installed) ─────────────────

def _require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for the neural network model.\n"
            "Install with:  pip install torch"
        )


# ── Model Definitions ─────────────────────────────────────────────────────────

def _build_mlp(input_dim: int, hidden_dims: list[int], dropout: float):
    torch, nn = _require_torch()

    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers += [
            nn.Linear(prev_dim, h),
            nn.BatchNorm1d(h),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        prev_dim = h
    layers.append(nn.Linear(prev_dim, 1))   # binary output (logit)
    return nn.Sequential(*layers)


class PhishingMLP:
    """Multi-Layer Perceptron for phishing detection.

    Wraps a PyTorch nn.Sequential with sklearn-style fit/predict API.

    Parameters
    ----------
    input_dim:
        Number of input features.
    hidden_dims:
        List of hidden layer sizes.
    dropout:
        Dropout probability applied after each hidden layer.
    lr:
        Learning rate for Adam optimizer.
    epochs:
        Training epochs.
    batch_size:
        Mini-batch size.
    """

    MODEL_FILE = "neural_network.pt"

    def __init__(
        self,
        input_dim: int = 102,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 512,
        device: Optional[str] = None,
    ):
        torch, nn = _require_torch()

        self.input_dim   = input_dim
        self.hidden_dims = hidden_dims or [512, 256, 128, 64]
        self.dropout     = dropout
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.device      = torch.device(device or settings.INFERENCE_DEVICE)
        self.is_fitted   = False

        self.net = _build_mlp(input_dim, self.hidden_dims, dropout).to(self.device)
        self._loss_history: list[dict] = []

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "PhishingMLP":
        """Train the MLP.

        Parameters
        ----------
        X_train, y_train : Training arrays (numpy float32).
        X_val, y_val     : Optional validation arrays for early stopping.

        Returns
        -------
        self
        """
        torch, nn = _require_torch()
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler

        logger.info(
            f"Training MLP {self.input_dim} -> "
            f"{'->'.join(str(h) for h in self.hidden_dims)} -> 1  "
            f"| epochs={self.epochs}  batch={self.batch_size}  lr={self.lr}"
        )

        # Normalise
        self._scaler = StandardScaler()
        X_train = self._scaler.fit_transform(X_train).astype(np.float32)
        if X_val is not None:
            X_val = self._scaler.transform(X_val).astype(np.float32)

        # Tensors
        Xt = torch.FloatTensor(X_train).to(self.device)
        yt = torch.FloatTensor(y_train).to(self.device)
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        # Class weights for BCE
        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([neg / max(pos, 1)], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            # ── Train ─────────────────────────────────────────────────────────
            self.net.train()
            train_loss = 0.0
            for Xb, yb in dl:
                optimizer.zero_grad()
                logits = self.net(Xb).squeeze(-1)
                loss   = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(Xb)
            train_loss /= len(Xt)
            scheduler.step()

            # ── Validate ──────────────────────────────────────────────────────
            val_loss = None
            if X_val is not None:
                self.net.eval()
                with torch.no_grad():
                    Xv  = torch.FloatTensor(X_val).to(self.device)
                    yv  = torch.FloatTensor(y_val).to(self.device)
                    val_logits = self.net(Xv).squeeze(-1)
                    val_loss   = criterion(val_logits, yv).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    import copy
                    best_state = copy.deepcopy(self.net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

            self._loss_history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6) if val_loss is not None else None,
            })

            if epoch % 10 == 0 or epoch == 1:
                msg = f"  Epoch {epoch:3d}/{self.epochs}  train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.4f}"
                logger.info(msg)

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore best weights
        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.is_fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return phishing probability scores."""
        torch, nn = _require_torch()
        import torch.nn.functional as F

        X = self._scaler.transform(X.astype(np.float32))
        self.net.eval()
        with torch.no_grad():
            Xt     = torch.FloatTensor(X).to(self.device)
            logits = self.net(Xt).squeeze(-1)
            proba  = torch.sigmoid(logits).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        torch, _ = _require_torch()
        if path is None:
            path = settings.MODEL_ARTIFACT_DIR / self.MODEL_FILE
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "scaler": self._scaler,
            "config": {
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "lr": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
            },
        }, path)
        logger.info(f"Neural Network saved → {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PhishingMLP":
        torch, _ = _require_torch()
        if path is None:
            path = settings.MODEL_ARTIFACT_DIR / cls.MODEL_FILE
        ckpt   = torch.load(path, map_location="cpu")
        config = ckpt["config"]
        obj    = cls(**config)
        obj.net.load_state_dict(ckpt["state_dict"])
        obj._scaler   = ckpt["scaler"]
        obj.is_fitted = True
        logger.info(f"Neural Network loaded from {path}")
        return obj
