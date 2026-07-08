"""
features/nlp_features.py
─────────────────────────
Extracts semantic NLP features from URL text and page content using
two complementary approaches:

1. **Keyword/Statistical** (fast, no GPU required):
   - TF-IDF-style token analysis
   - Suspicious keyword density
   - Phishing brand impersonation score

2. **Transformer Embeddings** (DistilBERT, optional):
   - 768-dim pooled [CLS] embedding of the URL/page text
   - Used by the neural network branch of the ensemble
   - Falls back gracefully if transformers not installed

The transformer model is lazily loaded (singleton) so the first call
is slow (~1-2s) but subsequent calls are fast (<50ms on CPU).

Feature groups
--------------
A. Statistical  : token_count, unique_token_ratio, avg_token_length,
                  suspicious_kw_density, brand_impersonation_score,
                  phishing_phrase_count
B. Embeddings   : bert_embedding (768-dim vector, optional)
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Optional

import numpy as np

from utils.config import settings
from utils.helpers import suspicious_keyword_count, strip_html
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Phishing phrase lexicon ───────────────────────────────────────────────────

_PHISHING_PHRASES = [
    "verify your account",
    "confirm your identity",
    "update your information",
    "your account has been suspended",
    "click here to verify",
    "your account will be closed",
    "unusual activity detected",
    "sign in to continue",
    "your payment failed",
    "confirm your email",
    "limited time offer",
    "act now",
    "click immediately",
    "your account is at risk",
    "security alert",
    "we detected suspicious",
]

_BRAND_PATTERNS = [
    re.compile(r"\bpaypal\b", re.I),
    re.compile(r"\bamazon\b", re.I),
    re.compile(r"\bapple\b", re.I),
    re.compile(r"\bmicrosoft\b", re.I),
    re.compile(r"\bgoogle\b", re.I),
    re.compile(r"\bnetflix\b", re.I),
    re.compile(r"\bfacebook\b", re.I),
    re.compile(r"\bebay\b", re.I),
    re.compile(r"\bchase\b", re.I),
    re.compile(r"\bwells\s*fargo\b", re.I),
    re.compile(r"\bbank\s*of\s*america\b", re.I),
]

_TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")


# ── Statistical NLP features ──────────────────────────────────────────────────

def extract_statistical_features(text: str) -> dict[str, Any]:
    """Fast, CPU-only lexical NLP features.

    Parameters
    ----------
    text:
        URL string or page text (HTML stripped before processing).

    Returns
    -------
    dict
        Statistical NLP feature dict.
    """
    # Strip any HTML tags first
    clean = strip_html(text).lower()

    tokens = [t for t in _TOKEN_SPLIT_RE.split(clean) if len(t) >= 2]
    n = max(len(tokens), 1)

    unique_ratio = len(set(tokens)) / n
    avg_len      = sum(len(t) for t in tokens) / n
    sus_count    = suspicious_keyword_count(clean)
    sus_density  = round(sus_count / n, 4)

    # Brand mention count
    brand_hits = sum(1 for p in _BRAND_PATTERNS if p.search(clean))

    # Phishing phrase hits
    phrase_hits = sum(1 for phrase in _PHISHING_PHRASES if phrase in clean)

    # Urgency words
    urgency_words = frozenset([
        "urgent", "immediately", "now", "expire", "alert",
        "warning", "limited", "final", "last", "critical",
    ])
    urgency_count = sum(1 for t in tokens if t in urgency_words)

    return {
        "token_count":              len(tokens),
        "unique_token_ratio":       round(unique_ratio, 4),
        "avg_token_length":         round(avg_len, 4),
        "suspicious_kw_density":    sus_density,
        "brand_impersonation_score": brand_hits,
        "phishing_phrase_count":    phrase_hits,
        "urgency_word_count":       urgency_count,
    }


# ── Transformer embedding ─────────────────────────────────────────────────────

class _BertEncoder:
    """Singleton DistilBERT encoder for URL/text embeddings.

    Lazy-loaded on first use to avoid slow startup when embeddings
    are not needed (e.g. URL-only analysis).
    """

    _instance: Optional["_BertEncoder"] = None

    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModel

        model_name = settings.BERT_MODEL_NAME
        device_str = settings.INFERENCE_DEVICE

        logger.info(f"Loading BERT encoder: {model_name} on {device_str}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name)
        self.device    = torch.device(device_str)
        self.model.to(self.device)
        self.model.eval()
        logger.info("BERT encoder loaded.")

    @classmethod
    def get(cls) -> "Optional[_BertEncoder]":
        """Return the singleton, creating it on first call.

        Returns None if transformers/torch are not available.
        """
        if cls._instance is None:
            try:
                cls._instance = cls()
            except ImportError:
                logger.warning(
                    "transformers / torch not installed. "
                    "BERT embeddings disabled — install with: "
                    "pip install transformers torch"
                )
                return None
            except Exception as exc:
                logger.error(f"Failed to load BERT encoder: {exc}")
                return None
        return cls._instance

    def encode(self, text: str, max_length: int = 128) -> np.ndarray:
        """Encode *text* and return the [CLS] pooled embedding.

        Parameters
        ----------
        text:
            Input text (URL or page snippet).
        max_length:
            Maximum token length (128 is sufficient for URLs).

        Returns
        -------
        np.ndarray
            Shape (768,) float32 embedding vector.
        """
        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use [CLS] token representation (index 0 of last hidden state)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding.astype(np.float32)


def get_bert_embedding(text: str) -> Optional[np.ndarray]:
    """Return a 768-dim DistilBERT embedding for *text*, or None.

    Parameters
    ----------
    text:
        Input text to embed.

    Returns
    -------
    np.ndarray or None
        Float32 embedding vector, or None if BERT is unavailable.
    """
    encoder = _BertEncoder.get()
    if encoder is None:
        return None
    try:
        return encoder.encode(text)
    except Exception as exc:
        logger.warning(f"BERT encoding failed: {exc}")
        return None


# ── Combined NLP extractor ────────────────────────────────────────────────────

def extract_nlp_features(
    url: str,
    page_text: str = "",
    use_bert: bool = False,
) -> dict[str, Any]:
    """Extract all NLP features for a URL / page.

    Parameters
    ----------
    url:
        The URL string.
    page_text:
        Raw page text (HTML stripped externally or passed raw).
    use_bert:
        If True, also compute BERT embedding (much slower).
        The embedding is returned as ``"bert_embedding"`` key with
        a numpy array value — handled separately by the feature pipeline.

    Returns
    -------
    dict
        Statistical features + optionally ``"bert_embedding": np.ndarray``.
    """
    # Combine URL + page text for analysis
    combined = f"{url} {strip_html(page_text)}"
    features = extract_statistical_features(combined)

    if use_bert:
        emb = get_bert_embedding(url[:512])
        features["bert_embedding"] = emb   # np.ndarray(768,) or None

    return features


def get_feature_names(include_bert: bool = False) -> list[str]:
    """Return canonical NLP feature names (excludes embedding vector)."""
    names = [
        "token_count", "unique_token_ratio", "avg_token_length",
        "suspicious_kw_density", "brand_impersonation_score",
        "phishing_phrase_count", "urgency_word_count",
    ]
    return names
