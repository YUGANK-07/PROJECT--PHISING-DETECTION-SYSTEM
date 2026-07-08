"""
data/preprocess.py
───────────────────
Merges raw PhishTank, OpenPhish, and Tranco datasets into a clean,
balanced, deduplicated training corpus.

Pipeline
--------
1. Load raw CSVs (phishtank, openphish, tranco).
2. Merge phishing sources; drop duplicates across all sources.
3. Balance classes (undersample majority class if needed).
4. Validate URLs: length, scheme, reachability checks (optional).
5. Split into train / val / test sets (stratified).
6. Save to data/processed/.

Usage
-----
    python -m data.preprocess \
        [--phishtank data/raw/phishtank.csv] \
        [--openphish data/raw/openphish.csv] \
        [--tranco    data/raw/tranco.csv] \
        [--output-dir data/processed] \
        [--balance-ratio 1.5] \
        [--test-size 0.15] \
        [--val-size  0.10]

Output files
------------
    data/processed/train.csv   Training set (url, label)
    data/processed/val.csv     Validation set
    data/processed/test.csv    Hold-out test set
    data/processed/full.csv    Full cleaned dataset (no split)
    data/processed/stats.json  Class distribution statistics
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.config import settings
from utils.helpers import normalize_url
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_URL_LENGTH = 10
MAX_URL_LENGTH = 2048
VALID_SCHEMES = {"http", "https"}

# Regex: rough URL sanity check (must have a hostname)
_HOSTNAME_RE = re.compile(r"^[a-zA-Z0-9._\-\[\]:]+$")


# ── Validation ────────────────────────────────────────────────────────────────

def _is_valid_url(url: str) -> bool:
    """Return True if *url* passes basic structural validation."""
    if not url or not isinstance(url, str):
        return False
    if len(url) < MIN_URL_LENGTH or len(url) > MAX_URL_LENGTH:
        return False
    try:
        parsed = urlparse(url)
        if parsed.scheme not in VALID_SCHEMES:
            return False
        host = parsed.hostname or ""
        if not host or not _HOSTNAME_RE.match(host):
            return False
    except Exception:
        return False
    return True


# ── Loading ───────────────────────────────────────────────────────────────────

def _load_csv(path: Path, required_cols: list[str]) -> Optional[pd.DataFrame]:
    """Load a CSV file, verify required columns exist.

    Parameters
    ----------
    path:
        Path to CSV.
    required_cols:
        Columns that must be present.

    Returns
    -------
    pd.DataFrame or None
        None if file does not exist.
    """
    if not path.exists():
        logger.warning(f"File not found: {path} — skipping")
        return None

    df = pd.read_csv(path, dtype=str, low_memory=False)
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")

    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


# ── Merging & cleaning ────────────────────────────────────────────────────────

def _merge_phishing_sources(
    phishtank_path: Path,
    openphish_path: Optional[Path],
) -> pd.DataFrame:
    """Merge PhishTank / seed and optional OpenPhish into one labelled frame."""
    frames = []

    pt = _load_csv(phishtank_path, ["url", "label"])
    if pt is not None:
        frames.append(pt[["url", "label"]])

    if openphish_path is not None:
        op = _load_csv(openphish_path, ["url", "label"])
        if op is not None:
            frames.append(op[["url", "label"]])

    if not frames:
        raise RuntimeError("No data sources found.")

    merged = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined sources: {len(merged):,} raw entries")
    return merged


def _clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply URL validation, normalisation, and deduplication.

    Parameters
    ----------
    df:
        DataFrame with 'url' and 'label' columns.

    Returns
    -------
    pd.DataFrame
        Cleaned frame.
    """
    logger.info("Validating and normalising URLs …")

    valid_mask = []
    normalised_urls = []

    for raw_url in tqdm(df["url"].tolist(), desc="Cleaning URLs"):
        try:
            norm = normalize_url(str(raw_url))
            is_valid = _is_valid_url(norm)
        except Exception:
            norm = str(raw_url)
            is_valid = False

        valid_mask.append(is_valid)
        normalised_urls.append(norm)

    df = df.copy()
    df["url"] = normalised_urls
    df = df[valid_mask].copy()

    before = len(df)
    df.drop_duplicates(subset=["url"], inplace=True)
    logger.info(f"After deduplication: {before:,} → {len(df):,} rows")

    return df.reset_index(drop=True)


def _balance_classes(
    df: pd.DataFrame,
    balance_ratio: float = 1.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Undersample the majority class to achieve a target ratio.

    Parameters
    ----------
    df:
        Cleaned, labelled frame.
    balance_ratio:
        ``n_majority / n_minority`` target ratio (≥ 1).
        Default 1.5 means 40% phishing, 60% legitimate.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Balanced (possibly undersampled) frame.
    """
    counts = df["label"].value_counts()
    logger.info(f"Class distribution before balancing: {counts.to_dict()}")

    minority_label = counts.idxmin()
    majority_label = counts.idxmax()
    n_minority = counts[minority_label]
    n_majority_target = min(counts[majority_label], int(n_minority * balance_ratio))

    majority_df = (
        df[df["label"] == majority_label]
        .sample(n=n_majority_target, random_state=random_state)
    )
    minority_df = df[df["label"] == minority_label]

    balanced = pd.concat([majority_df, minority_df], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(f"Class distribution after balancing: {balanced['label'].value_counts().to_dict()}")
    return balanced


# ── Splitting ─────────────────────────────────────────────────────────────────

def _split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / val / test split.

    Parameters
    ----------
    df:
        Balanced, cleaned full dataset.
    test_size:
        Fraction of data to hold out as the test set.
    val_size:
        Fraction of *remaining* data to hold out as the validation set.
    random_state:
        Seed.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    # val_size is expressed as a fraction of the full dataset
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val["label"],
        random_state=random_state,
    )

    logger.info(
        f"Split sizes — train: {len(train):,} | val: {len(val):,} | test: {len(test):,}"
    )
    return train, val, test


# ── Main pipeline ─────────────────────────────────────────────────────────────

def preprocess(
    phishtank_path: Path = Path("data/raw/phishtank.csv"),
    openphish_path: Path = Path("data/raw/openphish.csv"),
    tranco_path: Path = Path("data/raw/tranco.csv"),
    output_dir: Path = Path("data/processed"),
    balance_ratio: float = 1.5,
    test_size: float = 0.15,
    val_size: float = 0.10,
) -> dict[str, pd.DataFrame]:
    """Run the full data preprocessing pipeline.

    Parameters
    ----------
    phishtank_path, openphish_path, tranco_path:
        Paths to raw source CSVs.
    output_dir:
        Directory to write processed CSVs and stats JSON.
    balance_ratio:
        ``n_legitimate / n_phishing`` target ratio.
    test_size, val_size:
        Train / val / test fractions.

    Returns
    -------
    dict
        ``{"train": df, "val": df, "test": df, "full": df}``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load sources
    phishing_df = _merge_phishing_sources(phishtank_path, openphish_path)

    # Detect if the primary source is already a balanced pre-labelled dataset
    # (seed_dataset.csv has both label=0 and label=1)
    is_pre_labelled = (
        "label" in phishing_df.columns
        and phishing_df["label"].astype(int).nunique() > 1
    )

    if is_pre_labelled:
        logger.info("Pre-labelled dataset detected (contains both classes). Skipping Tranco merge.")
        raw_full = phishing_df.copy()
        raw_full["label"] = raw_full["label"].astype(int)
    else:
        # Force phishing label then merge with Tranco legit
        phishing_df["label"] = 1
        # 2. Load legitimate source
        if tranco_path is None or not Path(tranco_path).exists():
            raise RuntimeError("Tranco dataset not found and no pre-labelled seed. Run fetch_tranco.py first.")
        tranco_df = _load_csv(tranco_path, ["url", "label"])
        tranco_df["label"] = 0
        raw_full = pd.concat(
            [phishing_df[["url", "label"]], tranco_df[["url", "label"]]],
            ignore_index=True,
        )

    logger.info(f"Total combined rows before cleaning: {len(raw_full):,}")
    logger.info(f"Label distribution: {raw_full['label'].value_counts().to_dict()}")


    # 4. Clean
    clean_full = _clean_dataset(raw_full)

    # 5. Balance
    balanced = _balance_classes(clean_full, balance_ratio=balance_ratio)

    # 6. Split
    train_df, val_df, test_df = _split_dataset(
        balanced, test_size=test_size, val_size=val_size
    )

    # 7. Save
    balanced.to_csv(output_dir / "full.csv", index=False)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # 8. Stats
    stats = {
        "total": len(balanced),
        "phishing": int((balanced["label"] == 1).sum()),
        "legitimate": int((balanced["label"] == 0).sum()),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "balance_ratio": balance_ratio,
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Preprocessing complete. Stats: {stats}")

    return {"train": train_df, "val": val_df, "test": test_df, "full": balanced}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess phishing detection datasets")
    parser.add_argument("--phishtank", type=Path, default=Path("data/raw/phishtank.csv"))
    parser.add_argument("--openphish", type=Path, default=Path("data/raw/openphish.csv"))
    parser.add_argument("--tranco", type=Path, default=Path("data/raw/tranco.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--balance-ratio", type=float, default=1.5)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.10)
    args = parser.parse_args()

    preprocess(
        phishtank_path=args.phishtank,
        openphish_path=args.openphish,
        tranco_path=args.tranco,
        output_dir=args.output_dir,
        balance_ratio=args.balance_ratio,
        test_size=args.test_size,
        val_size=args.val_size,
    )
