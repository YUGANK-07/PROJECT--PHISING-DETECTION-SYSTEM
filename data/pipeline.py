"""
data/pipeline.py
─────────────────
End-to-end data collection → preprocessing orchestrator.

This script ties together the three fetch scripts and the preprocessor
so you can bootstrap the entire dataset in a single command.

Usage
-----
    # Full pipeline (download + preprocess):
    python -m data.pipeline

    # Skip download if raw CSVs already exist:
    python -m data.pipeline --skip-download

    # Use seed dataset (avoids API rate limits — recommended first run):
    python -m data.pipeline --use-seed

    # Limit Tranco to 50 000 domains (faster for dev/testing):
    python -m data.pipeline --tranco-limit 50000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(
    skip_download: bool = False,
    use_seed: bool = True,
    tranco_limit: int = 200_000,
    balance_ratio: float = 1.5,
    test_size: float = 0.15,
    val_size: float = 0.10,
) -> None:
    """Orchestrate the full data pipeline.

    Parameters
    ----------
    skip_download:
        If True, assume raw CSVs already exist and go straight to preprocessing.
    use_seed:
        If True, use the seed dataset generator for phishing URLs instead of
        live API feeds (avoids rate limits, recommended for first run).
    tranco_limit:
        Maximum Tranco rows to download.
    balance_ratio, test_size, val_size:
        Passed through to the preprocessor.
    """
    start = time.perf_counter()

    raw_dir = settings.raw_data_dir
    phishtank_path = raw_dir / "phishtank.csv"
    openphish_path = raw_dir / "openphish.csv"
    tranco_path = raw_dir / "tranco.csv"

    # ── Step 1: Download / Generate ───────────────────────────────────────────
    seed_csv = raw_dir / "seed_dataset.csv"

    if not skip_download:
        if use_seed:
            logger.info("=" * 60)
            logger.info("STEP 1 — Building seed dataset (legit + phishing) …")
            logger.info("=" * 60)
            try:
                from data.scripts.fetch_seed_dataset import generate_seed_dataset
                generate_seed_dataset(
                    n_legit=75_000, n_phish=75_000, output_dir=raw_dir
                )
            except Exception as exc:
                logger.error(f"Seed dataset generation failed: {exc}")
                sys.exit(1)
        else:
            # Try live feeds (PhishTank requires API key for unrestricted access)
            logger.info("=" * 60)
            logger.info("STEP 1 — Downloading PhishTank feed …")
            logger.info("=" * 60)
            try:
                from data.scripts.fetch_phishtank import fetch_phishtank
                fetch_phishtank(output_path=phishtank_path)
            except Exception as exc:
                logger.warning(f"PhishTank download failed: {exc}. Falling back to seed dataset.")
                from data.scripts.fetch_seed_dataset import fetch_seed_dataset
                fetch_seed_dataset(output_path=phishtank_path)

            logger.info("=" * 60)
            logger.info("STEP 2 — Downloading OpenPhish feed …")
            logger.info("=" * 60)
            try:
                from data.scripts.fetch_openphish import fetch_openphish
                fetch_openphish(output_path=openphish_path)
            except Exception as exc:
                logger.warning(f"OpenPhish download failed: {exc}. Skipping.")

        logger.info("=" * 60)
        logger.info("STEP 3 — Downloading Tranco list …")
        logger.info("=" * 60)
        try:
            from data.scripts.fetch_tranco import fetch_tranco
            fetch_tranco(output_path=tranco_path, limit=tranco_limit)
        except Exception as exc:
            logger.error(f"Tranco download failed: {exc}")
            sys.exit(1)
    else:
        logger.info("--skip-download flag set; skipping data collection.")
        for p in [phishtank_path, openphish_path, tranco_path]:
            if p.exists():
                logger.info(f"  Found: {p}")
            else:
                logger.warning(f"  Missing: {p}")

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 — Preprocessing and splitting datasets …")
    logger.info("=" * 60)

    from data.preprocess import preprocess

    # If seed CSV exists (has both legit+phish), pass it as phishtank_path
    # and skip tranco (legit already included)
    _phish_path = seed_csv if seed_csv.exists() else phishtank_path
    _tranco_path = None if seed_csv.exists() else (tranco_path if tranco_path.exists() else None)

    splits = preprocess(
        phishtank_path=_phish_path,
        openphish_path=openphish_path if openphish_path.exists() else None,
        tranco_path=_tranco_path,
        output_dir=settings.processed_data_dir,
        balance_ratio=1.0,   # seed already balanced
        test_size=test_size,
        val_size=val_size,
    )

    elapsed = time.perf_counter() - start
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"  Train : {len(splits['train']):,} rows")
    logger.info(f"  Val   : {len(splits['val']):,} rows")
    logger.info(f"  Test  : {len(splits['test']):,} rows")
    logger.info(f"  Output: {settings.processed_data_dir}")
    logger.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end data collection + preprocessing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading; use existing raw CSVs",
    )
    parser.add_argument(
        "--use-seed",
        action="store_true",
        default=True,
        help="Use seed dataset generator instead of live API feeds (recommended)",
    )
    parser.add_argument(
        "--live-feeds",
        action="store_true",
        default=False,
        help="Use live PhishTank/OpenPhish feeds (requires API key for PhishTank)",
    )
    parser.add_argument(
        "--tranco-limit",
        type=int,
        default=200_000,
        help="Maximum Tranco domains to include",
    )
    parser.add_argument("--balance-ratio", type=float, default=1.5)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.10)
    args = parser.parse_args()

    use_seed = not args.live_feeds
    run_pipeline(
        skip_download=args.skip_download,
        use_seed=use_seed,
        tranco_limit=args.tranco_limit,
        balance_ratio=args.balance_ratio,
        test_size=args.test_size,
        val_size=args.val_size,
    )
