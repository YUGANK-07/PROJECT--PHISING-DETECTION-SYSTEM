"""
data/scripts/fetch_phishtank.py
────────────────────────────────
Downloads the PhishTank verified phishing URL feed and saves it as a
normalised CSV in data/raw/phishtank.csv.

PhishTank provides a JSON feed (~40 MB compressed) that is updated
every 60 minutes.  An API key is optional — anonymous access is allowed
but rate-limited to 1 request / 5 minutes per IP.

Usage
-----
    python -m data.scripts.fetch_phishtank [--output data/raw/phishtank.csv]

Output schema
-------------
    url          str   Verified phishing URL
    phish_id     str   PhishTank internal ID
    target       str   Brand being spoofed (may be empty)
    verified_at  str   ISO-8601 verification timestamp
    label        int   Always 1 (phishing)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from utils.config import settings
from utils.helpers import normalize_url
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_TIMEOUT = 120          # seconds
_MAX_RETRIES = 3
_RETRY_BACKOFF = 5      # seconds


def _build_request_params() -> dict[str, str]:
    params: dict[str, str] = {"format": "json"}
    if settings.PHISHTANK_API_KEY:
        params["app_key"] = settings.PHISHTANK_API_KEY
    return params


def _download_feed() -> list[dict[str, Any]]:
    """Download the PhishTank JSON feed with retry logic.

    Returns
    -------
    list[dict]
        Raw list of PhishTank entry dicts.

    Raises
    ------
    RuntimeError
        If all retries are exhausted.
    """
    url = settings.PHISHTANK_FEED_URL
    params = _build_request_params()

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info(f"Fetching PhishTank feed (attempt {attempt}/{_MAX_RETRIES}): {url}")
            response = requests.get(
                url,
                params=params,
                timeout=_TIMEOUT,
                headers={"User-Agent": "PhishingDetector/1.0 (+https://github.com/your-org)"},
                stream=True,
            )
            response.raise_for_status()

            # Stream the response to handle the large payload
            raw_bytes = bytearray()
            total = int(response.headers.get("content-length", 0))
            with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    raw_bytes.extend(chunk)
                    pbar.update(len(chunk))

            data = json.loads(raw_bytes.decode("utf-8"))
            logger.info(f"Downloaded {len(data):,} PhishTank entries")
            return data

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt}")
        except requests.exceptions.HTTPError as exc:
            logger.error(f"HTTP error: {exc.response.status_code} {exc.response.reason}")
            if exc.response.status_code == 429:
                wait = _RETRY_BACKOFF * attempt * 10
                logger.info(f"Rate-limited. Waiting {wait}s before retry …")
                time.sleep(wait)
        except Exception as exc:
            logger.error(f"Unexpected error: {exc}")

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_BACKOFF * attempt)

    raise RuntimeError(f"Failed to download PhishTank feed after {_MAX_RETRIES} attempts")


def _parse_entries(raw: list[dict[str, Any]]) -> pd.DataFrame:
    """Parse raw JSON entries into a clean DataFrame.

    Parameters
    ----------
    raw:
        List of PhishTank JSON entry dicts.

    Returns
    -------
    pd.DataFrame
        Cleaned frame with columns: url, phish_id, target, verified_at, label.
    """
    records = []
    for entry in tqdm(raw, desc="Parsing entries"):
        url = entry.get("url", "").strip()
        if not url:
            continue

        # Normalise URL
        try:
            norm_url = normalize_url(url)
        except Exception:
            norm_url = url

        records.append(
            {
                "url": norm_url,
                "phish_id": str(entry.get("phish_id", "")),
                "target": entry.get("target", ""),
                "verified_at": entry.get("verification_time", ""),
                "label": 1,   # all PhishTank entries are phishing
            }
        )

    df = pd.DataFrame(records)

    # Drop rows with empty or invalid URLs
    df = df[df["url"].str.len() > 4].copy()

    # Deduplicate by URL
    before = len(df)
    df.drop_duplicates(subset=["url"], inplace=True)
    after = len(df)
    logger.info(f"Removed {before - after:,} duplicate PhishTank URLs. Kept {after:,}")

    return df.reset_index(drop=True)


def fetch_phishtank(output_path: Path | None = None) -> pd.DataFrame:
    """Main entry point: download, parse, and save the PhishTank feed.

    Parameters
    ----------
    output_path:
        Path to save the CSV.  Defaults to ``data/raw/phishtank.csv``.

    Returns
    -------
    pd.DataFrame
        Parsed PhishTank entries.
    """
    if output_path is None:
        output_path = settings.raw_data_dir / "phishtank.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = _download_feed()
    df = _parse_entries(raw)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} PhishTank URLs → {output_path}")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch PhishTank phishing URL feed")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/raw/phishtank.csv)",
    )
    args = parser.parse_args()
    fetch_phishtank(output_path=args.output)
