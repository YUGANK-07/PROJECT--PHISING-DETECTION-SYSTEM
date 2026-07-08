"""
data/scripts/fetch_openphish.py
────────────────────────────────
Downloads the OpenPhish community phishing URL feed.

OpenPhish publishes a plain-text file with one URL per line.
No API key is required for the community feed.

Usage
-----
    python -m data.scripts.fetch_openphish [--output data/raw/openphish.csv]

Output schema
-------------
    url    str   Phishing URL
    label  int   Always 1 (phishing)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from utils.config import settings
from utils.helpers import normalize_url
from utils.logger import get_logger

logger = get_logger(__name__)

_TIMEOUT = 60
_MAX_RETRIES = 3
_RETRY_BACKOFF = 5


def _download_feed(url: str) -> list[str]:
    """Download the OpenPhish text feed and return raw lines.

    Parameters
    ----------
    url:
        Feed URL.

    Returns
    -------
    list[str]
        Raw URL strings from the feed.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info(f"Fetching OpenPhish feed (attempt {attempt}): {url}")
            response = requests.get(
                url,
                timeout=_TIMEOUT,
                headers={"User-Agent": "PhishingDetector/1.0"},
            )
            response.raise_for_status()
            lines = response.text.splitlines()
            logger.info(f"Downloaded {len(lines):,} OpenPhish lines")
            return lines
        except requests.exceptions.HTTPError as exc:
            logger.warning(f"HTTP {exc.response.status_code} on attempt {attempt}")
        except Exception as exc:
            logger.warning(f"Error on attempt {attempt}: {exc}")

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_BACKOFF * attempt)

    raise RuntimeError("Failed to download OpenPhish feed")


def _parse_lines(lines: list[str]) -> pd.DataFrame:
    """Parse text-feed lines into a clean DataFrame.

    Parameters
    ----------
    lines:
        Raw URL strings.

    Returns
    -------
    pd.DataFrame
        Cleaned frame with columns: url, label.
    """
    records = []
    for raw_url in tqdm(lines, desc="Parsing OpenPhish URLs"):
        raw_url = raw_url.strip()
        if not raw_url or raw_url.startswith("#"):
            continue
        try:
            norm = normalize_url(raw_url)
        except Exception:
            norm = raw_url

        records.append({"url": norm, "label": 1})

    df = pd.DataFrame(records)
    df = df[df["url"].str.len() > 4]

    before = len(df)
    df.drop_duplicates(subset=["url"], inplace=True)
    logger.info(f"Removed {before - len(df):,} duplicates. Kept {len(df):,} OpenPhish URLs")

    return df.reset_index(drop=True)


def fetch_openphish(output_path: Path | None = None) -> pd.DataFrame:
    """Main entry point: download, parse, and save the OpenPhish feed.

    Parameters
    ----------
    output_path:
        Destination CSV path.  Defaults to ``data/raw/openphish.csv``.

    Returns
    -------
    pd.DataFrame
        Parsed OpenPhish entries.
    """
    if output_path is None:
        output_path = settings.raw_data_dir / "openphish.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = _download_feed(settings.OPENPHISH_FEED_URL)
    df = _parse_lines(lines)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} OpenPhish URLs → {output_path}")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OpenPhish phishing URL feed")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    fetch_openphish(output_path=args.output)
