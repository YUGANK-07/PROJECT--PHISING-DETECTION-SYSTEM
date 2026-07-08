"""
data/scripts/fetch_tranco.py
─────────────────────────────
Downloads the Tranco Top-1M list of legitimate domains and converts
them to synthetic "benign" URLs (https://<domain>/) for training.

Tranco (https://tranco-list.eu/) is a research-grade list that combines
and de-noises the Alexa, Majestic, Cisco Umbrella, and Quantcast lists.

Usage
-----
    python -m data.scripts.fetch_tranco \
        [--output data/raw/tranco.csv] \
        [--limit 100000]

Output schema
-------------
    url    str   Legitimate base URL (https://<domain>/)
    rank   int   Tranco rank (1 = most popular)
    label  int   Always 0 (legitimate)
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_TIMEOUT = 120
_CHUNK_SIZE = 1024 * 64   # 64 KB


def _download_zip(url: str) -> bytes:
    """Download the Tranco ZIP archive with a progress bar.

    Parameters
    ----------
    url:
        Direct URL to the Tranco CSV ZIP file.

    Returns
    -------
    bytes
        Raw ZIP bytes.
    """
    logger.info(f"Downloading Tranco list from {url} …")
    response = requests.get(
        url,
        timeout=_TIMEOUT,
        headers={"User-Agent": "PhishingDetector/1.0"},
        stream=True,
    )
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    buf = bytearray()
    with tqdm(total=total, unit="B", unit_scale=True, desc="Tranco ZIP") as pbar:
        for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
            buf.extend(chunk)
            pbar.update(len(chunk))

    logger.info(f"Downloaded {len(buf) / 1e6:.1f} MB")
    return bytes(buf)


def _parse_zip(raw_zip: bytes, limit: Optional[int] = None) -> pd.DataFrame:
    """Extract the CSV from the ZIP and return a clean DataFrame.

    Parameters
    ----------
    raw_zip:
        Raw ZIP archive bytes.
    limit:
        Maximum number of rows to include (None = all).

    Returns
    -------
    pd.DataFrame
        Cleaned frame: url, rank, label.
    """
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        logger.info(f"Extracting {csv_name} from ZIP …")
        with zf.open(csv_name) as f:
            df = pd.read_csv(
                f,
                header=None,
                names=["rank", "domain"],
                dtype={"rank": int, "domain": str},
            )

    if limit:
        df = df.iloc[:limit]

    # Build base URLs
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building URLs"):
        domain = str(row["domain"]).strip().lower()
        if not domain:
            continue
        records.append(
            {
                "url": f"https://{domain}/",
                "rank": int(row["rank"]),
                "label": 0,   # legitimate
            }
        )

    result = pd.DataFrame(records)
    result.drop_duplicates(subset=["url"], inplace=True)
    logger.info(f"Parsed {len(result):,} Tranco legitimate URLs")
    return result.reset_index(drop=True)


def fetch_tranco(
    output_path: Path | None = None,
    limit: int = 200_000,
) -> pd.DataFrame:
    """Main entry point: download, parse, and save the Tranco list.

    Parameters
    ----------
    output_path:
        Destination CSV path.  Defaults to ``data/raw/tranco.csv``.
    limit:
        Maximum number of domains to include.  Keeping this bounded
        avoids an enormous dataset that slows training.

    Returns
    -------
    pd.DataFrame
        Parsed Tranco legitimate URLs.
    """
    if output_path is None:
        output_path = settings.raw_data_dir / "tranco.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_zip = _download_zip(settings.TRANCO_LIST_URL)
    df = _parse_zip(raw_zip, limit=limit)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} Tranco URLs → {output_path}")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Tranco top-1M legitimate domain list")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=200_000,
        help="Maximum number of rows to download (default: 200 000)",
    )
    args = parser.parse_args()
    fetch_tranco(output_path=args.output, limit=args.limit)
