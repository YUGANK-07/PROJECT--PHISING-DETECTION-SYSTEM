"""
features/feature_pipeline.py
──────────────────────────────
Unified feature extractor that combines URL, domain, webpage, and NLP
features into a single fixed-length numeric vector for model inference.

Architecture
------------

    URL ──► url_features       (45 features)
         ──► domain_features   (19 features, optional network)
         ──► nlp_features      (7 features)
    HTML ──► webpage_features  (31 features, optional)
    ────────────────────────────────────────────────────
    Total (structural):         102 features
    + BERT embedding (optional): 768 features

The pipeline can run in three modes:
  - "url_only"   : URL lexical + NLP (no network, <1ms)
  - "full"       : URL + domain lookups + NLP (WHOIS/DNS/SSL, ~5s)
  - "with_html"  : full + webpage analysis (requires fetched HTML)

Usage
-----
    from features.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline(mode="url_only")
    vector, meta = pipeline.extract("https://example.com")

    # With pre-fetched HTML:
    vector, meta = pipeline.extract("https://example.com", html=html_str)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import aiohttp

from features.url_features import (
    extract_url_features,
    get_feature_names as url_feature_names,
)
from features.domain_features import (
    extract_domain_features,
    get_feature_names as domain_feature_names,
)
from features.webpage_features import (
    extract_webpage_features,
    get_feature_names as webpage_feature_names,
)
from features.nlp_features import (
    extract_nlp_features,
    get_feature_names as nlp_feature_names,
)
from features.obfuscation_features import (
    extract_obfuscation_features,
    get_feature_names as obfuscation_feature_names,
    _zero_obfuscation_features,
)
from utils.helpers import normalize_url
from utils.logger import get_logger

logger = get_logger(__name__)

PipelineMode = Literal["url_only", "full", "with_html"]

# ── Fetch constants ───────────────────────────────────────────────────────────
_FETCH_TIMEOUT    = aiohttp.ClientTimeout(total=10, connect=5)
_FETCH_HEADERS    = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_MAX_HTML_BYTES   = 512 * 1024   # 512 KB cap


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Container for feature extraction output."""

    vector: np.ndarray                  # Fixed-length float32 feature vector
    feature_names: list[str]            # Ordered names matching vector dims
    raw_features: dict[str, Any]        # All features as a named dict
    bert_embedding: Optional[np.ndarray] = None   # 768-dim, if enabled
    extraction_time_ms: float = 0.0
    html_fetched: bool = False
    error: Optional[str] = None


# ── Pipeline ──────────────────────────────────────────────────────────────────

class FeaturePipeline:
    """Unified feature extraction pipeline.

    Parameters
    ----------
    mode:
        Extraction mode:
        - ``"url_only"``  — lexical URL + NLP features (no I/O)
        - ``"full"``      — adds WHOIS/DNS/SSL lookups
        - ``"with_html"`` — adds webpage analysis (auto-fetches if no HTML given)
    use_bert:
        If True, compute DistilBERT embeddings (requires transformers+torch).
    enable_whois:
        Toggle WHOIS lookups individually (ignored in url_only mode).
    enable_dns:
        Toggle DNS lookups individually.
    enable_ssl:
        Toggle SSL checks individually.
    """

    def __init__(
        self,
        mode: PipelineMode = "full",
        use_bert: bool = False,
        enable_whois: bool = True,
        enable_dns: bool = True,
        enable_ssl: bool = True,
    ):
        self.mode          = mode
        self.use_bert      = use_bert
        self.enable_whois  = enable_whois
        self.enable_dns    = enable_dns
        self.enable_ssl    = enable_ssl
        self._feature_names: Optional[list[str]] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, url: str, html: str = "") -> ExtractionResult:
        """Synchronous extraction entry point.

        Wraps ``extract_async`` for use outside of async contexts.

        Parameters
        ----------
        url:
            URL to analyse.
        html:
            Pre-fetched HTML string.  If empty and mode is ``"with_html"``,
            the HTML will be fetched automatically.

        Returns
        -------
        ExtractionResult
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already inside an async context (e.g. FastAPI)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run, self.extract_async(url, html)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.extract_async(url, html))
        except Exception:
            return asyncio.run(self.extract_async(url, html))

    async def extract_async(self, url: str, html: str = "") -> ExtractionResult:
        """Async extraction — preferred entry point from FastAPI handlers.

        Parameters
        ----------
        url:
            URL to analyse.
        html:
            Pre-fetched HTML.  Auto-fetched if empty and mode includes HTML.

        Returns
        -------
        ExtractionResult
        """
        t0 = time.perf_counter()
        html_fetched = False
        error: Optional[str] = None

        # ── Normalise URL ─────────────────────────────────────────────────────
        try:
            url = normalize_url(url)
        except Exception as e:
            error = f"URL normalisation error: {e}"

        all_features: dict[str, Any] = {}

        # ── 1. URL features (always, no I/O) ──────────────────────────────────
        try:
            url_feats = extract_url_features(url)
            all_features.update(url_feats)
        except Exception as e:
            logger.warning(f"URL feature extraction failed: {e}")
            from features.url_features import _zero_features as url_zeros
            all_features.update(url_zeros())

        # ── 2. Domain features (full / with_html modes) ───────────────────────
        if self.mode in ("full", "with_html"):
            try:
                dom_feats = extract_domain_features(
                    url,
                    enable_whois=self.enable_whois,
                    enable_dns=self.enable_dns,
                    enable_ssl=self.enable_ssl,
                )
                all_features.update(dom_feats)
            except Exception as e:
                logger.warning(f"Domain feature extraction failed: {e}")
                all_features.update({k: 0 for k in domain_feature_names()})
        else:
            all_features.update({k: 0 for k in domain_feature_names()})

        # ── 3. Webpage features (with_html mode) ─────────────────────────────
        if self.mode == "with_html":
            if not html:
                html, html_fetched, fetch_err = await _fetch_html(url)
                if fetch_err:
                    error = fetch_err
            else:
                html_fetched = False

            try:
                web_feats = extract_webpage_features(html, base_url=url)
                all_features.update(web_feats)
            except Exception as e:
                logger.warning(f"Webpage feature extraction failed: {e}")
                from features.webpage_features import _zero_features as web_zeros
                all_features.update(web_zeros())
        else:
            from features.webpage_features import _zero_features as web_zeros
            all_features.update(web_zeros())

        # ── 4. Obfuscation features (always, no I/O) ───────────────────────
        try:
            obf_feats = extract_obfuscation_features(url)
            all_features.update(obf_feats)
        except Exception as e:
            logger.warning(f"Obfuscation feature extraction failed: {e}")
            all_features.update(_zero_obfuscation_features())

        # ── 5. NLP features ────────────────────────────────────────────────────
        page_text = html if self.mode == "with_html" else ""
        try:
            nlp_feats = extract_nlp_features(
                url,
                page_text=page_text,
                use_bert=self.use_bert,
            )
            bert_emb = nlp_feats.pop("bert_embedding", None)
            all_features.update(nlp_feats)
        except Exception as e:
            logger.warning(f"NLP feature extraction failed: {e}")
            all_features.update({k: 0 for k in nlp_feature_names()})
            bert_emb = None

        # ── Build ordered vector ──────────────────────────────────────────────
        feat_names = self.feature_names
        vector = np.array(
            [float(all_features.get(k, 0.0)) for k in feat_names],
            dtype=np.float32,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return ExtractionResult(
            vector=vector,
            feature_names=feat_names,
            raw_features=all_features,
            bert_embedding=bert_emb,
            extraction_time_ms=round(elapsed_ms, 2),
            html_fetched=html_fetched,
            error=error,
        )

    # ── Batch extraction ──────────────────────────────────────────────────────

    async def extract_batch_async(
        self,
        urls: list[str],
        htmls: Optional[list[str]] = None,
    ) -> list[ExtractionResult]:
        """Extract features for a batch of URLs concurrently.

        Parameters
        ----------
        urls:
            List of URLs.
        htmls:
            Optional pre-fetched HTML list (same length as urls).

        Returns
        -------
        list[ExtractionResult]
        """
        if htmls is None:
            htmls = [""] * len(urls)

        tasks = [
            self.extract_async(url, html)
            for url, html in zip(urls, htmls)
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def extract_batch(
        self,
        urls: list[str],
        htmls: Optional[list[str]] = None,
    ) -> list[ExtractionResult]:
        """Synchronous batch extraction."""
        return asyncio.run(self.extract_batch_async(urls, htmls))

    # ── DataFrame builder ─────────────────────────────────────────────────────

    def to_dataframe(self, results: list[ExtractionResult]):
        """Convert a list of ExtractionResults into a pandas DataFrame.

        Parameters
        ----------
        results:
            Output of ``extract_batch``.

        Returns
        -------
        pd.DataFrame
            Rows = samples, columns = feature names.
        """
        import pandas as pd
        rows = [r.raw_features for r in results]
        df = pd.DataFrame(rows, columns=self.feature_names)
        return df.astype(np.float32)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of all feature names in the output vector."""
        if self._feature_names is None:
            self._feature_names = (
                url_feature_names()
                + domain_feature_names()
                + webpage_feature_names()
                + obfuscation_feature_names()
                + nlp_feature_names()
            )
        return self._feature_names

    @property
    def n_features(self) -> int:
        """Dimensionality of the structural feature vector."""
        return len(self.feature_names)


# ── HTML fetcher ──────────────────────────────────────────────────────────────

async def _fetch_html(url: str) -> tuple[str, bool, Optional[str]]:
    """Fetch HTML content from *url* asynchronously.

    Parameters
    ----------
    url:
        Target URL.

    Returns
    -------
    (html_str, fetched_ok, error_msg)
    """
    try:
        async with aiohttp.ClientSession(
            headers=_FETCH_HEADERS,
            timeout=_FETCH_TIMEOUT,
        ) as session:
            async with session.get(url, allow_redirects=True, ssl=False) as resp:
                raw = await resp.content.read(_MAX_HTML_BYTES)
                html = raw.decode("utf-8", errors="replace")
                return html, True, None
    except asyncio.TimeoutError:
        return "", False, f"Timeout fetching {url}"
    except Exception as exc:
        return "", False, f"Fetch error: {exc}"


# ── Convenience function ──────────────────────────────────────────────────────

def build_feature_matrix(urls: list[str], mode: PipelineMode = "url_only") -> np.ndarray:
    """Build a 2-D feature matrix from a list of URLs.

    Parameters
    ----------
    urls:
        List of URL strings.
    mode:
        Extraction mode.

    Returns
    -------
    np.ndarray
        Shape (n_urls, n_features) float32 matrix.
    """
    pipeline = FeaturePipeline(mode=mode)
    results  = pipeline.extract_batch(urls)
    return np.vstack([r.vector for r in results])
