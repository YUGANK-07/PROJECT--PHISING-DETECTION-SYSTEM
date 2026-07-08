"""
features/url_features.py
─────────────────────────
Extracts lexical and structural features directly from the URL string.
No network requests are made here — everything is computed locally.

Feature groups
--------------
A. Length-based      : url_length, hostname_len, path_len, query_len
B. Entropy           : url_entropy, hostname_entropy, path_entropy
C. Character counts  : num_dots, num_hyphens, num_underscores, num_slashes,
                       num_at, num_equals, num_question, num_percent,
                       num_ampersand, num_hash, digit_ratio, letter_ratio
D. Structure         : subdomain_depth, has_ip, has_port, has_fragment,
                       num_query_params, path_depth
E. Suspicion signals : has_at_symbol, has_double_slash, tld_risk_score,
                       suspicious_keyword_count, is_shortened_url,
                       url_entropy_high, num_digits_in_domain
F. TLD & brand       : tld (encoded), brand_name_in_domain
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse, parse_qs

import numpy as np

# ── Typosquat helpers ─────────────────────────────────────────────────────────

# Visually similar / keyboard-adjacent character substitutions used by phishers
_CHAR_SUB = str.maketrans({
    '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's',
    '6': 'g', '7': 't', '8': 'b', '9': 'g', '@': 'a',
})

# Multi-char substitutions (apply before single-char)
_MULTI_SUB = [
    ('rn', 'm'), ('vv', 'w'), ('cl', 'd'), ('nn', 'm'),
    ('ii', 'u'), ('lI', 'u'), ('Il', 'u'),
]


def _normalise_typo(s: str) -> str:
    """Normalise common visual substitutions to detect typosquats."""
    s = s.lower()
    for src, dst in _MULTI_SUB:
        s = s.replace(src, dst)
    return s.translate(_CHAR_SUB)


def _levenshtein(a: str, b: str) -> int:
    """Iterative Levenshtein distance."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = curr
    return prev[-1]


def _typosquat_score(hostname: str, brands: frozenset) -> float:
    """Return a score in [0,1]: 1 = certain typosquat, 0 = no match.

    Strategy
    --------
    - Strip the registered domain (e.g. 'rnicrosoft' from 'rnicrosoft.com')
    - Normalise character substitutions
    - For each brand: if normalised domain == brand AND raw domain != brand
      → confirmed typosquat (score = 1.0)
    - Else use edit-distance: score = max(1 - dist/len(brand), 0)
      if edit distance <= 2 and domain != brand
    """
    parts  = hostname.rstrip('.').split('.')
    # Registered domain = second-to-last part (skip TLD)
    reg_domain = parts[-2] if len(parts) >= 2 else parts[0]

    norm = _normalise_typo(reg_domain)
    best = 0.0

    for brand in brands:
        # Exact match after normalisation but NOT an exact raw match → typosquat
        if norm == brand and reg_domain != brand:
            return 1.0
        # Edit distance check on both raw and normalised
        for candidate in {reg_domain, norm}:
            if candidate == brand:
                continue
            dist = _levenshtein(candidate, brand)
            if dist <= 2:
                sim = 1.0 - dist / max(len(brand), 1)
                best = max(best, sim)

    return round(best, 4)

from utils.helpers import (
    extract_domain,
    has_at_symbol,
    has_double_slash_redirect,
    is_ip_address,
    shannon_entropy,
    subdomain_depth,
    suspicious_keyword_count,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# URL shortening services — phishers often use these to obfuscate
_SHORTENERS = frozenset([
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "short.io", "tiny.cc", "bl.ink", "rebrand.ly",
    "cutt.ly", "shorte.st", "bc.vc", "clk.sh",
])

# Common brands targeted by phishers
_BRANDS = frozenset([
    "paypal", "amazon", "apple", "microsoft", "google", "netflix",
    "facebook", "instagram", "twitter", "ebay", "chase", "wellsfargo",
    "bankofamerica", "citibank", "hsbc", "linkedin", "dropbox",
    "icloud", "outlook", "yahoo", "dhl", "fedex", "usps", "ups",
    "steam", "roblox", "coinbase", "binance", "kraken",
])

# TLD risk scoring (higher = more suspicious)
_TLD_RISK: dict[str, float] = {
    # Very high risk
    ".xyz": 0.95, ".top": 0.92, ".click": 0.90, ".link": 0.88,
    ".online": 0.87, ".site": 0.85, ".info": 0.80, ".biz": 0.75,
    ".tk": 0.97, ".ml": 0.97, ".ga": 0.97, ".cf": 0.97, ".gq": 0.95,
    # Medium risk
    ".net": 0.25, ".org": 0.20, ".co": 0.30, ".io": 0.30,
    ".cc": 0.55, ".pw": 0.70, ".ws": 0.60, ".name": 0.50,
    # Low risk
    ".com": 0.10, ".edu": 0.02, ".gov": 0.01, ".mil": 0.01,
    ".uk": 0.15, ".de": 0.12, ".fr": 0.12, ".jp": 0.12,
}

_IP_IN_URL_RE = re.compile(
    r"(?:https?://)?(?:\d{1,3}\.){3}\d{1,3}"
)

_PORT_RE = re.compile(r":(\d+)(?:/|$)")


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_url_features(url: str) -> dict[str, Any]:
    """Extract a comprehensive set of lexical URL features.

    Parameters
    ----------
    url:
        Raw URL string (will be parsed internally).

    Returns
    -------
    dict
        Feature name → numeric value mapping.
        All values are Python int or float so they can be fed directly
        into numpy/pandas without further conversion.
    """
    features: dict[str, Any] = {}

    # ── Parse URL ─────────────────────────────────────────────────────────────
    try:
        if "://" not in url:
            url = "http://" + url
        parsed = urlparse(url)
        scheme   = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()
        path     = parsed.path or ""
        query    = parsed.query or ""
        fragment = parsed.fragment or ""
        netloc   = parsed.netloc or ""
    except Exception:
        # Return zero-vector on parse failure
        return _zero_features()

    full_url = url

    # ── A. Length-based ───────────────────────────────────────────────────────
    features["url_length"]      = len(full_url)
    features["hostname_len"]    = len(hostname)
    features["path_len"]        = len(path)
    features["query_len"]       = len(query)
    features["fragment_len"]    = len(fragment)

    # ── B. Entropy ────────────────────────────────────────────────────────────
    features["url_entropy"]      = round(shannon_entropy(full_url), 4)
    features["hostname_entropy"] = round(shannon_entropy(hostname), 4)
    features["path_entropy"]     = round(shannon_entropy(path), 4)

    # ── C. Character counts ───────────────────────────────────────────────────
    features["num_dots"]         = full_url.count(".")
    features["num_hyphens"]      = full_url.count("-")
    features["num_underscores"]  = full_url.count("_")
    features["num_slashes"]      = full_url.count("/")
    features["num_at"]           = full_url.count("@")
    features["num_equals"]       = full_url.count("=")
    features["num_question"]     = full_url.count("?")
    features["num_percent"]      = full_url.count("%")
    features["num_ampersand"]    = full_url.count("&")
    features["num_hash"]         = full_url.count("#")
    features["num_semicolon"]    = full_url.count(";")
    features["num_tilde"]        = full_url.count("~")
    features["num_colon"]        = full_url.count(":")
    features["num_comma"]        = full_url.count(",")

    letters     = sum(c.isalpha() for c in full_url)
    digits      = sum(c.isdigit() for c in full_url)
    url_len     = max(len(full_url), 1)
    features["digit_ratio"]      = round(digits / url_len, 4)
    features["letter_ratio"]     = round(letters / url_len, 4)
    features["num_digits_in_domain"] = sum(c.isdigit() for c in hostname)

    # ── D. Structure ──────────────────────────────────────────────────────────
    features["subdomain_depth"]  = subdomain_depth(url)
    features["has_ip"]           = int(is_ip_address(hostname))
    features["has_port"]         = int(bool(_PORT_RE.search(netloc)))
    features["has_fragment"]     = int(bool(fragment))
    features["is_https"]         = int(scheme == "https")

    try:
        query_params = parse_qs(query)
        features["num_query_params"] = len(query_params)
    except Exception:
        features["num_query_params"] = 0

    features["path_depth"] = path.count("/") if path else 0

    # ── E. Suspicion signals ──────────────────────────────────────────────────
    features["has_at_symbol"]        = int(has_at_symbol(url))
    features["has_double_slash"]     = int(has_double_slash_redirect(url))
    features["suspicious_kw_count"]  = suspicious_keyword_count(full_url)

    # TLD risk
    tld = _extract_tld(hostname)
    features["tld_risk_score"]  = _TLD_RISK.get(tld, 0.35)
    features["tld_is_free"]     = int(tld in {".tk", ".ml", ".ga", ".cf", ".gq"})

    # URL shortener
    domain_no_port = hostname.split(":")[0]
    features["is_shortened_url"] = int(domain_no_port in _SHORTENERS)

    # High entropy flag (obfuscated URLs tend to have entropy > 3.8)
    features["url_entropy_high"] = int(features["url_entropy"] > 3.8)

    # ── F. Brand & phishing patterns ─────────────────────────────────────────
    features["brand_in_domain"]    = int(_brand_in_text(hostname))
    features["brand_in_path"]      = int(_brand_in_text(path))
    features["brand_in_subdomain"] = int(_brand_in_subdomain(hostname))

    # Hostname contains hex-encoded characters
    features["has_hex_chars"] = int("%2" in full_url.lower() or "%3" in full_url.lower())

    # Punycode / IDN domain (homograph attack)
    features["has_punycode"] = int("xn--" in hostname)

    # Repeated TLD in URL (e.g. paypal.com.evil.com)
    features["multiple_tlds"] = int(
        sum(1 for t in _TLD_RISK if t in full_url) > 2
    )

    # ── G. Typosquat detection ────────────────────────────────────────────────
    ts = _typosquat_score(hostname, _BRANDS)
    features["typosquat_score"]    = ts
    features["is_typosquat"]       = int(ts >= 0.8)

    # Digit substitution in domain (e.g. paypa1, g00gle)
    features["digit_sub_in_domain"] = int(
        any(c.isdigit() for c in hostname.split(".")[-2] if len(hostname.split(".")) >= 2)
    )

    # Token count (number of words in full URL)
    features["token_count"] = len(re.findall(r"[a-zA-Z0-9]+", full_url))

    return features


def get_feature_names() -> list[str]:
    """Return the ordered list of URL feature names.

    Useful for building feature matrices and SHAP plots.
    """
    return list(_zero_features().keys())


def features_to_vector(features: dict[str, Any]) -> np.ndarray:
    """Convert a feature dict to a fixed-length numpy array.

    Parameters
    ----------
    features:
        Output of ``extract_url_features``.

    Returns
    -------
    np.ndarray
        1-D float32 array in canonical feature order.
    """
    order = get_feature_names()
    return np.array([features.get(k, 0.0) for k in order], dtype=np.float32)


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_tld(hostname: str) -> str:
    """Return the TLD of a hostname (e.g. '.com')."""
    parts = hostname.rstrip(".").split(".")
    if len(parts) >= 2:
        return f".{parts[-1]}"
    return ""


def _brand_in_text(text: str) -> bool:
    """Return True if any known brand name appears in *text*."""
    tokens = set(re.split(r"[^a-zA-Z]+", text.lower()))
    return bool(tokens & _BRANDS)


def _brand_in_subdomain(hostname: str) -> bool:
    """Return True if a brand name appears in the subdomain portion only."""
    parts = hostname.split(".")
    # Subdomains are all parts except the last two (registered domain + TLD)
    subdomains = parts[:-2] if len(parts) > 2 else []
    for sub in subdomains:
        if sub in _BRANDS:
            return True
    return False


def _zero_features() -> dict[str, float]:
    """Return a zero-valued feature dict with all canonical keys."""
    keys = [
        "url_length", "hostname_len", "path_len", "query_len", "fragment_len",
        "url_entropy", "hostname_entropy", "path_entropy",
        "num_dots", "num_hyphens", "num_underscores", "num_slashes",
        "num_at", "num_equals", "num_question", "num_percent",
        "num_ampersand", "num_hash", "num_semicolon", "num_tilde",
        "num_colon", "num_comma",
        "digit_ratio", "letter_ratio", "num_digits_in_domain",
        "subdomain_depth", "has_ip", "has_port", "has_fragment", "is_https",
        "num_query_params", "path_depth",
        "has_at_symbol", "has_double_slash", "suspicious_kw_count",
        "tld_risk_score", "tld_is_free", "is_shortened_url",
        "url_entropy_high",
        "brand_in_domain", "brand_in_path", "brand_in_subdomain",
        "has_hex_chars", "has_punycode", "multiple_tlds",
        # Typosquat features
        "typosquat_score", "is_typosquat", "digit_sub_in_domain", "token_count",
    ]
    return {k: 0.0 for k in keys}
