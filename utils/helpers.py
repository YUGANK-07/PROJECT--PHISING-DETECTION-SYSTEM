"""
utils/helpers.py
─────────────────
General-purpose utility functions shared across the codebase.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from urllib.parse import urlparse, unquote
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ── URL utilities ─────────────────────────────────────────────────────────────

_IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)

_HEX_URL_PATTERN = re.compile(r"%[0-9a-fA-F]{2}")
_UNICODE_ESCAPE = re.compile(r"\\u[0-9a-fA-F]{4}")


def normalize_url(url: str) -> str:
    """Normalise a URL for consistent processing.

    Steps
    -----
    1. Strip leading/trailing whitespace and NUL bytes.
    2. Decode percent-encoded characters.
    3. Lower-case scheme and host only (path is case-sensitive).
    4. Remove default ports (80 for http, 443 for https).
    5. Normalise Unicode to NFC form.

    Parameters
    ----------
    url:
        Raw URL string.

    Returns
    -------
    str
        Normalised URL.
    """
    url = url.strip().strip("\x00")
    # Decode percent-encoding
    url = unquote(url)
    # Normalise Unicode
    url = unicodedata.normalize("NFC", url)

    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Strip default ports
        if ":" in netloc:
            host, _, port = netloc.rpartition(":")
            if (scheme == "http" and port == "80") or (
                scheme == "https" and port == "443"
            ):
                netloc = host

        reconstructed = parsed._replace(scheme=scheme, netloc=netloc)
        return reconstructed.geturl()
    except Exception as exc:
        logger.warning(f"URL normalisation failed for '{url}': {exc}")
        return url


def extract_domain(url: str) -> str:
    """Return the registered domain (netloc without port) from a URL."""
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = parsed.hostname or ""
        return host.lower()
    except Exception:
        return ""


def is_ip_address(host: str) -> bool:
    """Return True if *host* is a raw IPv4 address."""
    return bool(_IP_PATTERN.match(host.strip()))


def shannon_entropy(text: str) -> float:
    """Compute the Shannon entropy (bits) of a string.

    Higher entropy → more random text (common in obfuscated payloads).

    Parameters
    ----------
    text:
        Input string.

    Returns
    -------
    float
        Entropy in bits.  0.0 for empty or single-character strings.
    """
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return -sum(
        (count / total) * math.log2(count / total)
        for count in freq.values()
        if count > 0
    )


def count_special_chars(text: str, chars: str = "@-=?#%+~") -> dict[str, int]:
    """Count occurrences of each special character in *text*.

    Parameters
    ----------
    text:
        String to inspect.
    chars:
        Characters to count.

    Returns
    -------
    dict
        ``{char: count}`` for every character in *chars*.
    """
    return {ch: text.count(ch) for ch in chars}


# ── Hashing ───────────────────────────────────────────────────────────────────

def sha256_hex(data: str | bytes) -> str:
    """Return the SHA-256 hex digest of *data*."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def url_cache_key(prefix: str, url: str) -> str:
    """Build a Redis cache key for a URL.

    Uses a short hash to avoid key-length limits.

    Parameters
    ----------
    prefix:
        Namespace (e.g. ``"dns"``, ``"whois"``, ``"prediction"``).
    url:
        The URL to cache.

    Returns
    -------
    str
        Cache key string.
    """
    short = sha256_hex(url.lower())[:16]
    return f"phish:{prefix}:{short}"


# ── Text sanitisation ─────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = _HTML_TAG_RE.sub(" ", html)
    return _WHITESPACE_RE.sub(" ", text).strip()


# ── Subdomain utilities ───────────────────────────────────────────────────────

def subdomain_depth(url: str) -> int:
    """Return the number of subdomain labels in the URL's host.

    Example
    -------
    >>> subdomain_depth("https://login.secure.bank.com/signin")
    2  # "login" and "secure" are subdomains of "bank.com"
    """
    host = extract_domain(url)
    parts = host.split(".")
    # Public suffix estimation: assume last 2 parts are registered domain
    return max(0, len(parts) - 2)


def has_at_symbol(url: str) -> bool:
    """Return True if the URL contains an @ symbol (common phishing trick)."""
    parsed = urlparse(url if "://" in url else f"http://{url}")
    return "@" in parsed.netloc


def has_double_slash_redirect(url: str) -> bool:
    """Detect ``//`` in the URL path which can indicate a redirect."""
    parsed = urlparse(url if "://" in url else f"http://{url}")
    return "//" in parsed.path


# ── Risk scoring helpers ───────────────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = frozenset([
    "secure", "account", "update", "confirm", "verify", "login",
    "signin", "banking", "password", "credential",
    "webscr", "cmd", "dispatch", "support", "alert",
])


def suspicious_keyword_count(text: str) -> int:
    """Count how many suspicious phishing keywords appear in *text*."""
    tokens = set(re.split(r"[^a-zA-Z]+", text.lower()))
    return len(tokens & SUSPICIOUS_KEYWORDS)
