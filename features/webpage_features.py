"""
features/webpage_features.py
──────────────────────────────
Extracts structural and behavioural features from raw HTML/JS content.
Works on already-fetched HTML strings — network fetching is done by
the API layer (async) to keep this module pure and testable.

Feature groups
--------------
A. Form analysis    : num_forms, num_inputs, num_password_fields,
                      num_hidden_fields, form_action_external,
                      form_has_no_action
B. Link analysis    : num_links, num_external_links, external_link_ratio,
                      num_null_links, links_pointing_to_different_domain
C. Script analysis  : num_scripts, num_external_scripts,
                      js_obfuscation_score, has_eval, has_document_write,
                      has_window_location, num_iframes, num_hidden_iframes
D. Content signals  : has_favicon, favicon_external,
                      num_images, num_meta_refresh,
                      page_title_suspicious, has_login_form,
                      has_copyright, page_text_entropy
E. Resource ratios  : resource_external_ratio, style_external_ratio
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup

from utils.helpers import (
    extract_domain,
    shannon_entropy,
    suspicious_keyword_count,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────

# JS obfuscation indicators
_OBFUSCATION_PATTERNS = [
    re.compile(r"\beval\s*\("),
    re.compile(r"\bunescape\s*\("),
    re.compile(r"\bString\.fromCharCode\s*\("),
    re.compile(r"\\x[0-9a-fA-F]{2}"),          # hex escapes
    re.compile(r"\\u[0-9a-fA-F]{4}"),          # unicode escapes
    re.compile(r"(?:atob|btoa)\s*\("),         # base64
    re.compile(r"document\.write\s*\("),
    re.compile(r"window\.location\s*[=.]"),
    re.compile(r"\[(\s*['\"][^'\"]+['\"]\s*,?\s*)+\]\.join\s*\("),  # array join
]

_WINDOW_LOCATION_RE = re.compile(
    r"window\.location\s*(?:\.href\s*)?=|window\.location\.replace\s*\("
)
_DOCUMENT_WRITE_RE  = re.compile(r"document\.write\s*\(")
_EVAL_RE            = re.compile(r"\beval\s*\(")

# Suspicious title keywords
_SUSPICIOUS_TITLE_KEYWORDS = frozenset([
    "verify", "confirm", "update", "secure", "login",
    "signin", "account", "suspended", "limited", "alert",
    "paypal", "amazon", "apple", "microsoft", "bank",
])


# ── HTML parsing helpers ──────────────────────────────────────────────────────

def _parse(html: str) -> BeautifulSoup:
    """Parse HTML with lxml (fast) or fallback to html.parser."""
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def _is_external(href: str | None, base_domain: str) -> bool:
    """Return True if *href* points to a different domain than *base_domain*."""
    if not href:
        return False
    href = href.strip()
    if href.startswith(("javascript:", "mailto:", "tel:", "#", "")):
        return False
    try:
        parsed = urlparse(href)
        if not parsed.netloc:
            return False   # relative link
        link_domain = extract_domain(href)
        return link_domain != base_domain and link_domain != ""
    except Exception:
        return False


def _is_null_link(href: str | None) -> bool:
    """Return True for href values that are intentionally empty or void."""
    if not href:
        return True
    s = href.strip().lower()
    return s in ("#", "javascript:void(0)", "javascript:;", "javascript:", "")


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_webpage_features(html: str, base_url: str = "") -> dict[str, Any]:
    """Extract structural HTML and JS features from page content.

    Parameters
    ----------
    html:
        Raw HTML string of the page.
    base_url:
        The URL from which the HTML was fetched (used for external link
        detection).  Can be empty for offline analysis.

    Returns
    -------
    dict
        Feature name → numeric value.
    """
    if not html:
        return _zero_features()

    features: dict[str, Any] = {}
    soup  = _parse(html)
    base_domain = extract_domain(base_url) if base_url else ""

    # ── A. Form analysis ──────────────────────────────────────────────────────
    forms = soup.find_all("form")
    features["num_forms"] = len(forms)

    all_inputs     = soup.find_all("input")
    password_fields = [i for i in all_inputs if (i.get("type", "") or "").lower() == "password"]
    hidden_fields   = [i for i in all_inputs if (i.get("type", "") or "").lower() == "hidden"]

    features["num_inputs"]          = len(all_inputs)
    features["num_password_fields"] = len(password_fields)
    features["num_hidden_fields"]   = len(hidden_fields)
    features["has_login_form"]      = int(bool(password_fields))

    form_action_external = 0
    form_has_no_action   = 0
    for form in forms:
        action = form.get("action", "")
        if not action or action.strip() in ("#", ""):
            form_has_no_action += 1
        elif _is_external(action, base_domain):
            form_action_external += 1

    features["form_action_external"] = form_action_external
    features["form_has_no_action"]   = form_has_no_action

    # ── B. Link analysis ──────────────────────────────────────────────────────
    all_links = soup.find_all("a", href=True)
    features["num_links"] = len(all_links)

    external_links = [a for a in all_links if _is_external(a["href"], base_domain)]
    null_links     = [a for a in all_links if _is_null_link(a.get("href"))]

    features["num_external_links"]    = len(external_links)
    features["num_null_links"]        = len(null_links)
    features["external_link_ratio"]   = round(
        len(external_links) / max(len(all_links), 1), 4
    )

    # Count unique domains in links
    link_domains = set()
    for a in all_links:
        d = extract_domain(a["href"])
        if d and d != base_domain:
            link_domains.add(d)
    features["links_pointing_to_different_domain"] = len(link_domains)

    # ── C. Script analysis ────────────────────────────────────────────────────
    all_scripts      = soup.find_all("script")
    external_scripts = [s for s in all_scripts if s.get("src") and _is_external(s.get("src"), base_domain)]
    inline_js        = " ".join(s.get_text() for s in all_scripts if not s.get("src"))

    features["num_scripts"]          = len(all_scripts)
    features["num_external_scripts"] = len(external_scripts)

    # Obfuscation signals
    obf_score = sum(1 for p in _OBFUSCATION_PATTERNS if p.search(inline_js))
    features["js_obfuscation_score"]  = obf_score
    features["has_eval"]              = int(bool(_EVAL_RE.search(inline_js)))
    features["has_document_write"]    = int(bool(_DOCUMENT_WRITE_RE.search(inline_js)))
    features["has_window_location"]   = int(bool(_WINDOW_LOCATION_RE.search(inline_js)))

    # JS entropy (high entropy inline JS → obfuscated)
    features["inline_js_entropy"] = round(shannon_entropy(inline_js[:5000]), 4)

    # ── iframes ───────────────────────────────────────────────────────────────
    iframes = soup.find_all("iframe")
    hidden_iframes = [
        f for f in iframes
        if _is_hidden_element(f)
    ]
    features["num_iframes"]        = len(iframes)
    features["num_hidden_iframes"] = len(hidden_iframes)

    # ── D. Content signals ────────────────────────────────────────────────────
    # Favicon
    favicon_tag = soup.find("link", rel=lambda r: r and "icon" in " ".join(r).lower())
    if favicon_tag:
        favicon_href = favicon_tag.get("href", "")
        features["has_favicon"]      = 1
        features["favicon_external"] = int(_is_external(favicon_href, base_domain))
    else:
        features["has_favicon"]      = 0
        features["favicon_external"] = 0

    # Meta refresh (common in phishing redirect chains)
    meta_refresh = soup.find_all("meta", attrs={"http-equiv": re.compile(r"refresh", re.I)})
    features["num_meta_refresh"] = len(meta_refresh)

    # Images
    features["num_images"] = len(soup.find_all("img"))

    # Title suspicion
    title_tag = soup.find("title")
    title_text = title_tag.get_text().lower() if title_tag else ""
    features["page_title_suspicious"] = int(
        any(kw in title_text for kw in _SUSPICIOUS_TITLE_KEYWORDS)
    )

    # Copyright notice (legitimacy signal)
    page_text = soup.get_text(" ", strip=True).lower()
    features["has_copyright"] = int("©" in page_text or "copyright" in page_text)

    # Suspicious keyword count in page text
    features["page_text_suspicious_kw"] = suspicious_keyword_count(page_text[:3000])

    # Page text entropy
    features["page_text_entropy"] = round(shannon_entropy(page_text[:3000]), 4)

    # ── E. Resource ratios ────────────────────────────────────────────────────
    # External CSS
    stylesheets = soup.find_all("link", rel=lambda r: r and "stylesheet" in " ".join(r).lower())
    ext_styles  = [s for s in stylesheets if _is_external(s.get("href"), base_domain)]
    features["style_external_ratio"] = round(
        len(ext_styles) / max(len(stylesheets), 1), 4
    )

    # Overall external resource ratio
    total_resources   = features["num_scripts"] + len(stylesheets) + features["num_images"]
    ext_resources     = (
        features["num_external_scripts"] +
        len(ext_styles) +
        len([i for i in soup.find_all("img") if _is_external(i.get("src"), base_domain)])
    )
    features["resource_external_ratio"] = round(
        ext_resources / max(total_resources, 1), 4
    )

    return features


def _is_hidden_element(tag) -> bool:
    """Return True if an element is hidden via CSS or attributes."""
    style = (tag.get("style") or "").lower()
    if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", ""):
        return True
    width  = tag.get("width", "")
    height = tag.get("height", "")
    if width in ("0", "1", 0, 1) or height in ("0", "1", 0, 1):
        return True
    return False


def get_feature_names() -> list[str]:
    """Return canonical webpage feature names."""
    return list(_zero_features().keys())


def _zero_features() -> dict[str, float]:
    keys = [
        "num_forms", "num_inputs", "num_password_fields", "num_hidden_fields",
        "has_login_form", "form_action_external", "form_has_no_action",
        "num_links", "num_external_links", "num_null_links",
        "external_link_ratio", "links_pointing_to_different_domain",
        "num_scripts", "num_external_scripts",
        "js_obfuscation_score", "has_eval", "has_document_write",
        "has_window_location", "inline_js_entropy",
        "num_iframes", "num_hidden_iframes",
        "has_favicon", "favicon_external", "num_meta_refresh",
        "num_images", "page_title_suspicious", "has_copyright",
        "page_text_suspicious_kw", "page_text_entropy",
        "style_external_ratio", "resource_external_ratio",
    ]
    return {k: 0.0 for k in keys}
