"""
features/obfuscation_features.py
──────────────────────────────────
Detects visual-similarity attacks, Unicode homographs, URL encoding tricks,
and other obfuscation techniques used by phishing URLs.

Feature groups
--------------
A. Unicode / homograph attacks
   - unicode_confusable_count   : # of visually confusable Unicode chars in hostname
   - has_mixed_scripts          : hostname mixes Latin with Cyrillic / Greek etc.
   - has_unicode_attack         : confirmed IDN homograph (confusable + normalises to brand)
   - visual_confusable_score    : after confusable normalisation, typosquat similarity [0,1]
   - has_rtl_chars              : right-to-left override / mark characters present
   - has_zero_width_chars       : zero-width joiner/non-joiner (invisible chars)

B. URL-encoding obfuscation
   - pct_encoded_host_chars     : # of %XX sequences in hostname (should be 0)
   - has_double_encoding        : %25 (double URL-encoded percent sign)
   - encoded_dot_in_host        : %2E used instead of '.' in hostname
   - encoded_slash_in_path      : excessive %2F in path
   - pct_encoded_path_ratio     : fraction of path chars that are encoded

C. Alternative IP representations
   - has_hex_ip                 : 0xAABBCCDD format IP in URL
   - has_decimal_ip             : decimal (long int) IP e.g. http://3232235777/
   - has_octal_ip               : octal dotted notation e.g. 0300.0250...

D. Visual digit-letter substitution (g00gle, paypa1)
   - digit_letter_subs_count    : # of classic digit↔letter substitutions in hostname
   - has_brand_digit_sub        : brand name with digit substitutions detected
   - homoglyph_score            : composite visual similarity after digit normalisation [0,1]

E. Composite obfuscation score
   - obfuscation_score          : weighted composite [0,1]
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse, unquote
from typing import Any

# ── Unicode confusable table ───────────────────────────────────────────────────
# Maps confusable Unicode codepoints → their Latin-script equivalent character.
# Sourced from Unicode CLDR / confusables.txt (subset of most-exploited chars).
_UNICODE_CONFUSABLES: dict[str, str] = {
    # Cyrillic → Latin
    "\u0430": "a",   # а → a
    "\u0435": "e",   # е → e
    "\u043e": "o",   # о → o
    "\u0440": "p",   # р → p (Cyrillic р looks like p)
    "\u0441": "c",   # с → c
    "\u0445": "x",   # х → x
    "\u0456": "i",   # і → i (Ukrainian i)
    "\u0443": "y",   # у → y
    "\u0455": "s",   # ѕ → s
    "\u0458": "j",   # ј → j
    "\u0459": "lj",  # љ → lj
    "\u0491": "g",   # ґ → g
    "\u0410": "A",   # А → A
    "\u0412": "B",   # В → B
    "\u0415": "E",   # Е → E
    "\u041a": "K",   # К → K
    "\u041c": "M",   # М → M
    "\u041d": "H",   # Н → H
    "\u041e": "O",   # О → O
    "\u0420": "P",   # Р → P
    "\u0421": "C",   # С → C
    "\u0422": "T",   # Т → T
    "\u0425": "X",   # Х → X
    # Greek → Latin
    "\u03bf": "o",   # ο (omicron) → o
    "\u03b1": "a",   # α → a
    "\u03b5": "e",   # ε → e
    "\u03b9": "i",   # ι → i
    "\u03bd": "v",   # ν → v
    "\u03c1": "p",   # ρ → p
    "\u039f": "O",   # Ο → O
    "\u0391": "A",   # Α → A
    "\u0392": "B",   # Β → B
    "\u0395": "E",   # Ε → E
    "\u0396": "Z",   # Ζ → Z
    "\u0397": "H",   # Η → H
    "\u0399": "I",   # Ι → I
    "\u039a": "K",   # Κ → K
    "\u039c": "M",   # Μ → M
    "\u039d": "N",   # Ν → N
    "\u03a1": "P",   # Ρ → P
    "\u03a4": "T",   # Τ → T
    "\u03a5": "Y",   # Υ → Y
    "\u03a7": "X",   # Χ → X
    # Full-width Latin (often used in IDN)
    "\uff41": "a", "\uff42": "b", "\uff43": "c", "\uff44": "d",
    "\uff45": "e", "\uff46": "f", "\uff47": "g", "\uff48": "h",
    "\uff49": "i", "\uff4a": "j", "\uff4b": "k", "\uff4c": "l",
    "\uff4d": "m", "\uff4e": "n", "\uff4f": "o", "\uff50": "p",
    "\uff51": "q", "\uff52": "r", "\uff53": "s", "\uff54": "t",
    "\uff55": "u", "\uff56": "v", "\uff57": "w", "\uff58": "x",
    "\uff59": "y", "\uff5a": "z",
    # Miscellaneous lookalikes
    "\u0131": "i",   # ı (dotless i)
    "\u02c0": "h",   # ʀ-like
    "\u0269": "i",   # ɩ
    "\u0251": "a",   # ɑ
    "\u0261": "g",   # ɡ
    "\u04bb": "h",   # һ (Cyrillic shha)
    "\u0455": "s",   # ѕ
    "\u0501": "d",   # Ԁ
    "\u0509": "q",   # Ԉ
    "\u0273": "n",   # ɳ
    "\u217c": "l",   # ℼ
    "\u2170": "i",   # ⅰ
}

# Script detection regex groups
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_GREEK_RE    = re.compile(r"[\u0370-\u03ff]")
_ARABIC_RE   = re.compile(r"[\u0600-\u06ff]")
_CJK_RE      = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE    = re.compile(r"[a-zA-Z]")

# RTL / invisible chars
_RTL_RE      = re.compile(r"[\u200f\u202b\u202e\u2067]")
_ZWJ_RE      = re.compile(r"[\u200b\u200c\u200d\ufeff]")

# URL encoding patterns
_PCT_ENC_RE       = re.compile(r"%[0-9a-fA-F]{2}")
_DOUBLE_ENC_RE    = re.compile(r"%25[0-9a-fA-F]{2}", re.IGNORECASE)
_ENCODED_DOT_RE   = re.compile(r"%2[eE]")
_ENCODED_SLASH_RE = re.compile(r"%2[fF]")

# IP representations
_HEX_IP_RE     = re.compile(r"0x[0-9a-fA-F]{8}", re.IGNORECASE)
_DECIMAL_IP_RE = re.compile(r"(?:^|//|@)(\d{8,10})(?:/|$)")  # large int IP
_OCTAL_IP_RE   = re.compile(r"0\d{3}\.0\d{3}\.0\d{3}\.0\d{3}")

# Digit ↔ letter substitution map
_DIGIT_SUB: dict[str, str] = {
    "0": "o", "1": "l", "3": "e", "4": "a",
    "5": "s", "6": "g", "7": "t", "8": "b", "9": "g",
}

# Known brands (must match url_features._BRANDS for consistency)
_BRANDS = frozenset([
    "paypal", "amazon", "apple", "microsoft", "google", "netflix",
    "facebook", "instagram", "twitter", "ebay", "chase", "wellsfargo",
    "bankofamerica", "citibank", "hsbc", "linkedin", "dropbox",
    "icloud", "outlook", "yahoo", "dhl", "fedex", "usps", "ups",
    "steam", "roblox", "coinbase", "binance", "kraken",
])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalise_confusables(text: str) -> str:
    """Replace confusable Unicode chars with their Latin equivalents."""
    result = []
    for ch in text.lower():
        result.append(_UNICODE_CONFUSABLES.get(ch, ch))
    return "".join(result)


def _normalise_digit_subs(text: str) -> str:
    """Replace digit substitutions (0→o, 1→l, …) with letter equivalents."""
    return "".join(_DIGIT_SUB.get(c, c) for c in text.lower())


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
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1,
                          prev[j - 1] + (0 if ca == cb else 1))
        prev = curr
    return prev[-1]


def _brand_similarity(domain_part: str) -> float:
    """Return max similarity [0,1] to any known brand."""
    best = 0.0
    for brand in _BRANDS:
        if domain_part == brand:
            return 0.0  # exact match = not phishing
        dist = _levenshtein(domain_part, brand)
        if dist <= 3:
            sim = 1.0 - dist / max(len(brand), 1)
            best = max(best, sim)
    return round(best, 4)


def _reg_domain(hostname: str) -> str:
    """Extract registered domain (strip TLD and subdomains)."""
    parts = hostname.rstrip(".").split(".")
    return parts[-2] if len(parts) >= 2 else parts[0]


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_obfuscation_features(url: str) -> dict[str, Any]:
    """Extract obfuscation and visual-similarity features from *url*.

    Returns
    -------
    dict
        Feature name → numeric value (int or float).
    """
    feats: dict[str, Any] = {}

    # Parse
    try:
        if "://" not in url:
            url = "http://" + url
        parsed  = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        path     = parsed.path or ""
        query    = parsed.query or ""
        netloc   = parsed.netloc or ""
        full_url = url
    except Exception:
        return _zero_obfuscation_features()

    reg_dom = _reg_domain(hostname)

    # ── A. Unicode / homograph ─────────────────────────────────────────────────

    # Count confusable chars in hostname
    conf_count = sum(1 for ch in hostname if ch in _UNICODE_CONFUSABLES)
    feats["unicode_confusable_count"] = conf_count
    feats["has_unicode_attack"]       = int(conf_count > 0)

    # Mixed scripts
    has_latin    = bool(_LATIN_RE.search(hostname))
    has_cyrillic = bool(_CYRILLIC_RE.search(hostname))
    has_greek    = bool(_GREEK_RE.search(hostname))
    has_arabic   = bool(_ARABIC_RE.search(hostname))
    n_scripts = sum([has_latin, has_cyrillic, has_greek, has_arabic])
    feats["has_mixed_scripts"] = int(n_scripts > 1)

    # After normalising confusables, check brand similarity
    norm_conf = _normalise_confusables(reg_dom)
    conf_brand_sim = 0.0
    for brand in _BRANDS:
        if norm_conf == brand and reg_dom != brand:
            conf_brand_sim = 1.0
            break
        if norm_conf == brand:
            continue  # exact ASCII match — legitimate domain
        dist = _levenshtein(norm_conf, brand)
        if 0 < dist <= 2:  # dist==0 means norm_conf IS the brand (legit)
            sim = 1.0 - dist / max(len(brand), 1)
            conf_brand_sim = max(conf_brand_sim, sim)
    feats["visual_confusable_score"] = round(conf_brand_sim, 4)

    # RTL and zero-width chars
    feats["has_rtl_chars"]        = int(bool(_RTL_RE.search(full_url)))
    feats["has_zero_width_chars"] = int(bool(_ZWJ_RE.search(full_url)))

    # ── B. URL-encoding obfuscation ────────────────────────────────────────────

    # Percent-encoded chars in hostname (legitimate: 0)
    pct_in_host = len(_PCT_ENC_RE.findall(netloc))
    feats["pct_encoded_host_chars"] = pct_in_host

    # Double-encoding (%25XX)
    feats["has_double_encoding"]  = int(bool(_DOUBLE_ENC_RE.search(full_url)))

    # Encoded dot in hostname (%2E) — check full URL (hostname may be pre-decoded)
    feats["encoded_dot_in_host"]  = int(bool(_ENCODED_DOT_RE.search(full_url)))

    # Encoded slash in path (%2F) — search raw full_url for maximum coverage
    feats["encoded_slash_in_path"] = int(bool(_ENCODED_SLASH_RE.search(full_url)))

    # Path encoding ratio
    path_len = max(len(path), 1)
    pct_in_path = len(_PCT_ENC_RE.findall(path))
    feats["pct_encoded_path_ratio"] = round((pct_in_path * 3) / path_len, 4)  # each %XX = 3 chars

    # Deep URL-decoding attempt (detect layers of encoding)
    try:
        decoded_once  = unquote(full_url)
        decoded_twice = unquote(decoded_once)
        feats["multi_decode_changes"] = int(decoded_twice != decoded_once)
    except Exception:
        feats["multi_decode_changes"] = 0

    # ── C. Alternative IP representations ──────────────────────────────────────
    feats["has_hex_ip"]     = int(bool(_HEX_IP_RE.search(full_url)))
    feats["has_decimal_ip"] = int(bool(_DECIMAL_IP_RE.search(full_url)))
    feats["has_octal_ip"]   = int(bool(_OCTAL_IP_RE.search(full_url)))

    # ── D. Digit-letter substitution (g00gle, paypa1) ─────────────────────────
    # Count digit substitutions in the registered domain
    digit_sub_count = sum(1 for c in reg_dom if c in _DIGIT_SUB)
    feats["digit_letter_subs_count"] = digit_sub_count

    # After normalising digit subs, check brand similarity
    norm_digit = _normalise_digit_subs(reg_dom)
    digit_brand_sim = 0.0
    for brand in _BRANDS:
        if norm_digit == brand and reg_dom != brand:
            digit_brand_sim = 1.0
            break
        if norm_digit == brand:
            continue  # exact ASCII match — legitimate domain
        dist = _levenshtein(norm_digit, brand)
        if 0 < dist <= 2:  # dist==0 means norm_digit IS the brand (legit)
            sim = 1.0 - dist / max(len(brand), 1)
            digit_brand_sim = max(digit_brand_sim, sim)
    feats["has_brand_digit_sub"] = int(digit_brand_sim >= 0.8)
    feats["homoglyph_score"]     = round(max(conf_brand_sim, digit_brand_sim), 4)

    # ── E. Composite obfuscation score ─────────────────────────────────────────
    score = 0.0
    score += min(conf_count * 0.3, 0.9)              # Unicode confusables
    score += feats["has_mixed_scripts"] * 0.8
    score += conf_brand_sim * 0.9
    score += feats["has_rtl_chars"] * 0.95
    score += feats["has_zero_width_chars"] * 0.95
    score += min(pct_in_host * 0.4, 0.9)             # encoded hostname chars
    score += feats["has_double_encoding"] * 0.7
    score += feats["encoded_dot_in_host"] * 0.8
    score += feats["encoded_slash_in_path"] * 0.55   # encoded slashes in path
    score += min(feats["pct_encoded_path_ratio"] * 0.8, 0.6)  # high encoding ratio
    score += feats["has_hex_ip"] * 0.85
    score += feats["has_decimal_ip"] * 0.85
    score += feats["has_octal_ip"] * 0.85
    score += min(digit_sub_count * 0.2, 0.6)
    score += digit_brand_sim * 0.85
    feats["obfuscation_score"] = round(min(score, 1.0), 4)

    return feats


def get_feature_names() -> list[str]:
    """Return ordered list of obfuscation feature names."""
    return list(_zero_obfuscation_features().keys())


def _zero_obfuscation_features() -> dict[str, float]:
    return {
        # Unicode
        "unicode_confusable_count":  0.0,
        "has_unicode_attack":        0.0,
        "has_mixed_scripts":         0.0,
        "visual_confusable_score":   0.0,
        "has_rtl_chars":             0.0,
        "has_zero_width_chars":      0.0,
        # URL encoding
        "pct_encoded_host_chars":    0.0,
        "has_double_encoding":       0.0,
        "encoded_dot_in_host":       0.0,
        "encoded_slash_in_path":     0.0,
        "pct_encoded_path_ratio":    0.0,
        "multi_decode_changes":      0.0,
        # Alt IP
        "has_hex_ip":                0.0,
        "has_decimal_ip":            0.0,
        "has_octal_ip":              0.0,
        # Digit-letter sub
        "digit_letter_subs_count":   0.0,
        "has_brand_digit_sub":       0.0,
        "homoglyph_score":           0.0,
        # Composite
        "obfuscation_score":         0.0,
    }
