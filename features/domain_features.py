"""
features/domain_features.py
─────────────────────────────
Extracts network-level domain features:
  - WHOIS data  (registration date, registrar, expiry, age)
  - DNS records (A, MX, NS, TXT presence and counts)
  - SSL/TLS certificate (validity, days to expiry, issuer trust)

All network lookups are designed to be called with a Redis cache layer
(see api/cache.py) so live requests are minimised during inference.

Feature groups
--------------
A. WHOIS   : domain_age_days, days_to_expiry, registrar_suspicious,
             whois_available, is_recently_registered
B. DNS     : has_a_record, has_mx_record, has_ns_record, has_txt_record,
             num_a_records, num_ns_records, dns_available
C. SSL     : has_ssl, ssl_days_to_expiry, ssl_valid, ssl_self_signed,
             ssl_issuer_trusted
"""

from __future__ import annotations

import socket
import ssl
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

import dns.resolver
import whois as pywhois

from utils.helpers import extract_domain, is_ip_address
from utils.logger import get_logger

logger = get_logger(__name__)

_DNS_TIMEOUT = 5.0    # seconds
_WHOIS_TIMEOUT = 10   # seconds

# Registrars associated with high phishing activity (from academic studies)
_SUSPICIOUS_REGISTRARS = frozenset([
    "namecheap", "namesilo", "name.com", "dynadot",
    "porkbun", "epik", "publicdomainregistry", "godaddy",  # high volume, not necessarily bad
])

# Trusted SSL Certificate Authorities
_TRUSTED_ISSUERS = frozenset([
    "let's encrypt", "digicert", "comodo", "sectigo", "geotrust",
    "rapidssl", "globalsign", "entrust", "thawte", "trustwave",
    "amazon", "google trust services", "microsoft",
])


# ── WHOIS ─────────────────────────────────────────────────────────────────────

def _get_whois_features(domain: str) -> dict[str, Any]:
    """Perform a WHOIS lookup and return domain registration features.

    Parameters
    ----------
    domain:
        Bare hostname (no scheme, no port).

    Returns
    -------
    dict
        WHOIS features.  Returns safe defaults on failure.
    """
    defaults = {
        "domain_age_days": -1,
        "days_to_expiry": -1,
        "registrar_suspicious": 0,
        "whois_available": 0,
        "is_recently_registered": 1,  # pessimistic default
    }

    try:
        w = pywhois.whois(domain)
        if not w:
            return defaults

        now = datetime.now(tz=timezone.utc)

        # ── Creation date → domain age ─────────────────────────────────────
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation:
            if creation.tzinfo is None:
                creation = creation.replace(tzinfo=timezone.utc)
            age_days = (now - creation).days
        else:
            age_days = -1

        # ── Expiry date ────────────────────────────────────────────────────
        expiry = w.expiration_date
        if isinstance(expiry, list):
            expiry = expiry[0]
        if expiry:
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            days_to_expiry = (expiry - now).days
        else:
            days_to_expiry = -1

        # ── Registrar ──────────────────────────────────────────────────────
        registrar = (w.registrar or "").lower()
        registrar_suspicious = int(
            any(r in registrar for r in _SUSPICIOUS_REGISTRARS)
        )

        return {
            "domain_age_days": max(age_days, -1),
            "days_to_expiry": max(days_to_expiry, -1),
            "registrar_suspicious": registrar_suspicious,
            "whois_available": 1,
            "is_recently_registered": int(0 <= age_days <= 30),
        }

    except Exception as exc:
        logger.debug(f"WHOIS lookup failed for {domain}: {exc}")
        return defaults


# ── DNS ───────────────────────────────────────────────────────────────────────

def _get_dns_features(domain: str) -> dict[str, Any]:
    """Query DNS records for *domain* and return feature dict.

    Parameters
    ----------
    domain:
        Bare hostname.

    Returns
    -------
    dict
        DNS features.
    """
    resolver = dns.resolver.Resolver()
    resolver.timeout = _DNS_TIMEOUT
    resolver.lifetime = _DNS_TIMEOUT

    def _query(rtype: str) -> list:
        try:
            answers = resolver.resolve(domain, rtype)
            return list(answers)
        except Exception:
            return []

    a_records  = _query("A")
    mx_records = _query("MX")
    ns_records = _query("NS")
    txt_records = _query("TXT")

    dns_available = int(bool(a_records))

    return {
        "has_a_record":   int(bool(a_records)),
        "has_mx_record":  int(bool(mx_records)),
        "has_ns_record":  int(bool(ns_records)),
        "has_txt_record": int(bool(txt_records)),
        "num_a_records":  len(a_records),
        "num_ns_records": len(ns_records),
        "dns_available":  dns_available,
        # Suspicious: domain resolves to private IP (RFC 1918)
        "dns_private_ip": int(_resolves_to_private(a_records)),
    }


def _resolves_to_private(a_records: list) -> bool:
    """Return True if the domain resolves to an RFC-1918 private address."""
    private_prefixes = ("10.", "172.16.", "172.17.", "192.168.", "127.")
    for rr in a_records:
        ip = str(rr)
        if any(ip.startswith(p) for p in private_prefixes):
            return True
    return False


# ── SSL ───────────────────────────────────────────────────────────────────────

def _get_ssl_features(hostname: str, port: int = 443) -> dict[str, Any]:
    """Connect to *hostname:port* and inspect the TLS certificate.

    Parameters
    ----------
    hostname:
        Domain to check.
    port:
        Default 443 (HTTPS).

    Returns
    -------
    dict
        SSL features.
    """
    defaults = {
        "has_ssl": 0,
        "ssl_days_to_expiry": -1,
        "ssl_valid": 0,
        "ssl_self_signed": 1,   # pessimistic
        "ssl_issuer_trusted": 0,
    }

    try:
        ctx = ssl.create_default_context()
        conn = ctx.wrap_socket(
            socket.create_connection((hostname, port), timeout=5),
            server_hostname=hostname,
        )
        cert = conn.getpeercert()
        conn.close()

        if not cert:
            return defaults

        # Expiry
        not_after = datetime.strptime(
            cert.get("notAfter", ""), "%b %d %H:%M:%S %Y %Z"
        ).replace(tzinfo=timezone.utc)
        days_to_expiry = (not_after - datetime.now(tz=timezone.utc)).days

        # Issuer trust
        issuer_dict = dict(x[0] for x in cert.get("issuer", []))
        issuer_org  = issuer_dict.get("organizationName", "").lower()
        issuer_cn   = issuer_dict.get("commonName", "").lower()
        issuer_trusted = int(
            any(t in issuer_org or t in issuer_cn for t in _TRUSTED_ISSUERS)
        )

        # Self-signed: issuer == subject
        subject_dict = dict(x[0] for x in cert.get("subject", []))
        self_signed = int(issuer_dict == subject_dict)

        return {
            "has_ssl": 1,
            "ssl_days_to_expiry": max(days_to_expiry, 0),
            "ssl_valid": int(days_to_expiry > 0),
            "ssl_self_signed": self_signed,
            "ssl_issuer_trusted": issuer_trusted,
        }

    except ssl.SSLCertVerificationError:
        # Certificate present but invalid
        return {**defaults, "has_ssl": 1, "ssl_valid": 0}
    except Exception as exc:
        logger.debug(f"SSL check failed for {hostname}: {exc}")
        return defaults


# ── Combined extractor ────────────────────────────────────────────────────────

def extract_domain_features(
    url: str,
    enable_whois: bool = True,
    enable_dns: bool = True,
    enable_ssl: bool = True,
) -> dict[str, Any]:
    """Extract all domain-level features for a URL.

    Parameters
    ----------
    url:
        Full URL string.
    enable_whois, enable_dns, enable_ssl:
        Toggle individual lookup types (useful for testing/offline mode).

    Returns
    -------
    dict
        All domain features as a flat dict of numeric values.
    """
    features: dict[str, Any] = {}

    try:
        parsed   = urlparse(url if "://" in url else f"http://{url}")
        hostname = (parsed.hostname or "").lower()
        port     = parsed.port or (443 if parsed.scheme == "https" else 80)
    except Exception:
        hostname = ""
        port     = 443

    # If hostname is an IP, skip WHOIS/DNS lookups
    is_ip = is_ip_address(hostname)
    features["hostname_is_ip"] = int(is_ip)

    if enable_whois and not is_ip and hostname:
        features.update(_get_whois_features(hostname))
    else:
        features.update({
            "domain_age_days": -1, "days_to_expiry": -1,
            "registrar_suspicious": 0, "whois_available": 0,
            "is_recently_registered": int(is_ip),
        })

    if enable_dns and not is_ip and hostname:
        features.update(_get_dns_features(hostname))
    else:
        features.update({
            "has_a_record": 0, "has_mx_record": 0,
            "has_ns_record": 0, "has_txt_record": 0,
            "num_a_records": 0, "num_ns_records": 0,
            "dns_available": 0, "dns_private_ip": 0,
        })

    if enable_ssl and hostname:
        ssl_port = 443 if parsed.scheme == "https" else port
        features.update(_get_ssl_features(hostname, ssl_port))
    else:
        features.update({
            "has_ssl": 0, "ssl_days_to_expiry": -1,
            "ssl_valid": 0, "ssl_self_signed": 1, "ssl_issuer_trusted": 0,
        })

    return features


def get_feature_names() -> list[str]:
    """Return canonical domain feature names."""
    return [
        "hostname_is_ip",
        "domain_age_days", "days_to_expiry", "registrar_suspicious",
        "whois_available", "is_recently_registered",
        "has_a_record", "has_mx_record", "has_ns_record", "has_txt_record",
        "num_a_records", "num_ns_records", "dns_available", "dns_private_ip",
        "has_ssl", "ssl_days_to_expiry", "ssl_valid",
        "ssl_self_signed", "ssl_issuer_trusted",
    ]
