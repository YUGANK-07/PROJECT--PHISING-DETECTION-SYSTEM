"""
api/security.py
────────────────
JWT authentication, API-key validation, and input sanitisation.

Usage
-----
    # In a route:
    @router.post("/predict")
    async def predict(req: PredictRequest, token: str = Depends(require_auth)):
        ...
"""

from __future__ import annotations

import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_bearer = HTTPBearer(auto_error=False)

# ── Hardcoded demo API keys (replace with DB lookup in production) ────────────
# Format: {api_key: username}
_API_KEYS: dict[str, str] = {
    "demo-key-phishguard-2024": "demo_user",
    "admin-key-phishguard-9999": "admin",
}


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(subject: str, expires_minutes: Optional[int] = None) -> str:
    """Create a signed JWT access token.

    Parameters
    ----------
    subject:
        Username / user ID to embed in the token.
    expires_minutes:
        Token lifetime.  Defaults to ``settings.JWT_EXPIRE_MINUTES``.

    Returns
    -------
    str
        Encoded JWT string.
    """
    try:
        from jose import jwt
    except ImportError:
        raise ImportError("python-jose required: pip install python-jose[cryptography]")

    expire = datetime.now(tz=timezone.utc) + timedelta(
        minutes=expires_minutes or settings.JWT_EXPIRE_MINUTES
    )
    payload = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.now(tz=timezone.utc),
        "jti": secrets.token_hex(8),
    }
    return jwt.encode(payload, settings.APP_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token.

    Returns
    -------
    dict or None
        Decoded payload, or None if invalid/expired.
    """
    try:
        from jose import jwt, JWTError
        payload = jwt.decode(
            token,
            settings.APP_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except Exception:
        return None


# ── Dependency ────────────────────────────────────────────────────────────────

async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> str:
    """FastAPI dependency that enforces authentication.

    Accepts either:
    - ``Authorization: Bearer <jwt_token>``
    - ``Authorization: Bearer <api_key>``

    Returns
    -------
    str
        The authenticated username.

    Raises
    ------
    HTTPException 401
        If no valid credential is provided.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Include: Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # ── Try API key first (fast path) ─────────────────────────────────────────
    if token in _API_KEYS:
        return _API_KEYS[token]

    # ── Try JWT ───────────────────────────────────────────────────────────────
    payload = decode_access_token(token)
    if payload and "sub" in payload:
        return payload["sub"]

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ── Optional auth (for health / metrics endpoints) ────────────────────────────

async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> Optional[str]:
    """Like ``require_auth`` but returns None instead of raising for unauthenticated requests."""
    if credentials is None:
        return None
    token = credentials.credentials
    if token in _API_KEYS:
        return _API_KEYS[token]
    payload = decode_access_token(token)
    if payload:
        return payload.get("sub")
    return None
