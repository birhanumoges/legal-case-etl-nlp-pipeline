"""api/dependencies.py — FastAPI dependency injection: auth, DB session, pagination."""

from __future__ import annotations
import os
from typing import Generator
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from utils.logger import get_logger

logger = get_logger(__name__)

# ── JWT config ────────────────────────────────────────────────────
SECRET_KEY  = os.getenv("SECRET_KEY", "legal-nlp-secret-key-change-in-production")
ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Simple in-memory user store (replace with DB in production)
USERS_DB: dict[str, str] = {
    "admin": "admin123",
    "analyst": "analyst123",
}

bearer_scheme = HTTPBearer(auto_error=False)


# ── Token helpers ─────────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")


# ── Auth dependency ───────────────────────────────────────────────
def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = verify_token(credentials.credentials)
    return payload.get("sub", "unknown")


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str | None:
    """Returns None instead of raising if no token — for public routes."""
    if credentials is None:
        return None
    try:
        payload = verify_token(credentials.credentials)
        return payload.get("sub")
    except HTTPException:
        return None


# ── Pagination dependency ─────────────────────────────────────────
class Pagination:
    def __init__(self, page: int = 1, size: int = 20):
        self.page   = max(1, page)
        self.size   = min(max(1, size), 100)
        self.offset = (self.page - 1) * self.size


def get_pagination(page: int = 1, size: int = 20) -> Pagination:
    return Pagination(page=page, size=size)
